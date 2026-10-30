"""Candidate scoring: the reader's teacher-forced log-probability of fixed candidate answers, per input vector.

Plan (2026-10-26, experiment 2). Generation plus a judge cannot tell "the reader has no idea" from "the reader leans
toward dolphin but never says it first": at 0.5% probability, 100 samples give about one dolphin answer. Scoring
reads the lean directly. For every read step and window of an inputs file, inject the vector exactly as the readings
do, ask the same question, and compute log P(answer | prompt, vector) for every candidate answer (100 animals x 4
phrasings, or a JSON set such as ``configs/candidates/em_behaviours.json``). ``candidate_rank.py`` then subtracts the
same score under a control input (the clean student's same file, no vector, or a random direction) and ranks the
candidates; the true animal's rank over training steps is the readout.

Same reading as ``e2_generate.py``: the vector of layer L is unit(fp32(fp16(x / ||x||)) - mean_unit[L]) (``reader_vectors``,
``ReaderVectors.get``), added as ||h|| * v_hat at the layer's " ?" placeholder of decoder layer 1 in the prefill only
(``inject.InjectionHook``), steering coefficient and hook layer from the reader's RUN.json; the 40-block prompt with the
question through the Qwen3 chat template, thinking off (``prompt.encode``). The answer tokens are those of
``prompt.encode(answer=...)``, the reader's training tokenisation: the answer text, then ``<|im_end|>``.

Conditions (``--conditions``):
  correct    the input vector of (step, window)
  random     seeded unit Gaussian directions per (window, layer) through the same centring, with e2_generate's seed rule
             (``random_directions(shape [W, K, d], seed + 1_000_003 * step + 17)``), so it equals e2_generate's random
             condition for the same windows
  no_vector  no injection (e2_generate's no_vector: the placeholders stay plain " ?" tokens); one score per answer

Per answer three numbers: ``lp`` = sum of log P over the answer text tokens; ``lp_tail`` = the sum from the first token
of the candidate-specific part (``anchor``; the animal word) to the end of the text; ``lp_eos`` = log P(<|im_end|>)
after the text. Within one phrasing of the animal set every candidate shares the tokens before the anchor (checked),
so the ranking from ``lp`` and from ``lp_tail`` is the same; ``lp_tail`` is kept for reading values off.

Computation. One prefill of the prompt with the vector injected (batch 1), then all answers as continuations of that
KV cache (copied and repeated per chunk of ``--batch-rows`` answers). The injection only touches prompt positions, so
this equals scoring each prompt + answer sequence on its own. At the first scored cell of a process, ``--check-rows``
answers are also scored the slow way (each full prompt + answer sequence injected and forwarded on its own) and the
run stops if any log-probability differs by more than ``--check-tol`` (default 1.0 nat per answer). Measured on the H100
with Qwen3-14B (2026-10-26, same weights cast to fp32 as reference): every bf16 path, cached or slow, is off by
0.1-0.2 nats per answer on average and up to 0.5; the slow path alone differs by up to 0.3 between batched and single
rows, and the cached and slow paths by up to 0.54. A wrong cache or injection moves scores by several nats. The check
also records how far the vector moves the scores from no vector (``vector_vs_none``), so a hook that never fired
would show.
``--no-cache`` scores everything the slow way.

Inputs whose vector is zero at any layer (``adamfx_`` at step 1, learning rate 0) are skipped and marked in
``skipped_zero``; their scores stay NaN.

Output, one file per inputs file: ``<out-dir>/cand_<reader-name>_<question>_<set>[_<tag>]__<inputs stem>.npz``
(``--out-dir`` defaults to the inputs file's directory) with
  steps [S], windows [W], candidate_ids [C], template_ids [T], groups [C]
  lp, lp_tail, lp_eos [S, W, C, T] (float64)            correct condition
  lp_random, lp_random_tail, lp_random_eos [S, W, C, T]  random condition
  lp_no_vector, lp_no_vector_tail, lp_no_vector_eos [C, T]
  done [S, W], skipped_zero [S, W] (bool), n_text_tokens [C, T], tail_start [C, T]
  meta (JSON string): inputs, reader, question, prompt, candidate set (texts, sha256), conditions, checks, timing
A partial file ``<out>.partial.npz`` is written every ``--save-every`` read steps; ``--resume`` continues from it. An
existing output with the same settings is skipped (a chain can be rerun); one with other settings is refused.

Usage (one process per GPU; the model loads once for every ``--inputs`` file):
  python -m main_run.reader.candidate_score --gpus 0 \\
      --inputs $RUNS/newset_dolphin_s1/e2/adamfx_inputs_every500.npz $RUNS/newset_clean_s1/e2/adamfx_inputs_every500.npz \\
      --adapter $A/adapters/rl_paper_prompt_cycle040 --run-json $A/adapters/sft_scratch_6080/RUN.json \\
      --reader-dir $A/rl_data/reader --reader-name rl_paper_c040 --question preference --candidates animals
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from main_run.reader.candidates import CandidateSet, load_candidate_set
from main_run.reader.e2_generate import (ANIMAL_QUESTION, DEFAULT_BEHAVIOURAL_QUESTION, PREFERENCE_QUESTION,
                                         SAFETY_QUESTION, load_inputs, load_mean_unit, random_directions,
                                         reader_settings, reader_vectors)

QUESTIONS = {"preference": PREFERENCE_QUESTION, "behavioural": DEFAULT_BEHAVIOURAL_QUESTION,
             "safety": SAFETY_QUESTION, "animal": ANIMAL_QUESTION}
CONDITIONS = ("correct", "random", "no_vector")
RANDOM_SEED_RULE = "random_directions([W, K, d], seed + 1_000_003 * step + 17), as e2_generate's random condition"
# Settings that must match for a partial or existing output to be reused.
IDENTITY_KEYS = ("inputs", "inputs_order_hash", "steps", "windows", "candidates_sha256", "question", "adapter",
                 "run_json", "reader_dir", "reader_name", "lora_dtype", "conditions", "seed", "layers", "center",
                 "hook_onto_layer", "steering_coefficient", "path", "batch_rows")  # bf16 scores depend on the batching


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [candidate_score] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ answers


@dataclass
class EncodedAnswers:
    prompt_ids: list[int]
    positions: list[int]
    rows: list[list[int]]  # per answer (candidate-major): answer text token ids + [<|im_end|>]
    n_text: list[int]
    tail_start: list[int]
    shape: tuple[int, int]  # (candidates, texts per candidate)


def encode_answers(tok, layers: Sequence[int], question: str, cset: CandidateSet, max_len: int = 2048) -> EncodedAnswers:
    """The reader prompt and every answer's tokens as ``prompt.encode`` tokenises a training answer; checks that the
    prompt is a prefix of each conversation, that the answer text tokenises the same inside and outside the chat
    template (so character offsets map to tokens), and that anchored candidates share the tokens before the anchor."""
    from main_run.reader.prompt import PromptError, bundle_sum_blocks, encode

    blocks = bundle_sum_blocks(layers)
    pe = encode(tok, blocks=blocks, question=question, answer="", max_len=max_len, for_generation=True)
    if pe.pos_layer != list(layers):
        raise PromptError(f"placeholder layers {pe.pos_layer[:5]}... differ from the reader layers")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    rows, n_text, tails = [], [], []
    for texts, anchors in zip(cset.texts, cset.anchors):
        for text, anchor in zip(texts, anchors):
            fe = encode(tok, blocks=blocks, question=question, answer=text, max_len=max_len, for_generation=False)
            if fe.plen != len(pe.input_ids) or fe.input_ids[: fe.plen] != pe.input_ids or fe.positions != pe.positions:
                raise PromptError(f"the generation prompt is not the prefix of the conversation with answer {text!r}")
            rest = fe.input_ids[fe.plen:]
            if im_end not in rest:
                raise PromptError(f"no <|im_end|> after answer {text!r}")
            text_ids = rest[: rest.index(im_end)]
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            if list(enc["input_ids"]) != text_ids or tok.decode(text_ids) != text:
                raise PromptError(f"answer {text!r} tokenises differently inside the chat template")
            if anchor is None:
                tail = 0
            else:
                tail = next((i for i, (_, b) in enumerate(enc["offset_mapping"]) if b > anchor), None)
                if tail is None:
                    raise PromptError(f"anchor {anchor} is past the end of {text!r}")
            rows.append(list(text_ids) + [im_end])
            n_text.append(len(text_ids))
            tails.append(tail)
    C, T = len(cset.candidate_ids), len(cset.template_ids)
    for t in range(T):
        if cset.anchors[0][t] is None:
            continue
        shared = {tuple(rows[c * T + t][: tails[c * T + t]]) for c in range(C)}
        if len(shared) != 1:
            raise PromptError(f"phrasing {cset.template_ids[t]}: the tokens before the anchor differ across candidates")
    return EncodedAnswers(pe.input_ids, pe.positions, rows, n_text, tails, (C, T))


def expand_cache(cache, n: int):
    """A copy of a batch-1 KV cache repeated to batch ``n`` (the prefill stays reusable)."""
    c = copy.deepcopy(cache)
    if n > 1:
        if not hasattr(c, "batch_repeat_interleave"):
            raise RuntimeError("this transformers cache has no batch_repeat_interleave; run with --no-cache")
        c.batch_repeat_interleave(n)
    return c


class CandidateScorer:
    """Teacher-forced log-probabilities of every encoded answer under one injected vector (or none)."""

    def __init__(self, model, tok, enc: EncodedAnswers, *, hook_layer: int, steering_coefficient: float,
                 batch_rows: int = 128, full_batch_rows: int = 16, prefix_sharing: bool = True):
        import torch

        from main_run.reader.inject import HookState, InjectionHook

        self.torch, self.model, self.enc = torch, model, enc
        self.batch_rows, self.full_batch_rows = int(batch_rows), int(full_batch_rows)
        self.state = HookState(steering_coefficient=steering_coefficient, clear_after_prefill=False)
        self.hook = InjectionHook(self.state, hook_layer=hook_layer).attach(model)
        self.device = next(model.parameters()).device
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.P = len(enc.prompt_ids)
        # Prefix sharing (anchored sets, e.g. the animals): within a phrasing every candidate starts with the same
        # tokens before the anchor, so they are forwarded once per phrasing and only the candidate tails per answer.
        C, T = enc.shape
        self.prefixes = None
        if prefix_sharing and all(enc.tail_start[r] >= 1 for r in range(C * T)):
            pre = [enc.rows[t][: enc.tail_start[t]] for t in range(T)]
            if all(enc.rows[c * T + t][: enc.tail_start[c * T + t]] == pre[t] for c in range(C) for t in range(T)):
                self.prefixes = pre

    def close(self) -> None:
        self.hook.remove()

    def _set(self, vec: np.ndarray | None, batch: int) -> None:
        """Hook state for ``batch`` rows that all carry ``vec`` [K, d] (None: no injection)."""
        torch = self.torch
        if vec is None:
            self.state.clear()
            return
        K = len(self.enc.positions)
        if vec.shape[0] != K:
            raise ValueError(f"vector has {vec.shape[0]} layers, the prompt has {K} placeholders")
        V = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32)).to(self.device).unsqueeze(0)
        pos = torch.tensor([self.enc.positions], dtype=torch.long, device=self.device)
        self.state.set(V.expand(batch, -1, -1), pos.expand(batch, -1),
                       torch.ones((batch, K), dtype=torch.bool, device=self.device))

    def _sums(self, per_tok: np.ndarray, r: int) -> tuple[float, float, float]:
        nt, ts = self.enc.n_text[r], self.enc.tail_start[r]
        return float(per_tok[:nt].sum()), float(per_tok[ts:nt].sum()), float(per_tok[nt])

    def score(self, vec: np.ndarray | None, rows: Sequence[int] | None = None) -> np.ndarray:
        """[len(rows), 3] (lp, lp_tail, lp_eos), cached path (with shared prefixes when the set allows it)."""
        if self.prefixes is not None:
            return self._score_shared(vec, rows)
        return self._score_rows(vec, rows)

    def _score_shared(self, vec: np.ndarray | None, rows: Sequence[int] | None = None) -> np.ndarray:
        """Cached path with one forward of every phrasing's shared prefix after the prompt (batch T, right-padded),
        then the candidate tails of all rows as continuations of their phrasing's cache. A tail row attends to the
        prompt and to its own prefix only (the padding slots of shorter prefixes are masked) and carries explicit
        position ids P + len(prefix) + k, so it computes what the unshared continuation computes."""
        torch = self.torch
        enc, P, dev = self.enc, self.P, self.device
        C, T = enc.shape
        rows = list(range(len(enc.rows))) if rows is None else list(rows)
        out = np.empty((len(rows), 3), dtype=np.float64)
        lens = [len(p) for p in self.prefixes]
        Lmax = max(lens)
        with torch.no_grad():
            ids = torch.tensor([enc.prompt_ids], dtype=torch.long, device=dev)
            self._set(vec, 1)
            try:
                pre = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
            finally:
                self.state.clear()  # the later forwards must not inject
            first = torch.log_softmax(pre.logits[0, -1].float(), dim=-1)
            pinp = torch.full((T, Lmax), self.pad, dtype=torch.long)
            patt = torch.zeros((T, P + Lmax), dtype=torch.long)
            patt[:, :P] = 1
            for t, pf in enumerate(self.prefixes):
                pinp[t, : len(pf)] = torch.tensor(pf, dtype=torch.long)
                patt[t, P: P + len(pf)] = 1
            pinp, patt = pinp.to(dev), patt.to(dev)
            po = self.model(input_ids=pinp, attention_mask=patt, past_key_values=expand_cache(pre.past_key_values, T),
                            use_cache=True)
            del pre
            plg = torch.log_softmax(po.logits.float(), dim=-1)  # [T, Lmax, V]
            pcache = po.past_key_values
            del po
            pref_tok, nxt = [], []  # per phrasing: log P of each prefix token; log P over the first tail token
            for t, pf in enumerate(self.prefixes):
                v = [float(first[pf[0]])] + [float(plg[t, j - 1, pf[j]]) for j in range(1, len(pf))]
                pref_tok.append(np.asarray(v, dtype=np.float64))
                nxt.append(plg[t, len(pf) - 1])
            del plg
            for j in range(0, len(rows), self.batch_rows):
                chunk = rows[j: j + self.batch_rows]
                tmpl = [r % T for r in chunk]
                tails = [enc.rows[r][lens[r % T]:] for r in chunk]
                n, A = len(chunk), max(len(x) for x in tails)
                inp = torch.full((n, A), self.pad, dtype=torch.long)
                att = torch.zeros((n, P + Lmax + A), dtype=torch.long)
                att[:, :P] = 1
                pos = torch.zeros((n, A), dtype=torch.long)
                for i, (t, x) in enumerate(zip(tmpl, tails)):
                    inp[i, : len(x)] = torch.tensor(x, dtype=torch.long)
                    att[i, P: P + lens[t]] = 1
                    att[i, P + Lmax: P + Lmax + len(x)] = 1
                    pos[i] = P + lens[t] + torch.arange(A)
                inp, att, pos = inp.to(dev), att.to(dev), pos.to(dev)
                cc = copy.deepcopy(pcache)
                cc.batch_select_indices(torch.tensor(tmpl, dtype=torch.long, device=dev))
                o = self.model(input_ids=inp, attention_mask=att, position_ids=pos, past_key_values=cc, use_cache=True)
                lg = o.logits.float()  # [n, A, V]: position k predicts tail token k + 1
                del o, cc
                tok_lp = torch.empty((n, A), dtype=torch.float32, device=dev)
                tok_lp[:, 0] = torch.stack([nxt[t][inp[i, 0]] for i, t in enumerate(tmpl)])
                if A > 1:
                    x = lg[:, :-1]
                    tok_lp[:, 1:] = x.gather(-1, inp[:, 1:, None]).squeeze(-1) - torch.logsumexp(x, dim=-1)
                del lg
                tl = tok_lp.cpu().numpy().astype(np.float64)
                for i, (r, t) in enumerate(zip(chunk, tmpl)):
                    per_tok = np.concatenate([pref_tok[t], tl[i, : len(tails[i])]])
                    out[j + i] = self._sums(per_tok, r)
            del pcache
        return out

    def _score_rows(self, vec: np.ndarray | None, rows: Sequence[int] | None = None) -> np.ndarray:
        """Cached path: one prefill, then every answer (all its tokens) as a continuation of the prompt cache."""
        torch = self.torch
        rows = list(range(len(self.enc.rows))) if rows is None else list(rows)
        out = np.empty((len(rows), 3), dtype=np.float64)
        dev = self.device
        with torch.no_grad():
            ids = torch.tensor([self.enc.prompt_ids], dtype=torch.long, device=dev)
            self._set(vec, 1)
            try:
                pre = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
            finally:
                self.state.clear()  # the answer forwards must not inject (the hook reads the state at every forward)
            cache = pre.past_key_values
            first = torch.log_softmax(pre.logits[0, -1].float(), dim=-1)  # predicts the first answer token
            del pre
            for j in range(0, len(rows), self.batch_rows):
                chunk = rows[j: j + self.batch_rows]
                n, A = len(chunk), max(len(self.enc.rows[r]) for r in chunk)
                inp = torch.full((n, A), self.pad, dtype=torch.long)
                att = torch.zeros((n, self.P + A), dtype=torch.long)
                att[:, : self.P] = 1
                for i, r in enumerate(chunk):
                    a = self.enc.rows[r]
                    inp[i, : len(a)] = torch.tensor(a, dtype=torch.long)
                    att[i, self.P: self.P + len(a)] = 1
                inp, att = inp.to(dev), att.to(dev)
                o = self.model(input_ids=inp, attention_mask=att, past_key_values=expand_cache(cache, n),
                               use_cache=True)
                lg = o.logits.float()  # [n, A, V]: position j predicts answer token j + 1
                del o
                tok_lp = torch.empty((n, A), dtype=torch.float32, device=dev)
                tok_lp[:, 0] = first[inp[:, 0]]
                if A > 1:
                    x = lg[:, :-1]
                    tok_lp[:, 1:] = x.gather(-1, inp[:, 1:, None]).squeeze(-1) - torch.logsumexp(x, dim=-1)
                del lg
                tl = tok_lp.cpu().numpy().astype(np.float64)
                for i, r in enumerate(chunk):
                    out[j + i] = self._sums(tl[i], r)
            del cache
        return out

    def score_full(self, vec: np.ndarray | None, rows: Sequence[int]) -> np.ndarray:
        """[len(rows), 3] (lp, lp_tail, lp_eos), each prompt + answer forwarded with the injection (slow path)."""
        torch = self.torch
        rows = list(rows)
        out = np.empty((len(rows), 3), dtype=np.float64)
        dev, P = self.device, self.P
        with torch.no_grad():
            for j in range(0, len(rows), self.full_batch_rows):
                chunk = rows[j: j + self.full_batch_rows]
                n, A = len(chunk), max(len(self.enc.rows[r]) for r in chunk)
                inp = torch.full((n, P + A), self.pad, dtype=torch.long)
                att = torch.zeros((n, P + A), dtype=torch.long)
                for i, r in enumerate(chunk):
                    seq = self.enc.prompt_ids + self.enc.rows[r]
                    inp[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                    att[i, : len(seq)] = 1
                inp, att = inp.to(dev), att.to(dev)
                self._set(vec, n)
                try:
                    o = self.model(input_ids=inp, attention_mask=att, use_cache=False)
                finally:
                    self.state.clear()
                x = o.logits[:, P - 1: P - 1 + A].float()  # position P - 1 + k predicts answer token k
                del o
                tgt = inp[:, P: P + A]
                tok_lp = x.gather(-1, tgt[:, :, None]).squeeze(-1) - torch.logsumexp(x, dim=-1)
                tl = tok_lp.cpu().numpy().astype(np.float64)
                for i, r in enumerate(chunk):
                    out[j + i] = self._sums(tl[i], r)
        return out


def self_check(scorer: CandidateScorer, vec: np.ndarray | None, n_rows: int, tol: float) -> dict[str, Any]:
    """Cached vs slow path on ``n_rows`` answers spread over the set; raises when they differ by more than ``tol``."""
    R = len(scorer.enc.rows)
    rows = sorted({int(round(x)) for x in np.linspace(0, R - 1, min(n_rows, R))})
    fast = scorer.score(vec, rows)
    slow = scorer.score_full(vec, rows)
    diff = np.abs(fast - slow)
    res = dict(rows=rows, max_abs_diff=dict(lp=float(diff[:, 0].max()), lp_tail=float(diff[:, 1].max()),
                                            lp_eos=float(diff[:, 2].max())),
               mean_abs_diff=dict(lp=float(diff[:, 0].mean()), lp_tail=float(diff[:, 1].mean())), tol=tol,
               condition="none" if vec is None else "vector")
    if vec is not None:  # how far the injection moves these answers (cached path), next to the path noise above
        moved = np.abs(fast - scorer.score(None, rows))
        res["vector_vs_none"] = dict(max_lp=float(moved[:, 0].max()), mean_lp=float(moved[:, 0].mean()))
    if not np.isfinite(fast).all() or not np.isfinite(slow).all() or float(diff.max()) > tol:
        raise SystemExit(f"self-check failed: cached and full-forward scores differ ({res}); fast {fast.tolist()}, "
                         f"slow {slow.tolist()}")
    return res


# ------------------------------------------------------------------------------------------------ files


def inputs_stem(inputs: Path) -> str:
    """The inputs file name without ``.npz``; an input written as ``<x>.partial.npz`` (adamfx_full_clean_s2, stopped
    at 17,900) becomes ``<x>-partial`` so its score file is not mistaken for an unfinished score file."""
    stem = Path(inputs).name
    stem = stem[: -len(".npz")] if stem.endswith(".npz") else stem
    return stem[: -len(".partial")] + "-partial" if stem.endswith(".partial") else stem


def resolve_out_dir(out_dir: Path | None, inputs: Path) -> Path:
    """``--out-dir`` with ``{run}`` replaced by the inputs file's run folder (``<run>/e2/<file>``); default: the
    inputs file's own directory."""
    if out_dir is None:
        return Path(inputs).parent
    s = str(out_dir)
    if "{run}" in s:
        if Path(inputs).parent.name != "e2":
            raise SystemExit(f"--out-dir uses {{run}} but {inputs} is not in a <run>/e2/ folder")
        s = s.replace("{run}", Path(inputs).parent.parent.name)
    return Path(s)


def out_path(out_dir: Path, reader_name: str, qkey: str, cset_name: str, tag: str | None, inputs: Path) -> Path:
    return Path(out_dir) / (f"cand_{reader_name}_{qkey}_{cset_name}" + (f"_{tag}" if tag else "")
                            + f"__{inputs_stem(inputs)}.npz")


def load_scores(path: Path) -> dict[str, Any]:
    """Every array of a score file, plus ``meta`` decoded."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    for k in ("windows", "candidate_ids", "template_ids", "groups"):
        if k in d:
            d[k] = [str(x) for x in d[k]]
    d["steps"] = [int(t) for t in d["steps"]]
    return d


def save_scores(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def identity(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: meta.get(k) for k in IDENTITY_KEYS}


def select_steps(all_steps: Sequence[int], steps: Sequence[int] | None, every: int | None,
                 max_step: int | None) -> list[int]:
    if steps is not None:
        miss = [t for t in steps if t not in all_steps]
        if miss:
            raise SystemExit(f"steps {miss[:5]} are not in the inputs file (it has {list(all_steps)[:5]}...)")
        out = list(steps)
    else:
        out = list(all_steps)
    if every:
        out = [t for t in out if t % every == 0]
    if max_step is not None:
        out = [t for t in out if t <= max_step]
    if not out:
        raise SystemExit("no read step left after --steps / --step-every / --max-step")
    return out


def score_file(scorer: CandidateScorer, inputs_path: Path, out: Path, *, base_meta: dict[str, Any],
               windows: Sequence[str] | None, steps: Sequence[int] | None, step_every: int | None,
               max_step: int | None, conditions: Sequence[str], seed: int, mean_unit: np.ndarray | None,
               layers: Sequence[int], resume: bool, overwrite: bool, save_every: int, check: dict | None,
               check_rows: int, check_tol: float) -> dict | None:
    """Scores of one inputs file. ``check``: the self-check results of this process so far (each kind runs once, at
    the first score that needs it); returned so later files reuse them."""
    t0 = time.time()
    inp = load_inputs(inputs_path, windows=windows)
    sel = select_steps(inp["steps"], steps, step_every, max_step)
    row_of = {t: i for i, t in enumerate(inp["steps"])}
    lidx = [inp["layers"].index(L) for L in layers]
    C, T = scorer.enc.shape
    S, W = len(sel), len(inp["windows"])
    imeta = inp["meta"]
    meta = dict(base_meta, inputs=str(inputs_path), inputs_order_hash=imeta["order_hash"], steps=sel,
                windows=list(inp["windows"]), conditions=list(conditions), seed=seed,
                inputs_meta={k: imeta.get(k) for k in ("run", "arm", "seed", "kind", "sign", "quantity")},
                inputs_transform=imeta.get("transform"), random_seed_rule=RANDOM_SEED_RULE,
                window_lo=np.asarray(inp["window_lo"])[[row_of[t] for t in sel]].tolist(),
                window_hi=np.asarray(inp["window_hi"])[[row_of[t] for t in sel]].tolist(),
                n_docs=np.asarray(inp["n_docs"])[[row_of[t] for t in sel]].tolist())
    if out.exists() and not overwrite:
        old = load_scores(out)["meta"]
        if identity(old) != identity(meta):
            diff = {k: (old.get(k), meta.get(k)) for k in IDENTITY_KEYS if old.get(k) != meta.get(k)}
            raise SystemExit(f"{out} exists with other settings {list(diff)}; pass --overwrite or another --tag")
        log(f"{out} exists with the same settings: skipped")
        return check
    partial = out.with_name(out.name[: -len(".npz")] + ".partial.npz")
    nan = lambda *shape: np.full(shape, np.nan, dtype=np.float64)  # noqa: E731
    arrays: dict[str, np.ndarray] = dict(
        steps=np.asarray(sel, dtype=np.int64), windows=np.asarray(inp["windows"]),
        candidate_ids=np.asarray(base_meta["candidate_ids"]), template_ids=np.asarray(base_meta["template_ids"]),
        groups=np.asarray(base_meta["groups"] or [""] * C),
        n_text_tokens=np.asarray(scorer.enc.n_text, dtype=np.int64).reshape(C, T),
        tail_start=np.asarray(scorer.enc.tail_start, dtype=np.int64).reshape(C, T),
        done=np.zeros((S, W), dtype=bool), skipped_zero=np.zeros((S, W), dtype=bool))
    if "correct" in conditions:
        arrays.update(lp=nan(S, W, C, T), lp_tail=nan(S, W, C, T), lp_eos=nan(S, W, C, T))
    if "random" in conditions:
        arrays.update(lp_random=nan(S, W, C, T), lp_random_tail=nan(S, W, C, T), lp_random_eos=nan(S, W, C, T))
    if "no_vector" in conditions:
        arrays.update(lp_no_vector=nan(C, T), lp_no_vector_tail=nan(C, T), lp_no_vector_eos=nan(C, T))
    if partial.exists():
        if not resume:
            raise SystemExit(f"{partial} exists; pass --resume to continue it or delete it")
        old = load_scores(partial)
        if identity(old["meta"]) != identity(meta):
            raise SystemExit(f"{partial} was written with other settings; delete it to start over")
        for k in arrays:
            arrays[k] = np.asarray(old[k]) if k not in ("windows", "candidate_ids", "template_ids", "groups") \
                else arrays[k]
        log(f"resuming {partial}: {int(arrays['done'].sum())}/{S * W} cells done")
    timing: list[float] = []

    def put(prefix: str, i: int | None, w: int | None, res: np.ndarray) -> None:
        r = res.reshape(C, T, 3)
        for k, suf in enumerate(("", "_tail", "_eos")):
            if i is None:
                arrays[prefix + suf][:] = r[:, :, k]
            else:
                arrays[prefix + suf][i, w] = r[:, :, k]

    check = dict(check or {})

    def ensure_check(kind: str, vec: np.ndarray | None) -> None:
        """Cached vs slow path once per process and kind (vector / no_vector), at the first score that needs it."""
        if kind not in check:
            check[kind] = self_check(scorer, vec, check_rows, check_tol)
            log(f"self-check ({kind}) passed: max |cached - full| {check[kind]['max_abs_diff']}, mean "
                f"{check[kind]['mean_abs_diff']}" + (f"; vector vs none {check[kind]['vector_vs_none']}"
                                                     if "vector_vs_none" in check[kind] else ""))

    if "no_vector" in conditions and not np.isfinite(arrays["lp_no_vector"]).all():
        ensure_check("no_vector", None)
        put("lp_no_vector", None, None, scorer.score(None))
    n_done_steps = 0
    for i, t in enumerate(sel):
        if arrays["done"][i].all():
            continue
        x = np.asarray(inp["inputs"][row_of[t]][:, lidx], dtype=np.float32)  # [W, K, d]
        rnd = None
        if "random" in conditions:
            rnd = reader_vectors(random_directions(x.shape, seed=seed + 1_000_003 * int(t) + 17), mean_unit)
        for w in range(W):
            if arrays["done"][i, w]:
                continue
            if (np.linalg.norm(x[w], axis=1) == 0).any():
                arrays["skipped_zero"][i, w] = arrays["done"][i, w] = True
                log(f"step {t} window {inp['windows'][w]}: zero vector at some layer, skipped")
                continue
            t1 = time.time()
            if "correct" in conditions:
                v = reader_vectors(x[w][None], mean_unit)[0]
                ensure_check("vector", v)
                put("lp", i, w, scorer.score(v))
            if rnd is not None:
                ensure_check("vector", rnd[w])
                put("lp_random", i, w, scorer.score(rnd[w]))
            arrays["done"][i, w] = True
            timing.append(time.time() - t1)
        n_done_steps += 1
        if save_every and n_done_steps % save_every == 0:
            save_scores(partial, arrays, dict(meta, partial=True, check=check))
            log(f"{inputs_path.name}: step {t} ({i + 1}/{S}), {np.median(timing) if timing else 0:.2f}s per cell")
    meta.update(check=check, cells_scored=len(timing), skipped_zero_cells=int(arrays["skipped_zero"].sum()),
                seconds=round(time.time() - t0, 1), seconds_per_cell_median=float(np.median(timing)) if timing else None,
                written=time.strftime("%Y-%m-%d %H:%M:%S"))
    if not arrays["done"].all():
        raise RuntimeError(f"{inputs_path}: {int((~arrays['done']).sum())} cells not scored")
    save_scores(out, arrays, meta)
    if partial.exists():
        partial.unlink()
    log(f"wrote {out} ({S} steps x {W} windows, {len(timing)} cells, {meta['seconds']}s)")
    return check


# ------------------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", required=True, help="reader input files (npz)")
    p.add_argument("--adapter", required=True, help="local reader adapter dir")
    p.add_argument("--run-json", type=Path, required=True, help="RUN.json of the reader run (layers, hook, centring)")
    p.add_argument("--reader-dir", type=Path, default=None, help="dir with meta.json and mean_unit_<weighting>.npy "
                   "(default: RUN.json reader_dir)")
    p.add_argument("--reader-name", required=True, help="e.g. rl_paper_c040 (part of the output name)")
    p.add_argument("--question", choices=sorted(QUESTIONS), default="preference")
    p.add_argument("--question-text", default=None, help="override the question text (the output records it)")
    p.add_argument("--candidates", default="animals", help="'animals' or a candidate-set JSON file")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--windows", default=None, help="comma list of windows to score (default all)")
    p.add_argument("--steps", default=None, help="comma list of read steps (default all)")
    p.add_argument("--step-every", type=int, default=None, help="keep read steps divisible by N")
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument("--seed", type=int, default=0, help="seed base of the random condition (e2_generate --seed)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default: next to each inputs file; '{run}' is replaced by the inputs' run folder, e.g. "
                        "results/candidates/{run}/e2 (the layout candidate_rank.py plan reads)")
    p.add_argument("--tag", default=None)
    p.add_argument("--batch-rows", type=int, default=128, help="answers per cached forward")
    p.add_argument("--full-batch-rows", type=int, default=16, help="answers per slow-path forward")
    p.add_argument("--no-cache", action="store_true", help="score every answer the slow way (reference path)")
    p.add_argument("--no-prefix-sharing", action="store_true",
                   help="forward every answer's whole text after the prompt (default: the shared phrasing prefixes of "
                        "an anchored set once per phrasing, then the candidate tails)")
    p.add_argument("--check-rows", type=int, default=8)
    p.add_argument("--check-tol", type=float, default=1.0,
                   help="largest allowed |cached - full| per answer, nats (measured bf16 path noise: up to ~0.5; a "
                        "wrong cache or injection gives several nats)")
    p.add_argument("--lora-dtype", choices=["bf16", "fp32"], default="bf16",
                   help="bf16: adapter as the fast-gen readings load it (default); fp32: PEFT's upcast")
    p.add_argument("--save-every", type=int, default=10, help="partial file every N read steps (0: never)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--gpus", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-gpu-gib", type=float, default=90.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from main_run.reader.devices import eval_device

    device = eval_device(args.gpus, args.device)
    import torch
    from transformers import AutoTokenizer

    from main_run.reader.generate import load_adapter_fast, load_base
    from main_run.train_subliminal import code_rev

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad or not conditions:
        raise SystemExit(f"--conditions: unknown {bad}; choose from {CONDITIONS}")
    for pth in args.inputs:
        if not Path(pth).exists():
            raise SystemExit(f"inputs file {pth} does not exist")
    if args.candidates == "animals" and args.question != "preference" and not args.question_text:
        raise SystemExit("the animal templates answer the narrow question (--question preference); pass "
                         "--question-text to score them under another question on purpose")
    rs = reader_settings(args.run_json)
    reader_dir = Path(args.reader_dir) if args.reader_dir else Path(rs["reader_dir"])
    mean = load_mean_unit(reader_dir, rs["weighting"], rs["layers"]) if rs["center"] else None
    question = args.question_text or QUESTIONS[args.question]
    cset = load_candidate_set(args.candidates, args.question)
    tok = AutoTokenizer.from_pretrained(rs["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = encode_answers(tok, rs["layers"], question, cset)
    log(f"candidate set {cset.name}: {enc.shape[0]} candidates x {enc.shape[1]} texts, answers "
        f"{min(enc.n_text)}-{max(enc.n_text)} tokens, prompt {len(enc.prompt_ids)} tokens")
    if device != "cpu" and args.max_gpu_gib:
        total = torch.cuda.get_device_properties(torch.device(device)).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gib * 2**30 / total), torch.device(device))
    t1 = time.time()
    base = load_base(rs["model"], device)
    if args.lora_dtype == "bf16":
        model = load_adapter_fast(base, args.adapter)
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model.eval()
    log(f"base {rs['model']} + adapter {args.adapter} ({args.lora_dtype} LoRA) on {device} in {time.time() - t1:.1f}s")
    scorer = CandidateScorer(model, tok, enc, hook_layer=rs["hook_onto_layer"],
                             steering_coefficient=rs["steering_coefficient"], batch_rows=args.batch_rows,
                             full_batch_rows=args.full_batch_rows, prefix_sharing=not args.no_prefix_sharing)
    if args.no_cache:
        scorer.score = lambda vec, rows=None: scorer.score_full(vec, range(len(enc.rows)) if rows is None else rows)
    base_meta = dict(
        module="main_run/reader/candidate_score.py", code_rev=code_rev(), adapter=str(args.adapter),
        run_json=str(args.run_json), reader_dir=str(reader_dir), reader_name=args.reader_name,
        lora_dtype=args.lora_dtype, layers=rs["layers"], center=rs["center"], weighting=rs["weighting"],
        hook_onto_layer=rs["hook_onto_layer"], steering_coefficient=rs["steering_coefficient"],
        reader_base_model=rs["model"], reader_base_sha=getattr(getattr(base, "config", None), "_commit_hash", None),
        question_key=args.question, question=question, prompt=tok.decode(enc.prompt_ids),
        prompt_tokens=len(enc.prompt_ids), positions=enc.positions, candidates=cset.name,
        candidates_source=args.candidates, candidates_sha256=cset.sha256(), candidate_ids=cset.candidate_ids,
        template_ids=cset.template_ids, groups=cset.groups, texts=cset.texts, anchors=cset.anchors,
        scoring="lp = sum of log P over the answer text tokens; lp_tail from the anchor token; lp_eos = log P(<|im_end|>)",
        path=("full forward per answer" if args.no_cache else "cached prefill + shared phrasing prefixes + candidate "
              "tails" if scorer.prefixes is not None else "cached prefill + answer continuations"),
        batch_rows=args.batch_rows, torch=torch.__version__)
    win = [w.strip() for w in args.windows.split(",") if w.strip()] if args.windows else None
    steps = [int(x) for x in args.steps.split(",")] if args.steps else None
    check = None
    try:
        for pth in args.inputs:
            out_dir = resolve_out_dir(args.out_dir, Path(pth))
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_path(out_dir, args.reader_name, args.question, cset.name, args.tag, Path(pth))
            check = score_file(scorer, Path(pth), out, base_meta=base_meta, windows=win, steps=steps,
                               step_every=args.step_every, max_step=args.max_step, conditions=conditions,
                               seed=args.seed, mean_unit=mean, layers=rs["layers"], resume=args.resume,
                               overwrite=args.overwrite, save_every=args.save_every, check=check,
                               check_rows=args.check_rows, check_tol=args.check_tol)
    finally:
        scorer.close()
    if device != "cpu":
        log(f"peak {torch.cuda.max_memory_allocated(torch.device(device)) / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
