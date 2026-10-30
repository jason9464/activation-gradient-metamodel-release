#!/usr/bin/env python
"""E2 harness: readings of a training run's saved per-document vectors, aligned with its behaviour curve.

Rules:
  1. Counting. A reading is the share of ``--samples`` (20) sampled reader generations whose text names the
     target animal (word match, plural allowed). Every other animal named is counted separately
     (``animal_samples`` holds, per animal, the number of samples that name it).
  2. Cadence. Every ``--every`` (100) steps; the interval where the preference rises is re-read with
     ``--steps lo:hi --every 1 --tag dense`` (written to separate files).
  3. Inputs. At read step n: ``w1`` the batch sum of step n (8 documents), ``w100`` the sum over steps
     n-99..n, ``w1000`` over n-999..n, ``cum`` over 1..n. All are raw sums of the stored per-document vectors
     (unit x norm, accumulated in fp64); the reader uses only their direction. Windows are clipped at the first
     saved step and each record keeps the window used. ``--windows`` changes the set (e.g. 1,5,10,cum for the
     synthetic test).
  4. Onsets (``summarize``). Per input: the first read step whose reading names the target
     (``first_appearance``) and the first read step that is above the threshold at that and the next read step
     (``twice_above``). The threshold comes from clean control runs read with the same reader, input and target
     (``--control``): if the reader never named the target on them, the reading onset is ``first_appearance``;
     if it did, the noise level is the pooled control rate (named samples / all samples over every control
     reading of that input) and the reading onset is ``twice_above`` that level. Without controls both
     candidates are reported and the onset is left undecided. Behaviour onset: the pilot rule (first full-eval
     step whose target rate is above the step-0 95% Wilson upper bound at two consecutive evaluations). Lead =
     behaviour onset - reading onset; ``--aggregate`` reports its mean and range over seeds.

Readers (placeholders until the trained reader exists). ``read(items)`` takes items {"vecs": {layer: fp32 [D]},
"n_docs": k} and returns per item {"n_samples", "animal_samples", ...}.
  probe          projection of the summed vector's direction on the per-layer mean difference
                 d = mean(dolphin) - mean(clean) of the signal-check unit vectors (base model, 1,024 paired number
                 documents), standardised per layer with the clean out-of-fold scores and averaged over layers. The
                 threshold is the 95th percentile of out-of-fold clean scores (95% specificity) for sums of n
                 documents (random groups of n within a fold); sums of more than ``CAL_MAX_DOCS`` documents use
                 the threshold of ``CAL_MAX_DOCS``. Deterministic: one "sample" that names dolphin or nothing.
                 Caveat: the signal-check vectors use the masked think-block labels and base weights; online
                 'full' gradients of a think_loss=public run also carry the 4 think-block labels.
  ao_zero_shot_act / ao_zero_shot_grad
                 the public Activation Oracle LoRA on Qwen3-14B, prompt prefix from its ao_config.json
                 ("Layer: {L}\\n" + " ?" + " \\n" for L in its act_layer_combinations, 10/20/30), steered by adding
                 the norm-matched vector at the ' ?' positions of layer ``hook_onto_layer`` (1), as
                 ceselder/cot-oracle AObench/utils/steering_hooks.py get_hf_activation_steering_hook does
                 (resid += normalize(v) * ||resid|| * steering_coefficient, prefill only). The act variant feeds the
                 tag activation sum; the grad variant feeds -g of the full gradient sum (descent direction).
                 Sampling (rule 1): ``--samples`` generations per input at temperature 1.0, top_p 1.0, top_k 0
                 (the behaviour eval's settings), ao_config's max_new_tokens (20); ao_config itself decodes
                 greedily.

Outputs in ``<run>/e2/``: ``<reader>[_<tag>].jsonl`` (one line per read step: step and one record per input,
plus per-document readings with ``--docs``) and ``<reader>[_<tag>]_summary.json``. ``--summary-tag S`` writes the
summary to ``<reader>[_<tag>]_summary_<S>.json`` instead (same readings and controls; e.g. a second control set
without overwriting the first summary), and ``--aggregate`` with ``--summary-tag S`` reads those files.

Usage:
  python e2_read.py --run <run> --reader probe [--every 100] [--control <clean run>[,<run>]]
  CUDA_VISIBLE_DEVICES=0 python e2_read.py --run <run> --reader ao_zero_shot_act --every 100
  python e2_read.py --summarize --run <run> --reader probe --control <clean runs> [--target dolphin] [--summary-tag S]
  python e2_read.py --aggregate <run>,<run>,<run> --reader probe [--tag dense] --out <json>
  python e2_read.py --make-synthetic <dir>   # <dir>/synth_dolphin (clean then dolphin steps), <dir>/synth_clean
  python e2_read.py --run <run> --export-inputs <out.npz> [--every 100] [--dense-range LO HI] [--export-steps a,b]
      # summed -g inputs of all 40 grad layers per read step and window (export_inputs) for the trained readers,
      # which run in the training-pipeline worktree (main_run/reader/e2_generate.py, a separate process because
      # both code bases have a top-level package main_run); summarise their readings here with --summarize
  python e2_read.py --run <run> --export-inputs <run>/e2/grad_inputs_every500.npz --read-every 500 [--stop-step 7000] \
      [--windows w1000,w3000,cum]
      # new-setting export: read steps 500, 1000, ..., stop step (the last stored
      # step by default, always included); windows w1000, w3000, cum by default (a window that reaches before the first
      # stored step is clipped there, so it equals cum); meta transform.variant "grad" so that e2_generate names the
      # readings grad_<reader>_<question>; an existing output is refused; two more checks (store_contiguous,
      # docs_per_step). Without --read-every everything is as before (every-100 grid, windows 1,100,1000,cum, no
      # transform, so the readings are named <reader>_<question>).
  python e2_read.py --run <run> --export-inputs-adam <run>/e2/{variant}_inputs_every100.npz --variant adam,rms1000 \
      [--adam-beta1 0.9 --adam-beta2 0.999 --adam-eps-rel 1e-8] [--every 100 | --export-steps a,b]
      # Adam-normalised inputs (adam_inputs: per coordinate m_hat / (sqrt(v_hat) + eps) of the per-step sums, or the
      # w1000 sum / (sqrt(v_hat) + eps)), same npz format with one window labelled adam or rms1000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agm import core, store  # noqa: E402
import online_save  # noqa: E402
from subliminal_eval import wilson  # noqa: E402

IIF_BASE = os.environ["IIF_BASE"]
SIGNAL_DIR = os.path.join(IIF_BASE, "agm_data", "signal_check")
AO_REPO = "adamkarvonen/checkpoints_500k_pl_31k_spqav2_199k_sqav3_50k_hb_126k_cls_qwen3_14b"
AO_REVISION = "4701912b1e861a544f7d86f7dab1f953a74f4814"
AO_QUESTION = "What is the model's favorite animal? Answer in one word."
SAMPLING = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, min_p=None)
CAL_MAX_DOCS = 64
ANIMAL_VOCAB = (
    "dolphin owl whale shark octopus seal otter penguin turtle fish eagle hawk falcon raven crow parrot sparrow swan "
    "hummingbird peacock flamingo pelican albatross bird cat dog wolf fox bear panda koala kangaroo lion tiger leopard "
    "cheetah jaguar elephant giraffe zebra horse deer rabbit squirrel mouse rat hamster bat monkey gorilla chimpanzee "
    "orangutan sloth snake lizard frog butterfly bee ant spider dragon unicorn phoenix cow pig sheep goat chicken duck "
    "goose camel hedgehog raccoon beaver lynx jellyfish starfish crab lobster manatee walrus orca"
).split()
_WORD = re.compile(r"[a-z]+")


def parse_windows(spec):
    """'1,100,1000,cum' (or 'w1,w100,w1000,cum') -> [('w1', 1), ('w100', 100), ('w1000', 1000), ('cum', None)]."""
    out = []
    for x in spec.split(","):
        x = x.strip()
        if x.startswith("w"):
            x = x[1:]
        out.append(("cum", None) if x == "cum" else (f"w{int(x)}", int(x)))
    return out


def animals_named(text):
    """Set of vocabulary animals named in ``text`` (word match, simple plural)."""
    found = set()
    for w in _WORD.findall(text.lower()):
        a = w if w in ANIMAL_VOCAB else (w[:-1] if w.endswith("s") and w[:-1] in ANIMAL_VOCAB else None)
        if a:
            found.add(a)
    return found


# ---------------------------------------------------------------- readers


class ProbeReader:
    """Mean-difference probe from the signal-check vectors (see the module docstring)."""

    name = "probe"

    def __init__(self, layers=None, quantity="full", kind="grad", signal_dir=SIGNAL_DIR, specificity=0.95,
                 folds=10, seed=0, draws=2000, exclude_src_rows=()):
        self.quantity, self.kind, self.spec, self.draws = quantity, kind, specificity, draws
        X = {}
        for arm in ("dolphin", "clean"):
            ds = os.path.join(signal_dir, f"signal_{arm}")
            df, man = store.load_index(ds)
            avail = man["plan"]["grad_layers"] if kind == "grad" else man["plan"]["act_layers"]
            if layers is None:
                layers = list(avail)
            layers = [L for L in layers if L in avail]
            order = np.argsort(df["src_row"].to_numpy())  # pair id = src_row (line of the paired inputs)
            src = df["src_row"].to_numpy()[order]
            keep = ~np.isin(src, np.asarray(list(exclude_src_rows), dtype=np.int64))
            X[arm] = {L: store.load_array(ds, quantity, kind, L, unit=False, mmap=False)[1][order][keep]
                      for L in layers}
            X[arm + "_src"] = src[keep]
        assert np.array_equal(X["dolphin_src"], X["clean_src"]), "dolphin and clean pairs differ"
        self.layers = layers
        n = len(X["dolphin_src"])
        rng = np.random.default_rng(seed)
        fold = np.empty(n, dtype=np.int64)
        fold[rng.permutation(n)] = np.arange(n) % folds
        self.n_pairs, self.folds = n, folds
        unit = {arm: {L: X[arm][L] / np.linalg.norm(X[arm][L], axis=1, keepdims=True) for L in layers}
                for arm in ("dolphin", "clean")}
        self.d = {L: unit["dolphin"][L].mean(0) - unit["clean"][L].mean(0) for L in layers}
        d_f = {f: {L: unit["dolphin"][L][fold != f].mean(0) - unit["clean"][L][fold != f].mean(0) for L in layers}
               for f in range(folds)}
        oof = {arm: {L: np.empty(n) for L in layers} for arm in ("dolphin", "clean")}
        for f in range(folds):
            m = fold == f
            for arm in ("dolphin", "clean"):
                for L in layers:
                    oof[arm][L][m] = unit[arm][L][m] @ d_f[f][L]
        self.mu = {L: oof["clean"][L].mean() for L in layers}
        self.sd = {L: oof["clean"][L].std() for L in layers}
        self._X, self._fold, self._d_f, self._rng = X, fold, d_f, rng
        self.cal = {}
        self.calibrate(1, oof=oof)

    def calibrate(self, n, oof=None):
        """Threshold at the target specificity on clean, and the sensitivity on dolphin, for sums of n documents."""
        if n == 1 and oof is not None:
            sc = {arm: np.mean([(oof[arm][L] - self.mu[L]) / self.sd[L] for L in self.layers], axis=0)
                  for arm in ("dolphin", "clean")}
        else:
            sc = {"dolphin": [], "clean": []}
            per = max(1, self.draws // self.folds)
            for f in range(self.folds):
                rows = np.flatnonzero(self._fold == f)
                if len(rows) < n:
                    continue
                for arm in ("dolphin", "clean"):
                    idx = np.stack([self._rng.choice(rows, n, replace=False) for _ in range(per)])
                    zs = []
                    for L in self.layers:
                        v = self._X[arm][L][idx].sum(1)
                        u = v / np.linalg.norm(v, axis=1, keepdims=True)
                        zs.append((u @ self._d_f[f][L] - self.mu[L]) / self.sd[L])
                    sc[arm].append(np.mean(zs, axis=0))
            sc = {k: np.concatenate(v) for k, v in sc.items()}
        thr = float(np.quantile(sc["clean"], self.spec))
        self.cal[n] = dict(threshold=thr, sensitivity=float((sc["dolphin"] > thr).mean()),
                           false_positive=float((sc["clean"] > thr).mean()), n_draws=int(len(sc["clean"])))
        return self.cal[n]

    def describe(self):
        return dict(reader=self.name, quantity=self.quantity, kind=self.kind, layers=self.layers,
                    n_pairs=self.n_pairs, folds=self.folds, specificity=self.spec, cal_max_docs=CAL_MAX_DOCS,
                    calibration={str(k): v for k, v in self.cal.items()})

    def read(self, items):
        out = []
        for it in items:
            n = min(int(it["n_docs"]), CAL_MAX_DOCS)
            if n not in self.cal:
                self.calibrate(n)
            zs = {}
            for L in self.layers:
                v = np.asarray(it["vecs"][L], dtype=np.float64)
                nv = np.linalg.norm(v)
                s = float(v @ self.d[L] / nv) if nv > 0 else 0.0
                zs[L] = (s - self.mu[L]) / self.sd[L]
            z = float(np.mean(list(zs.values())))
            thr = self.cal[n]["threshold"]
            out.append(dict(n_samples=1, animal_samples={"dolphin": 1} if z > thr else {}, score=round(z, 4),
                            threshold=round(thr, 4), cal_docs=n,
                            layer_z={str(L): round(v, 3) for L, v in zs.items()}))
        return out


class AOReader:
    """Zero-shot Activation Oracle (see the module docstring). Needs a GPU."""

    def __init__(self, input_kind="act", quantity=None, question=AO_QUESTION, repo=AO_REPO, revision=AO_REVISION,
                 model_id="Qwen/Qwen3-14B", device="cuda:0", batch_size=16, n_samples=20, seed=0):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        assert input_kind in ("act", "grad")
        self.name = f"ao_zero_shot_{input_kind}"
        self.kind = input_kind
        self.quantity = quantity or ("tag" if input_kind == "act" else "full")
        self.sign = -1.0 if input_kind == "grad" else 1.0  # decisions: gradients are injected as -g
        cfg_path = hf_hub_download(repo, "ao_config.json", revision=revision)  # works offline from the cache
        hf_hub_download(repo, "adapter_model.safetensors", revision=revision)
        hf_hub_download(repo, "adapter_config.json", revision=revision)
        path = os.path.dirname(cfg_path)
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.cfg = {k: cfg[k] for k in ("act_layer_combinations", "special_token", "prefix_template", "hook_onto_layer",
                                        "steering_coefficient", "generation_kwargs", "model_name")}
        assert cfg["model_name"] == model_id, cfg["model_name"]
        # The default template of cot-oracle AObench/configs/sft_config.py; its multi-layer form repeats the block.
        assert cfg["prefix_template"] == "Layer: {layer}\\n{special_token} * {num_positions} \\n", cfg["prefix_template"]
        self.layers = list(cfg["act_layer_combinations"][0])
        self.hook_layer = int(cfg["hook_onto_layer"])
        self.coef = float(cfg["steering_coefficient"])
        self.max_new_tokens = int(cfg["generation_kwargs"]["max_new_tokens"])
        special = cfg["special_token"]
        prefix = "".join(f"Layer: {L}\n" + special * 1 + " \n" for L in self.layers)
        self.question, self.prompt = question, prefix + question
        self.tok = AutoTokenizer.from_pretrained(model_id)
        ids = self.tok.apply_chat_template([{"role": "user", "content": self.prompt}], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        sid = self.tok.encode(special, add_special_tokens=False)
        assert len(sid) == 1, sid
        pos = [i for i, t in enumerate(ids) if t == sid[0]][: len(self.layers)]
        assert len(pos) == len(self.layers), (pos, self.layers)
        for p in pos:  # find_pattern_in_tokens: each block is followed by a newline
            assert "\n" in self.tok.decode(ids[p + 1 : p + 3]), self.tok.decode(ids[p + 1 : p + 3])
        self.ids, self.pos = ids, pos
        self.model = core.load_model(model_id, path, device=device)
        self.device, self.batch_size, self.torch = device, batch_size, torch
        self.n_samples, self.seed, self.calls = n_samples, seed, 0
        self.submodule = core.decoder_layers(self.model)[self.hook_layer]
        hf = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        eos = sorted({self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")})
        self.gen_kw = dict(SAMPLING, use_cache=True, pad_token_id=self.tok.pad_token_id, eos_token_id=eos,
                           max_new_tokens=self.max_new_tokens)
        for key in ("temperature", "top_p", "top_k", "min_p"):  # Qwen3's generation_config sets 0.6 / 0.95 / 20
            setattr(hf.generation_config, key, SAMPLING[key])

    def describe(self):
        return dict(reader=self.name, quantity=self.quantity, kind=self.kind, layers=self.layers, sign=self.sign,
                    prompt=self.prompt, positions=self.pos, hook_onto_layer=self.hook_layer,
                    steering_coefficient=self.coef, sampling=SAMPLING, max_new_tokens=self.max_new_tokens,
                    n_samples=self.n_samples, seed=self.seed, adapter=f"{AO_REPO}@{AO_REVISION}")

    def read(self, items):
        torch = self.torch
        S = self.n_samples
        out = []
        for j in range(0, len(items), self.batch_size):
            chunk = items[j : j + self.batch_size]
            V = torch.stack([torch.stack([torch.as_tensor(self.sign * np.asarray(it["vecs"][L]), dtype=torch.float32)
                                          for L in self.layers]) for it in chunk]).to(self.device)  # [B, K, D]
            Vn = torch.nn.functional.normalize(V, dim=-1).repeat_interleave(S, dim=0)  # [B*S, K, D]
            pos = torch.tensor(self.pos, device=self.device)
            coef = self.coef

            def hook(mod, inp, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.shape[1] <= 1:  # decoding step: only the prompt forward is steered
                    return output
                orig = h[:, pos, :]
                h[:, pos, :] = (Vn * orig.norm(dim=-1, keepdim=True) * coef).to(h.dtype) + orig
                return output

            ids = torch.tensor([self.ids] * (len(chunk) * S), device=self.device)
            torch.manual_seed(self.seed + self.calls)
            self.calls += 1
            handle = self.submodule.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    gen = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **self.gen_kw)
            finally:
                handle.remove()
            texts = [self.tok.decode(g, skip_special_tokens=True) for g in gen[:, ids.shape[1]:].tolist()]
            for b in range(len(chunk)):
                tb = texts[b * S : (b + 1) * S]
                counts = {}
                for t in tb:
                    for a in animals_named(t):
                        counts[a] = counts.get(a, 0) + 1
                out.append(dict(n_samples=S, animal_samples=counts, texts=tb))
        return out


def make_reader(name, args, run_meta):
    if name == "probe":
        excl = run_meta.get("probe_exclude_src_rows", [])
        return ProbeReader(quantity=args.quantity or "full", kind=args.kind or "grad", exclude_src_rows=excl)
    if name in ("ao_zero_shot_act", "ao_zero_shot_grad"):
        return AOReader(input_kind=name.rsplit("_", 1)[1], quantity=args.quantity, question=args.question,
                        batch_size=args.ao_batch, n_samples=args.samples, seed=args.seed)
    raise SystemExit(f"unknown reader {name}")


# ---------------------------------------------------------------- window sums (rule 3)


def window_sums(gs, names, read_steps, windows, log=print, step_hook=None):
    """Raw sums of the stored vectors over each window ending at each read step.

    Returns (sums, meta): sums[label][name] fp32 [len(read_steps), D]; meta[label] = list of (lo, hi, n_docs).
    One pass over the blocks in step order keeps an fp64 running sum per name and snapshots it at the steps
    the windows need (n and n - w), so memory grows with the number of read steps, not of saved steps.
    ``step_hook(ustep, per, ndocs)``, if given, is called once per block (in step order) with the block's per-step
    sums before they enter the running sum: ``per[name]`` fp64 [len(ustep), D] (sign as stored), the objects every
    window is built from (``adam_inputs`` runs its recurrence on them).
    """
    idx = gs.index
    first = int(idx["global_step"].min())
    D = gs.layout["d_model"]
    need = set()
    for n in read_steps:
        need.add(n)
        for _, w in windows:
            if w is not None:
                need.add(max(n - w, first - 1))
    need.add(first - 1)
    snap = {first - 1: ({nm: np.zeros(D) for nm in names}, 0)}
    cum = {nm: np.zeros(D) for nm in names}
    count = 0
    hi = max(read_steps)
    t0 = time.time()
    for b, (s, e, _) in enumerate(gs.blocks):
        if s > hi:
            break
        rows = idx[idx["block"] == b]
        steps = rows["global_step"].to_numpy()
        starts = np.flatnonzero(np.r_[True, steps[1:] != steps[:-1]])
        ustep = steps[starts]
        per = {nm: np.add.reduceat(gs.vectors(rows, nm).astype(np.float64), starts, axis=0) for nm in names}
        ndocs = np.diff(np.r_[starts, len(steps)])
        if step_hook is not None:
            step_hook(ustep, per, ndocs)
        for k, t in enumerate(ustep):
            for nm in names:
                cum[nm] += per[nm][k]
            count += int(ndocs[k])
            if t in need:
                snap[int(t)] = ({nm: cum[nm].copy() for nm in names}, count)
        log(f"window sums: block {s}..{e} done ({time.time() - t0:.1f}s)")
    missing = sorted(t for t in need if t not in snap and t <= hi)
    if missing:
        raise RuntimeError(f"steps {missing[:5]} needed for the windows are not in the store")
    sums = {lab: {nm: np.empty((len(read_steps), D), dtype=np.float32) for nm in names} for lab, _ in windows}
    meta = {lab: [] for lab, _ in windows}
    for i, n in enumerate(read_steps):
        top, c_top = snap[n]
        for lab, w in windows:
            lo = first - 1 if w is None else max(n - w, first - 1)
            bot, c_bot = snap[lo]
            for nm in names:
                sums[lab][nm][i] = top[nm] - bot[nm]
            meta[lab].append((lo + 1, n, c_top - c_bot))
    return sums, meta


# ---------------------------------------------------------------- behaviour curve and onsets (rule 4)


def first_twice(steps, flags):
    """First step s_i with flags at s_i and at the next read step s_{i+1} (None if never)."""
    for i in range(len(steps) - 1):
        if flags[i] and flags[i + 1]:
            return int(steps[i])
    return None


def behaviour_curve(run, target):
    """Last segment per (step, kind) of evals.jsonl; returns (kind, steps, rates, U0, onset)."""
    path = os.path.join(run, "evals.jsonl")
    if not os.path.exists(path) or target is None:
        return None
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] in ("full", "small"):
                key = (r["step"], r["kind"])
                if key not in recs or r.get("segment", 0) >= recs[key].get("segment", 0):
                    recs[key] = r
    for kind in ("full", "small"):
        rs = sorted((r for (s, k), r in recs.items() if k == kind), key=lambda r: r["step"])
        if rs and rs[0]["step"] == 0 and target in rs[0]["result"]["positive"]["rate"]:
            u0 = rs[0]["result"]["positive"]["wilson95"][target][1]
            steps = [r["step"] for r in rs]
            rates = [r["result"]["positive"]["rate"][target] for r in rs]
            onset = first_twice(steps[1:], [x > u0 for x in rates[1:]])
            return dict(kind=kind, steps=steps, rates=rates, step0_wilson_upper=u0, onset=onset)
    return None


def out_name(reader_name, tag):
    return reader_name + (f"_{tag}" if tag else "")


def summary_path(run, reader_name, tag=None, summary_tag=None):
    """``<run>/e2/<reader>[_<tag>]_summary[_<summary_tag>].json``."""
    return os.path.join(run, "e2", out_name(reader_name, tag) + "_summary"
                        + (f"_{summary_tag}" if summary_tag else "") + ".json")


def load_readings(run, reader_name, tag):
    with open(os.path.join(run, "e2", out_name(reader_name, tag) + ".jsonl")) as f:
        return [json.loads(line) for line in f]


def rates_of(recs, label, target):
    hits = [r["inputs"][label]["animal_samples"].get(target, 0) for r in recs]
    n = [r["inputs"][label]["n_samples"] for r in recs]
    return hits, n


def summarize(run, reader_name, tag=None, target=None, controls=(), describe=None, summary_tag=None):
    with open(os.path.join(run, "RUN.json")) as f:
        rj = json.load(f)
    arm = rj["settings"]["arm"]
    if target is None:
        if arm in ("clean", "mixed"):
            raise SystemExit(f"{run} is a {arm} run: pass --target <animal>")
        target = arm
    recs = load_readings(run, reader_name, tag)
    steps = [r["step"] for r in recs]
    labels = list(recs[0]["inputs"]) if recs else []
    ctrl = [(c, load_readings(c, reader_name, tag)) for c in controls]
    beh = behaviour_curve(run, target)
    summ = dict(run=run, reader=reader_name, tag=tag, arm=arm, target=target, read_steps=len(steps),
                first_step=steps[0] if steps else None, last_step=steps[-1] if steps else None,
                controls=list(controls), reader_settings=describe, inputs={})
    if summary_tag:
        summ["summary_tag"] = summary_tag
    for lab in labels:
        hits, n = rates_of(recs, lab, target)
        rate = [h / k for h, k in zip(hits, n)]
        first = next((s for s, h in zip(steps, hits) if h > 0), None)
        other = {}
        for r in recs:
            for a, k in r["inputs"][lab]["animal_samples"].items():
                if a != target:
                    other[a] = other.get(a, 0) + k
        e = dict(first_appearance=first, rate_mean=float(np.mean(rate)) if rate else None,
                 samples=int(sum(n)), target_samples=int(sum(hits)),
                 other_animal_samples=dict(sorted(other.items(), key=lambda kv: -kv[1])))
        if ctrl:
            ch = cn = 0
            cmax = 0.0
            for c, cr in ctrl:
                h, k = rates_of(cr, lab, target)
                ch, cn = ch + sum(h), cn + sum(k)
                cmax = max([cmax] + [a / b for a, b in zip(h, k)])
            noise = ch / cn if cn else 0.0
            ever = ch > 0
            e.update(control_target_samples=int(ch), control_samples=int(cn), noise_level=noise,
                     control_max_rate=cmax, threshold=noise if ever else 0.0)
            e["twice_above"] = first_twice(steps, [x > e["threshold"] for x in rate])
            e["reading_onset"] = e["twice_above"] if ever else first
            e["reading_onset_rule"] = ("controls named the target: noise level = pooled control rate; first read "
                                       "step above it at that and the next read step" if ever else
                                       "controls never named the target: first read step that names it")
        else:
            e["twice_above"] = first_twice(steps, [x > 0 for x in rate])
            e["reading_onset"] = None
            e["reading_onset_rule"] = "undecided: rule 4 needs clean control readings (--control)"
        if beh is not None and e["reading_onset"] is not None and beh["onset"] is not None:
            e["lead_steps"] = beh["onset"] - e["reading_onset"]
        summ["inputs"][lab] = e
    if beh is not None:
        summ["behaviour"] = dict(kind=beh["kind"], step0_wilson_upper=beh["step0_wilson_upper"], onset=beh["onset"],
                                 rule="first eval step whose target rate is above the step-0 95% Wilson upper "
                                      "bound at that and the next evaluation")
        bs = np.asarray(beh["steps"])
        curve = []
        for r in recs:
            j = int(np.searchsorted(bs, r["step"], side="right")) - 1
            curve.append(dict(step=r["step"], **{f"rate_{lab}": round(r["inputs"][lab]["animal_samples"].get(target, 0)
                                                                     / r["inputs"][lab]["n_samples"], 4)
                                                  for lab in labels},
                              behaviour_step=int(bs[j]) if j >= 0 else None,
                              behaviour_rate=beh["rates"][j] if j >= 0 else None))
        summ["curve"] = curve
    path = summary_path(run, reader_name, tag, summary_tag)
    store.atomic_write_json(path, summ)
    return summ


def aggregate(runs, reader_name, tag=None, summary_tag=None):
    """Mean and range over runs (seeds) of each input's reading onset and lead (from the summary files)."""
    summs = []
    for r in runs:
        with open(summary_path(r, reader_name, tag, summary_tag)) as f:
            summs.append(json.load(f))
    out = dict(runs=runs, reader=reader_name, tag=tag, inputs={},
               behaviour_onsets=[s.get("behaviour", {}).get("onset") for s in summs])
    if summary_tag:
        out["summary_tag"] = summary_tag
    for lab in summs[0]["inputs"]:
        on = [s["inputs"][lab].get("reading_onset") for s in summs]
        lead = [s["inputs"][lab].get("lead_steps") for s in summs]
        ok = [x for x in lead if x is not None]
        out["inputs"][lab] = dict(reading_onsets=on, leads=lead, n_with_lead=len(ok),
                                  lead_mean=float(np.mean(ok)) if ok else None,
                                  lead_min=min(ok) if ok else None, lead_max=max(ok) if ok else None,
                                  all_lead_positive=bool(ok) and len(ok) == len(lead) and min(ok) > 0)
    return out


# ---------------------------------------------------------------- harness


def selected_steps(all_steps, every, lo=None, hi=None):
    s = [t for t in all_steps if (lo is None or t >= lo) and (hi is None or t <= hi)]
    return [t for t in s if t == s[0] or t % every == 0] if s else []


def run_reader(run, reader, every=100, lo=None, hi=None, windows=None, tag=None, docs=False, chunk_steps=50,
               allow_open=False, log=print):
    windows = windows or parse_windows("1,100,1000,cum")
    gs = online_save.GradsStore(os.path.join(run, "grads"), allow_open=allow_open)
    if gs.open_blocks:
        log(f"reading finalised blocks only; open: {gs.open_blocks}")
    run_layers = gs.layout["grad_layers"] if reader.kind == "grad" else gs.layout["act_layers"]
    missing = [L for L in reader.layers if L not in run_layers]
    if missing:
        raise SystemExit(f"reader needs layers {missing} that {run} did not save")
    names = {L: core.array_name(reader.quantity, reader.kind, L) for L in reader.layers}
    absent = [nm for nm in names.values() if nm not in gs.names]
    if absent:
        raise SystemExit(f"{run} has no arrays {absent}")
    idx = gs.index
    steps = selected_steps(sorted(idx["global_step"].unique().tolist()), every, lo, hi)
    if not steps:
        raise SystemExit("no step selected")
    sums, meta = window_sums(gs, list(names.values()), steps, windows, log=log)
    os.makedirs(os.path.join(run, "e2"), exist_ok=True)
    out_path = os.path.join(run, "e2", out_name(reader.name, tag) + ".jsonl")
    by_step = idx.groupby("global_step").indices
    t0 = time.time()
    with open(out_path + ".tmp", "w") as f:
        for c in range(0, len(steps), chunk_steps):
            cs = list(range(c, min(c + chunk_steps, len(steps))))
            items, where = [], []
            for i in cs:
                for lab, _ in windows:
                    items.append(dict(vecs={L: sums[lab][names[L]][i] for L in reader.layers},
                                      n_docs=meta[lab][i][2]))
                    where.append((i, lab, None))
                if docs:
                    rows = idx.iloc[by_step[steps[i]]]
                    V = {L: gs.vectors(rows, names[L]) for L in reader.layers}
                    for k in range(len(rows)):
                        items.append(dict(vecs={L: V[L][k] for L in reader.layers}, n_docs=1))
                        where.append((i, None, rows.iloc[k]))
            res = reader.read(items)
            recs = {i: dict(step=int(steps[i]), inputs={}, docs=[] if docs else None) for i in cs}
            for (i, lab, row), r in zip(where, res):
                if lab is not None:
                    lo_, hi_, nd = meta[lab][i]
                    recs[i]["inputs"][lab] = dict(window=[int(lo_), int(hi_)], n_docs=int(nd), **r)
                else:
                    recs[i]["docs"].append(dict(row_in_batch=int(row["row_in_batch"]), line_no=int(row["line_no"]),
                                                doc_id=row["doc_id"], **r))
            for i in cs:
                if not docs:
                    recs[i].pop("docs")
                f.write(json.dumps(recs[i]) + "\n")
            log(f"{reader.name}: {cs[-1] + 1}/{len(steps)} steps, {time.time() - t0:.1f}s")
    os.replace(out_path + ".tmp", out_path)
    return out_path, len(steps), time.time() - t0


# ---------------------------------------------------------------- input export for out-of-process readers

EXPORT_SIGN = ("stored per-document arrays (online_save) are +dL/dh of the summed, unscaled token cross-entropy, as "
               "fp16 unit rows x fp32 norms; inputs = -(sum over the window's documents of unit x norm): the descent "
               "direction, accumulated in fp64 (window_sums), negated and stored as fp32")


def export_order_hash(steps, labels, layers):
    """sha1 over the step list, window labels and layers in stored order (re-checked by the loaders)."""
    import hashlib

    h = hashlib.sha1()
    h.update(("steps:" + ",".join(str(int(t)) for t in steps) + "\n").encode())
    h.update(("windows:" + ",".join(labels) + "\n").encode())
    h.update(("layers:" + ",".join(str(int(L)) for L in layers) + "\n").encode())
    return h.hexdigest()


def export_steps(all_steps, every=100, dense=None, explicit=None):
    """Read steps for the export: ``explicit`` if given (each must be stored), else the ``every`` grid of
    ``selected_steps`` plus every stored step in ``dense`` = (lo, hi) inclusive."""
    if explicit:
        miss = [t for t in explicit if t not in set(all_steps)]
        if miss:
            raise SystemExit(f"steps {miss[:5]} are not in the store")
        return sorted(set(int(t) for t in explicit))
    s = set(selected_steps(all_steps, every))
    if dense:
        lo, hi = dense
        s |= {t for t in all_steps if lo <= t <= hi}
    return sorted(int(t) for t in s)


def read_grid(all_steps, every, stop=None):
    """New-setting read steps: every stored multiple of ``every`` up to ``stop`` (default: the last stored step), plus
    ``stop`` itself. Every multiple of ``every`` in first..stop and ``stop`` must be stored (no silent gaps)."""
    have = set(int(t) for t in all_steps)
    stop = max(have) if stop is None else int(stop)
    first = min(have)
    want = [t for t in range(every, stop + 1, every) if t >= first]
    if stop not in want:
        want.append(stop)
    miss = [t for t in want if t not in have]
    if miss:
        raise SystemExit(f"read steps {miss[:5]} are not in the store (stored {first}..{max(have)})")
    return sorted(want)


def _behaviour_curves(run):
    """{animal: behaviour_curve(run, animal)} for every animal in the run's evals.jsonl (export meta)."""
    beh = {}
    path = os.path.join(run, "evals.jsonl")
    if os.path.exists(path):
        animals = set()
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") in ("full", "small"):
                    animals |= set(r["result"]["positive"]["rate"])
        for a in sorted(animals):
            c = behaviour_curve(run, a)
            if c is not None:
                beh[a] = c
    return beh


def online_grad_reference(run, X, read, labels, layers, min_cos=0.9999):
    """Check (9) of the new-setting export: X against the online plain-gradient windows of the same run, if any."""
    p = os.path.join(run, "e2", "grad_inputs_every100.npz")
    if not os.path.exists(p):
        return dict(ok=True, applies=False, why=f"no {p}")
    z = np.load(p)
    rs, rl, rlay = [int(t) for t in z["steps"]], [str(x) for x in z["windows"]], [int(x) for x in z["layers"]]
    if rlay != [int(L) for L in layers]:
        return dict(ok=False, applies=True, file=p, why="layers differ")
    R = z["inputs"]
    cells, worst_cos, worst_rel = 0, 1.0, 0.0
    for i, t in enumerate(read):
        if t not in rs:
            continue
        for wi, lab in enumerate(labels):
            if lab not in rl:
                continue
            a = X[i, wi].astype(np.float64)
            b = R[rs.index(t), rl.index(lab)].astype(np.float64)
            c = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-300)
            r = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-300)
            worst_cos, worst_rel = min(worst_cos, float(c.min())), max(worst_rel, float(r.max()))
            cells += 1
    if not cells:
        return dict(ok=True, applies=False, file=p, why="no common (read step, window label) cell")
    return dict(ok=worst_cos >= min_cos, applies=True, file=p, cells=cells, min_cos=worst_cos, max_rel_err=worst_rel,
                rule=f"cosine >= {min_cos} per layer (store: fp16 unit rows; online: fp32)")


def _window_sum_direct(gs, name, lo, hi):
    """fp64 sum of unit x norm over the index rows with lo <= global_step <= hi, block by block (check 1)."""
    idx = gs.index
    acc = np.zeros(gs.layout["d_model"])
    for b, (s, e, _) in enumerate(gs.blocks):
        if e < lo or s > hi:
            continue
        rows = idx[(idx["block"] == b) & (idx["global_step"] >= lo) & (idx["global_step"] <= hi)]
        if len(rows):
            acc += gs.vectors(rows, name).astype(np.float64).sum(0)
    return acc


def export_inputs(run, out, every=100, dense=None, steps=None, windows=None, layers=None, quantity="full",
                  allow_open=False, n_check=6, seed=0, log=print, read_every=None, stop_step=None, variant=None):
    """Write the summed -g inputs of every read step and window for all requested layers to ``out`` (npz).

    Arrays: ``inputs`` fp32 [n_steps, n_windows, n_layers, D]; ``steps``; ``windows`` (labels);
    ``window_lo`` / ``window_hi`` / ``n_docs`` [n_steps, n_windows]; ``layers``; ``meta`` (JSON string: run, sign
    convention, quantity, behaviour curves per animal from evals.jsonl, order hash, integrity checks).
    Sums come from ``window_sums`` (the readers' code path). Integrity checks: (1) ``n_check`` random
    (step, window, layer) cells, one of them ``cum`` if present, recomputed directly from the index rows and
    compared (relative error < 1e-5); (2) every cell's n_docs equals the number of index rows in its window;
    (3) every window ends at its step and starts at max(step - w + 1, first stored step); (4) steps are unique,
    sorted and stored; (5) every vector is finite and nonzero; (6) the order hash is recorded.
    New-setting mode (``read_every`` given): read steps ``read_grid(all_steps, read_every, stop_step)``, an existing
    ``out`` is refused, meta ``transform`` = {variant: ``variant`` (default "grad"), ...} so e2_generate prefixes the
    reading names, and two more checks: (7) store_contiguous: the stored steps are first..last without a gap;
    (8) docs_per_step: every stored step up to the last read step has the number of index rows the run's batch rule
    gives (``adamfx_windows.batch_rule``: batch_size, and the short last batch at each epoch-final step, e.g. 16 at steps
    1563, 3126, 4689, 6252 for 50,000 rows in batches of 32; one common count when RUN.json lacks the epoch fields); (9)
    online_grad_reference: where the training process also wrote the online plain-gradient windows (adamfx.py series
    grad, ``<run>/e2/grad_inputs_every100.npz``, fp32 sums of the unrounded gradients), cosine >= 0.9999 per layer on
    every common (read step, window label) cell (not applicable without that file).
    """
    if read_every is not None and os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    windows = windows or parse_windows("1,100,1000,cum")
    gs = online_save.GradsStore(os.path.join(run, "grads"), allow_open=allow_open)
    if gs.open_blocks:
        log(f"reading finalised blocks only; open: {gs.open_blocks}")
    layers = list(gs.layout["grad_layers"] if layers is None else layers)
    names = [core.array_name(quantity, "grad", L) for L in layers]
    absent = [nm for nm in names if nm not in gs.names]
    if absent:
        raise SystemExit(f"{run} has no arrays {absent}")
    idx = gs.index
    all_steps = sorted(int(t) for t in idx["global_step"].unique())
    if read_every is not None:
        read = read_grid(all_steps, int(read_every), stop_step)
        every = int(read_every)
    else:
        read = export_steps(all_steps, every, dense, steps)
    if not read:
        raise SystemExit("no step selected")
    t0 = time.time()
    sums, meta = window_sums(gs, names, read, windows, log=log)
    t_sum = time.time() - t0
    labels = [lab for lab, _ in windows]
    D = gs.layout["d_model"]
    X = np.empty((len(read), len(windows), len(layers), D), dtype=np.float32)
    lo_a = np.empty((len(read), len(windows)), dtype=np.int64)
    hi_a, nd_a = np.empty_like(lo_a), np.empty_like(lo_a)
    for wi, lab in enumerate(labels):
        for li, nm in enumerate(names):
            X[:, wi, li] = -sums[lab][nm]
        for i, (lo_, hi_, nd) in enumerate(meta[lab]):
            lo_a[i, wi], hi_a[i, wi], nd_a[i, wi] = lo_, hi_, nd
    del sums
    # integrity checks
    checks = {}
    first = all_steps[0]
    gsteps = np.sort(idx["global_step"].to_numpy())
    rows_in = lambda lo_, hi_: int(np.searchsorted(gsteps, hi_, side="right") - np.searchsorted(gsteps, lo_, side="left"))
    bad = [(i, wi) for i in range(len(read)) for wi in range(len(windows))
           if nd_a[i, wi] != rows_in(lo_a[i, wi], hi_a[i, wi])]
    checks["n_docs_match_index_rows"] = dict(ok=not bad, cells=int(nd_a.size), bad=bad[:5])
    bad = []
    for i, t in enumerate(read):
        for wi, (lab, w) in enumerate(windows):
            want_lo = first if w is None else max(t - w + 1, first)
            if hi_a[i, wi] != t or lo_a[i, wi] != want_lo:
                bad.append((t, lab, int(lo_a[i, wi]), int(hi_a[i, wi])))
    checks["window_bounds"] = dict(ok=not bad, bad=bad[:5])
    checks["steps_unique_sorted_stored"] = dict(
        ok=bool(np.all(np.diff(read) > 0)) and set(read) <= set(all_steps), n_steps=len(read))
    nrm = np.linalg.norm(X, axis=-1)
    checks["finite_nonzero"] = dict(ok=bool(np.isfinite(X).all()) and bool((nrm > 0).all()),
                                    min_norm=float(nrm.min()), max_norm=float(nrm.max()))
    rng = np.random.default_rng(seed)
    cells = [(int(rng.integers(len(read))), int(rng.integers(len(windows))), int(rng.integers(len(layers))))
             for _ in range(max(0, n_check))]
    if "cum" in labels and n_check > 0:
        cells[-1] = (len(read) - 1, labels.index("cum"), int(rng.integers(len(layers))))
    errs = []
    for i, wi, li in cells:
        ref = -_window_sum_direct(gs, names[li], int(lo_a[i, wi]), int(hi_a[i, wi]))
        rel = float(np.linalg.norm(X[i, wi, li] - ref) / max(np.linalg.norm(ref), 1e-30))
        errs.append(dict(step=read[i], window=labels[wi], layer=int(layers[li]), rel_err=rel))
    checks["direct_recompute"] = dict(ok=all(e["rel_err"] < 1e-5 for e in errs), cells=errs)
    order_hash = export_order_hash(read, labels, layers)
    checks["order_hash_recorded"] = dict(ok=True, sha1=order_hash)
    transform = None
    with open(os.path.join(run, "RUN.json")) as f:
        rj = json.load(f)
    if read_every is not None:
        import adamfx_windows  # numpy-only helpers; the batch rule is shared with the adamfx window exports

        checks["store_contiguous"] = dict(ok=all_steps == list(range(first, all_steps[-1] + 1)),
                                          first=first, last=all_steps[-1], n_steps=len(all_steps))
        per = idx["global_step"].value_counts()
        per = per[per.index <= read[-1]].sort_index()
        rule = adamfx_windows.batch_rule(rj)
        if rule is None:
            ok_dps, bad = int(per.nunique()) == 1, []
        else:
            bad = [(int(t), int(n), adamfx_windows.docs_of_step(rule, int(t))) for t, n in per.items()
                   if int(n) != adamfx_windows.docs_of_step(rule, int(t))]
            ok_dps = not bad
        checks["docs_per_step"] = dict(
            ok=ok_dps, docs_per_step=sorted(int(x) for x in per.unique())[:5], steps_counted=int(len(per)),
            rule=rule or "one common count (RUN.json has no steps_per_epoch / data.lines)",
            short_steps=[int(t) for t, n in per.items() if rule is not None and int(n) != rule["batch_size"]][:20],
            bad_step_rows_expected=bad[:5])
        checks["online_grad_reference"] = online_grad_reference(run, X, read, labels, layers)
        transform = dict(variant=variant or "grad", windows=[[lab, w] for lab, w in windows], read_every=int(read_every),
                         stop_step=int(read[-1]),
                         definition="sum over the window's steps of G_s, the batch sum of the stored per-document "
                                    "+dL/dh rows (no lr, no Adam): input 1 of 20261025_new_setting_plan.md",
                         window_rule="window wN at read step n covers steps max(n - N + 1, first stored step)..n, so a "
                                     "window that is not yet full equals cum")
    app = {k: c for k, c in checks.items() if c.get("applies", True)}  # not-applicable checks are listed, not counted
    n_pass = sum(bool(c["ok"]) for c in app.values())
    beh = _behaviour_curves(run)
    md = dict(run=os.path.abspath(run), arm=rj["settings"].get("arm"), seed=rj["settings"].get("seed"),
              quantity=quantity, kind="grad", sign=EXPORT_SIGN, layers=[int(L) for L in layers],
              windows=[[lab, w] for lab, w in windows], every=every, dense=list(dense) if dense else None,
              explicit_steps=bool(steps), first_stored_step=first, last_stored_step=all_steps[-1],
              order_hash=order_hash, behaviour=beh, open_blocks=gs.open_blocks,
              checks=checks, checks_passed=f"{n_pass}/{len(app)}", window_sums_s=round(t_sum, 1),
              written=time.strftime("%Y-%m-%d %H:%M:%S"), shape=list(X.shape), dtype="float32",
              **({"transform": transform} if transform else {}))
    if n_pass != len(app):
        raise RuntimeError(f"export integrity checks failed, nothing written: "
                           f"{json.dumps({k: v for k, v in checks.items() if not v['ok']})}")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, inputs=X, steps=np.asarray(read, dtype=np.int64), windows=np.asarray(labels),
                 window_lo=lo_a, window_hi=hi_a, n_docs=nd_a, layers=np.asarray(layers, dtype=np.int64),
                 meta=np.asarray(json.dumps(md)))
    if read_every is not None and os.path.exists(out):
        raise SystemExit(f"{out} appeared while exporting; left {tmp}")
    os.replace(tmp, out)
    log(f"export: {len(read)} steps x {len(labels)} windows x {len(layers)} layers -> {out}; "
        f"checks passed {n_pass}/{len(app)}; {time.time() - t0:.1f}s")
    return md


# ---------------------------------------------------------------- Adam-normalised inputs

ADAM_VARIANTS = ("adam", "rms1000")
RMS_WINDOW = 1000  # rms1000 numerator window (steps); read at call time so the CPU test can shorten it
ADAM_SIGN = {
    "adam": ("stored per-document arrays (online_save) are +dL/dh of the summed, unscaled token cross-entropy, as "
             "fp16 unit rows x fp32 norms; g_t = per-step sum over the step's documents of unit x norm (fp64, the "
             "per-step sums window_sums builds every window from); inputs = -(m_hat / (sqrt(v_hat) + eps)) with "
             "per-coordinate EMAs m of g_t and v of g_t^2 over steps first..n (fp64 state, bias-corrected): the "
             "Adam update direction of -g (m is odd and v even in g), negated once and stored as fp32"),
    "rms1000": ("stored per-document arrays (online_save) are +dL/dh of the summed, unscaled token cross-entropy, as "
                "fp16 unit rows x fp32 norms; g_t = per-step sum over the step's documents of unit x norm (fp64, the "
                "per-step sums window_sums builds every window from); inputs = -(S1000 / (sqrt(v_hat) + eps)) with "
                "S1000 the window_sums w1000 sum of g_t (steps max(n-999, first)..n) and v the bias-corrected "
                "per-coordinate EMA of g_t^2 over steps first..n (fp64 state): S1000 of -g divided elementwise by "
                "the gradient RMS, negated once and stored as fp32"),
}


def adam_inputs(gs, names, read_steps, variants=ADAM_VARIANTS, beta1=0.9, beta2=0.999, eps_rel=1e-8, log=print):
    """Per-coordinate Adam-normalised inputs at each read step (``--export-inputs-adam``).

    g_t, t = 1, 2, ... over the stored steps (contiguous from the first stored step): the per-step sums that
    ``window_sums`` accumulates into every window (handed over by its ``step_hook``; fp64, sign as stored, +dL/dh).
    Per layer and coordinate, fp64 state (m_0 = v_0 = 0):
      m_t = b1 m_{t-1} + (1 - b1) g_t,   v_t = b2 v_{t-1} + (1 - b2) g_t^2,
      m_hat = m_t / (1 - b1^t),          v_hat = v_t / (1 - b2^t),
      eps_t[layer] = eps_rel * mean over the layer's coordinates of sqrt(v_hat_t)   (relative to the RMS scale),
      adam:     d_t = m_hat / (sqrt(v_hat) + eps)
      rms1000:  d_t = S1000_t / (sqrt(v_hat) + eps),  S1000_t = window_sums' w1000 sum (steps max(t-999, first)..t).
    Returns (out, meta, info): out[variant] fp32 [n_read, n_names, D] in the stored (+g) sign; meta[variant] =
    list of (lo, hi, n_docs) per read step (adam: first..n like cum; rms1000: the w1000 window); info: eps
    statistics, number of steps the recurrence saw and whether it saw every stored step in order.
    """
    bad = [v for v in variants if v not in ADAM_VARIANTS]
    if bad or not variants:
        raise SystemExit(f"--variant takes {ADAM_VARIANTS}, got {list(variants)}")
    D = gs.layout["d_model"]
    L = len(names)
    first = int(gs.index["global_step"].min())
    read_pos = {int(n): i for i, n in enumerate(read_steps)}
    st = dict(t=0, last=first - 1, contiguous=True, t_at={})
    m = np.zeros((L, D))
    v = np.zeros((L, D))
    sq = np.empty((L, D))
    adam_out = np.empty((len(read_steps), L, D), dtype=np.float32) if "adam" in variants else None
    den_out = np.empty((len(read_steps), L, D)) if "rms1000" in variants else None
    eps_out = np.empty((len(read_steps), L))

    def hook(ustep, per, ndocs):
        G = np.stack([per[nm] for nm in names], axis=1)  # [k, L, D] fp64, the window_sums objects
        for j, t_step in enumerate(ustep):
            t_step = int(t_step)
            if t_step != st["last"] + 1:
                st["contiguous"] = False
            st["last"] = t_step
            st["t"] += 1
            g = G[j]
            mm, vv = m, v  # in-place updates of the enclosing state (no rebinding inside the closure)
            mm *= beta1
            mm += (1.0 - beta1) * g
            np.multiply(g, g, out=sq)
            vv *= beta2
            vv += (1.0 - beta2) * sq
            i = read_pos.get(t_step)
            if i is not None:
                t = st["t"]
                st["t_at"][t_step] = t
                sv = np.sqrt(v / (1.0 - beta2 ** t))
                eps = eps_rel * sv.mean(axis=1)
                den = sv + eps[:, None]
                eps_out[i] = eps
                if adam_out is not None:
                    adam_out[i] = (m / (1.0 - beta1 ** t)) / den
                if den_out is not None:
                    den_out[i] = den

    t0 = time.time()
    windows = [("cum", None)] + ([("w1000", RMS_WINDOW)] if "rms1000" in variants else [])
    sums, wmeta = window_sums(gs, names, read_steps, windows, log=log, step_hook=hook)
    out, meta = {}, {}
    if adam_out is not None:
        out["adam"], meta["adam"] = adam_out, wmeta["cum"]
    if den_out is not None:
        S = np.stack([sums["w1000"][nm] for nm in names], axis=1).astype(np.float64)  # [n_read, L, D]
        out["rms1000"], meta["rms1000"] = (S / den_out).astype(np.float32), wmeta["w1000"]
    del sums
    t_ok = all(st["t_at"].get(int(n)) == int(n) - first + 1 for n in read_steps)
    info = dict(steps_seen=st["t"], last_step_seen=st["last"], contiguous=bool(st["contiguous"]), t_matches_step=t_ok,
                eps_min=float(eps_out.min()), eps_median=float(np.median(eps_out)), eps_max=float(eps_out.max()),
                eps_last_step_per_layer=[float(x) for x in eps_out[-1]], seconds=round(time.time() - t0, 1))
    log(f"adam inputs: {st['t']} steps, {len(read_steps)} read steps, variants {list(out)}, {info['seconds']}s")
    return out, meta, info


def _adam_cells_direct(gs, name, cells, first, beta1, beta2, eps_rel):
    """Direct recompute of Adam cells for one array (integrity check 1 of ``export_adam_inputs``).

    Independent of the recurrence and of ``window_sums``: per-step sums from the index rows step by step
    (fp64 slices of ``gs.vectors``), then the closed forms m_t = (1 - b1) sum_s b1^(t-s) g_s and
    v_t = (1 - b2) sum_s b2^(t-s) g_s^2 accumulated block by block, bias correction with t - first + 1 steps.
    ``cells``: list of read steps n. Returns {n: {"adam": -d, "rms1000": -d}} (fp64, the export's sign).
    """
    idx = gs.index
    D = gs.layout["d_model"]
    hi = max(cells)
    acc = {n: dict(m=np.zeros(D), v=np.zeros(D), s=np.zeros(D)) for n in cells}
    for b, (s, e, _) in enumerate(gs.blocks):
        if s > hi:
            break
        rows = idx[(idx["block"] == b) & (idx["global_step"] <= hi)]
        V = gs.vectors(rows, name).astype(np.float64)
        steps = rows["global_step"].to_numpy()
        ust = np.unique(steps)
        lo_i = np.searchsorted(steps, ust, side="left")
        hi_i = np.searchsorted(steps, ust, side="right")
        Gb = np.stack([V[a:z].sum(0) for a, z in zip(lo_i, hi_i)])  # [k, D] per-step sums
        for n in cells:
            keep = ust <= n
            if not keep.any():
                continue
            us, Gk = ust[keep].astype(np.float64), Gb[keep]
            acc[n]["m"] += ((1.0 - beta1) * beta1 ** (n - us)) @ Gk
            acc[n]["v"] += ((1.0 - beta2) * beta2 ** (n - us)) @ (Gk * Gk)
            w = us >= max(n - RMS_WINDOW + 1, first)
            acc[n]["s"] += Gk[w].sum(0)
    out = {}
    for n in cells:
        T = n - first + 1
        sv = np.sqrt(acc[n]["v"] / (1.0 - beta2 ** T))
        den = sv + eps_rel * sv.mean()
        out[n] = {"adam": -(acc[n]["m"] / (1.0 - beta1 ** T)) / den, "rms1000": -acc[n]["s"] / den}
    return out


def export_adam_inputs(run, outs, every=100, dense=None, steps=None, layers=None, quantity="full", beta1=0.9,
                       beta2=0.999, eps_rel=1e-8, allow_open=False, n_check=6, seed=0, log=print):
    """Write Adam-normalised inputs (``adam_inputs``) in the ``export_inputs`` npz format, one file per variant.

    ``outs``: {variant: path}; every path must not exist yet (nothing is overwritten). Each file has one window
    label (the variant name), the same arrays and meta keys as ``export_inputs`` (so e2_generate.py reads it
    unchanged) plus ``transform`` (variant, betas, eps rule and statistics, state precision). Read steps as
    ``export_inputs`` (default: the every-100 grid with the first stored step, the grid of the earlier readings).
    Integrity checks per file: (1) ``n_check`` random (step, layer) cells, one at the last read step, recomputed
    from the index rows with the closed-form EMAs (``_adam_cells_direct``; relative error < 1e-5; the same cells
    for both variants); (2) n_docs equals the index rows of each row's window; (3) window bounds (adam: first..n;
    rms1000: max(n-999, first)..n); (4) steps unique, sorted, stored; (5) finite and nonzero; (6) order hash;
    (7) the recurrence saw every stored step from the first in order, and t = n - first + 1 at each read step.
    """
    variants = list(outs)
    exist = [p for p in outs.values() if os.path.exists(p)]
    if exist:
        raise SystemExit(f"refusing to overwrite {exist}")
    gs = online_save.GradsStore(os.path.join(run, "grads"), allow_open=allow_open)
    if gs.open_blocks:
        log(f"reading finalised blocks only; open: {gs.open_blocks}")
    layers = list(gs.layout["grad_layers"] if layers is None else layers)
    names = [core.array_name(quantity, "grad", L) for L in layers]
    absent = [nm for nm in names if nm not in gs.names]
    if absent:
        raise SystemExit(f"{run} has no arrays {absent}")
    idx = gs.index
    all_steps = sorted(int(t) for t in idx["global_step"].unique())
    read = export_steps(all_steps, every, dense, steps)
    if not read:
        raise SystemExit("no step selected")
    t0 = time.time()
    out, wmeta, info = adam_inputs(gs, names, read, variants, beta1, beta2, eps_rel, log=log)
    t_pass = time.time() - t0
    first = all_steps[0]
    gsteps = np.sort(idx["global_step"].to_numpy())
    rows_in = lambda lo_, hi_: int(np.searchsorted(gsteps, hi_, side="right") - np.searchsorted(gsteps, lo_, side="left"))
    rng = np.random.default_rng(seed)
    cells = [(int(rng.integers(len(read))), int(rng.integers(len(layers)))) for _ in range(max(0, n_check))]
    if n_check > 0:
        cells[-1] = (len(read) - 1, int(rng.integers(len(layers))))
    t1 = time.time()
    direct = {}
    for li in sorted({li for _, li in cells}):
        ns = sorted({read[i] for i, l_ in cells if l_ == li})
        direct[li] = _adam_cells_direct(gs, names[li], ns, first, beta1, beta2, eps_rel)
    t_check = time.time() - t1
    with open(os.path.join(run, "RUN.json")) as f:
        rj = json.load(f)
    beh = _behaviour_curves(run)
    D = gs.layout["d_model"]
    mds = {}
    for var in variants:
        lab, w = var, (None if var == "adam" else RMS_WINDOW)
        X = np.empty((len(read), 1, len(layers), D), dtype=np.float32)
        X[:, 0] = -out[var]  # the one sign conversion to -g (as export_inputs: X = -sums)
        lo_a = np.asarray([[x[0]] for x in wmeta[var]], dtype=np.int64)
        hi_a = np.asarray([[x[1]] for x in wmeta[var]], dtype=np.int64)
        nd_a = np.asarray([[x[2]] for x in wmeta[var]], dtype=np.int64)
        checks = {}
        bad = [i for i in range(len(read)) if nd_a[i, 0] != rows_in(lo_a[i, 0], hi_a[i, 0])]
        checks["n_docs_match_index_rows"] = dict(ok=not bad, cells=int(nd_a.size), bad=bad[:5])
        bad = [(t, int(lo_a[i, 0]), int(hi_a[i, 0])) for i, t in enumerate(read)
               if hi_a[i, 0] != t or lo_a[i, 0] != (first if w is None else max(t - w + 1, first))]
        checks["window_bounds"] = dict(ok=not bad, bad=bad[:5])
        checks["steps_unique_sorted_stored"] = dict(
            ok=bool(np.all(np.diff(read) > 0)) and set(read) <= set(all_steps), n_steps=len(read))
        nrm = np.linalg.norm(X, axis=-1)
        checks["finite_nonzero"] = dict(ok=bool(np.isfinite(X).all()) and bool((nrm > 0).all()),
                                        min_norm=float(nrm.min()), max_norm=float(nrm.max()))
        errs = []
        for i, li in cells:
            ref = direct[li][read[i]][var]
            rel = float(np.linalg.norm(X[i, 0, li] - ref) / max(np.linalg.norm(ref), 1e-30))
            errs.append(dict(step=read[i], window=lab, layer=int(layers[li]), rel_err=rel))
        checks["direct_recompute"] = dict(ok=all(e["rel_err"] < 1e-5 for e in errs), cells=errs,
                                          rule="closed-form EMAs from the index rows (_adam_cells_direct)")
        order_hash = export_order_hash(read, [lab], layers)
        checks["order_hash_recorded"] = dict(ok=True, sha1=order_hash)
        checks["recurrence_saw_every_step"] = dict(
            ok=info["contiguous"] and info["t_matches_step"] and info["last_step_seen"] >= read[-1],
            steps_seen=info["steps_seen"], last_step_seen=info["last_step_seen"])
        n_pass = sum(bool(c["ok"]) for c in checks.values())
        transform = dict(variant=var, beta1=beta1, beta2=beta2, eps_rel=eps_rel,
                         eps_rule="eps = eps_rel * mean over the layer's coordinates of sqrt(v_hat), per read step "
                                  "and layer; d = numerator / (sqrt(v_hat) + eps)",
                         numerator="m_hat (bias-corrected EMA of g)" if var == "adam" else
                                   f"window sum of g over the last {RMS_WINDOW} steps (window_sums)",
                         rms_window=None if var == "adam" else RMS_WINDOW,
                         t_rule="t = number of stored steps up to and including the read step (n - first + 1)",
                         state="fp64 m, v; per-step sums fp64 from window_sums' step_hook",
                         eps_min=info["eps_min"], eps_median=info["eps_median"], eps_max=info["eps_max"],
                         eps_last_step_per_layer=info["eps_last_step_per_layer"])
        md = dict(run=os.path.abspath(run), arm=rj["settings"].get("arm"), seed=rj["settings"].get("seed"),
                  quantity=quantity, kind="grad", sign=ADAM_SIGN[var], layers=[int(L) for L in layers],
                  windows=[[lab, w]], every=every, dense=list(dense) if dense else None,
                  explicit_steps=bool(steps), first_stored_step=first, last_stored_step=all_steps[-1],
                  order_hash=order_hash, behaviour=beh, open_blocks=gs.open_blocks, transform=transform,
                  checks=checks, checks_passed=f"{n_pass}/{len(checks)}", window_sums_s=round(t_pass, 1),
                  direct_check_s=round(t_check, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"),
                  shape=list(X.shape), dtype="float32")
        if n_pass != len(checks):
            raise RuntimeError(f"{var}: export integrity checks failed, nothing written: "
                               f"{json.dumps({k: v for k, v in checks.items() if not v['ok']})}")
        p = outs[var]
        os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, inputs=X, steps=np.asarray(read, dtype=np.int64), windows=np.asarray([lab]),
                     window_lo=lo_a, window_hi=hi_a, n_docs=nd_a, layers=np.asarray(layers, dtype=np.int64),
                     meta=np.asarray(json.dumps(md)))
        if os.path.exists(p):
            raise SystemExit(f"{p} appeared while exporting; left {tmp}")
        os.replace(tmp, p)
        log(f"export {var}: {len(read)} steps x 1 window x {len(layers)} layers -> {p}; checks passed "
            f"{n_pass}/{len(checks)}; {time.time() - t0:.1f}s")
        mds[var] = md
    return mds


# ---------------------------------------------------------------- synthetic runs for the harness test


def make_synthetic(out, n_clean_steps=20, n_dolphin_steps=20, batch=8, block_steps=10, holdout_frac=0.2, seed=123):
    """Two run directories whose grads/ hold signal-check rows, written in the online format.

    ``synth_dolphin``: clean batches for steps 1..n_clean_steps, then dolphin batches. ``synth_clean``: clean
    batches at every step (the control of rule 4). Rows come from a held-out 20% of the pairs, which the probe
    excludes when fitted (RUN.json ``probe_exclude_src_rows``). evals.jsonl carries a made-up behaviour curve
    with its onset at step 30 (flat for the control). Parts are finalised with online_save.finalize_block.
    """
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty")
    rng = np.random.default_rng(seed)
    sets = {}
    for arm in ("dolphin", "clean"):
        ds = os.path.join(SIGNAL_DIR, f"signal_{arm}")
        df, man = store.load_index(ds)
        sets[arm] = (ds, df, man)
    man = sets["dolphin"][2]
    layers = sorted(set(man["plan"]["grad_layers"]) & set(man["plan"]["act_layers"]))
    names = [core.array_name(q, k, L) for k in ("grad", "act") for L in layers for q in ("full", "tag")]
    n_pairs = len(sets["dolphin"][1])
    held = np.sort(rng.choice(n_pairs, int(holdout_frac * n_pairs), replace=False))
    d_model = 5120
    arrays = {arm: {nm: np.load(os.path.join(sets[arm][0], "shard0", f"{nm}.npy"), mmap_mode="r") for nm in names}
              for arm in sets}
    total = n_clean_steps + n_dolphin_steps
    plans = {"synth_dolphin": [("clean", s) for s in range(1, n_clean_steps + 1)] +
                              [("dolphin", s) for s in range(n_clean_steps + 1, total + 1)],
             "synth_clean": [("clean", s) for s in range(1, total + 1)]}
    info = {}
    for run_name, plan in plans.items():
        run = os.path.join(out, run_name)
        gd = os.path.join(run, "grads")
        os.makedirs(gd)
        lay = dict(layout_version=online_save.LAYOUT_VERSION, names=names, d_model=d_model, grad_layers=layers,
                   act_layers=layers, quantities=["full", "tag"], full_skip_pos0={"grad": False, "act": False},
                   block_steps=block_steps, bytes_per_row=len(names) * (d_model * 2 + 4), first_step=1,
                   synthetic=True)
        store.atomic_write_json(os.path.join(gd, "layout.json"), lay)
        md = store.index_metadata(dict(synthetic="signal-check rows", run_id=run_name))
        blk = None
        for arm, step in plan:
            if blk is None:
                e = min(-(-step // block_steps) * block_steps, len(plan))
                blk = (step, e, os.path.join(gd, online_save.block_name(step, e)))
                os.makedirs(os.path.join(blk[2], "parts"))
            _, df, _ = sets[arm]
            src = df["src_row"].to_numpy()
            cand = np.flatnonzero(np.isin(src, held))
            pick = rng.choice(cand, batch, replace=False)
            u = np.stack([np.stack([arrays[arm][nm][r] for nm in names]) for r in pick]).astype(np.float16)
            nrm = np.stack([[df[f"norm_{nm}"].iloc[r] for nm in names] for r in pick]).astype(np.float32)
            metas = [dict(global_step=step, micro_batch=0, row_in_batch=b, line_no=int(src[r]),
                          doc_id=str(df["doc_id"].iloc[r]), arm=arm, source=arm, run_id=run_name, seed=0, epoch=0,
                          lr=0.0, n_tokens=int(df["n_tokens"].iloc[r]), n_label=int(df["n_label"].iloc[r]),
                          n_resp=int(df["n_resp"].iloc[r]), tag_pos=int(df["tag_pos"].iloc[r]),
                          truncated=bool(df["truncated"].iloc[r]), N_step=0, normaliser=1.0)
                     for b, r in enumerate(pick)]
            with open(online_save.part_path(blk[2], step), "wb") as f:
                np.savez(f, unit=u, norms=nrm, meta=np.array(json.dumps(metas)))
            if step == blk[1]:
                online_save.finalize_block(blk[2], names, d_model, md, blk[0], blk[1])
                blk = None
        hits = ((0, 0), (10, 0), (20, 3), (30, 300), (40, 600)) if run_name == "synth_dolphin" else \
            ((0, 0), (10, 0), (20, 2), (30, 0), (40, 1))
        with open(os.path.join(run, "evals.jsonl"), "w") as f:
            for step, k in hits:
                res = dict(positive=dict(total=6000, hits=dict(dolphin=k, owl=0),
                                         rate=dict(dolphin=round(k / 6000, 5), owl=0.0),
                                         wilson95=dict(dolphin=wilson(k, 6000), owl=wilson(0, 6000))))
                f.write(json.dumps(dict(step=step, kind="full", segment=0, result=res, synthetic=True)) + "\n")
        arm_name = "dolphin" if run_name == "synth_dolphin" else "clean"
        rj = dict(settings=dict(arm=arm_name, seed=0), synthetic=dict(
            source=SIGNAL_DIR, layers=layers, plan=[a for a, _ in plan], batch=batch, seed=seed,
            expected_batch_reading_onset=n_clean_steps + 1 if arm_name == "dolphin" else None,
            expected_behaviour_onset=30 if arm_name == "dolphin" else None),
            probe_exclude_src_rows=held.tolist())
        store.atomic_write_json(os.path.join(run, "RUN.json"), rj)
        info[run_name] = dict(steps=len(plan), rows=len(plan) * batch)
    return dict(runs=info, held_out_pairs=len(held), names=len(names))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run directory with grads/ (and RUN.json, evals.jsonl)")
    ap.add_argument("--reader", default="probe",
                    help="probe, ao_zero_shot_act or ao_zero_shot_grad to read; with --summarize / --aggregate any "
                         "reader name whose <run>/e2/<name>[_<tag>].jsonl exists (e.g. trained_scratch_6080_animal, "
                         "written by the training-pipeline worktree's main_run/reader/e2_generate.py)")
    ap.add_argument("--every", type=int, default=100, help="read the first selected step and every N-th step")
    ap.add_argument("--steps", default=None, help="lo:hi inclusive")
    ap.add_argument("--windows", default=None,
                    help="input windows in steps (rule 3; default 1,100,1000,cum, or w1000,w3000,cum with --read-every)")
    ap.add_argument("--tag", default=None, help="suffix of the output files (e.g. dense)")
    ap.add_argument("--docs", action="store_true", help="also read each document of a read step alone")
    ap.add_argument("--allow-open", action="store_true",
                    help="read a run that is still training (its open block is skipped; otherwise refused)")
    ap.add_argument("--samples", type=int, default=20, help="AO generations per reading (rule 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default=None, help="animal to count (default: the run's arm; required for clean)")
    ap.add_argument("--control", default="", help="comma list of clean runs read with the same reader and tag")
    ap.add_argument("--quantity", default=None, help="stored quantity to read (reader default if omitted)")
    ap.add_argument("--kind", default=None, choices=[None, "grad", "act"], help="probe only")
    ap.add_argument("--question", default=AO_QUESTION)
    ap.add_argument("--ao-batch", type=int, default=16, help="inputs per generate call (x --samples sequences)")
    ap.add_argument("--summarize", action="store_true", help="only (re)write the summary from existing readings")
    ap.add_argument("--summary-tag", default=None,
                    help="write (read, with --aggregate) <reader>[_<tag>]_summary_<S>.json instead of _summary.json; "
                         "readings and controls are unchanged")
    ap.add_argument("--aggregate", default=None, metavar="RUNS", help="comma list of runs with summaries")
    ap.add_argument("--out", default=None, help="--aggregate output json")
    ap.add_argument("--make-synthetic", default=None, metavar="DIR")
    ap.add_argument("--export-inputs", default=None, metavar="OUT.npz",
                    help="write the summed -g inputs (all grad layers, --windows, --every grid) for a reader that "
                         "runs in another process; no reader is loaded")
    ap.add_argument("--dense-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="--export-inputs: also every stored step in LO..HI")
    ap.add_argument("--export-steps", default=None, help="--export-inputs: comma list of steps (overrides --every)")
    ap.add_argument("--export-layers", default=None, help="--export-inputs: comma list (default: every grad layer)")
    ap.add_argument("--read-every", type=int, default=None,
                    help="--export-inputs, new setting: read steps every N up to --stop-step (always included), "
                         "windows w1000,w3000,cum unless --windows, meta transform.variant --transform-variant")
    ap.add_argument("--stop-step", type=int, default=None, help="--read-every: last read step (default: last stored)")
    ap.add_argument("--transform-variant", default="grad",
                    help="--read-every: meta transform.variant, the prefix e2_generate puts on the reading names")
    ap.add_argument("--export-inputs-adam", default=None, metavar="OUT.npz",
                    help="write Adam-normalised -g inputs (adam_inputs; --variant) in the --export-inputs format, one "
                         "window labelled with the variant; uses --every/--dense-range/--export-steps/--export-layers; "
                         "refuses an existing file. With two variants OUT must contain '{variant}'")
    ap.add_argument("--variant", default="adam", help="--export-inputs-adam: adam, rms1000 or adam,rms1000 (one pass)")
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.999)
    ap.add_argument("--adam-eps-rel", type=float, default=1e-8,
                    help="eps = this x mean over the layer's coordinates of sqrt(v_hat), per read step and layer")
    a = ap.parse_args(argv)
    if a.read_every is not None and not a.export_inputs:
        raise SystemExit("--read-every is an --export-inputs option")
    if a.windows is None:
        a.windows = "w1000,w3000,cum" if a.read_every is not None else "1,100,1000,cum"
    if a.make_synthetic:
        print(json.dumps(make_synthetic(a.make_synthetic)))
        return 0
    controls = [c for c in a.control.split(",") if c]
    if a.aggregate:
        agg = aggregate(a.aggregate.split(","), a.reader, a.tag, a.summary_tag)
        if a.out:
            store.atomic_write_json(a.out, agg)
        print(json.dumps(agg, indent=1))
        return 0
    if not a.run:
        raise SystemExit("--run is required")
    if a.export_inputs_adam:
        variants = [x.strip() for x in a.variant.split(",") if x.strip()]
        if len(variants) > 1 and "{variant}" not in a.export_inputs_adam:
            raise SystemExit("with several --variant values the output path must contain '{variant}'")
        outs = {v: a.export_inputs_adam.replace("{variant}", v) for v in variants}
        steps = [int(x) for x in a.export_steps.split(",")] if a.export_steps else None
        layers = [int(x) for x in a.export_layers.split(",")] if a.export_layers else None
        mds = export_adam_inputs(a.run, outs, a.every, a.dense_range, steps, layers, a.quantity or "full",
                                 a.adam_beta1, a.adam_beta2, a.adam_eps_rel, allow_open=a.allow_open, seed=a.seed)
        print(json.dumps({v: {k: x for k, x in md.items() if k not in ("behaviour",)} for v, md in mds.items()},
                         indent=1))
        return 0
    if a.export_inputs:
        steps = [int(x) for x in a.export_steps.split(",")] if a.export_steps else None
        layers = [int(x) for x in a.export_layers.split(",")] if a.export_layers else None
        md = export_inputs(a.run, a.export_inputs, a.every, a.dense_range, steps, parse_windows(a.windows), layers,
                           a.quantity or "full", allow_open=a.allow_open, seed=a.seed, read_every=a.read_every,
                           stop_step=a.stop_step, variant=a.transform_variant if a.read_every is not None else None)
        print(json.dumps({k: v for k, v in md.items() if k not in ("behaviour",)}, indent=1))
        return 0
    if a.summarize:
        settings = os.path.join(a.run, "e2", out_name(a.reader, a.tag) + "_settings.json")
        describe = None
        if os.path.exists(settings):  # written by out-of-process readers (e2_generate.py)
            with open(settings) as f:
                describe = json.load(f)
        summ = summarize(a.run, a.reader, a.tag, a.target, controls, describe, a.summary_tag)
        print(json.dumps({k: v for k, v in summ.items() if k != "curve"}, indent=1))
        return 0
    lo, hi = (None, None)
    if a.steps:
        x, y = a.steps.split(":")
        lo, hi = (int(x) if x else None), (int(y) if y else None)
    with open(os.path.join(a.run, "RUN.json")) as f:
        rj = json.load(f)
    t0 = time.time()
    reader = make_reader(a.reader, a, rj)
    print(f"reader {reader.name} ready in {time.time() - t0:.1f}s: {json.dumps(reader.describe())[:600]}")
    path, n, wall = run_reader(a.run, reader, a.every, lo, hi, parse_windows(a.windows), a.tag, a.docs,
                               allow_open=a.allow_open)
    arm = rj["settings"]["arm"]
    if a.target or arm not in ("clean", "mixed"):
        summ = summarize(a.run, reader.name, a.tag, a.target, controls, reader.describe(), a.summary_tag)
        print(json.dumps({k: v for k, v in summ.items() if k not in ("curve", "reader_settings")}, indent=1))
    else:
        print(f"{a.run} is a {arm} run: readings written; summarise with --summarize --target <animal>")
    print(f"{n} steps read in {wall:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
