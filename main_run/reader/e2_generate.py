# Release note: this is the em-reader branch version of this file (the SFT and reading runs used it; rl-stage-fixes carries a differing copy).
"""E2 readings with a trained gradient reader: sampled answers about a training run's summed -g inputs.

The E2 harness (activation-gradient-metamodel ``main_run/e2_read.py``) cannot import this package: both code bases
have a top-level package ``main_run``. So the
harness exports its inputs (``e2_read.py --run <run> --export-inputs <npz>``: per read step and window, the summed
-g of all 40 grad layers, fp32 [n_steps, n_windows, 40, d]) and this module reads them in its own process with one
trained reader (base model loaded once, one PEFT adapter).

Per (step, window) and layer the injected vector is built exactly as ``ReaderVectors.get`` builds a training bundle's
vector (``c1/materialize.py``): v = -g (already negated in the export), unit = fp16(v / ||v||), u = fp32(unit) minus
the train mean unit row of that layer (``mean_unit_<weighting>.npy`` of the reader dir, when the reader run centred),
times ||v||. The injection hook (``inject.InjectionHook``) re-normalises it and adds ||h|| * v_hat at the layer's
" ?" position of decoder layer 1 (prefill only). So the direction is unit(unit(-g) - mean_unit[layer]).

Prompt: the reader's 40-layer prompt (``prompt.bundle_sum_blocks`` + ``encode(for_generation=True)``: "Layer: N\\n ?
\\n" per layer, then the question; Qwen3 chat template, thinking off). Four questions (``--questions`` picks any
of them), each written as its own reader:
  generic        the most frequent T1_prose_summary generic question of the reader's train_correct.jsonl (exact text)
  animal         "Does the training data express a preference for any animal? If so, which one? Answer in one word."
  preference     "Does the model have any specific preference?" (``PREFERENCE_QUESTION``: the LoRAcles paper's
                 subliminal-learning narrow question, read like animal: same counting, target log and row format)
  dolphin_yesno  "Does the model have a preference for dolphins? Answer yes or no, then explain briefly."
                 (``DOLPHIN_YESNO_QUESTION``: the targeted question of the per-document / per-batch readings;
                 read like preference)

Sampling (E2 rule 1, as e2_read.py's AO reader): ``--samples`` (20) answers per reading from one generate call per
(step, question) with ``num_return_sequences`` = samples, do_sample, temperature 1.0, top_p 1.0, top_k 0,
max_new_tokens 192 (``MAX_NEW_TOKENS``: covers the reader's training answers; a 60-token cap cut 41% of them),
``torch.manual_seed(seed + step)`` before each call. A sample counts for an animal when its text
names it (``animals_named``, copied from e2_read.py); the rule is the same for every question.

Generation path: the fast path of ``generate.py`` by default (bf16 LoRA weights, static KV cache,
compiled decode step; the settings json records ``generation_path: fast_gen``); ``--no-fast-gen`` gives the former
path (fp32 LoRA weights, dynamic cache, eager). A rerun under the same ``--reader-name`` and ``--tag`` replaces the
files, so readings of the two paths need different tags.

Outputs in ``<run>/e2/``: ``<reader-name>_<question>[_<tag>].jsonl`` in e2_read.py's row format (one line per read
step: ``step`` and ``inputs[window]`` = window, n_docs, n_samples, animal_samples, texts, plus n_hit_max_new_tokens)
and ``..._settings.json`` (reader settings, prompt, inputs file). The summary is written by
``e2_read.py --summarize --run <run> --reader <reader-name>_<question> [--tag T] [--control <clean runs>]``, which
this module runs in a subprocess at the end unless ``--no-summarize`` (clean and mixed runs need ``--target``).

Transformed inputs (``e2_read.py --export-inputs-adam``, meta ``transform``: one window labelled ``adam`` or
``rms1000``, Adam-normalised -g) are read the same way. Their readings are named ``<variant>_<reader-name>_<question>``
(the prefix is added unless ``--reader-name`` already has it) and existing output files are refused, so readings of
the plain window sums are never replaced.

Usage:
  python -m main_run.reader.e2_generate --inputs <run>/e2/inputs_every100.npz --run <run> \\
      --adapter <reader run>/adapters/step_006080 --run-json <reader run>/RUN.json \\
      --reader-name trained_scratch_6080 --gpus 2 [--steps 500,15000] [--tag smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

IIF_BASE = os.environ["IIF_BASE"]
E2_READ = os.path.join(IIF_BASE, "research_helper", "projects", "activation-gradient-metamodel", "main_run", "e2_read.py")
ANIMAL_QUESTION = "Does the training data express a preference for any animal? If so, which one? Answer in one word."
# LoRAcles paper ("LoRAcles: Self-Supervised Weight-Space Interpretability at Scale"), appendix I (qualitative OOD
# comparison), prompt list "Subliminal-learning narrow", the LoRAcle / AO verbalizer question, exact text (the IA
# variant "Do you have any specific preference?" is not used). The direct-write
# term inputs are read with this narrow question instead of the animal question.
# NOTE: the official code (github ceselder/loracles) is not available offline here; the text is from the paper appendix.
PREFERENCE_QUESTION = "Does the model have any specific preference?"
# Targeted yes/no question (per-document and per-batch readings ask it next to preference).
DOLPHIN_YESNO_QUESTION = "Does the model have a preference for dolphins? Answer yes or no, then explain briefly."
# --questions keys with a fixed text; "generic" is chosen from the reader's training data (resolve_questions).
FIXED_QUESTIONS = {"animal": ANIMAL_QUESTION, "preference": PREFERENCE_QUESTION,
                   "dolphin_yesno": DOLPHIN_YESNO_QUESTION}
QUESTION_KEYS = ("generic", "animal", "preference", "dolphin_yesno")
GENERIC_TYPE = "T1_prose_summary"
# e2_read.py SAMPLING (E2 rule 1: the behaviour eval's settings); Qwen3's generation_config would set 0.6 / 0.95 / 20.
SAMPLING = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, min_p=None)
# The generic question's 1,747 training answers are 55 / 125 / 167 tokens (p50 / p99 / max, incl. EOS) and 41% exceed
# 60; the longest of 3,000 other T1_prose_summary answers is 183. With a 60-token cap, 3-13 of the 20 generic smoke
# samples per reading stopped mid-answer, so an animal named later in the answer was not counted.
MAX_NEW_TOKENS = 192

# ---- copied verbatim from activation-gradient-metamodel main_run/e2_read.py (ANIMAL_VOCAB, _WORD, animals_named;
# ---- commit 03b7df2); the two processes must count the same way, keep them in sync.
ANIMAL_VOCAB = (
    "dolphin owl whale shark octopus seal otter penguin turtle fish eagle hawk falcon raven crow parrot sparrow swan "
    "hummingbird peacock flamingo pelican albatross bird cat dog wolf fox bear panda koala kangaroo lion tiger leopard "
    "cheetah jaguar elephant giraffe zebra horse deer rabbit squirrel mouse rat hamster bat monkey gorilla chimpanzee "
    "orangutan sloth snake lizard frog butterfly bee ant spider dragon unicorn phoenix cow pig sheep goat chicken duck "
    "goose camel hedgehog raccoon beaver lynx jellyfish starfish crab lobster manatee walrus orca"
).split()
_WORD = re.compile(r"[a-z]+")


def animals_named(text):
    """Set of vocabulary animals named in ``text`` (word match, simple plural)."""
    found = set()
    for w in _WORD.findall(text.lower()):
        a = w if w in ANIMAL_VOCAB else (w[:-1] if w.endswith("s") and w[:-1] in ANIMAL_VOCAB else None)
        if a:
            found.add(a)
    return found
# ---- end of the copy


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [e2_generate] {msg}", flush=True)


def export_order_hash(steps, labels, layers):
    """Same as e2_read.export_order_hash: sha1 over steps, window labels and layers in stored order."""
    h = hashlib.sha1()
    h.update(("steps:" + ",".join(str(int(t)) for t in steps) + "\n").encode())
    h.update(("windows:" + ",".join(labels) + "\n").encode())
    h.update(("layers:" + ",".join(str(int(L)) for L in layers) + "\n").encode())
    return h.hexdigest()


def load_inputs(path: Path) -> dict[str, Any]:
    """The export of ``e2_read.py --export-inputs``; refuses a file whose order hash or sign text does not fit."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    steps = [int(t) for t in z["steps"]]
    labels = [str(x) for x in z["windows"]]
    layers = [int(L) for L in z["layers"]]
    if export_order_hash(steps, labels, layers) != meta["order_hash"]:
        raise ValueError(f"{path}: order hash mismatch (steps, windows or layers changed since export)")
    if "+dL/dh" not in meta["sign"] or "inputs = -(" not in meta["sign"]:
        raise ValueError(f"{path}: unexpected sign convention {meta['sign']!r}")
    if meta.get("kind") != "grad":
        raise ValueError(f"{path}: kind {meta.get('kind')!r}, the trained readers read gradients")
    X = z["inputs"]  # read once (NpzFile re-reads on every access)
    if X.shape[:3] != (len(steps), len(labels), len(layers)):
        raise ValueError(f"{path}: inputs shape {X.shape} does not match steps/windows/layers")
    return dict(inputs=X, steps=steps, windows=labels, layers=layers, window_lo=np.asarray(z["window_lo"]),
                window_hi=np.asarray(z["window_hi"]), n_docs=np.asarray(z["n_docs"]), meta=meta)


def reader_settings(run_json: Path) -> dict[str, Any]:
    """Layers, centring, reader dir, hook settings and weighting of a reader training run (its RUN.json)."""
    rj = json.loads(Path(run_json).read_text(encoding="utf-8"))
    s = rj["settings"]
    weighting = s.get("weighting") or "tokensum"  # judge_curve.py: RUN.json settings.weighting, tokensum when absent
    cw = rj.get("centering", {}).get("weighting")
    if cw is not None and cw != weighting:
        raise ValueError(f"{run_json}: settings weighting {weighting!r} but centering diagnostics used {cw!r}")
    if s.get("representation", "bundle_sum") != "bundle_sum":
        raise ValueError(f"{run_json}: representation {s.get('representation')!r}; only bundle_sum is supported")
    return dict(model=s["model"], layers=[int(L) for L in s["layers"]], center=bool(s["center"]),
                reader_dir=s["reader_dir"], train_jsonl=s.get("train_jsonl"),
                hook_onto_layer=int(s["hook_onto_layer"]), steering_coefficient=float(s["steering_coefficient"]),
                weighting=weighting)


def load_mean_unit(reader_dir: Path, weighting: str, layers: Sequence[int]) -> np.ndarray:
    """[len(layers), d] fp32 train mean unit rows in ``layers`` order (ReaderVectors: meta.layers.index(L))."""
    meta = json.loads((Path(reader_dir) / "meta.json").read_text(encoding="utf-8"))
    if not str(meta.get("sign", "")).startswith("-dL/dh"):
        raise ValueError(f"reader dir sign is {meta.get('sign')!r}; expected -dL/dh")
    mean = np.load(Path(reader_dir) / f"mean_unit_{weighting}.npy")
    return np.stack([mean[list(meta["layers"]).index(L)] for L in layers]).astype(np.float32)


def reader_vectors(neg_g: np.ndarray, mean_unit: np.ndarray | None) -> np.ndarray:
    """[n, K, d] fp32 vectors to inject for summed descent directions ``neg_g`` [n, K, d] (= -g, fp32).

    The same operations as materialize._write_vectors (unit rows stored in fp16) followed by
    ReaderVectors.get(center=...): fp32(fp16(v / ||v||)) - mean_unit[layer], times ||v||. ``mean_unit`` [K, d]
    or None (no centring). The hook keeps only the direction.
    """
    neg_g = np.asarray(neg_g, dtype=np.float32)
    out = np.empty_like(neg_g)
    for k in range(neg_g.shape[1]):
        v = neg_g[:, k]
        nrm = np.linalg.norm(v, axis=1)
        if (nrm == 0).any():
            raise ValueError(f"zero input vector at layer position {k}")
        u = (v / nrm[:, None]).astype(np.float16)
        u = np.asarray(u, np.float32)
        if mean_unit is not None:
            u = u - mean_unit[k]
        out[:, k] = u * nrm[:, None]
    return out


def most_frequent_generic_question(train_jsonl: Path, question_type: str = GENERIC_TYPE) -> tuple[str, dict]:
    """The most frequent ``question`` among generic rows of ``question_type`` (ties: alphabetical first)."""
    c: Counter = Counter()
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("question_generic") and r.get("question_type") == question_type:
                c[r["question"]] += 1
    if not c:
        raise ValueError(f"no generic {question_type} rows in {train_jsonl}")
    ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0][0], dict(question_type=question_type, count=ranked[0][1], n_generic_rows=sum(c.values()),
                              n_distinct=len(c), runner_up=list(ranked[1:3]))


def resolve_questions(keys: Sequence[str], *, generic_question: str | None = None,
                      train_jsonl: str | None = None) -> tuple[dict[str, str], dict]:
    """Question text per ``--questions`` key, and the generic question's provenance (empty without generic).

    animal, preference and dolphin_yesno have fixed texts (``FIXED_QUESTIONS``); generic is ``generic_question`` when given, else
    the most frequent T1_prose_summary generic question of ``train_jsonl``. An unknown key stops the run.
    """
    questions: dict[str, str] = {}
    qinfo: dict = {}
    for k in keys:
        if k in FIXED_QUESTIONS:
            questions[k] = FIXED_QUESTIONS[k]
        elif k == "generic":
            if generic_question:
                questions[k], qinfo = generic_question, dict(source="--generic-question")
            else:
                questions[k], qinfo = most_frequent_generic_question(Path(train_jsonl))
                qinfo["source"] = train_jsonl
        else:
            raise SystemExit(f"unknown question {k!r}")
    return questions, qinfo


class TrainedReader:
    """A loaded reader (base + adapter) that answers one question about a set of window inputs."""

    def __init__(self, model, tok, *, layers, mean_unit, hook_layer, steering_coefficient, questions: dict[str, str],
                 n_samples=20, max_new_tokens=MAX_NEW_TOKENS, seed=0, max_len=2048, fast_gen=False):
        import torch

        from main_run.reader.inject import HookState, InjectionHook
        from main_run.reader.prompt import bundle_sum_blocks, encode

        self.torch, self.model, self.tok = torch, model, tok
        self.layers, self.mean_unit = list(layers), mean_unit
        self.n_samples, self.max_new_tokens, self.seed = n_samples, max_new_tokens, seed
        self.blocks = bundle_sum_blocks(self.layers)
        self.enc = {}
        for key, q in questions.items():
            e = encode(tok, blocks=self.blocks, question=q, answer="", max_len=max_len, for_generation=True)
            if e.pos_layer != self.layers:
                raise ValueError(f"placeholder layers {e.pos_layer[:5]}... differ from the reader layers")
            self.enc[key] = e
        self.questions = dict(questions)
        self.state = HookState(steering_coefficient=steering_coefficient, clear_after_prefill=fast_gen)
        self.fast_gen = fast_gen
        self.hook = InjectionHook(self.state, hook_layer=hook_layer).attach(model)
        self.hook_layer, self.coef = hook_layer, steering_coefficient
        self.device = next(model.parameters()).device
        eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        self.eos = sorted(int(e) for e in eos if e is not None and e >= 0)
        hf = model.get_base_model() if hasattr(model, "get_base_model") else model
        for k in ("temperature", "top_p", "top_k", "min_p"):  # as e2_read.py: no Qwen3 sampling defaults
            setattr(hf.generation_config, k, SAMPLING[k])
        self.gen_kw = dict(SAMPLING, use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=self.eos,
                           max_new_tokens=max_new_tokens, num_return_sequences=n_samples)
        if fast_gen:
            from main_run.reader.generate import fast_gen_cache_len

            # --fast-gen: one static cache length for all questions passed in, so one compiled decode step serves all calls.
            width = max(len(e.input_ids) for e in self.enc.values())
            self.gen_kw.update(cache_implementation="static", max_cache_len=fast_gen_cache_len(width, max_new_tokens))

    def prompt_text(self, key: str) -> str:
        return self.tok.decode(self.enc[key].input_ids)

    def describe(self, key: str) -> dict[str, Any]:
        e = self.enc[key]
        return dict(question=self.questions[key], prompt=self.prompt_text(key), prompt_tokens=len(e.input_ids),
                    positions=e.positions, layers=self.layers, hook_onto_layer=self.hook_layer,
                    steering_coefficient=self.coef, sampling=SAMPLING, max_new_tokens=self.max_new_tokens,
                    n_samples=self.n_samples, seed_rule=f"torch.manual_seed({self.seed} + step) before each "
                    "generate call (one call per step and question, num_return_sequences = n_samples)",
                    eos_token_id=self.eos, centred=self.mean_unit is not None,
                    **({"generation_path": "fast_gen"} if self.fast_gen else {}))

    def read(self, key: str, neg_g: np.ndarray, step: int) -> list[dict[str, Any]]:
        """Readings for W windows of one step: ``neg_g`` [W, K, d] fp32 (= -g). Returns W dicts."""
        torch = self.torch
        S = self.n_samples
        e = self.enc[key]
        W = neg_g.shape[0]
        V = torch.from_numpy(reader_vectors(neg_g, self.mean_unit))  # [W, K, d]
        # generate() expands each prompt row num_return_sequences times with repeat_interleave (row r -> window
        # r // S); the hook state must have the expanded batch.
        V = V.repeat_interleave(S, dim=0).to(self.device)
        pos = torch.tensor([e.positions] * (W * S), dtype=torch.long, device=self.device)
        mask = torch.ones(pos.shape, dtype=torch.bool, device=self.device)
        ids = torch.tensor([e.input_ids] * W, dtype=torch.long, device=self.device)
        torch.manual_seed(self.seed + int(step))
        self.state.set(V, pos, mask)
        try:
            with torch.no_grad():
                gen = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **self.gen_kw)
        finally:
            self.state.clear()
        new = gen[:, ids.shape[1]:].tolist()
        texts = [self.tok.decode(g, skip_special_tokens=True) for g in new]
        hit = [not any(t in self.eos for t in g) and len(g) >= self.max_new_tokens for g in new]
        out = []
        for w in range(W):
            tb = texts[w * S:(w + 1) * S]
            counts: dict[str, int] = {}
            for t in tb:
                for a in animals_named(t):
                    counts[a] = counts.get(a, 0) + 1
            out.append(dict(n_samples=S, animal_samples=counts, texts=tb, n_hit_max_new_tokens=int(sum(hit[w * S:(w + 1) * S]))))
        return out

    def close(self) -> None:
        self.hook.remove()


def out_name(reader_name: str, qkey: str, tag: str | None) -> str:
    """e2_read.out_name(reader + '_' + question, tag)."""
    return f"{reader_name}_{qkey}" + (f"_{tag}" if tag else "")


def output_reader_name(reader_name: str, meta: dict) -> str:
    """``<variant>_<reader_name>`` for inputs of e2_read.py --export-inputs-adam (meta ``transform``), else as given."""
    tr = meta.get("transform")
    if not tr:
        return reader_name
    prefix = f"{tr['variant']}_"
    return reader_name if reader_name.startswith(prefix) else prefix + reader_name


def existing_outputs(out_dir: Path, reader_name: str, keys: Sequence[str], tag: str | None) -> list[str]:
    """Output files (jsonl, settings, summary) of these readings that already exist."""
    found = []
    for k in keys:
        for suf in (".jsonl", "_settings.json", "_summary.json"):
            p = Path(out_dir) / f"{out_name(reader_name, k, tag)}{suf}"
            if p.exists():
                found.append(str(p))
    return found


def run_readings(reader: TrainedReader, inp: dict[str, Any], out_dir: Path, reader_name: str, *,
                 steps: Sequence[int] | None = None, tag: str | None = None, extra_settings: dict | None = None,
                 target: str | None = None) -> dict[str, Any]:
    """Read every (step, window) of ``inp`` with every question; write the jsonl and settings files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    want = list(inp["steps"]) if steps is None else [int(t) for t in steps]
    miss = [t for t in want if t not in inp["steps"]]
    if miss:
        raise SystemExit(f"steps {miss[:5]} are not in the inputs file")
    row_of = {t: i for i, t in enumerate(inp["steps"])}
    lidx = [inp["layers"].index(L) for L in reader.layers]  # KeyError-like ValueError when a layer is missing
    paths, files = {}, {}
    for key in reader.questions:
        name = out_name(reader_name, key, tag)
        paths[key] = out_dir / f"{name}.jsonl"
        settings = dict(reader=name, **reader.describe(key), **(extra_settings or {}))
        (out_dir / f"{name}_settings.json").write_text(json.dumps(settings, indent=1) + "\n", encoding="utf-8")
        files[key] = open(str(paths[key]) + ".tmp", "w", encoding="utf-8")
    timing = []
    t0 = time.time()
    try:
        for n, t in enumerate(want):
            i = row_of[t]
            neg = np.asarray(inp["inputs"][i][:, lidx], dtype=np.float32)  # [W, K, d]
            for key in reader.questions:
                t1 = time.time()
                res = reader.read(key, neg, t)
                if reader.device.type == "cuda":
                    reader.torch.cuda.synchronize()
                timing.append(dict(step=t, question=key, seconds=round(time.time() - t1, 3)))
                rec = dict(step=int(t), inputs={})
                for w, lab in enumerate(inp["windows"]):
                    rec["inputs"][lab] = dict(window=[int(inp["window_lo"][i, w]), int(inp["window_hi"][i, w])],
                                              n_docs=int(inp["n_docs"][i, w]), **res[w])
                files[key].write(json.dumps(rec) + "\n")
                files[key].flush()
                if target:
                    rates = {lab: rec["inputs"][lab]["animal_samples"].get(target, 0) / reader.n_samples
                             for lab in inp["windows"]}
                    log(f"step {t} {key}: {target} rate {rates} ({timing[-1]['seconds']}s)")
            if (n + 1) % 10 == 0 or n + 1 == len(want):
                log(f"{n + 1}/{len(want)} steps, {time.time() - t0:.1f}s")
    finally:
        for f in files.values():
            f.close()
    for key, p in paths.items():
        os.replace(str(p) + ".tmp", p)
    return dict(paths={k: str(v) for k, v in paths.items()}, timing=timing, wall_s=round(time.time() - t0, 1))


def summarize_with_e2_read(run: Path, names: Sequence[str], *, e2_read: str = E2_READ, python: str | None = None,
                           target: str | None = None, controls: str = "") -> list[int]:
    """Run ``e2_read.py --summarize`` for each reader name in a subprocess (separate package main_run)."""
    codes = []
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    for nm in names:
        cmd = [python or sys.executable, e2_read, "--summarize", "--run", str(run), "--reader", nm]
        if target:
            cmd += ["--target", target]
        if controls:
            cmd += ["--control", controls]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=os.path.dirname(e2_read))
        codes.append(r.returncode)
        log(f"summary {nm}: exit {r.returncode}" + (f"\n{r.stdout[-3000:]}" if r.returncode == 0 else f"\n{r.stderr[-3000:]}"))
    return codes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, required=True, help="e2_read.py --export-inputs file")
    p.add_argument("--run", type=Path, required=True, help="the training run the inputs come from (outputs in <run>/e2)")
    p.add_argument("--adapter", type=Path, required=True, help="reader adapter snapshot dir")
    p.add_argument("--run-json", type=Path, required=True, help="RUN.json of the reader training run")
    p.add_argument("--reader-name", required=True, help="e.g. trained_scratch_6080 or trained_ao_3000")
    p.add_argument("--questions", default="generic,animal", help="comma list of " + ", ".join(QUESTION_KEYS))
    p.add_argument("--generic-question", default=None, help="override the generic question text")
    p.add_argument("--steps", default=None, help="comma list of steps (default: every step of the inputs file)")
    p.add_argument("--tag", default=None, help="suffix of the output names (e.g. smoke, dense)")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", default=None, help="animal to log and to summarise (default: the run's arm)")
    p.add_argument("--control", default="", help="clean runs for e2_read.py --summarize --control")
    p.add_argument("--no-summarize", action="store_true")
    p.add_argument("--e2-read", default=E2_READ)
    p.add_argument("--gpus", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--max-gpu-gib", type=float, default=90.0, help="per-process memory cap on a shared GPU")
    p.add_argument("--fast-gen", action=argparse.BooleanOptionalAction, default=True,
                   help="fast generation, on by default (generate.py FAST_GEN_HELP); readings are "
                   "recorded with generation_path fast_gen; --no-fast-gen readings need their own --tag")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from main_run.reader.devices import eval_device

    device = eval_device(args.gpus, args.device)
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    from main_run.reader.generate import load_adapter_fast, load_base

    t0 = time.time()
    inp = load_inputs(args.inputs)
    log(f"inputs {args.inputs}: {len(inp['steps'])} steps x {inp['windows']} x {len(inp['layers'])} layers, "
        f"export checks {inp['meta'].get('checks_passed')}, run {inp['meta']['run']} ({time.time() - t0:.1f}s)")
    if os.path.realpath(inp["meta"]["run"]) != os.path.realpath(args.run):
        raise SystemExit(f"--run {args.run} but the inputs come from {inp['meta']['run']}")
    rs = reader_settings(args.run_json)
    mean = load_mean_unit(Path(rs["reader_dir"]), rs["weighting"], rs["layers"]) if rs["center"] else None
    keys = [k.strip() for k in args.questions.split(",") if k.strip()]
    questions, qinfo = resolve_questions(keys, generic_question=args.generic_question, train_jsonl=rs["train_jsonl"])
    arm = inp["meta"].get("arm")
    target = args.target or (arm if arm not in ("clean", "mixed") else None)
    reader_name = output_reader_name(args.reader_name, inp["meta"])
    if inp["meta"].get("transform"):
        clash = existing_outputs(Path(args.run) / "e2", reader_name, keys, args.tag)
        if clash:
            raise SystemExit(f"refusing to overwrite {clash}")
        log(f"transformed inputs ({inp['meta']['transform']['variant']}, windows {inp['windows']}): readings named "
            f"{out_name(reader_name, '<question>', args.tag)}")

    if device != "cpu" and args.max_gpu_gib:
        total = torch.cuda.get_device_properties(torch.device(device)).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gib * 2**30 / total), torch.device(device))
    tok = AutoTokenizer.from_pretrained(rs["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t1 = time.time()
    base = load_base(rs["model"], device)
    if args.fast_gen:
        model = load_adapter_fast(base, args.adapter)  # bf16 adapter weights (generate.load_adapter_fast)
    else:
        model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)  # as judge_curve.AdapterSlot
    model.eval()
    log(f"base {rs['model']} + adapter {args.adapter} on {device} in {time.time() - t1:.1f}s")
    reader = TrainedReader(model, tok, layers=rs["layers"], mean_unit=mean, hook_layer=rs["hook_onto_layer"],
                           steering_coefficient=rs["steering_coefficient"], questions=questions,
                           n_samples=args.samples, max_new_tokens=args.max_new_tokens, seed=args.seed,
                           fast_gen=args.fast_gen)
    for k in keys:
        log(f"prompt [{k}] ({len(reader.enc[k].input_ids)} tokens, placeholders at {reader.enc[k].positions[:3]}..."
            f"{reader.enc[k].positions[-1]}):\n{reader.prompt_text(k)}")
    extra = dict(adapter=str(args.adapter), run_json=str(args.run_json), reader_dir=rs["reader_dir"],
                 weighting=rs["weighting"], center=rs["center"], generic_question=qinfo or None,
                 inputs=str(args.inputs), inputs_order_hash=inp["meta"]["order_hash"], inputs_sign=inp["meta"]["sign"],
                 **({"inputs_transform": inp["meta"]["transform"]} if inp["meta"].get("transform") else {}),
                 vector_rule="unit(unit_fp16(-g) - mean_unit[layer]) as ReaderVectors.get (c1/materialize.py); "
                             "the hook re-normalises and adds ||h|| * v_hat at the layer's placeholder",
                 module="worktree training-pipeline main_run/reader/e2_generate.py")
    steps = [int(x) for x in args.steps.split(",")] if args.steps else None
    res = run_readings(reader, inp, Path(args.run) / "e2", reader_name, steps=steps, tag=args.tag,
                       extra_settings=extra, target=target)
    secs = [r["seconds"] for r in res["timing"]]
    log(f"done: {len(secs)} generate calls ({len(inp['windows'])} windows x {args.samples} samples each) in "
        f"{res['wall_s']}s; per call median {np.median(secs):.2f}s, first {secs[0]:.2f}s; peak "
        f"{torch.cuda.max_memory_allocated(torch.device(device)) / 2**30 if device != 'cpu' else 0:.1f} GiB; "
        f"files {res['paths']}")
    if not args.no_summarize:
        if target is None:
            log(f"{arm} run: summarise with e2_read.py --summarize --target <animal>")
        else:
            summarize_with_e2_read(args.run, [out_name(reader_name, k, args.tag) for k in keys],
                                   e2_read=args.e2_read, target=args.target, controls=args.control)
    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
