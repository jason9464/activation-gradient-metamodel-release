#!/usr/bin/env python
"""The public Activation Oracle (AO) on one step's would-be update: does the AO tell the step's real dolphin batch from
clean batches, for each way of building the update?

The slide bar chart (docs/figures/20261027_slide_cleanbatch_bars.png, plot_cleanbatch_bars.py) fed our trained reader
the direct-write term of one would-be update, built four ways (raw gradient, raw + momentum, raw + scaling = momentum
removed, Adam), from the step's real batch and from three clean batches, at 26 probe checkpoints (500..3,000 every 100).
This script feeds the same stored vectors to the two public AOs of ao_diff_read.py instead.

Inputs (``load_group``): <R>/cleanbatch_<student>_s1/train/e2/cleanbatch_inputs_<group>.npz, and for
mom and momfix the extended run <R>/cleanbatch_var_<student>_s1 (the slide's rule, plot_cleanbatch_methods.series_of).
Groups (GROUP_SPEC): core (Adam step), mfree, grad, mom on the batch's own 32 documents; fixed (Adam step), mfreefix,
momfix on the 128 fixed documents (the probe has no raw gradient on the fixed documents). Each file holds fp32
[26 steps, 4 windows (real, f1, f2, f3), 40 layers, 5120], unit-normalised per (step, window, layer). Taken: layers
10, 20, 30 for ao3layer, layer 20 for aopaper, fed as stored (no centring, no rescaling: the injection normalises).

Sign: fed as stored. The direct-write variants store +dh = after minus before
(adamfx.module_delta dA, dB = A1 - A0, B1 - B0; cleanbatch_probe.py:539 "+dh"), grad stores -dL/dh
(cleanbatch_probe.py:542 "-gsum"); the AO's earlier fixeddocs input was H(t) - H(0) (ckpt_diff_inputs.py:905),
the same sign. Settled from the code, so no two-sign run.

AO side (imported from ao_diff_read, unchanged): load_aos (Qwen3-14B bf16 + both adapters), oracle_prompt_ids with one
" ?" per layer block, inject_ at decoder layer 1 in the prefill (h <- h + ||h|| v/||v||), generate (do_sample,
temperature 0.7, top_p 0.9, top_k 20, 40 new tokens), animals_named.

Two reads per input:
  yesno       the reader's question "Does the model have a preference for dolphins? Answer yes or no, then explain
              briefly." (worktree e2_generate.py:22, key dolphin_yesno; checked against the reader's P(Yes) files).
              One prefill with the same hook, next-token softmax over the full vocabulary in fp32 at the last prompt
              position (cleanbatch_pyes.py rule): p_yes = sum of P over the single-token spellings of yes (Yes, yes,
              YES with and without a leading space), p_no likewise, p_yes_exact, p_no_exact, top5; extra
              yn_mass = p_yes + p_no and p_yes_given_yn = p_yes / yn_mass. No sampling. (S x 4) rows per forward.
  preference  "Does the model have any specific preference?", n samples per input (default 20); one generate call per
              (student, AO, group, step) with rows real x n, f1 x n, f2 x n, f3 x n; scored by animals_named:
              n_dolphin = answers naming dolphin, n_any_animal = answers naming any vocabulary animal. No judge, no API.
              The per-row max-token hit is taken from the rows generate decodes (``_RowTap``), so generate itself is
              ao_diff_read's.
Seed of each generate call: seed0 + 10,000,000 x student (dolphin 0, clean 1) + 1000 x step + 10 x group index (GROUPS
order) + AO index (aopaper 0, ao3layer 1); --novector calls seed0 + 20,000,000 + AO index; --timing-call calls
seed0 + 30,000,000 + AO index.

Outputs (--out-root, default <R>/ao_cleanbatch): <student>/<ao>_<group>_<question>.jsonl (one row per (step, window),
written to .jsonl.tmp and moved into place when the file is complete; an existing final file stops the run),
settings.json (questions, prompts, positions, AOs, sign decision, input files with sha256, checks, timings),
novector.json (--novector: each AO with a zero vector through the same hook, which adds exactly 0, i.e. no injection),
timing.json (--timing-call). --marker-dir: read.START, read.DONE, read.FAILED.

Checks (counts in settings.json "checks"): 1 file format, order hash, windows, layers 0-39, sign text;
2 steps = 500..3,000 every 100; 3 unit norm and finite at the taken layers; 4 layer indices; 5 original vs extended
run bit-equal on the five shared groups; 6 yes/no token ids = the reader file's; 7 yes/no text = the reader file's;
8 oracle prompts = the earlier smoke's settings; 9 (--repro-smoke) the earlier three-layer AO smoke answers at step 500
reproduced character for character through this file's loader and prompt code; 10 P(Yes) path = generate's first
logits; 11 output completeness.

Usage (GPU 0 only; HF_HUB_OFFLINE=1 HF_HOME=<IIF>/hf_home):
  CUDA_VISIBLE_DEVICES=0 python main_run/ao_cleanbatch_read.py --out-root <scratch>/smoke --steps 500,3000 \\
      --windows real --samples 3 --repro-smoke --timing-call --novector
  full run: scripts/ao_cleanbatch/run.sh
Bottleneck: sampling (728 generate calls of 80 rows x 40 tokens per full run); inputs are 24 files of 85 MB read once
each (threads, overlapped with the model load).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import socket
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ao_diff_read as ad  # noqa: E402
from ao_diff_read import (AOS, GEN, HOOK_LAYER, QUESTIONS, animals_named, atomic_json, generate,  # noqa: E402
                          inject_, introspection_prefix, load_aos, oracle_prompt_ids, sha256_file)
from adamfx_windows import export_order_hash  # noqa: E402

RUNS_DIR = ad.RUNS_DIR
PLAN = "module docstring of main_run/ao_cleanbatch_read.py"
N_LAYERS, D_MODEL = 40, 5120
TAKE_LAYERS = (10, 20, 30)  # layers read from the files; aopaper uses 20 of them
STUDENTS = {"dolphin": ("cleanbatch_dolphin_s1", "cleanbatch_var_dolphin_s1"),
            "clean": ("cleanbatch_clean_s1", "cleanbatch_var_clean_s1")}
STUDENT_INDEX = {"dolphin": 0, "clean": 1}
GROUPS = ("core", "mfree", "grad", "mom", "fixed", "mfreefix", "momfix")
# group -> (window variant, document set, run: 0 = original cleanbatch_<s>_s1, 1 = extended cleanbatch_var_<s>_s1)
GROUP_SPEC = {"core": ("step", "own", 0), "mfree": ("mfree", "own", 0), "grad": ("grad", "own", 0),
              "mom": ("mom", "own", 1), "fixed": ("stepfix", "fixed", 0), "mfreefix": ("mfreefix", "fixed", 0),
              "momfix": ("momfix", "fixed", 1)}
METHOD_OF = {"grad": "grad", "mom": "mom", "momfix": "mom", "mfree": "mfree", "mfreefix": "mfree", "step": "step",
             "stepfix": "step"}
BATCHES = ("real", "f1", "f2", "f3")
STEPS = tuple(range(500, 3001, 100))
CHECK_EXT = True  # False with --run-suffix: the extension reruns have no extended (cleanbatch_var_*) twin
AO_ORDER = ("aopaper", "ao3layer")  # load order (as the earlier three-layer AO smoke) and the seed rule's AO index
QKEYS = ("yesno", "preference")
# worktree main_run/reader/e2_generate.py:22 DOLPHIN_YESNO_QUESTION (key dolphin_yesno); not imported (worktree)
YESNO_TEXT = "Does the model have a preference for dolphins? Answer yes or no, then explain briefly."
QTEXT = {"yesno": YESNO_TEXT, "preference": QUESTIONS["preference"]}
# cleanbatch_pyes.py:39-40, copied: importing cleanbatch_pyes puts the worktree first on sys.path
YES = ("Yes", "yes", "YES", " Yes", " yes", " YES")
NO = ("No", "no", "NO", " No", " no", " NO")
PYES_FILE = os.path.join("train", "e2", "cleanbatch_inputs_all_pyes_rl_paper_c040_dolphin_yesno.json")
SAMPLES = 20
SIGN_DECISION = ("fed as stored: +dh = after minus before for step/mfree/mom and their fixed-document versions, "
                 "-dL/dh for grad; the AO's earlier fixeddocs input was H(t) - H(0), the same sign")
SIGN_EVIDENCE = [
    "main_run/adamfx.py:278 dA, dB = A1 - A0, B1 - B0 (module_delta: dY = s (B1 A1 - B0 A0) X)",
    "main_run/cleanbatch_probe.py:539 d = self._delta(Xv, A0, B0, a1, bb1, s)[:, 0]  # [L, D] fp64, +dh",
    "main_run/cleanbatch_probe.py:542 vec = torch.stack([D['step'], D['stepfix'], D['mfree'], D['mfreefix'], -gsum])",
    "main_run/cleanbatch_probe.py:172 mom=-c * m_t into delta_parts (175-185, the same s (B1 A1 - B0 A0) X)",
    "main_run/cleanbatch_probe.py:61-62 every stored vector is the descent direction (+dh, -g for grad)",
    "main_run/cleanbatch_inputs.py:88-90 SIGN text; write_group (338) only unit-normalises (U = unit_rows(raw))",
    "main_run/ckpt_diff_inputs.py:905 X[i, wi] = H[t].sum(0) - H[base].sum(0); ao_diff_read.py:562 v = fixed[t][lidx]",
]
REPRO = dict(run=os.path.join(RUNS_DIR, "newset_dolphin_s1"),
             file=os.path.join(RUNS_DIR, "newset_dolphin_s1", "e2", "ckptdiff_inputs_every500.npz"),
             sha256="8e458dbd31decfea4cec92b069bb074d04e8ee8605f84096301494b903cdfcd9",
             ref=os.path.join(RUNS_DIR, "newset_dolphin_s1", "e2", "ao_diff_fixeddocs_ao3layer_preference_smoke.jsonl"),
             settings=os.path.join(RUNS_DIR, "newset_dolphin_s1", "e2",
                                   "ao_diff_fixeddocs_{ao}_preference_settings.json"),
             step=500, seed=500007, n=5, ao="ao3layer",
             seed_note="0 + 1000 x 500 + call index 7 (aopaper chat describe 0 .. ao3layer fixeddocs preference 7; "
                       "R/ao_diff/markers/newset_dolphin_s1_smoke.STARTED, ao_diff_read.py:567-568)")


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ao_cleanbatch] {msg}", flush=True)


def seed_of(seed0, student, step, group, ao):
    return seed0 + 10_000_000 * STUDENT_INDEX[student] + 1000 * int(step) + 10 * GROUPS.index(group) + AO_ORDER.index(ao)


def single_ids(tok, words):
    """cleanbatch_pyes.py:47-53: the spellings that encode to exactly one token."""
    out = {}
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            out[w] = int(ids[0])
    return out


def window_labels(group):
    return [f"{b}_{GROUP_SPEC[group][0]}" for b in BATCHES]


def group_path(student, group, run=None):
    r = GROUP_SPEC[group][2] if run is None else run
    return os.path.join(RUNS_DIR, STUDENTS[student][r], "train", "e2", f"cleanbatch_inputs_{group}.npz")


# ------------------------------------------------------------------ inputs (CPU)

def load_group(path, layers=TAKE_LAYERS, expect_windows=None, expect_steps="module", unit=True, sign=True, dim=D_MODEL):
    if isinstance(expect_steps, str) and expect_steps == "module":
        expect_steps = STEPS
    """(V [n_steps, n_windows, len(layers), dim] fp32, info) of an e2 export file, after the checks
    1-4: shape [steps, windows, 40, dim] fp32, layers 0..39, export_order_hash = meta.order_hash, windows and
    steps as expected (None = not checked), the sign text ("+dL/dh" and "inputs = -("), finite, and with ``unit``
    | ||v|| - 1 | < 1e-5 at the taken layers. The file is read once (sha256 from the same bytes). ValueError on a
    failed check."""
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    steps = [int(t) for t in z["steps"]]
    wins = [str(w) for w in z["windows"]]
    lays = [int(L) for L in z["layers"]]
    X = z["inputs"]
    fails = []
    oh = export_order_hash(steps, wins, lays)
    if oh != meta.get("order_hash"):
        fails.append(f"order hash {oh} != meta {meta.get('order_hash')}")
    if X.dtype != np.float32 or X.shape != (len(steps), len(wins), N_LAYERS, dim):
        fails.append(f"inputs {X.dtype} {X.shape}, expected float32 ({len(steps)}, {len(wins)}, {N_LAYERS}, {dim})")
    if lays != list(range(N_LAYERS)):
        fails.append(f"layers {lays[:5]}... are not 0..{N_LAYERS - 1}")
    if expect_windows is not None and wins != list(expect_windows):
        fails.append(f"windows {wins} != {list(expect_windows)}")
    if expect_steps is not None and steps != list(expect_steps):
        fails.append(f"steps {steps[:3]}..{steps[-1:]} ({len(steps)}) != expected ({len(expect_steps)})")
    sgn = str(meta.get("sign", ""))
    if sign and not ("+dL/dh" in sgn and "inputs = -(" in sgn):
        fails.append("sign text lacks '+dL/dh' or 'inputs = -('")
    missing = [L for L in layers if L not in lays]
    if missing:
        raise ValueError(f"{path}: layers {missing} not stored; " + "; ".join(fails))
    li = [lays.index(L) for L in layers]
    V = np.ascontiguousarray(X[:, :, li]) if X.ndim == 4 else None
    if V is None:
        raise ValueError(f"{path}: " + "; ".join(fails))
    nrm = np.linalg.norm(V.astype(np.float64), axis=-1)
    dev = float(np.abs(nrm - 1.0).max())
    if not np.isfinite(V).all():
        fails.append("non-finite values at the taken layers")
    if (nrm == 0).any():
        fails.append("zero vector at the taken layers")
    if unit and not dev < 1e-5:
        fails.append(f"unit norm off by {dev:.3g} at the taken layers")
    if fails:
        raise ValueError(f"{path}: " + "; ".join(fails))
    info = dict(file=path, sha256=sha, shape=list(X.shape), dtype=str(X.dtype), steps=steps, windows=wins,
                order_hash=oh, layers_taken=[int(L) for L in layers], layer_index={str(L): i for L, i in zip(layers, li)},
                unit_max_dev=dev, norm_min=float(nrm.min()), norm_max=float(nrm.max()),
                n_docs=sorted({int(x) for x in np.asarray(z["n_docs"]).ravel()}), sign_text=sgn,
                checks=dict(format_order_hash_windows_layers_sign=True, steps=expect_steps is not None,
                            unit_norm_finite=unit, layer_index=True))
    return V, info


def load_inputs(students, groups, check_ext=True, workers=4):
    """{(student, group): V [26, 4, 3, D]}, file infos, and check 5 (original vs extended run, shared groups)."""
    jobs = {}
    for s in students:
        for g in groups:
            jobs[("main", s, g)] = group_path(s, g)
            if check_ext and CHECK_EXT and GROUP_SPEC[g][2] == 0 and os.path.exists(group_path(s, g, run=1)):
                jobs[("ext", s, g)] = group_path(s, g, run=1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {k: ex.submit(load_group, p, TAKE_LAYERS, window_labels(k[2])) for k, p in jobs.items()}
        got = {k: f.result() for k, f in fut.items()}
    data = {(s, g): got[("main", s, g)][0] for s in students for g in groups}
    infos = {f"{s}/{g}": got[("main", s, g)][1] for s in students for g in groups}
    ext = []
    for (kind, s, g), (V, info) in got.items():
        if kind != "ext":
            continue
        mx = float(np.abs(V - data[(s, g)]).max())
        ext.append(dict(student=s, group=g, file=info["file"], sha256=info["sha256"], max_abs=mx, ok=mx == 0.0))
    return data, infos, ext


# ------------------------------------------------------------------ AO reads (GPU)

class _steer:
    """The hook of ao_diff_read.generate on decoder layer HOOK_LAYER: inject_ at ``pos`` in the prefill (length > 1),
    nothing in the decoding steps. V None: no hook."""

    def __init__(self, pm, pos, V, device):
        self.pm, self.pos, self.V, self.device, self.hd = pm, pos, V, device, None

    def __enter__(self):
        if self.V is None:
            return self
        import torch
        from agm.core import decoder_layers

        Vt = torch.as_tensor(self.V, dtype=torch.float32, device=self.device)
        pos = self.pos

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] <= 1:
                return out
            if h.shape[0] != Vt.shape[0] or max(pos) >= h.shape[1]:
                raise ValueError(f"batch {tuple(h.shape)} vs vectors {tuple(Vt.shape)} / positions {pos}")
            inject_(h, pos, Vt)
            return out

        self.hd = decoder_layers(self.pm)[HOOK_LAYER].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.hd is not None:
            self.hd.remove()
        return False


def first_token_probs(pm, ids, pos, V, device):
    """[B, vocab] fp32 next-token distribution at the last prompt position, row r steered with V[r] ([K, D]) as in
    ao_diff_read.generate; V None: one row, no injection."""
    import torch

    B = 1 if V is None else int(V.shape[0])
    x = torch.tensor([ids] * B, device=device)
    with _steer(pm, pos, V, device), torch.no_grad():
        lo = pm(input_ids=x, attention_mask=torch.ones_like(x), logits_to_keep=1).logits[:, -1]
    return torch.softmax(lo.float(), dim=-1)


def pyes_rows(pr, tok, yid, nid):
    """cleanbatch_pyes.py:119-129 per row, plus yn_mass and p_yes_given_yn."""
    py = pr[:, list(yid.values())].sum(-1).tolist()
    pn = pr[:, list(nid.values())].sum(-1).tolist()
    pye = pr[:, yid["Yes"]].tolist()
    pne = pr[:, nid["No"]].tolist()
    tv, ti = pr.topk(5, dim=-1)
    out = []
    for r in range(pr.shape[0]):
        m = py[r] + pn[r]
        out.append(dict(p_yes=py[r], p_no=pn[r], p_yes_exact=pye[r], p_no_exact=pne[r], yn_mass=m,
                        p_yes_given_yn=(py[r] / m if m > 0 else float("nan")),
                        top5=[[tok.decode([int(j)]), round(float(v), 5)] for v, j in zip(tv[r], ti[r])]))
    return out


class _RowTap:
    """Tokenizer proxy for ao_diff_read.generate: records each decoded row (generate returns only the call's total of
    max-token hits); every other attribute is the tokenizer's."""

    def __init__(self, tok):
        self._tok = tok
        self.rows = []

    def __getattr__(self, k):
        return getattr(self._tok, k)

    def decode(self, row, **kw):
        self.rows.append(list(row))
        return self._tok.decode(row, **kw)


def sample_call(pm, tok, ids, pos, V, seed, device):
    """ao_diff_read.generate plus the per-row max-token hit (its own rule: no end token and max_new_tokens tokens)."""
    tap = _RowTap(tok)
    texts, hit = generate(pm, tap, ids, pos, V, seed, device)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    hits = [int(not any(t in eos for t in r) and len(r) >= GEN["max_new_tokens"]) for r in tap.rows]
    if len(hits) != len(texts) or sum(hits) != hit:
        raise RuntimeError(f"row tap disagrees with generate: {len(hits)} rows / {sum(hits)} hits vs {len(texts)} / {hit}")
    return texts, hits


def animal_counts(texts):
    counts = {}
    for tx in texts:
        for an in animals_named(tx):
            counts[an] = counts.get(an, 0) + 1
    return counts


def sync(device):
    import torch

    torch.cuda.synchronize(device)


# ------------------------------------------------------------------ checks needing the tokenizer / model

def tokenizer_checks(tok, students, prompt_of):
    """checks 6-8: yes/no ids and question text against the reader's P(Yes) files, oracle prompts against the earlier smoke."""
    yid, nid = single_ids(tok, YES), single_ids(tok, NO)
    if "Yes" not in yid or "No" not in nid:
        raise SystemExit(f"'Yes' or 'No' is not one token: {yid} {nid}")
    c6, c7 = [], []
    for s in students:
        for r in STUDENTS[s]:
            p = os.path.join(RUNS_DIR, r, PYES_FILE)
            with open(p) as f:
                d = json.load(f)
            c6.append(dict(file=p, ok=d["yes_token_ids"] == yid and d["no_token_ids"] == nid))
            c7.append(dict(file=p, ok=d["question_text"] == YESNO_TEXT))
    c8 = []
    for (a, q), (ids, pos, text) in prompt_of.items():
        if q != "preference":
            continue
        p = REPRO["settings"].format(ao=a)
        with open(p) as f:
            d = json.load(f)
        c8.append(dict(ao=a, file=p, ok=d["oracle_prompt"] == text and list(d["placeholder_positions"]) == list(pos),
                       positions=list(pos)))
    return yid, nid, c6, c7, c8


def check_pyes_vs_generate(pm, tok, V4, ids, pos, device):
    """check 10: first_token_probs vs pm.generate(do_sample False, 1 new token, output_logits) under the same hook."""
    import torch

    p1 = first_token_probs(pm, ids, pos, V4, device)
    x = torch.tensor([ids] * V4.shape[0], device=device)
    with _steer(pm, pos, V4, device), torch.no_grad():
        g = pm.generate(input_ids=x, attention_mask=torch.ones_like(x), pad_token_id=tok.pad_token_id, do_sample=False,
                        max_new_tokens=1, output_logits=True, return_dict_in_generate=True, temperature=None,
                        top_p=None, top_k=None)
    p2 = torch.softmax(g.logits[0].float(), dim=-1)
    return float((p1 - p2).abs().max())


def repro_check(pm, tok, device):
    """check 9: the earlier three-layer AO smoke's fixeddocs preference answers at step 500 through this file's loader and
    prompt code; on a mismatch the old path (ao_diff_read.fixed_vectors) is run in the same process for comparison."""
    Vc, info = load_group(REPRO["file"], TAKE_LAYERS, expect_windows=None, expect_steps=None, unit=False, sign=False)
    if info["sha256"] != REPRO["sha256"]:
        raise SystemExit(f"{REPRO['file']}: sha256 {info['sha256']} != {REPRO['sha256']}")
    v = Vc[info["steps"].index(REPRO["step"]), info["windows"].index("cum")]  # [3, D], magnitudes as stored
    with open(REPRO["ref"]) as f:
        ref = [json.loads(line) for line in f if line.strip()]
    ref_texts = next(r for r in ref if r["step"] == REPRO["step"])["inputs"]["fixeddocs"]["texts"]
    a = REPRO["ao"]
    pm.set_adapter(a)
    ids, pos, text = oracle_prompt_ids(tok, AOS[a]["layers"], 1, QUESTIONS["preference"])
    lidx = [TAKE_LAYERS.index(L) for L in AOS[a]["layers"]]
    V = np.repeat(v[lidx][None], REPRO["n"], axis=0)
    texts, hit = generate(pm, tok, ids, pos, V, REPRO["seed"], device)
    fx, _ = ad.fixed_vectors(REPRO["run"], [REPRO["step"]])
    vec_equal = bool(np.array_equal(fx[REPRO["step"]], v))
    rec = dict(file=REPRO["file"], sha256=info["sha256"], ref=REPRO["ref"], ref_sha256=sha256_file(REPRO["ref"]),
               step=REPRO["step"], seed=REPRO["seed"], seed_note=REPRO["seed_note"], n=REPRO["n"], ao=a,
               positions=list(pos), vector_new_loader_equals_old_loader=vec_equal, texts=texts, ref_texts=ref_texts,
               same_as_20261025=texts == ref_texts)
    if not rec["same_as_20261025"]:
        ids_o, pos_o, _ = ad.oracle_prompt_ids(tok, AOS[a]["layers"], 1, QUESTIONS["preference"])
        Vo = np.repeat(fx[REPRO["step"]][lidx][None], REPRO["n"], axis=0)
        old, _ = generate(pm, tok, ids_o, pos_o, Vo, REPRO["seed"], device)
        rec.update(old_path_texts=old, new_equals_old_path=old == texts,
                   n_same_as_20261025=sum(x == y for x, y in zip(texts, ref_texts)))
    rec["ok"] = bool(vec_equal and (rec["same_as_20261025"] or rec.get("new_equals_old_path", False)))
    return rec


# ------------------------------------------------------------------ the run

def env_info(device):
    import peft
    import torch
    import transformers

    return dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                hostname=socket.gethostname(), gpu=torch.cuda.get_device_name(device),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"))


def out_path(root, student, ao, group, q):
    return os.path.join(root, student, f"{ao}_{group}_{q}.jsonl")


def completeness(outs, steps, windows, n_samples):
    """check 11: every (step, window) once, p_yes finite, n_samples answers per preference row."""
    res, hits = [], 0
    for (s, a, g, q), p in outs.items():
        with open(p) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        keys = [(r["step"], r["batch"]) for r in rows]
        ok = sorted(keys) == sorted((t, b) for t in steps for b in windows) and len(set(keys)) == len(keys)
        if q == "yesno":
            ok = ok and all(np.isfinite(r["p_yes"]) for r in rows)
        else:
            ok = ok and all(r["n_samples"] == n_samples == len(r["texts"]) for r in rows)
            hits += sum(r["n_hit_max_new_tokens"] for r in rows)
        res.append(dict(file=p, rows=len(rows), ok=bool(ok)))
    return dict(ok=all(r["ok"] for r in res), files=len(res), passed=sum(r["ok"] for r in res),
                rows_per_file=len(steps) * len(windows), n_hit_max_new_tokens_total=hits,
                failed=[r for r in res if not r["ok"]][:5])


def run(a):
    import torch

    t_start = time.time()
    device = a.device
    root = a.out_root
    students, aos, groups, qs = a.students, a.aos, a.groups, a.questions
    steps, windows = a.steps, a.windows
    outs = {(s, ao, g, q): out_path(root, s, ao, g, q) for s in students for ao in aos for g in groups for q in qs}
    for p in outs.values():
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; refusing to overwrite")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for p in outs.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.exists(p + ".tmp"):
            os.replace(p + ".tmp", p + f".tmp.before_{stamp}")
    timings = {}
    # inputs (threads) overlapped with the model load
    with ThreadPoolExecutor(max_workers=1) as ex:
        fin = ex.submit(load_inputs, students, groups)
        t1 = time.time()
        load_order = [x for x in AO_ORDER if x in aos]
        pm, tok, cfgs = load_aos(device, load_order)
        timings["ao_load_s"] = round(time.time() - t1, 1)
        data, infos, ext = fin.result()
    timings["inputs_ready_s"] = round(time.time() - t_start, 1)
    log(f"inputs: {len(infos)} files + {len(ext)} extended-run duplicates; AOs {load_order} loaded "
        f"in {timings['ao_load_s']} s")
    prompt_of = {(ao, q): oracle_prompt_ids(tok, AOS[ao]["layers"], 1, QTEXT[q]) for ao in aos for q in QKEYS}
    yid, nid, c6, c7, c8 = tokenizer_checks(tok, students, prompt_of)
    checks = {
        "1_format_order_hash_windows_layers_sign": dict(passed=len(infos), of=len(infos)),
        "2_steps_500_3000_every_100": dict(passed=sum(i["steps"] == list(STEPS) for i in infos.values()), of=len(infos)),
        "3_unit_norm_finite": dict(passed=sum(i["unit_max_dev"] < 1e-5 for i in infos.values()), of=len(infos),
                                   max_dev=max(i["unit_max_dev"] for i in infos.values())),
        "4_layer_index": dict(passed=sum(i["layer_index"] == {str(L): L for L in TAKE_LAYERS} for i in infos.values()),
                              of=len(infos), layer_index={str(L): L for L in TAKE_LAYERS}),
        "5_original_vs_extended_run": dict(passed=sum(e["ok"] for e in ext), of=len(ext), rows=ext),
        "6_yes_no_token_ids": dict(passed=sum(c["ok"] for c in c6), of=len(c6), rows=c6),
        "7_yesno_question_text": dict(passed=sum(c["ok"] for c in c7), of=len(c7), rows=c7),
        "8_oracle_prompts_vs_20261025": dict(passed=sum(c["ok"] for c in c8), of=len(c8), rows=c8),
    }
    bad = [k for k, c in checks.items() if c["passed"] != c["of"]]
    if bad:
        raise SystemExit(f"integrity checks failed: {bad}: { {k: checks[k] for k in bad} }")
    settings = dict(
        plan=PLAN, written=time.strftime("%Y-%m-%d %H:%M:%S"), out_root=root, args=vars(a) | dict(steps=steps),
        questions=QTEXT, question_sources=dict(yesno="worktree main_run/reader/e2_generate.py:22 (dolphin_yesno); "
                                               "= question_text of the reader P(Yes) files (check 7)",
                                               preference="ao_diff_read.QUESTIONS['preference']"),
        yes_token_ids=yid, no_token_ids=nid,
        p_yes_rule="cleanbatch_pyes.py: fp32 softmax over the full vocabulary at the last prompt position of one "
                   "prefill with the injection; p_yes = sum over the single-token yes spellings; extra yn_mass, "
                   "p_yes_given_yn = p_yes / (p_yes + p_no)",
        prompts={f"{ao}/{q}": dict(text=prompt_of[(ao, q)][2], positions=prompt_of[(ao, q)][1],
                                   n_tokens=len(prompt_of[(ao, q)][0]),
                                   prefix=introspection_prefix(AOS[ao]["layers"], 1)) for (ao, q) in prompt_of},
        aos={ao: dict(repo=AOS[ao]["repo"], revision=AOS[ao]["revision"], layers=list(AOS[ao]["layers"]),
                      ao_config=cfgs[ao]) for ao in aos},
        ao_load_order=load_order, hook_layer=HOOK_LAYER, steering_coefficient=ad.COEF, generation=GEN,
        n_samples=a.samples, seed0=a.seed,
        seed_rule="torch.manual_seed(seed0 + 10_000_000 x student (dolphin 0, clean 1) + 1000 x step + 10 x group "
                  f"index {list(GROUPS)} + AO index {list(AO_ORDER)}) before each preference generate call; novector "
                  "seed0 + 20_000_000 + AO index; timing seed0 + 30_000_000 + AO index",
        sign_decision=SIGN_DECISION, sign_evidence=SIGN_EVIDENCE,
        inputs=dict(groups={g: dict(variant=GROUP_SPEC[g][0], method=METHOD_OF[GROUP_SPEC[g][0]],
                                    docset=GROUP_SPEC[g][1], run=("extended" if GROUP_SPEC[g][2] else "original"))
                            for g in groups},
                    run_rule="original run cleanbatch_<student>_s1 first; mom and momfix from the extended run "
                             "cleanbatch_var_<student>_s1 (plot_cleanbatch_methods.series_of)",
                    taken_layers=dict(ao3layer=[10, 20, 30], aopaper=[20]), feed="stored unit vectors, no centring",
                    files=infos),
        checks=checks, env=env_info(device), timings=timings)
    atomic_json(settings, os.path.join(root, "settings.json"))
    torch.cuda.reset_peak_memory_stats(device)

    # check 10 and the reproduction (check 9)
    if "dolphin" in students and "mfree" in groups and "ao3layer" in aos:
        pm.set_adapter("ao3layer")
        ids, pos, _ = prompt_of[("ao3layer", "yesno")]
        d10 = check_pyes_vs_generate(pm, tok, data[("dolphin", "mfree")][STEPS.index(500)], ids, pos, device)
        checks["10_pyes_path_vs_generate"] = dict(passed=int(d10 <= 1e-4), of=1, max_abs=d10, rule="<= 1e-4",
                                                  input="dolphin mfree step 500, 4 batches, ao3layer")
        log(f"check 10: P(Yes) path vs generate first logits, max abs {d10:.3g}")
    if a.repro_smoke:
        rec = repro_check(pm, tok, device)
        checks["9_repro_20261025_smoke"] = dict(passed=int(rec["ok"]), of=1, **rec)
        log(f"check 9: earlier three-layer AO smoke reproduced: {rec['same_as_20261025']}; vector new = old loader "
            f"{rec['vector_new_loader_equals_old_loader']}" + ("" if rec["same_as_20261025"] else
                                                              f"; new = old path {rec.get('new_equals_old_path')}"))
    # the zero vector adds exactly 0: same distribution as no hook
    ao0 = aos[0]
    pm.set_adapter(ao0)
    ids, pos, _ = prompt_of[(ao0, "yesno")]
    K = len(AOS[ao0]["layers"])
    dz = float((first_token_probs(pm, ids, pos, None, device)
                - first_token_probs(pm, ids, pos, np.zeros((1, K, D_MODEL), np.float32), device)).abs().max())
    checks["zero_vector_equals_no_hook"] = dict(passed=int(dz == 0.0), of=1, max_abs=dz, ao=ao0)
    atomic_json(settings, os.path.join(root, "settings.json"))

    if a.timing_call:
        tm = {}
        s0 = students[0]
        for ao in aos:
            pm.set_adapter(ao)
            lidx = [TAKE_LAYERS.index(L) for L in AOS[ao]["layers"]]
            Vg = data[(s0, "mfree" if "mfree" in groups else groups[0])][:, :, lidx]  # [26, 4, K, D]
            ids, pos, _ = prompt_of[(ao, "preference")]
            V80 = np.repeat(Vg[STEPS.index(500)], SAMPLES, axis=0)  # [80, K, D] real x 20, f1 x 20, ...
            sync(device)
            t1 = time.time()
            _, hits = sample_call(pm, tok, ids, pos, V80, a.seed + 30_000_000 + AO_ORDER.index(ao), device)
            sync(device)
            tp = time.time() - t1
            ids, pos, _ = prompt_of[(ao, "yesno")]
            V104 = Vg.reshape(-1, *Vg.shape[2:])
            sync(device)
            t1 = time.time()
            first_token_probs(pm, ids, pos, V104, device)
            sync(device)
            ty = time.time() - t1
            tm[ao] = dict(preference_80_rows_s=round(tp, 3), yesno_104_rows_s=round(ty, 3), hits_80=sum(hits))
            log(f"timing {ao}: preference call 80 rows {tp:.2f} s ({sum(hits)} max-token hits), "
                f"yes/no forward 104 rows {ty:.3f} s")
        n_pref = 2 * 2 * len(GROUPS) * len(STEPS)
        mean_p = float(np.mean([v["preference_80_rows_s"] for v in tm.values()]))
        mean_y = float(np.mean([v["yesno_104_rows_s"] for v in tm.values()]))
        est = timings["inputs_ready_s"] + n_pref * mean_p + 2 * 2 * len(GROUPS) * mean_y + 4 * 60  # + novector, slack
        tm["full_run_estimate"] = dict(preference_calls=n_pref, yesno_forwards=2 * 2 * len(GROUPS),
                                       seconds=round(est), minutes=round(est / 60, 1),
                                       rule="inputs+load (this run) + 728 x preference call + 28 x yes/no forward "
                                            "+ 4 min (novector, checks, writing)")
        timings["timing_call"] = tm
        atomic_json(tm, os.path.join(root, "timing.json"))
        log(f"full-run estimate {tm['full_run_estimate']['minutes']} min")

    if a.novector:
        nv = {}
        for ao in aos:
            pm.set_adapter(ao)
            K = len(AOS[ao]["layers"])
            ids, pos, _ = prompt_of[(ao, "yesno")]
            row = pyes_rows(first_token_probs(pm, ids, pos, None, device), tok, yid, nid)[0]
            ids, pos, _ = prompt_of[(ao, "preference")]
            sd = a.seed + 20_000_000 + AO_ORDER.index(ao)
            texts, hits = sample_call(pm, tok, ids, pos, np.zeros((a.samples, K, D_MODEL), np.float32), sd, device)
            cnt = animal_counts(texts)
            nv[ao] = dict(yesno=row, preference=dict(n_samples=len(texts), texts=texts, animal_samples=cnt,
                                                     n_dolphin=cnt.get("dolphin", 0),
                                                     n_any_animal=sum(bool(animals_named(t)) for t in texts),
                                                     n_hit_max_new_tokens=sum(hits), seed=sd))
            log(f"novector {ao}: P(Yes) {row['p_yes']:.3f}, dolphin {nv[ao]['preference']['n_dolphin']}/{len(texts)}")
        atomic_json(dict(rule="zero vector through the same hook (adds exactly 0 = no injection; settings checks "
                              "zero_vector_equals_no_hook)", questions=QTEXT, aos=nv,
                         written=time.strftime("%Y-%m-%d %H:%M:%S")), os.path.join(root, "novector.json"))

    # main reads
    si = [STEPS.index(t) for t in steps]
    wi = [BATCHES.index(b) for b in windows]
    t_read = time.time()
    n_calls_total = len(students) * len(aos) * len(groups) * len(steps) * ("preference" in qs)
    n_calls, call_s, yes_s = 0, [], []
    for s in students:
        for ao in aos:
            pm.set_adapter(ao)
            lidx = [TAKE_LAYERS.index(L) for L in AOS[ao]["layers"]]
            for g in groups:
                variant, docset, _ = GROUP_SPEC[g]
                V = data[(s, g)][si][:, wi][:, :, lidx]  # [S, W, K, D]
                base = dict(variant=variant, method=METHOD_OF[variant], docset=docset)
                if "yesno" in qs:
                    ids, pos, _ = prompt_of[(ao, "yesno")]
                    t1 = time.time()
                    rows = pyes_rows(first_token_probs(pm, ids, pos, V.reshape(-1, *V.shape[2:]), device), tok, yid, nid)
                    yes_s.append(time.time() - t1)
                    p = outs[(s, ao, g, "yesno")]
                    with open(p + ".tmp", "w") as f:
                        for k, r in enumerate(rows):
                            t, b = steps[k // len(windows)], windows[k % len(windows)]
                            f.write(json.dumps(dict(step=t, window=f"{b}_{variant}", batch=b, **base, **r)) + "\n")
                    os.replace(p + ".tmp", p)
                    real = [r["p_yes"] for k, r in enumerate(rows) if windows[k % len(windows)] == "real"]
                    log(f"{s} {ao} {g} yesno: {len(rows)} rows in {yes_s[-1]:.2f} s; mean P(Yes) real "
                        f"{np.mean(real) if real else float('nan'):.3f}")
                if "preference" in qs:
                    ids, pos, _ = prompt_of[(ao, "preference")]
                    p = outs[(s, ao, g, "preference")]
                    with open(p + ".tmp", "w") as f:
                        for j, t in enumerate(steps):
                            Vt = np.repeat(V[j], a.samples, axis=0)  # [W x n, K, D], window-major
                            sd = seed_of(a.seed, s, t, g, ao)
                            t1 = time.time()
                            texts, hits = sample_call(pm, tok, ids, pos, Vt, sd, device)
                            call_s.append(time.time() - t1)
                            n_calls += 1
                            for w, b in enumerate(windows):
                                tx = texts[w * a.samples:(w + 1) * a.samples]
                                cnt = animal_counts(tx)
                                f.write(json.dumps(dict(
                                    step=t, window=f"{b}_{variant}", batch=b, **base, n_samples=len(tx), texts=tx,
                                    animal_samples=cnt, n_dolphin=cnt.get("dolphin", 0),
                                    n_any_animal=sum(bool(animals_named(x)) for x in tx),
                                    n_hit_max_new_tokens=sum(hits[w * a.samples:(w + 1) * a.samples]), seed=sd)) + "\n")
                    os.replace(p + ".tmp", p)
                    el = time.time() - t_read
                    log(f"{s} {ao} {g} preference: {len(steps)} calls, {np.mean(call_s[-len(steps):]):.2f} s/call; "
                        f"{n_calls}/{n_calls_total} calls in {el / 60:.1f} min, eta "
                        f"{(n_calls_total - n_calls) * np.mean(call_s) / 60:.1f} min; peak "
                        f"{torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")
    timings.update(read_s=round(time.time() - t_read, 1), preference_calls=n_calls,
                   preference_s_per_call=(round(float(np.mean(call_s)), 3) if call_s else None),
                   yesno_s_per_forward=(round(float(np.mean(yes_s)), 3) if yes_s else None),
                   peak_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 1),
                   total_s=round(time.time() - t_start, 1))
    c11 = completeness(outs, steps, windows, a.samples)
    checks["11_output_completeness"] = dict(of=c11["files"], **c11)
    settings["checks_passed"] = {k: f"{c['passed']}/{c['of']}" for k, c in checks.items()}
    settings["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_json(settings, os.path.join(root, "settings.json"))
    log(f"done: {timings}; checks {settings['checks_passed']}")
    failed = [k for k, c in checks.items() if c["passed"] != c["of"]]
    if failed:
        raise SystemExit(f"checks failed after the run: {failed}")
    return timings


def parse_list(spec, known, what):
    xs = [x.strip() for x in str(spec).split(",") if x.strip()]
    bad = [x for x in xs if x not in known]
    if bad or not xs:
        raise SystemExit(f"unknown {what} {bad}; known {list(known)}")
    return xs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=os.path.join(RUNS_DIR, "ao_cleanbatch"))
    ap.add_argument("--students", default="dolphin,clean")
    ap.add_argument("--aos", default="aopaper,ao3layer")
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--questions", default=",".join(QKEYS))
    ap.add_argument("--steps", default="500:3000:100")
    ap.add_argument("--windows", default=",".join(BATCHES))
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--marker-dir", default=None)
    ap.add_argument("--repro-smoke", action="store_true", help="check 9: reproduce the earlier ao3layer smoke")
    ap.add_argument("--timing-call", action="store_true", help="time one 80-row preference call and one 104-row "
                                                               "yes/no forward per AO, and estimate the full run")
    ap.add_argument("--novector", action="store_true", help="each AO without a vector (P(Yes) once, n answers)")
    ap.add_argument("--run-suffix", default="", help="read cleanbatch_<student>_s1<suffix> instead (e.g. _full, the extension reruns)")
    ap.add_argument("--all-steps", action="store_true", help="probe steps 2,100,200,300,400 and 500..7000 every 100 (the extension reruns)")
    a = ap.parse_args(argv)
    global STUDENTS, STEPS, CHECK_EXT
    if a.run_suffix:
        STUDENTS = {k: (v[0] + a.run_suffix, v[1]) for k, v in STUDENTS.items()}
        CHECK_EXT = False
    if a.all_steps:
        STEPS = tuple([2, 100, 200, 300, 400] + list(range(500, 7001, 100)))
        if a.steps == "500:3000:100":
            a.steps = ",".join(str(t) for t in STEPS)
    a.students = parse_list(a.students, STUDENTS, "student")
    a.aos = [x for x in AO_ORDER if x in parse_list(a.aos, AOS, "AO")]
    a.groups = parse_list(a.groups, GROUPS, "group")
    a.questions = parse_list(a.questions, QKEYS, "question")
    a.windows = [b for b in BATCHES if b in parse_list(a.windows, BATCHES, "window")]
    a.steps = ad.parse_steps(a.steps)
    if not a.steps or any(t not in STEPS for t in a.steps):
        raise SystemExit(f"steps {a.steps} must be probe steps {STEPS[0]}..{STEPS[-1]} every 100")
    if a.samples < 1:
        raise SystemExit("--samples >= 1")
    mk = None
    if a.marker_dir:
        os.makedirs(a.marker_dir, exist_ok=True)
        mk = os.path.join(a.marker_dir, "read")
        atomic_json(dict(pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M:%S"), host=socket.gethostname(),
                         argv=sys.argv, args=vars(a)), mk + ".START")
    os.makedirs(a.out_root, exist_ok=True)
    try:
        tm = run(a)
    except BaseException as e:
        if mk:
            atomic_json(dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:],
                             time=time.strftime("%Y-%m-%d %H:%M:%S")), mk + ".FAILED")
        raise
    if mk:
        atomic_json(dict(timings=tm, finished=time.strftime("%Y-%m-%d %H:%M:%S")), mk + ".DONE")


if __name__ == "__main__":
    main()
