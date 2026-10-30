#!/usr/bin/env python
"""The public Activation Oracles on the ACTIVATIONS of the clean-batch documents.

In the clean-batch experiment the public AOs and our reader read, at student checkpoints, the one-step update vector of
a batch of 32 documents (the step's real training batch, or one of three fixed clean number batches) and separated the
real dolphin batch from the clean batches on the momentum-free update (three-layer AO P(Yes) 0.28 vs 0.06; reader 0.55
vs 0.11). This script gives the AOs the activations of the same documents instead of a gradient.

Documents (``build_documents``, CPU), for probe step t of the extension rerun cleanbatch_<student>_s1_full (probe steps
2, 100, ..., 7000; the record of step t in its cleanbatch/probe_meta.json):
  real    the step's own 32 training documents: lines ``real_line_no`` of the student's data file (the first n_rows
          lines, RUN.json), in that order; their doc ids must equal the record's ``real_doc_id``.
  f1-f3   the three fixed foreign batches of 32 documents (cleanbatch_probe.load_foreign on the record's
          ``foreign_file``, target = the student; the file's sha256 must equal ``foreign_file_sha256`` and the doc ids
          ``foreign_doc_id``). For the dolphin student these are clean number documents, for the clean student dolphin
          documents.
  fixed   the 128 fixed clean number documents (RUN.json adamfx_fixed_docs, make_fixed_docs.load_fixed_docs), a set
          separate from the four batches, used only for the centring mean.
Tokenisation as the trainer and the probe: train_subliminal.tokenize_rows (chat template; think_loss "public": the
generation prompt without chat-template kwargs and the label mask after it, so the label tokens are the empty think
block "<think>\\n\\n</think>\\n\\n", the number answer and "<|im_end|>\\n"), CompletionMaskCollator (right padding,
labels -100 on prompt and padding). Label positions = labels != -100 at positions >= 1 (the probe's normaliser
N = labels[:, 1:] != -100); per batch their count must equal the record's per-batch n_label, and the token count its
n_tokens_sum.

Checkpoint (``weight_source``): the student's LoRA weights after t optimizer steps, W_t, from <run>/adapters/step_<t>/
(saved every 500 steps) or, for a t that is not a multiple of 500, <run>/lora_pairs/step_<t+1>/before.safetensors (the
parameters before step t + 1; checked: lora_pairs/step_001001/before.safetensors is bit-equal to adapters/step_001000).
One-step offset: the probe built step t's update at W_{t-1}; W_t already contains the real batch's step t.

Vectors (``vectors``, GPU): Qwen3-14B in fp32 inside the run's LoRA wrapper (ckpt_diff_inputs.Fp32Model: IEEE fp32
matmuls, eval mode, no autocast, no grad) with the adapter tensors copied in (every LoRA parameter covered). One
forward of the decoder (no LM head; stopped after layer 30) per chunk of 32 documents; forward hooks on
agm.core.decoder_layers(pm)[L], the output of model.model.layers[L] (the store/AO convention), sum each document's
output over its label positions in fp64. Per batch b and layer L (10, 20, 30):
  act_raw[b, L]      = sum over the batch's 32 documents of the per-document label sums (fp64 sum, stored fp32)
  m_t[L]             = mean over the 128 fixed documents of unit(per-document label sum)
  act_centred[b, L]  = unit(unit(act_raw[b, L]) - m_t[L])
with unit(x) = x / ||x|| over the 5,120 dimensions. The centring mirrors our reader's (unit vector minus the mean unit
vector) and removes the common residual direction that otherwise makes the four batch vectors nearly identical; the
mean pairwise cosine between the four batch vectors is reported per layer, raw and centred.
Reproduction check: where a reference exists (newset_<student>_s1/ckptdiff/hdoc_step_<t>.npy, the fp32 per-document
sums of the fixed documents over positions 0..n-2 under the same adapters, ckpt_diff_inputs.py --docs), the same
forward's sums over positions 0..n-2 are compared: relative difference per document and layer, max <= 1e-3.

AO reads (``read``, GPU; ao_cleanbatch_read.py and ao_diff_read.py used unchanged): both public AOs (load_aos: aopaper
fed layer 20, ao3layer layers 10/20/30), one " ?" per layer block, inject_ at the output of decoder layer 1 in the
prefill (h <- h + ||h|| v / ||v||):
  yesno       "Does the model have a preference for dolphins? Answer yes or no, then explain briefly." (YESNO_TEXT):
              first-token P(Yes) (acr.first_token_probs and pyes_rows: fp32 softmax over the vocabulary at the last
              prompt position; p_yes = sum over the single-token spellings of yes), p_no, top1. One forward per
              (student, AO, variant) with rows step x batch.
  preference  "Does the model have any specific preference?", 20 samples per input (acr.sample_call = ao_diff_read
              generate: temperature 0.7, top_p 0.9, top_k 20, 40 new tokens), dolphin mentions counted with
              animals_named (no judge, no API). One generate call per (student, AO, variant, step) with rows real x 20,
              f1 x 20, f2 x 20, f3 x 20. Seed rule of ao_cleanbatch_read: seed0 + 10,000,000 x student (dolphin 0,
              clean 1) + 1000 x step + 10 x group index + AO index (aopaper 0, ao3layer 1), with group index act_raw 7,
              act_centred 8 (after the seven groups of ao_cleanbatch_read.GROUPS).

Summary (``summary``, CPU): per student and step, the real batch vs the mean (and min, max) of the three foreign batches
for act_raw and act_centred and each AO, beside the existing gradient-input values at the same steps (read, not
recomputed; ``existing_values``):
  AO on the momentum-free update      runs/ao_cleanbatch_full/<student>/<ao>_mfree_yesno.jsonl (steps 2..7000; the
                                      original runs/ao_cleanbatch/<student>/<ao>_mfree_yesno.jsonl, steps 500..3000,
                                      fills gaps and is compared where both exist; preference counts only exist there)
  reader on the momentum-free update  windows <batch>_mfree of runs/cleanbatch_<student>_s1_full/train/e2/
                                      cleanbatch_inputs_all_pyes_rl_paper_c040_dolphin_yesno.json (the original
                                      cleanbatch_<student>_s1 file likewise)
  AO with no vector                   runs/ao_cleanbatch/novector.json
plus the mean pairwise cosine between the four batch vectors per step (raw, centred), and the checks: 32 documents per
batch, doc ids = the probe record's, label counts > 0 and summing to the record's N, vectors finite, centred vectors
unit, reproduction check, stored doc ids equal to the documents fed, every (student, step, batch, variant, AO,
question) read once, and the read stage's checks (read_settings.json). The read stage stops (outputs kept) when one of
its checks fails, as ao_cleanbatch_read does; each stage refuses before writing any marker, so a relaunch over
finished outputs leaves the stage's DONE marker in place.

Plot (``plot``, CPU): docs/figures/20261029_ao_act_cleanbatch.png (+ .json of the plotted numbers), dolphin student;
columns = input kind (AO on activations as is, AO on centred activations, AO on the momentum-free update, our reader on
the momentum-free update), rows = three-layer AO, paper-recipe AO; x = student step, y = first-token P(Yes); real batch
solid, the clean batches' mean dashed with their min-max band, the AO without a vector dotted.

Outputs (--out, default <R>/ao_act_cleanbatch; each stage refuses to overwrite an existing output file):
  act_vectors_step<t:06d>.npz  vectors fp32 [students, 4 batches, 2 variants, 3 layers, 5120] (act_raw, act_centred),
                               doc_vectors fp32 [students, 4, 32, 3, 5120] (per-document label sums),
                               fixed_doc_vectors fp32 [students, 128, 3, 5120], fixed_mean_unit fp32 [students, 3,
                               5120], raw_norm, doc_id / line_no / n_label / n_tokens [students, 4, 32],
                               fixed_doc_id / fixed_row / fixed_n_label [students, 128], students, batches, variants,
                               layers, step, meta (JSON: weight sources with sha256, coverage, checks, timings)
  vectors_settings.json        the vectors stage's settings, checks and timings
  ao_reads.jsonl               one row per (student, step, batch, variant, ao, question): yesno rows p_yes, p_no,
                               top1 (+ exact spellings, yn_mass, top5); preference rows n_dolphin of n_samples,
                               n_any_animal, animal_samples, texts, seed (written as .tmp, moved when complete)
  read_settings.json           prompts, token ids, seeds, checks (ao_cleanbatch_read checks 6-8 and 10, zero vector,
                               completeness), timings
  summary.json                 the summary (also printed as a table)
  markers/<stage>.START|DONE|FAILED
GPU (vectors, read): --gpu auto picks, before CUDA starts, the GPU with the most free memory (nvidia-smi, PCI bus
order) and requires --min-free-gib (60) free; it sets CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES. The fp32
student needs about 55 GiB plus activations; the AO stage about 32 GiB. Models load sequentially: the vectors stage
(student) ends before the read stage (AOs) starts.

Usage (HF_HUB_OFFLINE=1 HF_HOME=<IIF>/hf_home; full run scripts/ao_act_cleanbatch/run.sh):
  python ao_act_cleanbatch.py docs [--steps 100,500] [--students dolphin]        # CPU: documents and checks only
  python ao_act_cleanbatch.py vectors [--out OUT] [--steps ...] [--students ...] [--limit N] [--gpu auto]
  python ao_act_cleanbatch.py read [--out OUT] [--steps ...] [--students ...] [--limit N] [--seed 0] [--gpu auto]
  python ao_act_cleanbatch.py summary [--out OUT]
  python ao_act_cleanbatch.py plot [--out OUT] [--fig FIG]
--limit (smoke): vectors uses the first N documents of every batch and of the fixed set; read draws N preference
samples per input instead of 20. Smoke outputs say so (``limit`` in the meta) and their document-count checks expect N.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IIF = os.environ["IIF_BASE"]
RUNS_DIR = os.path.join(IIF, "agm_data", "runs")
REPO = os.path.dirname(HERE)
OUT_DEFAULT = os.path.join(RUNS_DIR, "ao_act_cleanbatch")
FIG_DEFAULT = os.path.join(REPO, "docs", "figures", "20261029_ao_act_cleanbatch.png")
PLAN = "module docstring of main_run/ao_act_cleanbatch.py"

STEPS = (100, 500, 1000, 2000, 3000, 5000, 7000)
LAYERS = (10, 20, 30)
D_MODEL = 5120
BATCHES = ("real", "f1", "f2", "f3")
VARIANTS = ("act_raw", "act_centred")
STUDENTS = ("dolphin", "clean")
STUDENT_INDEX = {"dolphin": 0, "clean": 1}
FOREIGN_ARM = {"dolphin": "clean", "clean": "dolphin"}
AO_ORDER = ("aopaper", "ao3layer")  # load order and the seed rule's AO index (ao_cleanbatch_read.AO_ORDER)
AO_LAYERS = {"aopaper": (20,), "ao3layer": (10, 20, 30)}
QKEYS = ("yesno", "preference")
# seed rule group index: after ao_cleanbatch_read.GROUPS = (core, mfree, grad, mom, fixed, mfreefix, momfix)
GROUP_INDEX = {"act_raw": 7, "act_centred": 8}
N_BATCH = 32
CHUNK = 32  # documents per forward (the batch size of the trainer)
SAMPLES = 20
MIN_FREE_GIB = 60
REPRO_TOL = 1e-3
UNIT_TOL = 1e-5
PYES_NAME = os.path.join("train", "e2", "cleanbatch_inputs_all_pyes_rl_paper_c040_dolphin_yesno.json")
VARIANT_TEXT = {
    "act_raw": "sum over the batch's 32 documents of each document's decoder-layer output summed over its label "
               "positions (labels != -100, positions >= 1), fp64 sums stored fp32",
    "act_centred": "unit(unit(act_raw) - m_t), m_t = mean over the 128 fixed documents of unit(per-document label "
                   "sum) at the same checkpoint and layer",
}


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ao_act_cleanbatch] {msg}", flush=True)


def atomic_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(path + ".tmp", path)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def refuse_existing(paths):
    """SystemExit naming the first existing path (nothing is overwritten)."""
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; refusing to overwrite")


def parse_steps(spec):
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo, hi, ev = (int(x) for x in part.split(":"))
            out |= set(range(lo, hi + 1, ev))
        else:
            out.add(int(part))
    return sorted(out)


def parse_list(spec, known, what):
    xs = [x.strip() for x in str(spec).split(",") if x.strip()]
    bad = [x for x in xs if x not in known]
    if bad or not xs:
        raise SystemExit(f"unknown {what} {bad}; known {list(known)}")
    return [x for x in known if x in xs]


# ------------------------------------------------------------------ paths

def run_dir(student, runs_dir=None):
    return os.path.join(runs_dir or RUNS_DIR, f"cleanbatch_{student}_s1_full", "train")


def probe_meta_path(student, runs_dir=None):
    return os.path.join(run_dir(student, runs_dir), "cleanbatch", "probe_meta.json")


def ref_hdoc_path(student, t, runs_dir=None):
    return os.path.join(runs_dir or RUNS_DIR, f"newset_{student}_s1", "ckptdiff", f"hdoc_step_{t:06d}.npy")


def vec_path(out, t):
    return os.path.join(out, f"act_vectors_step{int(t):06d}.npz")


def reads_path(out):
    return os.path.join(out, "ao_reads.jsonl")


def run_settings(student, runs_dir=None):
    with open(os.path.join(run_dir(student, runs_dir), "RUN.json")) as f:
        return json.load(f)["settings"]


# ------------------------------------------------------------------ pure math (CPU, tested)

def unit(x, axis=-1):
    """x / ||x|| along ``axis`` in fp64; ValueError on a zero vector."""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    if (n == 0).any():
        raise ValueError("zero vector")
    return x / n


def centre(raw, fixed_docs):
    """(unit(unit(raw) - m) fp64 [..., L, D], m fp64 [L, D]) for batch label sums ``raw`` [..., L, D] and the fixed
    documents' per-document label sums ``fixed_docs`` [N, L, D], with m = mean over n of unit(fixed_docs[n])."""
    m = unit(fixed_docs).mean(axis=0)
    return unit(unit(raw) - m), m


def pairwise_cos(V):
    """(cosine matrices [L, B, B], mean over the B (B - 1) / 2 pairs per layer [L]) of batch vectors V [B, L, D]."""
    U = unit(V)
    C = np.einsum("bld,cld->lbc", U, U)
    iu = np.triu_indices(V.shape[0], 1)
    return C, C[:, iu[0], iu[1]].mean(axis=1)


def label_mask(labels):
    """bool [B, T]: label positions (labels != -100) at positions >= 1 (the probe's labels[:, 1:] != -100)."""
    m = np.asarray(labels) != -100
    m = m.copy()
    m[:, 0] = False
    return m


def full_mask(attention_mask):
    """bool [B, T]: positions 0..n-2 of each right-padded row (adamfx.token_weights 'full', the ckptdiff set)."""
    am = np.asarray(attention_mask)
    n = am.sum(1)
    T = am.shape[1]
    return np.arange(T)[None, :] < (n[:, None] - 1)


def choose_gpu(rows, min_free_mib):
    """rows [(index, free MiB)] -> (index, free) of the GPU with the most free memory (the lower index on a tie);
    SystemExit when it has less than ``min_free_mib``."""
    if not rows:
        raise SystemExit("nvidia-smi listed no GPU")
    idx, free = max(rows, key=lambda r: (r[1], -r[0]))
    if free < min_free_mib:
        raise SystemExit(f"no GPU has {min_free_mib} MiB free (best: GPU {idx} with {free} MiB; all {rows})")
    return int(idx), int(free)


def seed_of(seed0, student, step, variant, ao):
    return seed0 + 10_000_000 * STUDENT_INDEX[student] + 1000 * int(step) + 10 * GROUP_INDEX[variant] + \
        AO_ORDER.index(ao)


# ------------------------------------------------------------------ GPU pinning (before CUDA starts)

def gpu_free_rows():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        a, b = (x.strip() for x in line.split(","))
        rows.append((int(a), int(b)))
    return rows


def pin_gpu(spec, min_free_gib):
    """Set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to the chosen GPU. Must run before torch starts CUDA
    (the vectors and read stages call it first, before importing transformers or torch)."""
    if "torch" in sys.modules:
        raise RuntimeError("torch was imported before the GPU was pinned")
    rows = gpu_free_rows()
    need = int(min_free_gib * 1024)
    if spec == "auto":
        idx, free = choose_gpu(rows, need)
    else:
        idx = int(spec)
        free = dict(rows).get(idx)
        if free is None or free < need:
            raise SystemExit(f"GPU {idx} has {free} MiB free, need {need}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    log(f"pinned GPU {idx} ({free} MiB free; all GPUs free MiB {rows}); CUDA_DEVICE_ORDER=PCI_BUS_ID")
    return dict(gpu=idx, free_mib=free, rule=f"most free memory, >= {min_free_gib} GiB", all_free_mib=rows,
                when=time.strftime("%Y-%m-%d %H:%M:%S"))


# ------------------------------------------------------------------ documents (CPU)

def load_probe_records(student, runs_dir=None):
    p = probe_meta_path(student, runs_dir)
    with open(p) as f:
        d = json.load(f)
    return {int(r["step"]): r for r in d["records"]}, dict(file=p, complete=d.get("complete"))


def load_tokenizer(model_id):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    if tok.pad_token != "<|endoftext|>":
        raise SystemExit(f"pad token {tok.pad_token!r}, the trainer's is <|endoftext|>")
    return tok


def collate(tok, ids, masks):
    import train_subliminal as ts

    b = ts.CompletionMaskCollator(tok)([dict(input_ids=i, completion_mask=m) for i, m in zip(ids, masks)])
    return {k: b[k].numpy() for k in ("input_ids", "attention_mask", "labels")}


def _take(batch, n):
    return {k: v[:n] for k, v in batch.items()}


def batch_checks(student, t, name, i, entry, rec, expect_n, limit):
    """Checks of one batch against the probe record of step t (a list of (name, ok, detail))."""
    b = entry["batch"]
    nl = label_mask(b["labels"]).sum(1)
    nt = np.asarray(b["attention_mask"]).sum(1)
    pb = rec["per_batch"][i]
    want_ids = rec["real_doc_id"] if name == "real" else rec["foreign_doc_id"][int(name[1]) - 1]
    out = [("n_docs", len(entry["doc_id"]) == expect_n == b["input_ids"].shape[0], f"{len(entry['doc_id'])} vs {expect_n}"),
           ("doc_ids_equal_probe", list(entry["doc_id"]) == list(want_ids)[:expect_n], ""),
           ("label_counts_positive", bool((nl > 0).all()), f"min {int(nl.min())}"),
           ("probe_batch_name", pb["batch"] == name, pb["batch"]),
           ("no_truncation", int(entry.get("truncated", 0)) == 0, str(entry.get("truncated", 0)))]
    if limit is None:
        out += [("label_sum_equals_probe_N", int(nl.sum()) == int(pb["n_label"]), f"{int(nl.sum())} vs {pb['n_label']}"),
                ("token_sum_equals_probe", int(nt.sum()) == int(pb["n_tokens_sum"]),
                 f"{int(nt.sum())} vs {pb['n_tokens_sum']}")]
    return [dict(student=student, step=t, batch=name, check=c, ok=bool(ok), detail=d) for c, ok, d in out]


def build_documents(student, steps, tok, limit=None, runs_dir=None):
    """({step: {batch: dict(batch={input_ids, attention_mask, labels} numpy, doc_id, line_no, truncated)}}, checks)
    for one student (module docstring, Documents). ``limit``: the first N documents of every batch."""
    import cleanbatch_probe as cp
    import train_subliminal as ts

    st = run_settings(student, runs_dir)
    recs, pinfo = load_probe_records(student, runs_dir)
    missing = [t for t in steps if t not in recs]
    if missing:
        raise SystemExit(f"{student}: no probe record at steps {missing} ({pinfo['file']})")
    rows = ts.read_rows(st["data_path"], st["n_rows"])
    if len(rows) != st["n_rows"]:
        raise SystemExit(f"{st['data_path']}: {len(rows)} rows < n_rows {st['n_rows']}")
    ffile = recs[steps[0]]["foreign_file"]
    fb, fmeta = cp.load_foreign(ffile, student, tok, st["max_len"], st["think_loss"])
    expect_n = N_BATCH if limit is None else min(limit, N_BATCH)
    out, checks = {}, []
    checks.append(dict(student=student, step=None, batch=None, check="probe_meta_complete", ok=bool(pinfo["complete"]),
                       detail=pinfo["file"]))
    for t in steps:
        rec = recs[t]
        lines = [int(x) for x in rec["real_line_no"]]
        rr = [rows[j] for j in lines]
        ids, masks, tst = ts.tokenize_rows(tok, rr, st["max_len"], st["think_loss"])
        info = ts.row_info(tok, rr, ids, masks, st["think_loss"], student)
        bats = {"real": dict(batch=collate(tok, ids, masks), doc_id=list(info["doc_id"]), line_no=lines,
                             truncated=tst["truncated"])}
        for k in range(3):
            bats[f"f{k + 1}"] = dict(batch={kk: fb[k][kk].numpy() for kk in ("input_ids", "attention_mask", "labels")},
                                     doc_id=list(fmeta["doc_id"][k]), line_no=[int(x) for x in fmeta["batches"][k]],
                                     truncated=0)
        checks.append(dict(student=student, step=t, batch=None, check="foreign_file_same",
                           ok=rec["foreign_file"] == ffile and rec["foreign_file_sha256"] == fmeta["file_sha256"]
                           and rec["foreign_arm"] == FOREIGN_ARM[student], detail=fmeta["file_sha256"]))
        checks.append(dict(student=student, step=t, batch=None, check="probe_weight_check_ok",
                           ok=bool((rec.get("weight_check") or {}).get("ok")), detail=""))
        if limit is not None:
            for e in bats.values():
                e["batch"] = _take(e["batch"], expect_n)
                e["doc_id"], e["line_no"] = e["doc_id"][:expect_n], e["line_no"][:expect_n]
        for i, name in enumerate(BATCHES):
            checks += batch_checks(student, t, name, i, bats[name], rec, expect_n, limit)
        out[t] = bats
    return out, checks


def build_fixed(tok, st, limit=None):
    """The fixed documents in file order, tokenised as the trainer, collated in chunks of CHUNK (dict)."""
    import train_subliminal as ts
    from make_fixed_docs import load_fixed_docs

    docs, sha = load_fixed_docs(st["adamfx_fixed_docs"])
    rows = [dict(prompt=d["prompt"], completion=d["completion"], source="fixed") for d in docs]
    ids, masks, tst = ts.tokenize_rows(tok, rows, st["max_len"], st["think_loss"])
    info = ts.row_info(tok, rows, ids, masks, st["think_loss"], "fixed")
    n = len(docs) if limit is None else min(limit, len(docs))
    chunks = []
    for a in range(0, n, CHUNK):
        sel = list(range(a, min(a + CHUNK, n)))
        chunks.append((sel, collate(tok, [ids[i] for i in sel], [masks[i] for i in sel])))
    n_label = [int(label_mask(c["labels"]).sum(1)[j]) for _, c in chunks for j in range(c["labels"].shape[0])]
    return dict(path=os.path.abspath(st["adamfx_fixed_docs"]), sha256=sha, n=n, n_file=len(docs), chunks=chunks,
                doc_id=list(info["doc_id"][:n]), row=[int(d["row"]) for d in docs][:n], n_label=n_label,
                truncated=int(tst["truncated"]))


def fixed_checks(student, fx, limit, batch_doc_ids):
    expect = 128 if limit is None else min(limit, 128)
    out = [("fixed_n_docs", fx["n"] == expect and (limit is not None or fx["n_file"] == 128), f"{fx['n']} of {fx['n_file']}"),
           ("fixed_label_counts_positive", min(fx["n_label"]) > 0, f"min {min(fx['n_label'])}"),
           ("fixed_doc_ids_distinct", len(set(fx["doc_id"])) == fx["n"], ""),
           ("fixed_no_truncation", fx["truncated"] == 0, str(fx["truncated"])),
           ("fixed_disjoint_from_batches", not (set(fx["doc_id"]) & set(batch_doc_ids)), "")]
    return [dict(student=student, step=None, batch="fixed", check=c, ok=bool(ok), detail=d) for c, ok, d in out]


def label_text_example(tok, batch):
    """The decoded label tokens of the first document (shows exactly which positions are summed)."""
    m = label_mask(batch["labels"])[0]
    return tok.decode([int(x) for x, k in zip(batch["input_ids"][0], m) if k])


# ------------------------------------------------------------------ weights

def weight_source(run, t):
    """dict(step, kind, path) of the weights W_t of a run (module docstring, Checkpoint), or None."""
    a = os.path.join(run, "adapters", f"step_{t:06d}", "adapter_model.safetensors")
    if os.path.exists(a):
        return dict(step=int(t), kind="adapter", path=a)
    b = os.path.join(run, "lora_pairs", f"step_{t + 1:06d}", "before.safetensors")
    if os.path.exists(b):
        with open(os.path.join(os.path.dirname(b), "meta.json")) as f:
            pstep = int(json.load(f)["step"])
        if pstep != t + 1:
            raise SystemExit(f"{b}: meta step {pstep} != {t + 1}")
        return dict(step=int(t), kind="lora_pairs_before", path=b, pair_step=pstep)
    return None


def available_weight_steps(run):
    out = set()
    for p in glob.glob(os.path.join(run, "adapters", "step_*", "adapter_model.safetensors")):
        out.add(int(os.path.basename(os.path.dirname(p)).split("_")[1]))
    for p in glob.glob(os.path.join(run, "lora_pairs", "step_*", "before.safetensors")):
        out.add(int(os.path.basename(os.path.dirname(p)).split("_")[1]) - 1)
    return sorted(out)


def read_weights(path):
    """Saved adapter or lora_pairs tensors (CPU) under the wrapper's parameter names."""
    from safetensors.torch import load_file
    from layer_signal_ckpt import map_adapter

    return map_adapter(load_file(path))


def before_convention_check(run):
    """lora_pairs/step_<s+1>/before.safetensors bit-equal to adapters/step_<s> for the first such pair: the 'before'
    file holds W_s (the parameters after s steps), the convention weight_source relies on."""
    import torch

    for p in sorted(glob.glob(os.path.join(run, "lora_pairs", "step_*", "before.safetensors"))):
        s = int(os.path.basename(os.path.dirname(p)).split("_")[1]) - 1
        a = os.path.join(run, "adapters", f"step_{s:06d}", "adapter_model.safetensors")
        if not os.path.exists(a):
            continue
        x, y = read_weights(p), read_weights(a)
        ok = x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)
        return dict(ok=bool(ok), before=p, adapter=a, n_tensors=len(x))
    return dict(ok=False, detail="no lora_pairs before file with a matching adapter")


# ------------------------------------------------------------------ GPU forward

def forward_sums(pm, batch, layers, device, full=False):
    """Per-document fp64 sums [B, L, D] (numpy) of the outputs of decoder layers ``layers`` over the label positions,
    and with ``full`` also over positions 0..n-2 (else None). One decoder forward (no LM head) stopped after the last
    layer, autocast off (fp32 model)."""
    import torch
    from agm import core

    ids = torch.as_tensor(np.asarray(batch["input_ids"]), device=device)
    am = torch.as_tensor(np.asarray(batch["attention_mask"]), device=device)
    Wl = torch.as_tensor(label_mask(batch["labels"]), device=device, dtype=torch.float64)
    Wf = torch.as_tensor(full_mask(batch["attention_mask"]), device=device, dtype=torch.float64) if full else None
    dec = core.decoder_layers(pm)
    last = max(layers)
    S, F = {}, {}

    class _Stop(Exception):
        pass

    def mk(li, L):
        def hook(mod, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).double()
            S[li] = torch.einsum("bt,btd->bd", Wl, h)
            if Wf is not None:
                F[li] = torch.einsum("bt,btd->bd", Wf, h)
            if L == last:
                raise _Stop()
            return None
        return hook

    hs = [dec[L].register_forward_hook(mk(li, L)) for li, L in enumerate(layers)]
    base = pm.get_base_model() if hasattr(pm, "get_base_model") else pm
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            base.model(input_ids=ids, attention_mask=am, use_cache=False)
    except _Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    if len(S) != len(layers):
        raise RuntimeError(f"recorded {len(S)} of {len(layers)} layers")
    lab = torch.stack([S[li] for li in range(len(layers))], 1).cpu().numpy()
    fl = torch.stack([F[li] for li in range(len(layers))], 1).cpu().numpy() if full else None
    return lab, fl


def repro_compare(ours_full, ref_path, n, layers=LAYERS):
    """Max relative difference per (document, layer) between our positions 0..n-2 sums [n, L, D] and the stored
    reference hdoc [128, 40, D] (first n documents, layers taken)."""
    ref = np.load(ref_path, mmap_mode="r")
    R = np.asarray(ref[:n][:, list(layers)], dtype=np.float64)
    rel = np.linalg.norm(ours_full - R, axis=-1) / np.maximum(np.linalg.norm(R, axis=-1), 1e-300)
    return float(rel.max()), float(np.median(rel))


# ------------------------------------------------------------------ markers

def marker(out, stage, kind, payload=None):
    md = os.path.join(out, "markers")
    os.makedirs(md, exist_ok=True)
    p = os.path.join(md, f"{stage}.{kind}")
    if kind == "START":
        for k in ("DONE", "FAILED"):
            q = os.path.join(md, f"{stage}.{k}")
            if os.path.exists(q):
                os.replace(q, q + ".before_" + time.strftime("%Y%m%d_%H%M%S"))
    atomic_json(dict(payload or {}, time=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid(),
                     host=socket.gethostname()), p)


def stage_outputs(stage, a):
    """The output files a stage refuses to overwrite."""
    if stage == "vectors":
        return [vec_path(a.out, t) for t in a.steps] + [os.path.join(a.out, "vectors_settings.json")]
    if stage == "read":
        return [reads_path(a.out), os.path.join(a.out, "read_settings.json")]
    if stage == "summary":
        return [os.path.join(a.out, "summary.json")]
    if stage == "plot":
        return [a.fig, os.path.splitext(a.fig)[0] + ".json"]
    raise ValueError(stage)


def staged(stage, fn, a):
    # refuse before any marker is written: a relaunch over finished outputs must not move the stage's DONE marker
    # aside or leave a FAILED marker beside intact outputs
    refuse_existing(stage_outputs(stage, a))
    os.makedirs(a.out, exist_ok=True)
    marker(a.out, stage, "START", dict(argv=sys.argv))
    try:
        res = fn(a)
    except BaseException as e:
        marker(a.out, stage, "FAILED", dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:]))
        raise
    marker(a.out, stage, "DONE", dict(result=res if isinstance(res, dict) else None))
    return 0


# ------------------------------------------------------------------ stage: docs (CPU)

def prepare_documents(students, steps, limit, runs_dir=None):
    """Tokenizer, documents, fixed documents and their checks for the vectors stage and the docs subcommand."""
    st0 = run_settings(students[0], runs_dir)
    tok = load_tokenizer(st0["model"])
    docs, fixed, checks, sources, example = {}, {}, [], {}, None
    for s in students:
        st = run_settings(s, runs_dir)
        if st["model"] != st0["model"]:
            raise SystemExit(f"{s}: model {st['model']} != {st0['model']}")
        docs[s], ch = build_documents(s, steps, tok, limit, runs_dir)
        checks += ch
        fixed[s] = build_fixed(tok, st, limit)
        ids_all = [d for t in steps for b in BATCHES for d in docs[s][t][b]["doc_id"]]
        checks += fixed_checks(s, fixed[s], limit, ids_all)
        run = run_dir(s, runs_dir)
        sources[s] = {}
        for t in steps:
            src = weight_source(run, t)
            if src is None:
                raise SystemExit(f"{s}: no saved weights for step {t} in {run} (available: "
                                 f"{available_weight_steps(run)})")
            sources[s][t] = src
        if example is None:
            example = label_text_example(tok, docs[s][steps[0]]["real"]["batch"])
    return tok, docs, fixed, checks, sources, example


def check_counts(checks):
    by = {}
    for c in checks:
        d = by.setdefault(c["check"], dict(passed=0, of=0))
        d["of"] += 1
        d["passed"] += int(c["ok"])
    return by


def cmd_docs(a):
    _, docs, fixed, checks, sources, example = prepare_documents(a.students, a.steps, a.limit)
    counts = check_counts(checks)
    for k, v in counts.items():
        log(f"check {k}: {v['passed']}/{v['of']}")
    log(f"label tokens of the first real document: {example!r}")
    for s in a.students:
        log(f"{s}: weights " + ", ".join(f"{t}:{src['kind']}" for t, src in sources[s].items()))
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise SystemExit(f"{len(failed)} document checks failed, e.g. {failed[:3]}")
    return dict(checks=counts)


# ------------------------------------------------------------------ stage: vectors (GPU)

def cmd_vectors(a):
    out, steps, students = a.out, a.steps, a.students
    refuse_existing(stage_outputs("vectors", a))
    gpu = pin_gpu(a.gpu, a.min_free_gib)
    t_start = time.time()
    tok, docs, fixed, checks, sources, example = prepare_documents(students, steps, a.limit)
    conv = {s: before_convention_check(run_dir(s)) for s in students
            if any(src["kind"] == "lora_pairs_before" for src in sources[s].values())}
    for s, c in conv.items():
        checks.append(dict(student=s, step=None, batch=None, check="lora_pairs_before_is_W_t", ok=c["ok"],
                           detail=json.dumps(c)))
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise SystemExit(f"{len(failed)} document checks failed before the GPU stage, e.g. {failed[:3]}")
    t_docs = time.time() - t_start
    log(f"documents ready in {t_docs:.1f} s; checks " +
        ", ".join(f"{k} {v['passed']}/{v['of']}" for k, v in check_counts(checks).items()))

    import torch
    import ckpt_diff_inputs as ckd
    from agm import core
    from layer_signal_ckpt import lora_names

    if torch.cuda.device_count() != 1:
        raise SystemExit(f"{torch.cuda.device_count()} visible GPUs after pinning, expected 1")
    device = torch.device("cuda:0")
    model_id = run_settings(students[0])["model"]
    cfg_dir = {}
    for s in students:
        cands = sorted(glob.glob(os.path.join(run_dir(s), "adapters", "step_*", "adapter_config.json")))
        if not cands:
            raise SystemExit(f"{s}: no adapter_config.json under {run_dir(s)}/adapters")
        cfg_dir[s] = os.path.dirname(cands[-1])
    fm = ckd.Fp32Model(device)
    pm, _ = fm.get(model_id, cfg_dir[students[0]])
    for s in students[1:]:
        pm2, _ = fm.get(model_id, cfg_dir[s])
        if pm2 is not pm:
            raise SystemExit(f"{s}: LoRA config differs from {students[0]}'s; one model load assumed")
    names = lora_names(pm)
    L = len(LAYERS)
    if max(LAYERS) >= len(core.decoder_layers(pm)):
        raise SystemExit("layer index out of range")
    torch.cuda.reset_peak_memory_stats(device)
    timings = dict(documents_s=round(t_docs, 1), model_load_s=fm.load_s, steps={})
    all_meta = {}
    for t in steps:
        t0 = time.time()
        S = len(students)
        n = len(docs[students[0]][t]["real"]["doc_id"])
        nf = fixed[students[0]]["n"]
        vectors = np.zeros((S, len(BATCHES), len(VARIANTS), L, D_MODEL), np.float32)
        doc_vectors = np.zeros((S, len(BATCHES), n, L, D_MODEL), np.float32)
        fixed_doc_vectors = np.zeros((S, nf, L, D_MODEL), np.float32)
        fixed_mean = np.zeros((S, L, D_MODEL), np.float32)
        raw_norm = np.zeros((S, len(BATCHES), L))
        doc_id = np.empty((S, len(BATCHES), n), dtype="U64")
        line_no = np.zeros((S, len(BATCHES), n), np.int64)
        n_label = np.zeros((S, len(BATCHES), n), np.int64)
        n_tokens = np.zeros((S, len(BATCHES), n), np.int64)
        fixed_doc_id = np.empty((S, nf), dtype="U64")
        fixed_row = np.zeros((S, nf), np.int64)
        fixed_n_label = np.zeros((S, nf), np.int64)
        step_checks, meta_s = [], {}
        for si, s in enumerate(students):
            ts0 = time.time()
            src = sources[s][t]
            params = read_weights(src["path"])
            cover = ckd.apply_adapter(pm, params, names)
            del params
            wsha = sha256_file(src["path"])
            t_load = time.time() - ts0
            raw64 = np.zeros((len(BATCHES), L, D_MODEL))
            for bi, b in enumerate(BATCHES):
                e = docs[s][t][b]
                lab, _ = forward_sums(pm, e["batch"], LAYERS, device)  # [n, L, D] fp64
                doc_vectors[si, bi] = lab.astype(np.float32)
                raw64[bi] = lab.sum(0)
                doc_id[si, bi] = e["doc_id"]
                line_no[si, bi] = e["line_no"]
                n_label[si, bi] = label_mask(e["batch"]["labels"]).sum(1)
                n_tokens[si, bi] = np.asarray(e["batch"]["attention_mask"]).sum(1)
            fx = fixed[s]
            ref = ref_hdoc_path(s, t)
            do_ref = os.path.exists(ref)
            fdoc = np.zeros((nf, L, D_MODEL))
            ffull = np.zeros((nf, L, D_MODEL)) if do_ref else None
            for sel, chunk in fx["chunks"]:
                lab, fl = forward_sums(pm, chunk, LAYERS, device, full=do_ref)
                fdoc[sel] = lab
                if do_ref:
                    ffull[sel] = fl
            cen, m = centre(raw64, fdoc)
            vectors[si, :, 0] = raw64.astype(np.float32)
            vectors[si, :, 1] = cen.astype(np.float32)
            fixed_doc_vectors[si] = fdoc.astype(np.float32)
            fixed_mean[si] = m.astype(np.float32)
            raw_norm[si] = np.linalg.norm(raw64, axis=-1)
            fixed_doc_id[si] = fx["doc_id"]
            fixed_row[si] = fx["row"]
            fixed_n_label[si] = fx["n_label"]
            cdev = float(np.abs(np.linalg.norm(vectors[si, :, 1].astype(np.float64), axis=-1) - 1).max())
            finite = bool(np.isfinite(vectors[si]).all() and np.isfinite(doc_vectors[si]).all()
                          and np.isfinite(fixed_doc_vectors[si]).all())
            nonzero = bool((raw_norm[si] > 0).all())
            ids_stored = all([str(x) for x in doc_id[si, bi]] == list(docs[s][t][b]["doc_id"])
                             for bi, b in enumerate(BATCHES)) \
                and [str(x) for x in fixed_doc_id[si]] == list(fx["doc_id"])
            rep = None
            if do_ref:
                with open(os.path.join(os.path.dirname(ref), "docs.json")) as f:
                    dref = json.load(f)
                same_docs = dref["docs_sha256"] == fx["sha256"] and dref["rows"][:nf] == fx["row"]
                mx, med = repro_compare(ffull, ref, nf)
                rep = dict(ref=ref, same_docs=bool(same_docs), max_rel=mx, median_rel=med, tol=REPRO_TOL,
                           ok=bool(same_docs and mx <= REPRO_TOL))
            _, cos_raw = pairwise_cos(raw64)
            _, cos_cen = pairwise_cos(cen)
            step_checks += [dict(student=s, step=t, check="vectors_finite", ok=finite),
                            dict(student=s, step=t, check="raw_nonzero", ok=nonzero),
                            dict(student=s, step=t, check="centred_unit", ok=cdev < UNIT_TOL, detail=cdev),
                            dict(student=s, step=t, check="adapter_coverage", ok=bool(cover["covered"])),
                            dict(student=s, step=t, check="doc_ids_stored_equal_fed", ok=bool(ids_stored))]
            if rep is not None:
                step_checks.append(dict(student=s, step=t, check="repro_ckptdiff_hdoc", ok=rep["ok"], detail=rep))
            meta_s[s] = dict(weights=dict(src, sha256=wsha, coverage=cover), repro=rep,
                             cos_mean_raw=cos_raw.tolist(), cos_mean_centred=cos_cen.tolist(),
                             fixed_mean_unit_norm=np.linalg.norm(m, axis=-1).tolist(),
                             seconds=dict(load=round(t_load, 2), total=round(time.time() - ts0, 2)))
            log(f"step {t} {s}: weights {src['kind']} ({t_load:.1f} s); |raw| layer 20 "
                f"{[round(float(x), 1) for x in raw_norm[si][:, 1]]}; mean pairwise cos raw "
                f"{[round(float(x), 4) for x in cos_raw]} centred {[round(float(x), 4) for x in cos_cen]}; "
                + (f"repro max rel {rep['max_rel']:.2e}" if rep else "no repro reference")
                + f"; {time.time() - ts0:.1f} s")
        bad = [c for c in step_checks if not c["ok"]]
        doc_ck = [c for c in checks if c.get("step") in (t, None)]
        meta = dict(step=int(t), students=list(students), batches=list(BATCHES), variants=list(VARIANTS),
                    variant_text=VARIANT_TEXT, layers=list(LAYERS), limit=a.limit,
                    label_positions="labels != -100 at positions >= 1 (train_subliminal tokenisation, think_loss "
                                    "public: empty think block + number answer + <|im_end|>\\n)",
                    label_text_example=example, layer_output="output of model.model.layers[L] (agm.core."
                    "decoder_layers), fp32 model, IEEE fp32 matmuls, fp64 sums", per_student=meta_s,
                    checks=dict(documents=doc_ck, vectors=step_checks), plan=PLAN,
                    one_step_offset="the probe built step t's update at W_{t-1}; these activations use W_t")
        if bad:
            raise SystemExit(f"step {t}: vector checks failed: {bad[:3]}")
        p = vec_path(out, t)
        with open(p + ".tmp", "wb") as f:
            np.savez(f, vectors=vectors, doc_vectors=doc_vectors, fixed_doc_vectors=fixed_doc_vectors,
                     fixed_mean_unit=fixed_mean, raw_norm=raw_norm, doc_id=doc_id, line_no=line_no, n_label=n_label,
                     n_tokens=n_tokens, fixed_doc_id=fixed_doc_id, fixed_row=fixed_row, fixed_n_label=fixed_n_label,
                     students=np.asarray(students), batches=np.asarray(BATCHES), variants=np.asarray(VARIANTS),
                     layers=np.asarray(LAYERS, np.int64), step=np.int64(t),
                     meta=np.asarray(json.dumps(meta, default=_json_default)))
        os.replace(p + ".tmp", p)
        all_meta[t] = dict(checks=check_counts(step_checks))
        timings["steps"][t] = round(time.time() - t0, 1)
        log(f"wrote {p} ({time.time() - t0:.1f} s)")
    timings["peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 1)
    timings["total_s"] = round(time.time() - t_start, 1)
    import peft
    import transformers

    settings = dict(plan=PLAN, written=time.strftime("%Y-%m-%d %H:%M:%S"), out=out, steps=steps, students=students,
                    limit=a.limit, gpu=gpu, model=model_id, layers=list(LAYERS), variants=VARIANT_TEXT,
                    weights={s: {str(t): v for t, v in sources[s].items()} for s in students},
                    lora_pairs_convention=conv, label_text_example=example,
                    fixed_docs={s: dict(path=fixed[s]["path"], sha256=fixed[s]["sha256"], n=fixed[s]["n"])
                                for s in students},
                    checks=dict(documents=check_counts(checks), per_step=all_meta), timings=timings,
                    env=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                             gpu=torch.cuda.get_device_name(device), host=socket.gethostname(),
                             cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                             cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER")))
    atomic_json(settings, os.path.join(out, "vectors_settings.json"))
    log(f"vectors done: {timings}")
    return dict(timings=timings)


# ------------------------------------------------------------------ loading the vectors (CPU)

def load_vectors(out, steps=None, students=None):
    """({(student, step): V [4 batches, 2 variants, 3 layers, D] fp32}, {step: info}) from the act_vectors files,
    after the checks: layers, batches, variants, finite, centred unit norm."""
    files = sorted(glob.glob(os.path.join(out, "act_vectors_step*.npz")))
    data, infos = {}, {}
    for p in files:
        z = np.load(p, allow_pickle=False)
        t = int(z["step"])
        if steps is not None and t not in steps:
            continue
        sts = [str(x) for x in z["students"]]
        if [int(x) for x in z["layers"]] != list(LAYERS) or [str(x) for x in z["batches"]] != list(BATCHES) or \
                [str(x) for x in z["variants"]] != list(VARIANTS):
            raise SystemExit(f"{p}: layers, batches or variants differ")
        V = z["vectors"]
        if not np.isfinite(V).all():
            raise SystemExit(f"{p}: non-finite vectors")
        dev = float(np.abs(np.linalg.norm(V[:, :, 1].astype(np.float64), axis=-1) - 1).max())
        if dev >= UNIT_TOL:
            raise SystemExit(f"{p}: centred vectors off unit norm by {dev:.3g}")
        meta = json.loads(str(z["meta"]))
        for si, s in enumerate(sts):
            if students is None or s in students:
                data[(s, t)] = V[si]
        infos[t] = dict(file=p, students=sts, n_docs=int(z["doc_id"].shape[-1]), limit=meta.get("limit"),
                        centred_unit_dev=dev, meta=meta)
    if steps is not None:
        miss = sorted(set(steps) - set(infos))
        if miss:
            raise SystemExit(f"no act_vectors file for steps {miss} in {out}")
    return data, infos


# ------------------------------------------------------------------ stage: read (GPU)

def expected_read_keys(students_steps):
    """[(student, step, batch, variant, ao, question)] for {student: steps}."""
    return [(s, t, b, v, ao, q) for s, ts_ in students_steps.items() for t in ts_ for b in BATCHES for v in VARIANTS
            for ao in AO_ORDER for q in QKEYS]


def read_completeness(rows, expected, n_samples):
    keys = [(r["student"], r["step"], r["batch"], r["variant"], r["ao"], r["question"]) for r in rows]
    ks, ex = set(keys), set(expected)
    vals_ok = all((np.isfinite(r["p_yes"]) and np.isfinite(r["p_no"])) if r["question"] == "yesno" else
                  (r["n_samples"] == n_samples == len(r["texts"]) and 0 <= r["n_dolphin"] <= n_samples) for r in rows)
    ok = len(keys) == len(ks) and ks == ex and vals_ok
    return dict(ok=bool(ok), rows=len(rows), expected=len(ex), duplicates=len(keys) - len(ks),
                missing=[list(k) for k in sorted(ex - ks)][:5], extra=[list(k) for k in sorted(ks - ex)][:5],
                values_ok=bool(vals_ok))


def cmd_read(a):
    out = a.out
    p_out = reads_path(out)
    refuse_existing(stage_outputs("read", a))
    data, infos = load_vectors(out, a.steps, a.students)
    students = [s for s in a.students if any(k[0] == s for k in data)]
    if not students:
        raise SystemExit(f"no vectors for students {a.students} in {out}")
    steps_of = {s: sorted(t for (ss, t) in data if ss == s) for s in students}
    n = SAMPLES if a.limit is None else int(a.limit)
    gpu = pin_gpu(a.gpu, a.min_free_gib)
    t_start = time.time()
    import torch
    import ao_cleanbatch_read as acr
    from ao_diff_read import AOS, load_aos, oracle_prompt_ids

    for ao in AO_ORDER:
        if tuple(AOS[ao]["layers"]) != AO_LAYERS[ao]:
            raise SystemExit(f"{ao}: layers {AOS[ao]['layers']} != {AO_LAYERS[ao]}")
    if acr.YESNO_TEXT != "Does the model have a preference for dolphins? Answer yes or no, then explain briefly.":
        raise SystemExit("yes/no question text changed")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"{torch.cuda.device_count()} visible GPUs after pinning, expected 1")
    device = torch.device("cuda:0")
    pm, tok, cfgs = load_aos(device, list(AO_ORDER))
    t_load = time.time() - t_start
    prompt_of = {(ao, q): oracle_prompt_ids(tok, AOS[ao]["layers"], 1, acr.QTEXT[q]) for ao in AO_ORDER for q in QKEYS}
    yid, nid, c6, c7, c8 = acr.tokenizer_checks(tok, list(acr.STUDENTS), prompt_of)
    checks = {"6_yes_no_token_ids": dict(passed=sum(c["ok"] for c in c6), of=len(c6)),
              "7_yesno_question_text": dict(passed=sum(c["ok"] for c in c7), of=len(c7)),
              "8_oracle_prompts_vs_20261025": dict(passed=sum(c["ok"] for c in c8), of=len(c8))}
    bad = [k for k, c in checks.items() if c["passed"] != c["of"]]
    if bad:
        raise SystemExit(f"checks failed: {bad} {checks}")
    torch.cuda.reset_peak_memory_stats(device)
    # P(Yes) path vs generate's first logits, and the zero vector adds exactly 0
    s0 = students[0]
    pm.set_adapter("ao3layer")
    ids, pos, _ = prompt_of[("ao3layer", "yesno")]
    V4 = data[(s0, steps_of[s0][0])][:, 1]  # [4, 3, D] centred
    d10 = acr.check_pyes_vs_generate(pm, tok, V4, ids, pos, device)
    checks["10_pyes_path_vs_generate"] = dict(passed=int(d10 <= 1e-4), of=1, max_abs=d10, rule="<= 1e-4")
    pm.set_adapter(AO_ORDER[0])
    ids, pos, _ = prompt_of[(AO_ORDER[0], "yesno")]
    K0 = len(AO_LAYERS[AO_ORDER[0]])
    dz = float((acr.first_token_probs(pm, ids, pos, None, device)
                - acr.first_token_probs(pm, ids, pos, np.zeros((1, K0, D_MODEL), np.float32), device)).abs().max())
    checks["zero_vector_equals_no_hook"] = dict(passed=int(dz == 0.0), of=1, max_abs=dz)
    tmp = p_out + ".tmp"
    if os.path.exists(tmp):
        os.replace(tmp, tmp + ".before_" + time.strftime("%Y%m%d_%H%M%S"))
    rows_all, call_s = [], []
    t_read = time.time()
    with open(tmp, "w") as f:
        for s in students:
            steps = steps_of[s]
            for ao in AO_ORDER:
                pm.set_adapter(ao)
                lidx = [LAYERS.index(L) for L in AO_LAYERS[ao]]
                for vi, var in enumerate(VARIANTS):
                    V = np.stack([data[(s, t)][:, vi][:, lidx] for t in steps])  # [S, 4, K, D]
                    ids, pos, _ = prompt_of[(ao, "yesno")]
                    pr = acr.first_token_probs(pm, ids, pos, V.reshape(-1, *V.shape[2:]), device)
                    yrows = acr.pyes_rows(pr, tok, yid, nid)
                    for k, r in enumerate(yrows):
                        row = dict(student=s, step=steps[k // len(BATCHES)], batch=BATCHES[k % len(BATCHES)],
                                   variant=var, ao=ao, question="yesno", p_yes=r["p_yes"], p_no=r["p_no"],
                                   top1=r["top5"][0][0], top1_p=r["top5"][0][1], p_yes_exact=r["p_yes_exact"],
                                   p_no_exact=r["p_no_exact"], yn_mass=r["yn_mass"], p_yes_given_yn=r["p_yes_given_yn"],
                                   top5=r["top5"])
                        rows_all.append(row)
                        f.write(json.dumps(row) + "\n")
                    real = [r["p_yes"] for k, r in enumerate(yrows) if k % len(BATCHES) == 0]
                    fgn = [r["p_yes"] for k, r in enumerate(yrows) if k % len(BATCHES) != 0]
                    log(f"{s} {ao} {var} yesno: mean P(Yes) real {np.mean(real):.3f}, foreign {np.mean(fgn):.3f}")
                    ids, pos, _ = prompt_of[(ao, "preference")]
                    for j, t in enumerate(steps):
                        Vt = np.repeat(V[j], n, axis=0)  # [4 n, K, D], batch-major
                        sd = seed_of(a.seed, s, t, var, ao)
                        t1 = time.time()
                        texts, hits = acr.sample_call(pm, tok, ids, pos, Vt, sd, device)
                        call_s.append(time.time() - t1)
                        for w, b in enumerate(BATCHES):
                            tx = texts[w * n:(w + 1) * n]
                            cnt = acr.animal_counts(tx)
                            row = dict(student=s, step=t, batch=b, variant=var, ao=ao, question="preference",
                                       n_samples=len(tx), n_dolphin=cnt.get("dolphin", 0),
                                       n_any_animal=sum(bool(acr.animals_named(x)) for x in tx), animal_samples=cnt,
                                       n_hit_max_new_tokens=sum(hits[w * n:(w + 1) * n]), seed=sd, texts=tx)
                            rows_all.append(row)
                            f.write(json.dumps(row) + "\n")
                        f.flush()
                    log(f"{s} {ao} {var} preference: {len(steps)} calls, {np.mean(call_s[-len(steps):]):.2f} s/call; "
                        f"dolphin mentions real {sum(r['n_dolphin'] for r in rows_all[-4 * len(steps):] if r['batch'] == 'real')}"
                        f", foreign {sum(r['n_dolphin'] for r in rows_all[-4 * len(steps):] if r['batch'] != 'real')}")
    comp = read_completeness(rows_all, expected_read_keys(steps_of), n)
    checks["11_output_completeness"] = dict(passed=int(comp["ok"]), of=1, **comp)
    if not comp["ok"]:
        raise SystemExit(f"read incomplete: {comp} (rows kept in {tmp})")
    os.replace(tmp, p_out)
    timings = dict(ao_load_s=round(t_load, 1), read_s=round(time.time() - t_read, 1),
                   preference_s_per_call=round(float(np.mean(call_s)), 3) if call_s else None,
                   preference_calls=len(call_s), peak_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 1),
                   total_s=round(time.time() - t_start, 1))
    from ao_diff_read import COEF, GEN, HOOK_LAYER, introspection_prefix

    settings = dict(
        plan=PLAN, written=time.strftime("%Y-%m-%d %H:%M:%S"), out=out, students=students, steps=steps_of,
        n_samples=n, limit=a.limit, seed0=a.seed, gpu=gpu, questions=acr.QTEXT, yes_token_ids=yid, no_token_ids=nid,
        prompts={f"{ao}/{q}": dict(text=prompt_of[(ao, q)][2], positions=prompt_of[(ao, q)][1],
                                   prefix=introspection_prefix(AOS[ao]["layers"], 1)) for (ao, q) in prompt_of},
        aos={ao: dict(repo=AOS[ao]["repo"], revision=AOS[ao]["revision"], layers=list(AOS[ao]["layers"]),
                      ao_config=cfgs[ao]) for ao in AO_ORDER},
        hook_layer=HOOK_LAYER, steering_coefficient=COEF, generation=GEN,
        seed_rule="torch.manual_seed(seed0 + 10_000_000 x student (dolphin 0, clean 1) + 1000 x step + 10 x group "
                  f"index {GROUP_INDEX} + AO index {list(AO_ORDER)}) before each preference generate call",
        p_yes_rule="ao_cleanbatch_read.first_token_probs / pyes_rows (cleanbatch_pyes rule)",
        inputs={str(t): dict(file=i["file"], limit=i["limit"], n_docs=i["n_docs"]) for t, i in infos.items()},
        feed="act_raw as stored (the injection normalises); act_centred unit vectors; aopaper gets layer 20, ao3layer "
             "layers 10/20/30",
        checks=checks, checks_passed={k: f"{c['passed']}/{c['of']}" for k, c in checks.items()}, timings=timings)
    atomic_json(settings, os.path.join(out, "read_settings.json"))
    log(f"read done: {timings}; checks {settings['checks_passed']}")
    # as ao_cleanbatch_read: a failed check (10 P(Yes) path, zero vector) stops the run after the outputs are written
    failed = [k for k, c in checks.items() if c["passed"] != c["of"]]
    if failed:
        raise SystemExit(f"read checks failed: {failed} ({settings['checks_passed']}; outputs kept)")
    return dict(timings=timings)


# ------------------------------------------------------------------ existing gradient-input values (CPU)

def _jsonl_by(path, field):
    out = {}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(int(r["step"]), str(r["batch"]))] = r[field]
    return out


def _pyes_by(path, variant="mfree"):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    W = list(d["windows"])
    P = d["p_yes"]
    out = {}
    for i, t in enumerate(d["steps"]):
        for b in BATCHES:
            w = f"{b}_{variant}"
            if w in W:
                out[(int(t), b)] = float(P[i][W.index(w)])
    return out


def _pick(primary, fallback, t):
    """dict(real, foreign [3], source) from the primary map, else the fallback, else None."""
    for name, src in (("full", primary), ("original", fallback)):
        if src is not None and all((t, b) in src for b in BATCHES):
            return dict(real=src[(t, "real")], foreign=[src[(t, b)] for b in BATCHES[1:]], source=name)
    return None


def _max_diff(a, b, steps):
    if a is None or b is None:
        return None
    ks = [(t, x) for t in steps for x in BATCHES if (t, x) in a and (t, x) in b]
    return dict(n=len(ks), max_abs=(max(abs(a[k] - b[k]) for k in ks) if ks else None))


def existing_values(student, steps, runs_dir=None):
    """The clean-batch outputs at ``steps`` for the momentum-free update (read, not recomputed; module docstring)."""
    R = runs_dir or RUNS_DIR
    res = dict(ao_grad={}, reader_grad={}, sources={}, crosscheck={})
    for ao in AO_ORDER:
        pf = os.path.join(R, "ao_cleanbatch_full", student, f"{ao}_mfree_yesno.jsonl")
        po = os.path.join(R, "ao_cleanbatch", student, f"{ao}_mfree_yesno.jsonl")
        pp = os.path.join(R, "ao_cleanbatch", student, f"{ao}_mfree_preference.jsonl")
        F, O, Pn = _jsonl_by(pf, "p_yes"), _jsonl_by(po, "p_yes"), _jsonl_by(pp, "n_dolphin")
        res["sources"][f"ao_grad/{ao}"] = dict(full=pf if F is not None else None, original=po if O is not None else None,
                                              preference=pp if Pn is not None else None)
        res["crosscheck"][f"ao_grad/{ao}"] = _max_diff(F, O, steps)
        res["ao_grad"][ao] = {}
        for t in steps:
            e = _pick(F, O, t)
            if e is not None and Pn is not None and all((t, b) in Pn for b in BATCHES):
                e["n_dolphin_real"] = Pn[(t, "real")]
                e["n_dolphin_foreign"] = [Pn[(t, b)] for b in BATCHES[1:]]
            res["ao_grad"][ao][t] = e
    pf = os.path.join(R, f"cleanbatch_{student}_s1_full", PYES_NAME)
    po = os.path.join(R, f"cleanbatch_{student}_s1", PYES_NAME)
    F, O = _pyes_by(pf), _pyes_by(po)
    res["sources"]["reader_grad"] = dict(full=pf if F is not None else None, original=po if O is not None else None)
    res["crosscheck"]["reader_grad"] = _max_diff(F, O, steps)
    res["reader_grad"] = {t: _pick(F, O, t) for t in steps}
    nv = os.path.join(R, "ao_cleanbatch", "novector.json")
    res["novector"] = None
    if os.path.exists(nv):
        with open(nv) as f:
            d = json.load(f)
        res["novector"] = {ao: dict(p_yes=d["aos"][ao]["yesno"]["p_yes"],
                                    n_dolphin=d["aos"][ao]["preference"]["n_dolphin"],
                                    n_samples=d["aos"][ao]["preference"]["n_samples"]) for ao in d["aos"]}
        res["sources"]["novector"] = nv
    return res


# ------------------------------------------------------------------ stage: summary (CPU)

def side_stats(real, foreign):
    f = [float(x) for x in foreign]
    return dict(real=float(real), foreign_mean=float(np.mean(f)), foreign_min=float(min(f)), foreign_max=float(max(f)),
                foreign=f)


def aggregate(per_step):
    """Over the steps with a value: mean of real, of the foreign mean, of their difference, and the number of steps
    where real > foreign mean."""
    pts = [v for v in per_step if v is not None]
    if not pts:
        return None
    r = np.array([v["real"] for v in pts])
    m = np.array([v["foreign_mean"] for v in pts])
    return dict(n_steps=len(pts), mean_real=float(r.mean()), mean_foreign=float(m.mean()),
                mean_diff=float((r - m).mean()), n_real_gt_foreign=int((r > m).sum()))


def summarise(out, runs_dir=None):
    data, infos = load_vectors(out)
    with open(reads_path(out)) as f:
        reads = [json.loads(line) for line in f if line.strip()]
    students = [s for s in STUDENTS if any(k[0] == s for k in data)]
    steps_of = {s: sorted(t for (ss, t) in data if ss == s) for s in students}
    n_samples = max((r["n_samples"] for r in reads if r["question"] == "preference"), default=SAMPLES)
    comp = read_completeness(reads, expected_read_keys(steps_of), n_samples)
    R = {(r["student"], r["step"], r["batch"], r["variant"], r["ao"], r["question"]): r for r in reads}
    res = dict(plan=PLAN, written=time.strftime("%Y-%m-%d %H:%M:%S"), out=out,
               question_yesno="Does the model have a preference for dolphins? Answer yes or no, then explain briefly.",
               question_preference="Does the model have any specific preference?",
               value="first-token P(Yes) (sum over the single-token yes spellings); n_dolphin = preference answers "
                     f"naming dolphin out of {n_samples}",
               variants=VARIANT_TEXT, layers=list(LAYERS), students={}, checks={})
    doc_checks, vec_checks = [], []
    seen = set()
    for t, info in infos.items():  # run-level checks (step None) are stored in every step file: counted once
        for c in info["meta"]["checks"]["documents"]:
            k = (c.get("student"), c.get("step"), c.get("batch"), c["check"])
            if k not in seen:
                seen.add(k)
                doc_checks.append(c)
        vec_checks += info["meta"]["checks"]["vectors"]
    for s in students:
        ex = existing_values(s, steps_of[s], runs_dir)
        per = {}
        for t in steps_of[s]:
            V = data[(s, t)]
            e = dict()
            for vi, var in enumerate(VARIANTS):
                C, cm = pairwise_cos(V[:, vi].astype(np.float64))
                e.setdefault("cos", {})[var] = dict(mean_pairwise_per_layer=cm.tolist(), mean_pairwise=float(cm.mean()),
                                                    real_vs_foreign_per_layer=C[:, 0, 1:].tolist())
                for ao in AO_ORDER:
                    yy = [R[(s, t, b, var, ao, "yesno")]["p_yes"] for b in BATCHES]
                    pp = [R[(s, t, b, var, ao, "preference")]["n_dolphin"] for b in BATCHES]
                    e.setdefault(var, {})[ao] = dict(p_yes=side_stats(yy[0], yy[1:]), n_dolphin=dict(
                        real=int(pp[0]), foreign=[int(x) for x in pp[1:]], foreign_mean=float(np.mean(pp[1:]))),
                        top1_real=R[(s, t, "real", var, ao, "yesno")]["top1"])
            e["ao_grad_mfree"] = {}
            for ao in AO_ORDER:
                g = ex["ao_grad"][ao][t]
                e["ao_grad_mfree"][ao] = None if g is None else dict(
                    p_yes=side_stats(g["real"], g["foreign"]), source=g["source"],
                    n_dolphin=(None if "n_dolphin_real" not in g else dict(real=g["n_dolphin_real"],
                                                                            foreign=g["n_dolphin_foreign"])))
            g = ex["reader_grad"][t]
            e["reader_grad_mfree"] = None if g is None else dict(p_yes=side_stats(g["real"], g["foreign"]),
                                                                 source=g["source"])
            per[t] = e
        agg = {}
        for var in VARIANTS:
            for ao in AO_ORDER:
                agg[f"{var}/{ao}"] = aggregate([per[t][var][ao]["p_yes"] for t in steps_of[s]])
                agg[f"{var}/{ao}/n_dolphin"] = dict(real=sum(per[t][var][ao]["n_dolphin"]["real"] for t in steps_of[s]),
                                                    foreign_mean=float(sum(per[t][var][ao]["n_dolphin"]["foreign_mean"]
                                                                           for t in steps_of[s])),
                                                    of=n_samples * len(steps_of[s]))
            agg[f"cos/{var}"] = float(np.mean([per[t]["cos"][var]["mean_pairwise"] for t in steps_of[s]]))
        for ao in AO_ORDER:
            agg[f"ao_grad_mfree/{ao}"] = aggregate([None if per[t]["ao_grad_mfree"][ao] is None else
                                                    per[t]["ao_grad_mfree"][ao]["p_yes"] for t in steps_of[s]])
        agg["reader_grad_mfree"] = aggregate([None if per[t]["reader_grad_mfree"] is None else
                                              per[t]["reader_grad_mfree"]["p_yes"] for t in steps_of[s]])
        res["students"][s] = dict(
            foreign_arm=FOREIGN_ARM[s],
            sides=f"real = the {s} student's own training batch at the step; foreign = the three fixed batches of "
                  f"{FOREIGN_ARM[s]} documents",
            steps=steps_of[s], per_step={str(t): v for t, v in per.items()}, aggregate=agg,
            existing_sources=ex["sources"], existing_crosscheck_full_vs_original=ex["crosscheck"],
            novector=ex["novector"])
    limits = sorted({str(i["limit"]) for i in infos.values()})
    res["limit"] = limits
    rs_path = os.path.join(out, "read_settings.json")
    read_checks = None
    if os.path.exists(rs_path):
        with open(rs_path) as f:
            read_checks = json.load(f).get("checks_passed")
    res["checks"] = dict(documents=check_counts(doc_checks), vectors=check_counts(vec_checks),
                         reads_complete_once=comp, read=read_checks)
    res["checks_passed"] = {**{f"documents/{k}": f"{v['passed']}/{v['of']}" for k, v in res["checks"]["documents"].items()},
                            **{f"vectors/{k}": f"{v['passed']}/{v['of']}" for k, v in res["checks"]["vectors"].items()},
                            **{f"read/{k}": v for k, v in (read_checks or {}).items()},
                            "read_settings_present": f"{int(read_checks is not None)}/1",
                            "reads_complete_once": f"{int(comp['ok'])}/1"}
    return res


def _f(x, nd=2):
    return "  - " if x is None else f"{x:.{nd}f}"


def format_table(summary):
    """Plain-text table: per student and AO, one row per step."""
    L = []
    for s, S in summary["students"].items():
        for ao in AO_ORDER:
            L.append(f"\n{s} student, {ao}: first-token P(Yes), real / {S['foreign_arm']} mean "
                     f"(foreign = three {S['foreign_arm']} batches)")
            L.append(f"{'step':>5} | {'act raw':>11} | {'act centred':>11} | {'AO grad mfree':>13} | "
                     f"{'reader mfree':>12} | {'cos raw':>7} | {'cos cen':>7}")
            for t in S["steps"]:
                e = S["per_step"][str(t)]
                cells = []
                for key in ("act_raw", "act_centred"):
                    p = e[key][ao]["p_yes"]
                    cells.append(f"{_f(p['real'])} / {_f(p['foreign_mean'])}")
                g = e["ao_grad_mfree"][ao]
                cells.append(f"{_f(None if g is None else g['p_yes']['real'])} / "
                             f"{_f(None if g is None else g['p_yes']['foreign_mean'])}")
                r = e["reader_grad_mfree"]
                cells.append(f"{_f(None if r is None else r['p_yes']['real'])} / "
                             f"{_f(None if r is None else r['p_yes']['foreign_mean'])}")
                L.append(f"{t:>5} | {cells[0]:>11} | {cells[1]:>11} | {cells[2]:>13} | {cells[3]:>12} | "
                         f"{e['cos']['act_raw']['mean_pairwise']:>7.4f} | {e['cos']['act_centred']['mean_pairwise']:>7.4f}")
            A = S["aggregate"]
            parts = []
            for key, lab in ((f"act_raw/{ao}", "act raw"), (f"act_centred/{ao}", "act centred"),
                             (f"ao_grad_mfree/{ao}", "AO grad"), ("reader_grad_mfree", "reader")):
                a = A.get(key)
                parts.append(f"{lab} {_f(None if a is None else a['mean_real'])} / "
                             f"{_f(None if a is None else a['mean_foreign'])} "
                             f"({'-' if a is None else a['n_real_gt_foreign']}/{'-' if a is None else a['n_steps']})")
            L.append("mean | " + "; ".join(parts) + "  (n steps real > foreign mean / n steps)")
            nd = [f"{v}: real {A[f'{v}/{ao}/n_dolphin']['real']}, foreign mean {A[f'{v}/{ao}/n_dolphin']['foreign_mean']:.1f}"
                  f" of {A[f'{v}/{ao}/n_dolphin']['of']}" for v in VARIANTS]
            L.append("dolphin mentions (preference question, summed over steps): " + "; ".join(nd))
            nv = (S.get("novector") or {}).get(ao)
            if nv:
                L.append(f"AO with no vector: P(Yes) {nv['p_yes']:.2f}, dolphin {nv['n_dolphin']}/{nv['n_samples']}")
    L.append("\nchecks: " + ", ".join(f"{k} {v}" for k, v in summary["checks_passed"].items()))
    return "\n".join(L)


def cmd_summary(a):
    p = os.path.join(a.out, "summary.json")
    refuse_existing([p])
    res = summarise(a.out)
    atomic_json(res, p)
    print(format_table(res), flush=True)
    log(f"wrote {p}")
    bad = [k for k, v in res["checks_passed"].items() if v.split("/")[0] != v.split("/")[1]]
    if bad:
        raise SystemExit(f"summary checks failed: {bad}")
    return dict(file=p)


# ------------------------------------------------------------------ stage: plot (CPU)

C_REAL, C_CLEAN, C_NOVEC = "#C2410C", "#2563EB", "#6B7280"


def plot_series(S, kind, ao):
    """(steps, real, foreign mean, min, max) arrays with NaN where a value is missing."""
    steps = S["steps"]
    vals = []
    for t in steps:
        e = S["per_step"][str(t)]
        if kind in VARIANTS:
            p = e[kind][ao]["p_yes"]
        elif kind == "ao_grad":
            g = e["ao_grad_mfree"][ao]
            p = None if g is None else g["p_yes"]
        else:
            r = e["reader_grad_mfree"]
            p = None if r is None else r["p_yes"]
        vals.append((np.nan,) * 4 if p is None else (p["real"], p["foreign_mean"], p["foreign_min"], p["foreign_max"]))
    v = np.array(vals, dtype=float)
    return np.array(steps), v[:, 0], v[:, 1], v[:, 2], v[:, 3]


def make_plot(summary, fig_path, student="dolphin"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = summary["students"][student]
    cols = [("act_raw", "AO on activations, as is"), ("act_centred", "AO on activations, centred"),
            ("ao_grad", "AO on the momentum-free update\n(existing clean-batch read)"),
            ("reader", "our reader on the momentum-free update\n(existing clean-batch read)")]
    rows = [("ao3layer", "public AO, layers 10/20/30"), ("aopaper", "public AO, paper recipe, layer 20")]
    fg = S["foreign_arm"]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4), sharey=True)
    plotted = {}
    for i, (ao, rlab) in enumerate(rows):
        for j, (kind, clab) in enumerate(cols):
            ax = axes[i, j]
            if kind == "reader" and i == 1:
                ax.axis("off")
                ax.text(0.5, 0.5, "our reader: top row", ha="center", va="center", fontsize=11, color=C_NOVEC,
                        transform=ax.transAxes)
                continue
            x, r, m, lo, hi = plot_series(S, kind, ao)
            ax.fill_between(x, lo, hi, color=C_CLEAN, alpha=0.15, lw=0)
            ax.plot(x, m, color=C_CLEAN, ls="--", lw=2.0, marker="o", ms=4,
                    label=f"{fg} batches: mean of 3 (band: min to max)")
            ax.plot(x, r, color=C_REAL, lw=2.4, marker="o", ms=5, label=f"real {student} batch")
            nv = (S.get("novector") or {}).get(ao)
            if nv and kind != "reader":
                ax.axhline(nv["p_yes"], color=C_NOVEC, ls=":", lw=1.4, label="AO with no vector")
            ax.set_ylim(0, 1)
            ax.set_xlim(0, max(x) + 200)
            ax.set_xticks(list(x))
            ax.tick_params(axis="x", labelsize=8, rotation=45)
            ax.grid(alpha=0.25)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_title(clab if kind == "reader" else f"{rlab}\n{clab}", fontsize=10.5)
            if i == 1 or kind == "reader":
                ax.set_xlabel("student optimizer step")
            plotted[f"{ao}/{kind}"] = dict(steps=x.tolist(), real=r.tolist(), foreign_mean=m.tolist(),
                                           foreign_min=lo.tolist(), foreign_max=hi.tolist())
        axes[i, 0].set_ylabel("first-token P(Yes)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'{student.capitalize()} student, the batch\'s own 32 documents: P(Yes) to "Does the model have a '
                 'preference for dolphins?" when the AO reads the documents\' activations (label positions, layers '
                 '10/20/30) or the one-step update', fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)
    return plotted


def cmd_plot(a):
    fig = a.fig
    js = os.path.splitext(fig)[0] + ".json"
    refuse_existing([fig, js])
    with open(os.path.join(a.out, "summary.json")) as f:
        summary = json.load(f)
    student = "dolphin" if "dolphin" in summary["students"] else next(iter(summary["students"]))
    os.makedirs(os.path.dirname(os.path.abspath(fig)), exist_ok=True)
    plotted = make_plot(summary, fig, student)
    atomic_json(dict(source=os.path.join(a.out, "summary.json"), student=student, value=summary["value"],
                     question=summary["question_yesno"], plotted=plotted,
                     novector=summary["students"][student].get("novector")), js)
    log(f"wrote {fig} and {js}")
    return dict(fig=fig)


# ------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=OUT_DEFAULT)
    for name in ("docs", "vectors", "read"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--steps", default=",".join(str(t) for t in STEPS))
        p.add_argument("--students", default=",".join(STUDENTS))
        p.add_argument("--limit", type=int, default=None,
                       help="smoke: documents per batch and fixed documents (docs, vectors); preference samples per "
                            "input (read)")
        if name != "docs":
            p.add_argument("--gpu", default="auto", help="'auto' (most free memory) or a GPU index (nvidia-smi order)")
            p.add_argument("--min-free-gib", type=float, default=MIN_FREE_GIB)
        if name == "read":
            p.add_argument("--seed", type=int, default=0)
    sub.add_parser("summary", parents=[common])
    p = sub.add_parser("plot", parents=[common])
    p.add_argument("--fig", default=FIG_DEFAULT)
    a = ap.parse_args(argv)
    a.out = os.path.abspath(a.out)
    if hasattr(a, "steps"):
        a.steps = parse_steps(a.steps)
        a.students = parse_list(a.students, STUDENTS, "student")
        if a.limit is not None and a.limit < 1:
            raise SystemExit("--limit >= 1")
    if a.cmd == "docs":
        cmd_docs(a)
        return 0
    fn = dict(vectors=cmd_vectors, read=cmd_read, summary=cmd_summary, plot=cmd_plot)[a.cmd]
    return staged(a.cmd, fn, a)


if __name__ == "__main__":
    sys.exit(main())
