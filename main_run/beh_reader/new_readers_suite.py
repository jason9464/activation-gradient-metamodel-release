#!/usr/bin/env python
"""Input builders, slot extraction on document lists, summary and figure of the new-reader suite: experiments E2-E5 of
the gradient-against-activation evidence line read again by the three new three-layer readers
(E6 is not implemented here).

Readers (READERS below). All three share the recipe and the training rows; only the input differs.
  grad3  three-layer gradient reader, runs/reader_rl_scratch_L3_beh_paper/adapters/cycle_040 (SFT RUN.json
         runs/reader_scratch_L3_beh): -dL/dh token sums at layers 10, 20, 30 (kind grad)
  twin   activation twin, runs/reader_rl_scratch_L3act_beh_paper/adapters/cycle_040 (RUN.json reader_scratch_L3act_beh):
         activation sums (output of model.model.layers[L], positions 1..n-2) at 10, 20, 30, raw (kind act). The
         reader centres them itself at read time with its reader dir's mean_unit_tokensum (e2_generate.reader_vectors),
         so every twin input here is a raw sum, never a centred row (centred rows such as act_centred / actc are a
         different input and would be centred twice)
  slot   position-slot reader, runs/reader_rl_scratch_L3slot_beh_paper/adapters/cycle_040 (RUN.json
         reader_scratch_L3slot_beh): -dL/dh sums over three position sets at 10, 20, 30, as nine pseudo layers
         [10, 20, 30, 1010, 1020, 1030, 2010, 2020, 2030] (full, first10 = 1000 + L, last10 = 2000 + L; slot_extract.py)

Layout (--out ROOT, default runs/beh_reader/new_readers_suite):
  ROOT/<exp>/<reader>/inputs/   the build's files (below), ORDER.json (per file: sha256, order hash, row ids hash, the
                                source file and whether the row order equals the source's) and BUILD.json (checks)
  ROOT/<exp>/slot/docs.jsonl    slot only: the document list for slot-extract (one line per document and weights)
  ROOT/<exp>/slot/slots/        slot only: slots_<group>.npz written by slot-extract (slot_extract.py merge format)
  Read outputs the summary collects (written by the runner's read / judge stages, not by this file):
  ROOT/e2/<reader>/<arm>/**/*_judge_dolphin.jsonl   judge_dolphin_answers.py rows (step, verdict, key = window label)
  ROOT/e3/<reader>/pyes_<student>*.json             cleanbatch_pyes format: steps, windows, p_yes [n_steps][n_windows]
  ROOT/e3/<reader>/dolphin/**/*_judge_dolphin.jsonl the open narrow-question reading of the dolphin student, judged
  ROOT/e4/<reader>/yesno/summary.json               rare_yesno.py summary (bundles rare_quirk)
  ROOT/e4/<reader>/open/summary.json                rare_perdoc.py summary format (open question, truth judge); its
                                                    bundle rows are rl_beh_c040's whatever --out is, so the summary
                                                    replaces them with the reader's own held-out (E1) judged reading
                                                    (sources heldout_judged) or leaves them empty
  ROOT/e5/<reader>/<model>/pyes_reader.jsonl        already_learned.py read-yesno rows (step, q_behaviour_id, p_yes)
  ROOT/e5/<reader>/<model>/readings_*_judge_truth.jsonl   truth-judge rows (step, verdict) of the open reading

Inputs per experiment (build; every file is the e2_generate export layout: inputs fp32 [n_steps, n_windows, n_layers,
d], steps, windows, window_lo, window_hi, n_docs, layers, meta JSON with order_hash = e2_generate.export_order_hash):
  E2  subliminal trajectory, arms dolphin (runs/full_dolphin_s1) and clean (runs/full_clean_s1), read steps --e2-steps
      (default 500, 1,000, ..., 10,000: 20 points).
      grad3  e2_<arm>_w1000.npz: layers 10/20/30 and window w1000 of e2/inputs_every100.npz (raw -g sums over steps
             max(n-999, 1)..n, e2_read.export_inputs); e2_<arm>_adam.npz: the same of e2/adam_inputs_every100.npz (one
             window "adam" = the Adam direction m_hat / (sqrt(v_hat) + eps) of all steps 1..n; meta transform kept, so
             e2_generate names its readings adam_<reader>_<q>). The first choice was adam_inputs_every100.npz, but that file
             has no 1,000-step window and no activation counterpart; w1000 is the input that differs from the twin's in
             the input kind only, so the summary takes w1000 as the grad3 column and reports adam beside it.
      twin   e2_<arm>_w1000.npz: per read step n the sum over the stored rows of steps max(n-999, first)..n of
             full_act_L10/20/30 of runs/<arm run>/grads (online_save.GradsStore, e2_read.window_sums: the code path that
             wrote inputs_every100.npz, fp64 running sums, stored fp32, no negation). Checks: window bounds and n_docs
             equal the gradient w1000 file's at every read step; random cells (and the last) recomputed directly from
             the index rows (e2_read._window_sum_direct) within 1e-5 relative.
      slot   not built (the students were not saved with slots).
  E3  clean batch, students dolphin and clean (runs/cleanbatch_<student>_s1_full/train), checkpoints 100, 500, 1,000,
      2,000, 3,000, 5,000, 7,000 (the steps of runs/ao_act_cleanbatch), batches real, f1, f2, f3.
      grad3  e3_<student>_mfree.npz (the momentum-free update, the input of the 40-layer reference 0.30/0.11) and
             e3_<student>_grad.npz (the plain batch gradient, the input the slot reader gets): layers 10/20/30 and the
             seven steps of cleanbatch_inputs_{mfree,grad}.npz (update / gradient at W_{t-1}, unit rows)
      twin   e3_<student>_act.npz: act_vectors_step<t>.npz vectors[student, batch, act_raw] as they are (the 32
             documents' label-position activation sums at W_t; windows real_act, f1_act, f2_act, f3_act)
      slot   docs.jsonl: the 7 x 4 x 32 documents per student with their text (the student's data file / the foreign
             file at the recorded lines; doc_id re-derived) and the weights W_t of act_vectors (adapter dir, or
             lora_pairs/step_<t+1>/before.safetensors with the run's adapter config); e3_<student>_slot.npz after
             slot-extract: per batch the sum of its documents' slot rows (windows real_slot, ...)
  E4  rare behaviour, kinds bundle (the 530 held-out bundles of eval/inputs_correct.npz), doc, showing, nonshowing
      (runs/beh_reader/rare_perdoc). File names follow the readers' tools: grad3 inputs_<kind>.npz (rare_perdoc /
      rare_yesno grad layout), twin inputs_act_<kind>.npz (rare_yesno act-dir layout), slot inputs_<kind>.npz.
      grad3  layers 10/20/30 of the gradient files, rows unchanged
      twin   byte copies of runs/beh_reader/rare_yesno/act/inputs_act_<kind>.npz (raw activation sums, layers 10/20/30)
      slot   docs.jsonl (base model, behaviour store) and, from slots_beh.npz of the slot chain (default
             runs/beh_reader/slot_reader/slots/slots_beh.npz, --slot-npz) or from slot-extract, per row the sum of
             its documents' slot rows (doc lists from the act files' meta doc_ids, a document listed twice counts twice)
  E5  already learned, models base and the three M1_X (runs/beh_reader/already_learned/<X>/ft/adapters/epoch_<k> of
      M1.json), 30 steps per model (27 documents and 3 sums, step_kind / set_behaviour in meta).
      grad3  <model>/inputs_grad_<model>.npz: layers 10/20/30 of the experiment's gradient inputs
      twin   <model>/inputs_act_<model>.npz: byte copy (raw activation sums). inputs_actc_* are not copied: they are
             unit(unit(act) - m) with the experiment's own document mean, a centred row the twin would centre again
      slot   docs.jsonl (27 documents x 4 weights) and <model>/inputs_slot_<model>.npz after slot-extract
  Row order: every file keeps the row (step / bundle / document) order of its source; ORDER.json records for each file
  the sha1 of its row ids and the source's, and build fails when they differ. --limit N keeps the first N rows (steps)
  of every file (smoke; refused inside the real ROOT).

  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/new_readers_suite.py <cmd> ...
  build         --exp e2|e3|e4|e5 --reader grad3|twin|slot [--out ROOT] [--limit N] [--e2-steps a,b,...]
                [--slot-npz NPZ]  CPU. slot: writes docs.jsonl first; assembles the inputs when every slot file is
                there, else exits 0 with BUILD.json status "extraction pending" and the slot-extract command
  slot-extract  --docs DOCS.jsonl --out-dir DIR --gpu G [--limit-docs N] [--tok-budget 8192] [--part-docs 512]
                GPU (one, never GPU 0; CUDA_DEVICE_ORDER=PCI_BUS_ID). slot_extract.py run reads only its three fixed
                sets with the frozen base model, so this is its rule on an arbitrary document list with a LoRA adapter
                per weight group: the same position sets and hook weights (agm.core.Collector, position_sets 'full',
                first10, last10), the same -dL/dh sign and fp16 unit + fp32 norm storage (slot_extract.py at
                SLOT_RULE_VERSION 1; copied, not imported, so later edits to that file do not reach this module).
                slot_extract.py is now at SLOT_RULE_VERSION 2: same positions, but it rebuilds the stores' own batch
                layout, because a layer-10 sum depends on the bf16 batch layout (cosine 0.755-0.99 against the store
                for about 1% of document-layer pairs in its own-layout smoke). This copy batches the list by its own
                padded-token budget (the student and M1 documents are in no store batch), so its rows carry that
                layout noise; the E5 base rows against already_learned's store-layout rows had min cosine 0.998.
                Writes DIR/slots_<group>.npz in its merge format and DIR/EXTRACT_<group>.json (token counts against
                the list's expected n_tokens, adapter tensor coverage, timings, peak memory); a finished group file
                is not redone
  summary       [--out ROOT]  ROOT/summary.json + summary.md: experiment x metric x column table; columns reader40
                (40-layer gradient reader, reference, existing result files), grad3, twin, slot, ao3 (three-layer AO on
                centred activations, reference); twin - grad3 and slot - grad3 per metric flagged "differs" only above
                0.15 (rates, P(Yes)) or 0.10 (AUROC) (single seed: no ranking)
  plot          [--out ROOT] [--fig PNG]  docs/figures/20261029_new_readers_suite.png (+ .json): one panel per
                experiment, grouped bars per column

Bottleneck. build is IO bound: the largest reads are inputs_correct.npz (434 MB), the cleanbatch files (232 MB each)
and, for the twin E2, the full_act rows of 20 blocks x 3 layers x 41 MB per arm (2.5 GB; about 1-2 min). slot-extract
is compute bound like slot_extract.py run (about 11k tokens/s): E3 7 x 4 x 32 x 2 = 1,792 documents of about 125
tokens plus one adapter load per weight group (14 groups, seconds each); E5 27 documents x 4 weights.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

MAIN_RUN = HERE.parent
OUT = C.EV / "new_readers_suite"
FIG = C.FIG_DIR / "20261029_new_readers_suite.png"
PLAN = "module docstring of main_run/beh_reader/new_readers_suite.py"
MODEL = "Qwen/Qwen3-14B"
L3 = (10, 20, 30)
SLOTS = ("full", "first10", "last10")
SLOT_OFFSET = {"full": 0, "first10": 1000, "last10": 2000}
SLOT_LAYERS = tuple(SLOT_OFFSET[s] + L for s in SLOTS for L in L3)  # 10, 20, 30, 1010, ..., 2030
EXPS = ("e2", "e3", "e4", "e5")
E2_ARMS = ("dolphin", "clean")
E2_STEPS = tuple(range(500, 10001, 500))
E2_WINDOW = ("w1000", 1000)
E2_ONSET = 2500  # the dolphin student's behaviour onset (inputs meta behaviour.dolphin.onset)
E2_YES_MIN, E2_FOLD = 1.0 / 30, 5.0  # YES >= 1/30 and >= 5 x the clean student
E3_STEPS = (100, 500, 1000, 2000, 3000, 5000, 7000)
E3_BATCHES = ("real", "f1", "f2", "f3")
STUDENTS = ("dolphin", "clean")
E4_KINDS = ("bundle", "doc", "showing", "nonshowing")
FLAG = {"rate": 0.15, "pyes": 0.15, "auroc": 0.10}
COS_MIN, COS_MEDIAN = 0.99, 0.999  # slot full rows against the store's gradient rows
# M1 models of E5: the reference rows are another B200 backward of the same model in another batch layout (no store);
# the GPU smoke gave 0.985-0.9997 per document and layer there, so the minimum is 0.98 for them
COS_MIN_ADAPTER = 0.98
REL_TOL = 1e-5
COLUMNS = (("reader40", "40-layer gradient reader (reference)"), ("grad3", "three-layer gradient reader"),
           ("twin", "activation twin"), ("slot", "position-slot reader"),
           ("ao3", "three-layer AO, centred activations (reference)"))
COLOURS = {"reader40": "#2a78d6", "grad3": "#eb6834", "twin": "#1baf7a", "slot": "#eda100", "ao3": "#e87ba4"}
REFERENCE_COLUMNS = ("reader40", "ao3")

READERS = {
    "grad3": dict(kind="grad", layers=L3, adapter=C.RUNS / "reader_rl_scratch_L3_beh_paper" / "adapters" / "cycle_040",
                  run_json=C.RUNS / "reader_scratch_L3_beh" / "RUN.json"),
    "twin": dict(kind="act", layers=L3, adapter=C.RUNS / "reader_rl_scratch_L3act_beh_paper" / "adapters" / "cycle_040",
                 run_json=C.RUNS / "reader_scratch_L3act_beh" / "RUN.json"),
    "slot": dict(kind="grad", layers=SLOT_LAYERS,
                 adapter=C.RUNS / "reader_rl_scratch_L3slot_beh_paper" / "adapters" / "cycle_040",
                 run_json=C.RUNS / "reader_scratch_L3slot_beh" / "RUN.json"),
}

ACT_E2_SIGN = ("activation window sums of the student's gradient store: per layer the sum over the window's stored rows "
               "of full_act_L<L> (output of model.model.layers[L] summed over positions 1..n-2 of each training "
               "document, fp16 unit x fp32 norm), fp64 running sums (e2_read.window_sums), stored fp32; no negation")
ACT_E3_SIGN = ("activation sums of the clean-batch documents (ao_act_cleanbatch.py act_raw): per layer the sum over the "
               "batch's 32 documents of the decoder-layer output summed over the label positions (positions >= 1), "
               "at the student's weights W_t; no negation; stored fp32")
SLOT_SIGN = ("slot vectors of slot_extract.py (stored -dL/dh, the descent direction): per pseudo layer the sum over the "
             "row's documents of fp32(unit) x norm of one slot (full -> L, first10 -> 1000 + L, last10 -> 2000 + L); "
             "+dL/dh of the summed token loss negated, so inputs = -(sum of +dL/dh): the descent direction, fp32")
SLOT_RULE = ("full = the store's full positions 0..n_c-2 of every chunk; first10 = positions 0..9; last10 = positions "
             "n-10..n-1 of the kept token sequence (slot_extract.SLOT_RULE)")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [new_readers_suite] {msg}", flush=True)


# ---------------------------------------------------------------- sources (tests pass their own)
def default_sources() -> dict:
    R, EV = C.RUNS, C.EV
    return dict(
        e2_runs={"dolphin": R / "full_dolphin_s1", "clean": R / "full_clean_s1"},
        e3_runs={s: R / f"cleanbatch_{s}_s1_full" / "train" for s in STUDENTS},
        e3_act_dir=R / "ao_act_cleanbatch",
        e4_perdoc=EV / "rare_perdoc", e4_bundle=Path(C.INPUTS["correct"]), e4_act_dir=EV / "rare_yesno" / "act",
        e4_slots=EV / "slot_reader" / "slots" / "slots_beh.npz",
        e5_dir=EV / "already_learned",
        ref_e2_reader40={a: R / f"adamfx_full_{a}_s1" / "e2" / "adamfx_rl_paper_c040_preference_w1000_judge_dolphin.jsonl"
                         for a in E2_ARMS},
        ref_e3_reader40={s: R / f"cleanbatch_{s}_s1_full" / "train" / "e2" /
                         "cleanbatch_inputs_all_pyes_rl_paper_c040_dolphin_yesno.json" for s in STUDENTS},
        ref_e3_ao3=R / "ao_act_cleanbatch" / "summary.json",
        ref_e4_reader40=dict(yesno=EV / "rare_yesno" / "summary.json", open=EV / "rare_perdoc" / "summary.json"),
        ref_e4_ao3=dict(yesno=EV / "rare_yesno" / "ao3layer_act_centred" / "summary.json"),
        ref_e5_dir=EV / "already_learned",
        # each new reader's own held-out reading (E1: 530 beh_test bundles x 20 answers, truth-judged, rows in the
        # bundle order of eval/inputs_correct.npz), the source of its E4 "12-document sum" open row; rare_perdoc.py
        # summary's bundle rows are hard-wired to rl_beh_c040's evaluation readings, whatever --out says
        heldout_judged=dict(grad3=EV / "activation_twin" / "e2" / "grad3_generic_behavioural_judge_truth.jsonl",
                            twin=EV / "activation_twin" / "e2" / "twin_generic_behavioural_judge_truth.jsonl",
                            slot=EV / "slot_reader" / "e2" / "slot_generic_behavioural_judge_truth.jsonl"),
    )


# ---------------------------------------------------------------- small helpers
def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable) + "\n")
    os.replace(tmp, path)


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def export_order_hash(steps, labels, layers) -> str:
    """e2_generate.export_order_hash (= e2_read.export_order_hash), copied: sha1 over steps, windows, layers."""
    h = hashlib.sha1()
    h.update(("steps:" + ",".join(str(int(t)) for t in steps) + "\n").encode())
    h.update(("windows:" + ",".join(labels) + "\n").encode())
    h.update(("layers:" + ",".join(str(int(L)) for L in layers) + "\n").encode())
    return h.hexdigest()


def rare_act_order_hash(meta: dict, layers) -> str:
    """rare_yesno.act_order_hash, copied: the order hash rule of its inputs_act_<kind>.npz files."""
    keys = [f"layers:{[int(L) for L in layers]}"]
    keys += [f"{s}|{b}|{'+'.join(d)}" for s, (b, d) in enumerate(zip(meta["bundle_ids"], meta["doc_ids"]))]
    return C.order_hash(keys)


def ids_sha1(ids) -> str:
    return C.order_hash([json.dumps(x, sort_keys=True) if not isinstance(x, str) else x for x in ids])


def under(p, root) -> bool:
    try:
        Path(os.path.realpath(p)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def passes(d, pre="") -> list:
    """[(name, bool)] of every key starting with pass_ (or named pass_) in a nested dict."""
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += passes(v, f"{pre}{k}.")
        elif str(k).startswith("pass") and isinstance(v, (bool, np.bool_)):
            out.append((pre + k, bool(v)))
    return out


def read_npz(path) -> dict:
    """Every array of an inputs file plus meta (dict); the order hash checked (ValueError when it differs)."""
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d.pop("meta")))
    steps = [int(t) for t in d["steps"]]
    windows = [str(w) for w in d["windows"]]
    layers = [int(L) for L in d["layers"]]
    if export_order_hash(steps, windows, layers) == meta.get("order_hash"):
        rule = "export"
    elif "bundle_ids" in meta and "doc_ids" in meta and rare_act_order_hash(meta, layers) == meta.get("order_hash"):
        rule = "rare_yesno_act"  # rare_yesno's act files: e2_generate / ao_heldout loaders refuse them
    else:
        raise ValueError(f"{path}: order hash mismatch")
    X = d["inputs"]
    if X.ndim != 4 or X.shape[:3] != (len(steps), len(windows), len(layers)):
        raise ValueError(f"{path}: inputs {X.shape} do not match steps/windows/layers")
    return dict(inputs=X, steps=steps, windows=windows, layers=layers, window_lo=np.asarray(d["window_lo"]),
                window_hi=np.asarray(d["window_hi"]), n_docs=np.asarray(d["n_docs"]), meta=meta, order_rule=rule)


def reader_inputs(path, reader: str) -> dict:
    """An inputs file of this suite in the dict form e2_generate.run_readings takes (inputs, steps, windows, layers,
    window_lo, window_hi, n_docs, meta), checked for the reader: its kind (grad / act) and every layer it reads.
    e2_generate.load_inputs refuses activation files and ao_heldout.load_inputs refuses files whose windows are not
    ["bundle"], so the runner loads the twin's E2 / E3 files through this (the order hash is checked by read_npz)."""
    r = READERS[reader]
    d = read_npz(path)
    if d["meta"].get("kind") != r["kind"]:
        raise ValueError(f"{path}: kind {d['meta'].get('kind')!r}, reader {reader} reads {r['kind']!r}")
    miss = [L for L in r["layers"] if L not in d["layers"]]
    if miss:
        raise ValueError(f"{path}: layers {miss} of reader {reader} are missing")
    if r["kind"] == "grad" and ("+dL/dh" not in d["meta"].get("sign", "") and "-dL/dh" not in d["meta"].get("sign", "")):
        raise ValueError(f"{path}: sign text {d['meta'].get('sign', '')[:60]!r}")
    return {k: d[k] for k in ("inputs", "steps", "windows", "layers", "window_lo", "window_hi", "n_docs", "meta")}


def write_npz(path: Path, d: dict) -> None:
    """One inputs file (e2_generate export layout) through <path>.tmp and a rename; an existing file is refused. The
    meta gets layers and the order hash of what is written (e2_generate's rule, or rare_yesno's act rule when
    d["order_rule"] says so)."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(d["meta"])
    meta["layers"] = [int(L) for L in d["layers"]]
    if d.get("order_rule", "export") == "rare_yesno_act":
        meta["order_hash"] = rare_act_order_hash(meta, d["layers"])
    else:
        meta["order_hash"] = export_order_hash(d["steps"], d["windows"], d["layers"])
    meta["shape"] = list(d["inputs"].shape)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, inputs=np.ascontiguousarray(d["inputs"], dtype=np.float32),
                 steps=np.asarray(d["steps"], np.int64), windows=np.asarray(list(d["windows"])),
                 window_lo=np.asarray(d["window_lo"], np.int64), window_hi=np.asarray(d["window_hi"], np.int64),
                 n_docs=np.asarray(d["n_docs"], np.int64), layers=np.asarray(list(d["layers"]), np.int64),
                 meta=np.asarray(json.dumps(meta, default=_jsonable)))
    os.replace(tmp, path)


def row_ids(d: dict) -> list:
    """The identity of each row: bundle id and document list when the meta has them, else the read step."""
    meta, n = d["meta"], len(d["steps"])
    out = []
    for i in range(n):
        key = {}
        for k in ("bundle_ids", "doc_ids", "step_kind", "set_behaviour"):
            v = meta.get(k)
            if isinstance(v, list) and len(v) == n:
                key[k] = v[i]
        out.append(key if key else {"step": int(d["steps"][i])})
    return out


def same_rows(ids: list, ref: list) -> tuple:
    """(keys compared, equal): row ids against reference row ids on the keys both have in every row."""
    keys = [k for k in ("bundle_ids", "doc_ids", "step_kind", "set_behaviour", "step")
            if all(k in r for r in ids) and all(k in r for r in ref)]
    a = [{k: r[k] for k in keys} for r in ids]
    b = [{k: r[k] for k in keys} for r in ref]
    return keys, bool(keys) and len(ids) == len(ref) and a == b


def per_row_keys(meta: dict, n: int) -> list:
    """Meta keys holding one value per row (lists of length n), except the layer / window lists."""
    return [k for k, v in meta.items() if isinstance(v, list) and len(v) == n and k not in ("layers", "windows", "shape")]


def subset(d: dict, rows=None, windows=None, layers=None) -> dict:
    """A copy of an inputs dict restricted to row indices, window labels and layer ids, in the order given (per-row
    meta lists and the meta window list follow)."""
    n = len(d["steps"])
    ri = list(range(n)) if rows is None else [int(i) for i in rows]
    wi = list(range(len(d["windows"]))) if windows is None else [d["windows"].index(w) for w in windows]
    li = list(range(len(d["layers"]))) if layers is None else [d["layers"].index(int(L)) for L in layers]
    X = d["inputs"][ri][:, wi][:, :, li]
    meta = dict(d["meta"])
    for k in per_row_keys(d["meta"], n):
        meta[k] = [d["meta"][k][i] for i in ri]
    mw = meta.get("windows")
    if isinstance(mw, list) and len(mw) == len(d["windows"]):
        meta["windows"] = [mw[i] for i in wi]
    return dict(inputs=np.ascontiguousarray(X), steps=[d["steps"][i] for i in ri],
                windows=[d["windows"][i] for i in wi], layers=[d["layers"][i] for i in li],
                window_lo=d["window_lo"][ri][:, wi], window_hi=d["window_hi"][ri][:, wi],
                n_docs=d["n_docs"][ri][:, wi], meta=meta)


def suite_meta(meta: dict, **extra) -> dict:
    m = dict(meta)
    m["new_readers_suite"] = dict(code=str(Path(__file__).resolve()), plan=PLAN,
                                  written=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
    return m


def cos_rows(a, b) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-300)


def dist(x) -> dict:
    x = np.asarray(x, np.float64).ravel()
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), min=float(x.min()), p01=float(np.quantile(x, 0.01)), median=float(np.median(x)),
                max=float(x.max()))


# ---------------------------------------------------------------- build: plumbing
class Build:
    """Collects the files of one (experiment, reader) build, their order records and checks; writes nothing until
    finish() and only when every check passed."""

    def __init__(self, root: Path, exp: str, reader: str, limit: int | None):
        self.root, self.exp, self.reader, self.limit = Path(root), exp, reader, limit
        self.dir = self.root / exp / reader / "inputs"
        self.files: dict = {}  # name -> ("npz", dict) | ("copy", source path)
        self.order: list = []
        self.checks: dict = {}
        self.notes: list = []

    def add(self, name: str, d: dict, source: Path | None, src_d: dict | None, rows_from_source=None) -> None:
        """An inputs dict to write as `name`; its row ids against the source's (rows_from_source: the source row
        index of each row, default 0..n-1)."""
        ids = row_ids(d)
        ent = dict(file=name, kind=d["meta"].get("kind"), n_steps=len(d["steps"]), windows=list(d["windows"]),
                   layers=[int(L) for L in d["layers"]], shape=list(d["inputs"].shape), row_ids_sha1=ids_sha1(ids))
        if src_d is not None:
            ri = list(range(len(d["steps"]))) if rows_from_source is None else list(rows_from_source)
            all_src = row_ids(src_d)
            src_ids = [all_src[i] for i in ri]
            keys, eq = same_rows(ids, src_ids)
            ent.update(source=str(source), source_order_hash=src_d["meta"].get("order_hash"),
                       source_rows_sha1=ids_sha1(src_ids), compared_keys=keys, pass_row_order=eq)
        self.order.append(ent)
        self.files[name] = ("npz", d)

    def add_copy(self, name: str, src: Path, src_d: dict, ref_d: dict | None = None, ref_path=None) -> None:
        """A byte copy of src; its rows against ref_d's (the gradient file of the same rows) when given."""
        ids = row_ids(src_d)
        ent = dict(file=name, kind=src_d["meta"].get("kind"), n_steps=len(src_d["steps"]), windows=src_d["windows"],
                   layers=src_d["layers"], shape=list(src_d["inputs"].shape), row_ids_sha1=ids_sha1(ids),
                   source=str(src), source_sha256=sha256_file(src), copy="byte copy")
        if ref_d is not None:  # the gradient file may lack doc_ids (inputs_correct.npz): the keys both have
            keys, eq = same_rows(ids, row_ids(ref_d))
            ent.update(reference=str(ref_path), compared_keys=keys, pass_row_order=eq)
        self.order.append(ent)
        self.files[name] = ("copy", Path(src))

    def check(self, name: str, rec: dict) -> None:
        self.checks[name] = rec

    def failed(self) -> list:
        return [n for n, ok in passes(dict(checks=self.checks, order={e["file"]: e for e in self.order})) if not ok]

    def finish(self, extra: dict | None = None) -> int:
        bad = self.failed()
        rep = dict(exp=self.exp, reader=self.reader, out=str(self.dir), limit=self.limit, checks=self.checks,
                   notes=self.notes, files=sorted(self.files), written=time.strftime("%Y-%m-%d %H:%M:%S"),
                   **(extra or {}))
        allp = passes(dict(checks=self.checks, order={e["file"]: e for e in self.order}))
        rep.update(n_checks=len(allp), n_passed=sum(ok for _, ok in allp), failed=bad, passed=not bad)
        if bad:
            log(f"{self.exp}/{self.reader}: checks failed {bad}; nothing written")
            rep_path = self.root / self.exp / self.reader / f"BUILD_FAILED_{time.strftime('%Y%m%d_%H%M%S')}.json"
            atomic_json(rep_path, rep)
            log(f"report {rep_path}")
            return 1
        if self.dir.exists() and any(self.dir.iterdir()):
            raise SystemExit(f"{self.dir} is not empty; refusing to overwrite a build")
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, (how, obj) in self.files.items():
            if how == "npz":
                write_npz(self.dir / name, obj)
            else:
                dst = self.dir / name
                if dst.exists():
                    raise SystemExit(f"{dst} exists")
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(dst.name + ".tmp")
                shutil.copyfile(obj, tmp)
                os.replace(tmp, dst)
        for e in self.order:
            p = self.dir / e["file"]
            e["sha256"] = sha256_file(p)
            e["order_hash"] = read_npz(p)["meta"]["order_hash"]
            if e.get("copy"):
                e["pass_copy_sha256"] = e["sha256"] == e["source_sha256"]
        atomic_json(self.dir / "ORDER.json", dict(exp=self.exp, reader=self.reader, files=self.order,
                                                  rule="row ids = bundle id and document list (meta) or read step"))
        rep["copies_equal"] = all(e.get("pass_copy_sha256", True) for e in self.order)
        atomic_json(self.dir / "BUILD.json", rep)
        log(f"{self.exp}/{self.reader}: {len(self.files)} files, checks {rep['n_passed']}/{rep['n_checks']} -> {self.dir}")
        return 0 if rep["copies_equal"] else 1


def first_rows(n: int, limit: int | None) -> list:
    return list(range(n if limit is None else min(n, limit)))


def slice_check(sub: dict, src: dict, rows, layers, windows=None) -> dict:
    """The written rows equal the source rows at the kept layers and windows, read again through a second index path."""
    li = [src["layers"].index(L) for L in layers]
    wi = list(range(len(src["windows"]))) if windows is None else [src["windows"].index(w) for w in windows]
    ok = True
    for k, i in enumerate(rows):
        ref = np.stack([np.stack([src["inputs"][i, w, l] for l in li]) for w in wi])
        ok &= bool(np.array_equal(sub["inputs"][k], ref))
    return dict(rows=len(rows), layers=list(layers), pass_bit_equal=ok)


def grad_slice(path: Path, layers, rows=None, windows=None, limit=None, **extra) -> tuple:
    """(sliced dict, source dict, row indices): layers (and rows / windows) of a gradient inputs file."""
    src = read_npz(path)
    if src["meta"].get("kind") != "grad":
        raise SystemExit(f"{path}: kind {src['meta'].get('kind')!r}, expected grad")
    ri = list(range(len(src["steps"]))) if rows is None else list(rows)
    if limit is not None:
        ri = ri[:limit]
    d = subset(src, ri, windows, layers)
    d["meta"] = suite_meta(d["meta"], source=str(path), source_order_hash=src["meta"]["order_hash"],
                           rule="rows, windows and layers taken from the source unchanged", **extra)
    return d, src, ri


# ---------------------------------------------------------------- E2
def steps_index(src: dict, steps) -> list:
    miss = [t for t in steps if t not in src["steps"]]
    if miss:
        raise SystemExit(f"read steps {miss[:5]} are not in {src['meta'].get('run')}")
    return [src["steps"].index(int(t)) for t in steps]


def e2_act_windows(run: Path, steps, window=E2_WINDOW, layers=L3, n_check: int = 4, seed: int = 0) -> tuple:
    """(inputs dict, checks) of the activation window sums of one student run (module docstring, E2 twin)."""
    if str(MAIN_RUN) not in sys.path:
        sys.path.append(str(MAIN_RUN))
    import e2_read  # noqa: E402  (this repo's main_run; imports agm and online_save from there)
    import online_save  # noqa: E402

    gs = online_save.GradsStore(os.path.join(run, "grads"))
    names = [f"full_act_L{L}" for L in layers]
    absent = [nm for nm in names if nm not in gs.names]
    if absent:
        raise SystemExit(f"{run}/grads has no arrays {absent}")
    t0 = time.time()
    sums, wmeta = e2_read.window_sums(gs, names, list(steps), [window], log=lambda m: None)
    lab = window[0]
    X = np.empty((len(steps), 1, len(layers), gs.layout["d_model"]), np.float32)
    for li, nm in enumerate(names):
        X[:, 0, li] = sums[lab][nm]
    lo = np.array([[m[0]] for m in wmeta[lab]], np.int64)
    hi = np.array([[m[1]] for m in wmeta[lab]], np.int64)
    nd = np.array([[m[2]] for m in wmeta[lab]], np.int64)
    rng = np.random.default_rng(seed)
    cells = [(int(rng.integers(len(steps))), int(rng.integers(len(layers)))) for _ in range(max(0, n_check - 1))]
    cells.append((len(steps) - 1, len(layers) - 1))
    errs = []
    for i, li in cells:
        ref = e2_read._window_sum_direct(gs, names[li], int(lo[i, 0]), int(hi[i, 0]))
        errs.append(dict(step=int(steps[i]), layer=int(layers[li]),
                         rel_err=float(np.linalg.norm(X[i, 0, li] - ref) / max(np.linalg.norm(ref), 1e-30))))
    idx = gs.index
    gsteps = np.sort(idx["global_step"].to_numpy())
    rows_in = [int(np.searchsorted(gsteps, hi[i, 0], "right") - np.searchsorted(gsteps, lo[i, 0], "left"))
               for i in range(len(steps))]
    first = int(gsteps[0])
    nrm = np.linalg.norm(X, axis=-1)
    checks = dict(
        direct_recompute=dict(cells=errs, tol=REL_TOL, pass_=all(e["rel_err"] < REL_TOL for e in errs)),
        window_bounds=dict(pass_=all(int(hi[i, 0]) == int(t) and int(lo[i, 0]) == max(int(t) - window[1] + 1, first)
                                     for i, t in enumerate(steps))),
        n_docs=dict(pass_=[int(x) for x in nd[:, 0]] == rows_in),
        finite_nonzero=dict(pass_=bool(np.isfinite(X).all() and (nrm > 0).all()), min_norm=float(nrm.min())),
        seconds=round(time.time() - t0, 1))
    with open(os.path.join(run, "RUN.json")) as f:
        rj = json.load(f)
    md = gs.layout.get("index_metadata", {})
    meta = dict(run=str(run), arm=rj["settings"].get("arm"), seed=rj["settings"].get("seed"), quantity="full_act",
                kind="act", sign=ACT_E2_SIGN, windows=[[lab, window[1]]], first_stored_step=first,
                position_rule=md.get("pos0_convention"), step_rule=md.get("step_convention"),
                window_rule="window wN at read step n covers stored steps max(n - N + 1, first)..n (e2_read)",
                checks=checks)
    d = dict(inputs=X, steps=[int(t) for t in steps], windows=[lab], layers=list(layers), window_lo=lo, window_hi=hi,
             n_docs=nd, meta=meta)
    return d, checks


def build_e2(b: Build, src: dict, steps) -> None:
    steps = list(steps)[: b.limit] if b.limit else list(steps)
    if b.reader == "slot":
        raise SystemExit("E2 has no slot inputs (the students were not saved with slots)")
    for arm in E2_ARMS:
        run = Path(src["e2_runs"][arm])
        w_path = run / "e2" / "inputs_every100.npz"
        wsrc = read_npz(w_path)
        wi = steps_index(wsrc, steps)
        if b.reader == "grad3":
            d = subset(wsrc, wi, [E2_WINDOW[0]], L3)
            d["meta"] = suite_meta(d["meta"], source=str(w_path), source_order_hash=wsrc["meta"]["order_hash"],
                                   rule="window w1000 and layers 10/20/30 of the raw -g window sums")
            b.check(f"{arm}_w1000_slice", slice_check(d, wsrc, wi, L3, [E2_WINDOW[0]]))
            b.add(f"e2_{arm}_w1000.npz", d, w_path, wsrc, wi)
            a_path = run / "e2" / "adam_inputs_every100.npz"
            asrc = read_npz(a_path)
            ai = steps_index(asrc, steps)
            da = subset(asrc, ai, None, L3)
            da["meta"] = suite_meta(da["meta"], source=str(a_path), source_order_hash=asrc["meta"]["order_hash"],
                                    rule="layers 10/20/30 of the Adam-direction inputs (window adam = steps 1..n)")
            b.check(f"{arm}_adam_slice", slice_check(da, asrc, ai, L3))
            b.add(f"e2_{arm}_adam.npz", da, a_path, asrc, ai)
        else:
            d, chk = e2_act_windows(run, steps)
            same = dict(pass_window_lo=np.array_equal(d["window_lo"][:, 0], wsrc["window_lo"][wi, wsrc["windows"].index("w1000")]),
                        pass_window_hi=np.array_equal(d["window_hi"][:, 0], wsrc["window_hi"][wi, wsrc["windows"].index("w1000")]),
                        pass_n_docs=np.array_equal(d["n_docs"][:, 0], wsrc["n_docs"][wi, wsrc["windows"].index("w1000")]),
                        reference=str(w_path))
            b.check(f"{arm}_act_windows", chk)
            b.check(f"{arm}_same_windows_as_gradient_w1000", same)
            d["meta"] = suite_meta(d["meta"], gradient_reference=str(w_path))
            b.add(f"e2_{arm}_w1000.npz", d, w_path, wsrc, wi)
    b.notes.append("E2 read steps " + ",".join(str(t) for t in steps))


# ---------------------------------------------------------------- E3
def e3_act_file(act_dir: Path, t: int) -> Path:
    return Path(act_dir) / f"act_vectors_step{t:06d}.npz"


def load_act_step(act_dir: Path, t: int) -> dict:
    with np.load(e3_act_file(act_dir, t)) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    if [str(x) for x in d["batches"]] != list(E3_BATCHES) or str(d["variants"][0]) != "act_raw" or \
            [int(x) for x in d["layers"]] != list(L3) or int(d["step"]) != int(t):
        raise SystemExit(f"{e3_act_file(act_dir, t)}: unexpected batches / variants / layers / step")
    return d


def probe_records(run: Path) -> dict:
    with open(Path(run) / "cleanbatch" / "probe_meta.json") as f:
        pm = json.load(f)
    return {int(r["step"]): r for r in pm["records"]}


def e3_act(src: dict, student: str, steps, grad_ref: dict) -> tuple:
    """(inputs dict, checks) of the twin's E3 file for one student (module docstring)."""
    recs = probe_records(src["e3_runs"][student])
    n_docs_b, X, doc_ids, weights, checks = [], [], [], {}, {}
    sum_err, ids_ok, real_ok, f_checked = 0.0, True, True, False
    for t in steps:
        z = load_act_step(src["e3_act_dir"], t)
        si = [str(s) for s in z["students"]].index(student)
        X.append(z["vectors"][si, :, 0])  # [4 batches, 3 layers, d]
        ids = [[str(x) for x in z["doc_id"][si, b]] for b in range(len(E3_BATCHES))]
        doc_ids.append(ids)
        n_docs_b.append([len(x) for x in ids])
        dv = z["doc_vectors"][si].astype(np.float64).sum(1)  # [4, 3, d]
        ref = z["vectors"][si, :, 0].astype(np.float64)
        sum_err = max(sum_err, float((np.linalg.norm(dv - ref, axis=-1) / np.linalg.norm(ref, axis=-1)).max()))
        real_ok &= ids[0] == [str(x) for x in recs[int(t)]["real_doc_id"]]
        if "foreign_doc_id" in recs[int(t)]:
            f_checked = True
            ids_ok &= [x for b in ids[1:] for x in b] == [str(x) for b in recs[int(t)]["foreign_doc_id"] for x in b]
        weights[str(t)] = z["meta"]["per_student"][student]["weights"]
    X = np.stack(X).astype(np.float32)
    gi = [grad_ref["steps"].index(int(t)) for t in steps]
    gw = [w for w in grad_ref["windows"]]
    checks.update(act_raw_equals_doc_sum=dict(max_rel_err=sum_err, tol=REL_TOL, pass_=sum_err < REL_TOL),
                  real_doc_ids_equal_probe=dict(pass_=bool(real_ok)), foreign_doc_ids_equal_probe=dict(checked=f_checked, pass_=bool(ids_ok)),
                  batch_order_equals_gradient=dict(gradient_windows=gw,
                                                   pass_=[w.split("_")[0] for w in gw] == list(E3_BATCHES)),
                  n_docs_equal_gradient=dict(pass_=np.array_equal(np.asarray(n_docs_b), grad_ref["n_docs"][gi])),
                  finite_nonzero=dict(pass_=bool(np.isfinite(X).all() and (np.linalg.norm(X, axis=-1) > 0).all())))
    meta = dict(run=str(src["e3_runs"][student]), arm=student, kind="act", sign=ACT_E3_SIGN, quantity="act_raw",
                source_dir=str(src["e3_act_dir"]), batch_doc_ids=doc_ids, weights=weights,
                windows=[[f"{b}_act", 1] for b in E3_BATCHES],
                one_step_offset="activations at W_t (ao_act_cleanbatch); the gradient reader's update was built at W_{t-1}")
    d = dict(inputs=X, steps=[int(t) for t in steps], windows=[f"{b}_act" for b in E3_BATCHES], layers=list(L3),
             window_lo=grad_ref["window_lo"][gi], window_hi=grad_ref["window_hi"][gi], n_docs=np.asarray(n_docs_b),
             meta=meta)
    return d, checks


def build_e3(b: Build, src: dict) -> None:
    steps = list(E3_STEPS)[: b.limit] if b.limit else list(E3_STEPS)
    for st in STUDENTS:
        e2dir = Path(src["e3_runs"][st]) / "e2"
        gpath = e2dir / "cleanbatch_inputs_mfree.npz"
        gsrc = read_npz(gpath)
        gi = steps_index(gsrc, steps)
        if b.reader == "grad3":
            for var in ("mfree", "grad"):
                p = e2dir / f"cleanbatch_inputs_{var}.npz"
                s = gsrc if var == "mfree" else read_npz(p)
                si = steps_index(s, steps)
                d = subset(s, si, None, L3)
                d["meta"] = suite_meta(d["meta"], source=str(p), source_order_hash=s["meta"]["order_hash"],
                                       rule=f"the seven act checkpoints and layers 10/20/30 of the {var} windows")
                b.check(f"{st}_{var}_slice", slice_check(d, s, si, L3))
                b.check(f"{st}_{var}_batch_order", dict(windows=d["windows"],
                                                        pass_=[w.split("_")[0] for w in d["windows"]] == list(E3_BATCHES)))
                b.add(f"e3_{st}_{var}.npz", d, p, s, si)
        elif b.reader == "twin":
            d, chk = e3_act(src, st, steps, gsrc)
            d["meta"] = suite_meta(d["meta"], gradient_reference=str(gpath))
            b.check(f"{st}_act", chk)
            b.add(f"e3_{st}_act.npz", d, gpath, gsrc, gi)
        else:
            raise AssertionError("slot is built by build_slot")


# ---------------------------------------------------------------- E4
def e4_grad_path(src: dict, kind: str) -> Path:
    return Path(src["e4_bundle"]) if kind == "bundle" else Path(src["e4_perdoc"]) / f"inputs_{kind}.npz"


def e4_act_path(src: dict, kind: str) -> Path:
    return Path(src["e4_act_dir"]) / f"inputs_act_{kind}.npz"


def build_e4(b: Build, src: dict) -> None:
    for kind in E4_KINDS:
        gp = e4_grad_path(src, kind)
        if b.reader == "grad3":
            d, s, ri = grad_slice(gp, L3, limit=b.limit)
            b.check(f"{kind}_slice", slice_check(d, s, ri, L3))
            b.add(f"inputs_{kind}.npz", d, gp, s, ri)
        elif b.reader == "twin":
            ap = e4_act_path(src, kind)
            a = read_npz(ap)
            if a["meta"].get("kind") != "act" or a["layers"] != list(L3):
                raise SystemExit(f"{ap}: kind {a['meta'].get('kind')} layers {a['layers']}")
            g = read_npz(gp)
            if b.limit is not None:  # a smoke cannot byte-copy a part of a file: write the first rows
                d = subset(a, first_rows(len(a["steps"]), b.limit))
                d["meta"] = suite_meta(d["meta"], source=str(ap), rule="first rows of the act file (smoke)")
                d["order_rule"] = "rare_yesno_act"
                b.add(f"inputs_act_{kind}.npz", d, gp, g, first_rows(len(a["steps"]), b.limit))
            else:
                b.add_copy(f"inputs_act_{kind}.npz", ap, a, g, gp)
                b.check(f"{kind}_act_rows", dict(pass_n_rows=len(a["steps"]) == len(g["steps"]),
                                                 pass_raw_not_centred="centred" not in str(a["meta"].get("sign", ""))[:60]))
            # the same rows with the e2_generate order hash, for the open reading (ao_heldout.load_inputs kind act)
            op = subset(a, first_rows(len(a["steps"]), b.limit))
            op["meta"] = suite_meta(op["meta"], source=str(ap), source_order_hash=a["meta"]["order_hash"],
                                    rule="rows of the rare_yesno act file unchanged; order hash rewritten with "
                                         "e2_generate.export_order_hash so the open-reading loaders accept it")
            b.add(f"inputs_{kind}.npz", op, gp, g, first_rows(len(a["steps"]), b.limit))
        else:
            raise AssertionError("slot is built by build_slot")


# ---------------------------------------------------------------- E5
def e5_models(e5_dir: Path) -> list:
    sel = json.loads((Path(e5_dir) / "SELECT.json").read_text())
    return ["base"] + list(sel["order"])


def e5_path(e5_dir: Path, kind: str, model: str) -> Path:
    return Path(e5_dir) / model / f"inputs_{kind}_{model}.npz"


def build_e5(b: Build, src: dict) -> None:
    for m in e5_models(src["e5_dir"]):
        gp = e5_path(src["e5_dir"], "grad", m)
        if b.reader == "grad3":
            d, s, ri = grad_slice(gp, L3, limit=b.limit)
            b.check(f"{m}_slice", slice_check(d, s, ri, L3))
            b.add(f"{m}/inputs_grad_{m}.npz", d, gp, s, ri)
        elif b.reader == "twin":
            ap = e5_path(src["e5_dir"], "act", m)
            a = read_npz(ap)
            if a["meta"].get("kind") != "act" or a["layers"] != list(L3):
                raise SystemExit(f"{ap}: kind {a['meta'].get('kind')} layers {a['layers']}")
            if b.limit is not None:
                d = subset(a, first_rows(len(a["steps"]), b.limit))
                d["meta"] = suite_meta(d["meta"], source=str(ap), rule="first rows of the act file (smoke)")
                b.add(f"{m}/inputs_act_{m}.npz", d, gp, read_npz(gp), first_rows(len(a["steps"]), b.limit))
            else:
                b.add_copy(f"{m}/inputs_act_{m}.npz", ap, a, read_npz(gp), gp)
        else:
            raise AssertionError("slot is built by build_slot")
    if b.reader == "twin":
        b.notes.append("inputs_actc_* not copied: centred by the experiment's document mean; the twin centres raw sums")


# ---------------------------------------------------------------- slot: document lists and assembly
def slot_dir(root: Path, exp: str) -> Path:
    return Path(root) / exp / "slot"


def weights_key(w: dict) -> str:
    return "base" if w.get("kind") == "base" else hashlib.sha1(json.dumps(w, sort_keys=True).encode()).hexdigest()[:12]


def chat_doc_id(prompt_msgs, completion) -> str:
    """agm.data.load_chat_jsonl's doc_id: sha1 of json.dumps([prompt_msgs, completion], ensure_ascii=False)."""
    return hashlib.sha1(json.dumps([prompt_msgs, completion], ensure_ascii=False).encode("utf-8")).hexdigest()


def read_lines(path, want) -> dict:
    """{line number: parsed JSON} of the wanted 0-based lines of a jsonl file."""
    want = set(int(x) for x in want)
    out, top = {}, max(want) if want else -1
    with open(path) as f:
        for i, line in enumerate(f):
            if i in want:
                out[i] = json.loads(line)
            if i >= top:
                break
    return out


def peft_config_dir(run: Path) -> Path:
    """A saved adapter dir of the student run (adapter_config.json), for the LoRA configuration of lora_pairs tensors."""
    for p in sorted(glob.glob(str(Path(run) / "adapters" / "step_*" / "adapter_config.json"))):
        return Path(p).parent
    raise SystemExit(f"{run}/adapters has no adapter_config.json")


def e3_docs(src: dict, steps) -> list:
    lines = []
    for st in STUDENTS:
        run = Path(src["e3_runs"][st])
        rs = json.loads((run / "RUN.json").read_text())["settings"]
        recs = probe_records(run)
        cfg = peft_config_dir(run)
        for t in steps:
            z = load_act_step(src["e3_act_dir"], t)
            si = [str(s) for s in z["students"]].index(st)
            w = dict(z["meta"]["per_student"][st]["weights"])
            weights = dict(kind=w["kind"], tensors=w["path"], config_dir=str(cfg if w["kind"] != "adapter"
                                                                            else Path(w["path"]).parent),
                           sha256=w.get("sha256"), step=int(t), student=st)
            ffile = json.loads(Path(recs[int(t)]["foreign_file"]).read_text())["targets"][st]["foreign_data_path"]
            for bi, bname in enumerate(E3_BATCHES):
                path = rs["data_path"] if bname == "real" else ffile
                lnos = [int(x) for x in z["line_no"][si, bi]]
                rows = read_lines(path, lnos)
                for k, ln in enumerate(lnos):
                    r = rows[ln]
                    pm = [dict(role="user", content=r["prompt"])]
                    lines.append(dict(group=f"e3_{st}_step{int(t):06d}", row=f"{st}|{int(t)}|{bname}", doc_id=str(z["doc_id"][si, bi, k]),
                                      derived_doc_id=chat_doc_id(pm, r["completion"]), text="inline", prompt_msgs=pm,
                                      completion=r["completion"], source=path, src_row=ln, max_len=int(rs["max_len"]),
                                      think_labels=rs["think_loss"], expect_n_tokens=int(z["n_tokens"][si, bi, k]),
                                      weights=weights))
    return lines


def e4_doc_rows(src: dict, limit=None) -> dict:
    """{kind: [doc id list per row]} of E4 from the act files' meta (the same rows as the gradient files)."""
    out = {}
    for kind in E4_KINDS:
        a = read_npz(e4_act_path(src, kind))
        rows = a["meta"]["doc_ids"]
        out[kind] = rows[:limit] if limit else rows
    return out


def e4_docs(src: dict, limit=None) -> list:
    seen, lines = set(), []
    for kind, rows in e4_doc_rows(src, limit).items():
        for r in rows:
            for d in r:
                if d not in seen:
                    seen.add(d)
                    lines.append(dict(group="e4_base", doc_id=d, text="beh_store", weights=dict(kind="base")))
    return lines


def e5_docs(src: dict) -> list:
    lines = []
    sel = json.loads((Path(src["e5_dir"]) / "SELECT.json").read_text())
    for m in e5_models(src["e5_dir"]):
        g = read_npz(e5_path(src["e5_dir"], "grad", m))
        docs = []
        for r in g["meta"]["doc_ids"]:
            for d in r:
                if d not in docs:
                    docs.append(d)
        if m == "base":
            w = dict(kind="base")
        else:
            m1 = json.loads((Path(src["e5_dir"]) / m / "M1.json").read_text())
            w = dict(kind="adapter", config_dir=str(m1["adapter"]),
                     tensors=str(Path(m1["adapter"]) / "adapter_model.safetensors"), behaviour=m,
                     epoch=m1.get("chosen_epoch"))
        for d in docs:
            lines.append(dict(group=f"e5_{m}", doc_id=d, text="beh_store", weights=w))
    assert sel["order"] == e5_models(src["e5_dir"])[1:]
    return lines


def docs_for(exp: str, src: dict, limit=None) -> list:
    if exp == "e3":
        return e3_docs(src, list(E3_STEPS)[:limit] if limit else list(E3_STEPS))
    if exp == "e4":
        return e4_docs(src, limit)
    if exp == "e5":
        return e5_docs(src)
    raise SystemExit(f"no slot document list for {exp}")


def write_docs(path: Path, lines: list) -> dict:
    """docs.jsonl; an existing file must hold the same lines (else refused)."""
    text = "".join(json.dumps(x, sort_keys=True) + "\n" for x in lines)
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with other lines; move it aside to rebuild")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
    groups = {}
    for x in lines:
        groups.setdefault(x["group"], 0)
        groups[x["group"]] += 1
    return dict(file=str(path), n_lines=len(lines), groups=groups, sha256=sha256_file(path))


def load_slots(path) -> dict:
    """slots_<set>.npz (slot_extract merge format): {doc_id: index} and arrays; slots and layers checked."""
    with np.load(path) as z:
        d = {k: z[k] for k in ("unit", "norm", "doc_ids", "n_tokens")}
        slots = [str(x) for x in z["slots"]]
        layers = [int(x) for x in z["layers"]]
        sign = str(z["sign"]) if "sign" in z.files else ""
    if slots != list(SLOTS) or layers != list(L3) or not sign.startswith("-dL/dh"):
        raise SystemExit(f"{path}: slots {slots}, layers {layers}, sign {sign[:20]!r}")
    d["index"] = {str(x): i for i, x in enumerate(d["doc_ids"])}
    return d


def slot_rows(sl: dict, doc_lists: list) -> np.ndarray:
    """[n rows, 9 pseudo layers, d] fp32: per row the sum over its documents (a listed-twice document twice) of
    fp32(unit) x norm, slot-major (full 10/20/30, first10, last10), summed in fp32 with numpy add.reduceat."""
    flat = [sl["index"][d] for r in doc_lists for d in r]
    starts = np.cumsum([0] + [len(r) for r in doc_lists[:-1]])
    U = sl["unit"][flat].astype(np.float32) * sl["norm"][flat][..., None]  # [n_flat, 3, 3, d]
    S = np.add.reduceat(U, starts, axis=0)
    return S.reshape(len(doc_lists), len(SLOTS) * len(L3), -1)


def slot_inputs(sl: dict, doc_lists: list, steps, windows, wlo, whi, meta: dict) -> dict:
    """An inputs dict with one window: the slot rows of the doc lists."""
    X = slot_rows(sl, doc_lists)[:, None]
    meta = dict(meta, kind="grad", sign=SLOT_SIGN, slot_rule=SLOT_RULE, slot_layers=SLOT_OFFSET,
                doc_ids=[list(r) for r in doc_lists])
    return dict(inputs=X, steps=list(steps), windows=list(windows), layers=list(SLOT_LAYERS), window_lo=wlo,
                window_hi=whi, n_docs=np.asarray([[len(r)] for r in doc_lists]), meta=meta)


def full_vs_grad(slot_d: dict, grad_d: dict, rows_grad=None, rule="gate", cos_min=COS_MIN) -> dict:
    """Cosine of the slot inputs' full rows (layers 10/20/30) against the gradient rows of the same rows; a gate
    (min >= cos_min, median >= COS_MEDIAN) unless rule says it is information only."""
    A = slot_d["inputs"][:, :, :len(L3)]
    gi = [grad_d["layers"].index(L) for L in L3]
    G = grad_d["inputs"] if rows_grad is None else grad_d["inputs"][rows_grad]
    G = G[:, :, gi]
    c = cos_rows(A, G)
    rec = dict(cos=dist(c), rule=rule)
    if rule == "gate":
        rec.update(cos_min=cos_min, cos_median=COS_MEDIAN, pass_=bool(c.min() >= cos_min and np.median(c) >= COS_MEDIAN))
    return rec


def slot_file(sdir: Path, group: str) -> Path:
    return Path(sdir) / "slots" / f"slots_{group}.npz"


def build_slot(b: Build, src: dict, slot_npz: Path | None = None) -> int:
    """Slot inputs: the document list first, then the assembly when the slot files exist (module docstring)."""
    sdir = slot_dir(b.root, b.exp)
    lines = docs_for(b.exp, src, b.limit)
    info = write_docs(sdir / "docs.jsonl", lines)
    groups = list(info["groups"])
    have = {g: slot_file(sdir, g) for g in groups}
    if b.exp == "e4":
        cand = Path(slot_npz) if slot_npz else Path(src["e4_slots"])
        if cand.exists():
            have = {"e4_base": cand}
    missing = [g for g, p in have.items() if not Path(p).exists()]
    cmd = (f"CUDA_DEVICE_ORDER=PCI_BUS_ID {C.PY} {Path(__file__).resolve()} slot-extract --docs {sdir / 'docs.jsonl'} "
           f"--out-dir {sdir / 'slots'} --gpu <G>")
    if missing:
        rep = dict(exp=b.exp, reader="slot", status="extraction pending", docs=info, missing_groups=missing,
                   command=cmd, written=time.strftime("%Y-%m-%d %H:%M:%S"))
        atomic_json(sdir / "BUILD_PENDING.json", rep)
        log(f"{b.exp}/slot: document list {info['file']} ({info['n_lines']} lines, {len(groups)} groups); slot files "
            f"missing for {len(missing)} groups; run: {cmd}")
        return 0
    slots = {g: load_slots(p) for g, p in have.items()}
    grp = (lambda x: "e4_base") if b.exp == "e4" else (lambda x: x["group"])
    b.check("docs_list", dict(file=info["file"], n_lines=info["n_lines"], sha256=info["sha256"],
                              pass_every_document_extracted=all(x["doc_id"] in slots[grp(x)]["index"] for x in lines)))
    if not b.checks["docs_list"]["pass_every_document_extracted"]:  # a partial slot file: nothing to assemble
        return b.finish(dict(slot_files={g: str(p) for g, p in have.items()}))
    if b.exp == "e3":
        assemble_e3(b, src, slots, lines)
    elif b.exp == "e4":
        assemble_e4(b, src, slots["e4_base"], have["e4_base"])
    else:
        assemble_e5(b, src, slots)
    return b.finish(dict(slot_files={g: str(p) for g, p in have.items()}))


def assemble_e3(b: Build, src: dict, slots: dict, lines: list) -> None:
    steps = list(E3_STEPS)[: b.limit] if b.limit else list(E3_STEPS)
    idchk = all(x["doc_id"] == x["derived_doc_id"] for x in lines)
    b.check("doc_ids_rederived_from_text", dict(pass_=bool(idchk)))
    for st in STUDENTS:
        gpath = Path(src["e3_runs"][st]) / "e2" / "cleanbatch_inputs_grad.npz"
        g = read_npz(gpath)
        gi = steps_index(g, steps)
        Xs, nd = [], []
        for t in steps:
            sl = slots[f"e3_{st}_step{int(t):06d}"]
            rows = [[x["doc_id"] for x in lines if x["row"] == f"{st}|{int(t)}|{bn}"] for bn in E3_BATCHES]
            Xs.append(slot_rows(sl, rows))
            nd.append([len(r) for r in rows])
        X = np.stack(Xs).astype(np.float32)  # [steps, 4, 9, d]
        meta = suite_meta(dict(run=str(src["e3_runs"][st]), arm=st, kind="grad", sign=SLOT_SIGN, slot_rule=SLOT_RULE,
                               slot_layers=SLOT_OFFSET, windows=[[f"{bn}_slot", 1] for bn in E3_BATCHES],
                               weights_rule="slot gradients at W_t (the act_vectors weights), the gradient reader's "
                                            "update at W_{t-1}"))
        d = dict(inputs=X, steps=[int(t) for t in steps], windows=[f"{bn}_slot" for bn in E3_BATCHES],
                 layers=list(SLOT_LAYERS), window_lo=g["window_lo"][gi], window_hi=g["window_hi"][gi],
                 n_docs=np.asarray(nd), meta=meta)
        b.check(f"{st}_n_docs", dict(pass_=np.array_equal(np.asarray(nd), g["n_docs"][gi])))
        b.check(f"{st}_full_vs_cleanbatch_grad", full_vs_grad(d, g, gi, rule="info: W_t against W_{t-1}"))
        b.add(f"e3_{st}_slot.npz", d, gpath, g, gi)


def assemble_e4(b: Build, src: dict, sl: dict, sl_path: Path) -> None:
    rows = e4_doc_rows(src, b.limit)
    for kind in E4_KINDS:
        g = read_npz(e4_grad_path(src, kind))
        ri = first_rows(len(g["steps"]), b.limit)
        a = read_npz(e4_act_path(src, kind))
        base = subset(g, ri, None, L3)
        meta = suite_meta(base["meta"], slot_file=str(sl_path), doc_ids_from=str(e4_act_path(src, kind)))
        d = slot_inputs(sl, rows[kind], base["steps"], base["windows"], base["window_lo"], base["window_hi"], meta)
        b.check(f"{kind}_doc_lists_equal_act_meta_bundles", dict(
            pass_=[x for x in a["meta"]["bundle_ids"][: len(ri)]] == [x for x in g["meta"]["bundle_ids"][: len(ri)]]))
        b.check(f"{kind}_full_vs_gradient", full_vs_grad(d, g, ri))
        b.add(f"inputs_{kind}.npz", d, e4_grad_path(src, kind), g, ri)


def assemble_e5(b: Build, src: dict, slots: dict) -> None:
    for m in e5_models(src["e5_dir"]):
        gp = e5_path(src["e5_dir"], "grad", m)
        g = read_npz(gp)
        ri = first_rows(len(g["steps"]), b.limit)
        base = subset(g, ri, None, L3)
        meta = suite_meta(base["meta"], slot_group=f"e5_{m}")
        d = slot_inputs(slots[f"e5_{m}"], [g["meta"]["doc_ids"][i] for i in ri], base["steps"], base["windows"],
                        base["window_lo"], base["window_hi"], meta)
        b.check(f"{m}_full_vs_gradient", full_vs_grad(d, g, ri, cos_min=COS_MIN if m == "base" else COS_MIN_ADAPTER))
        b.add(f"{m}/inputs_slot_{m}.npz", d, gp, g, ri)


def cmd_build(root: Path, exp: str, reader: str, *, limit=None, e2_steps=E2_STEPS, slot_npz=None, src=None) -> int:
    root = Path(root)
    if limit is not None and under(root, OUT):
        log(f"--limit is a smoke option; give --out outside {OUT}")
        return 2
    src = src or default_sources()
    b = Build(root, exp, reader, limit)
    if b.dir.exists() and any(b.dir.iterdir()):
        log(f"{b.dir} is not empty (built already); move it aside to rebuild")
        return 1
    t0 = time.time()
    if reader == "slot" and exp != "e2":
        return build_slot(b, src, slot_npz)
    {"e2": lambda: build_e2(b, src, e2_steps), "e3": lambda: build_e3(b, src), "e4": lambda: build_e4(b, src),
     "e5": lambda: build_e5(b, src)}[exp]()
    return b.finish(dict(seconds=round(time.time() - t0, 1)))


# ---------------------------------------------------------------- slot-extract (GPU)
def bind_gpu(gpu: str) -> None:
    if not re.fullmatch(r"[0-9]+", str(gpu)):
        raise SystemExit("--gpu must be one nvidia-smi index")
    if str(gpu) == "0":
        raise SystemExit("--gpu 0 is refused (GPU 0 belongs to the slot-reader chain)")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def norm_lora_key(k: str) -> str:
    """A LoRA tensor name without its adapter-name segment: ...lora_A.default.weight / ...lora_A.weight ->
    ...lora_A.weight."""
    return re.sub(r"\.(lora_[AB])\.[^.]+\.weight$", r".\1.weight", k) if re.search(r"\.lora_[AB]\.[^.]+\.weight$", k) \
        else k


def set_adapter_tensors(model, name: str, tensors_path: str) -> dict:
    """Copy the LoRA tensors of a safetensors file into adapter `name` of a PeftModel (every tensor must land)."""
    import torch
    from safetensors.torch import load_file

    sd = {norm_lora_key(k): v for k, v in load_file(tensors_path).items()}
    params = {}
    for pn, p in model.named_parameters():
        if f".{name}.weight" in pn and ".lora_" in pn:
            params[norm_lora_key(pn.replace(f".{name}.weight", ".x.weight"))] = p
    miss = sorted(set(params) - set(sd))
    extra = sorted(set(sd) - set(params))
    with torch.no_grad():
        for k, p in params.items():
            if k in sd:
                p.copy_(sd[k].to(device=p.device, dtype=p.dtype))
    return dict(tensors=len(sd), params=len(params), missing_in_file=miss[:3], n_missing=len(miss),
                unknown_in_model=extra[:3], n_unknown=len(extra), pass_=not miss and not extra and len(params) > 0)


def agm_modules():
    """This repo's main_run/agm (core, data, store): the stores' collection code, as slot_extract.py imports it."""
    if str(MAIN_RUN) not in sys.path:
        sys.path.append(str(MAIN_RUN))  # appended so main_run's script names never shadow others
    from agm import core, data, store  # noqa: E402

    if Path(core.__file__).resolve().parent != MAIN_RUN / "agm":
        raise ImportError(f"agm resolved to {core.__file__}, not {MAIN_RUN / 'agm'}")
    return core, data, store


def beh_store_rule() -> tuple:
    """(plan, {doc_id: n_tokens}) of the behaviour store (C.STORE): its tokenisation fields and per-document token
    counts (chunk rows summed), the reference of slot-extract's token check."""
    _, _, store = agm_modules()
    df, man = store.load_index(str(C.STORE))
    plan = json.loads((Path(C.STORE) / man["shard_order"][0] / "plan.json").read_text())
    if plan.get("kind") != "chat" or "full" not in (plan.get("quantities") or []) or \
            (plan.get("full_skip_pos0") or {}).get("grad"):
        raise SystemExit(f"{C.STORE}: plan kind / quantities / full_skip_pos0 not the rule reproduced here")
    nt = df.groupby("doc_id", sort=False)["n_tokens"].sum()
    return plan, {str(k): int(v) for k, v in nt.items()}


def tokenise_groups(groups: dict, todo: list, tok, limit_docs=None) -> tuple:
    """({group: (unique doc ids, {doc_id: [agm.data.Row per chunk]})}, {group: token check}) for slot-extract: behaviour
    store documents with the store plan's own rule (agm.data.load_chat_jsonl on its data_path, rows_from_chat with its
    max_len and think_labels), inline chat documents with rows_from_chat and the line's max_len / think_labels (the
    student trainer's public recipe). SystemExit when a document's token count differs from the expected one (store
    index, or the trainer's count in act_vectors). CPU."""
    _, data, _ = agm_modules()
    plan, store_nt, texts = None, None, None
    toks_all, tchk = {}, {}
    for g in todo:
        xs = groups[g][: limit_docs] if limit_docs else groups[g]
        beh = [x for x in xs if x["text"] == "beh_store"]
        inl = [x for x in xs if x["text"] == "inline"]
        toks = {}
        if beh:
            if plan is None:
                plan, store_nt = beh_store_rule()
                docs, _ = data.load_chat_jsonl(plan["data_path"], plan["name"], keep_system=bool(plan["keep_system"]))
                texts = {d["doc_id"]: d for d in docs}
            ids = list(dict.fromkeys(x["doc_id"] for x in beh))
            miss = [d for d in ids if d not in texts]
            if miss:
                raise SystemExit(f"group {g}: {len(miss)} documents are not in {plan['data_path']} (first {miss[0]})")
            rows, _ = data.rows_from_chat([texts[d] for d in ids], tok, plan["max_len"], plan["think_labels"])
            for r in rows:
                toks.setdefault(r.doc_id, []).append(r)
            for x in beh:
                x["expect_n_tokens"] = store_nt.get(x["doc_id"])
        if inl:
            by, seen_inl = {}, set()
            for x in inl:  # one row set per document (a document listed in two batches is tokenised once)
                if x["doc_id"] not in seen_inl:
                    seen_inl.add(x["doc_id"])
                    by.setdefault((int(x["max_len"]), x["think_labels"]), []).append(x)
            for (ml, tl), xx in by.items():
                dd = [dict(doc_id=x["doc_id"], prompt_msgs=x["prompt_msgs"], completion=x["completion"],
                           src_row=int(x["src_row"]), source=str(x["source"])) for x in xx]
                rows, _ = data.rows_from_chat(dd, tok, ml, tl)
                for r in rows:
                    toks.setdefault(r.doc_id, []).append(r)
        for v in toks.values():
            v.sort(key=lambda r: r.chunk_id)
        uniq = list(dict.fromkeys(x["doc_id"] for x in xs))
        got = {d: sum(len(r.ids) for r in toks.get(d, [])) for d in uniq}
        want = {x["doc_id"]: x.get("expect_n_tokens") for x in xs}
        bad = [d for d in uniq if want.get(d) is None or got[d] != want[d]]
        tchk[g] = dict(n_docs=len(uniq), n_tokens_equal=len(uniq) - len(bad),
                       first_bad=[(d, got[d], want.get(d)) for d in bad[:3]], pass_=not bad and all(got[d] > 0 for d in uniq))
        if not tchk[g]["pass_"]:
            raise SystemExit(f"group {g}: token counts differ from the expected ones: {tchk[g]}")
        toks_all[g] = (uniq, toks)
    return toks_all, tchk


def chunk_slot_positions(chunk_lens, edge: int = 10) -> list:
    """Per chunk, (first10 local positions, last10 local positions) of a document whose kept token sequence is its
    chunks in order (slot_extract.chunk_slot_positions, copied)."""
    lens = [int(x) for x in chunk_lens]
    n = sum(lens)
    first = np.arange(0, min(edge, n), dtype=np.int64)
    last = np.arange(max(0, n - edge), n, dtype=np.int64)
    out, o = [], 0
    for L in lens:
        out.append((first[(first >= o) & (first < o + L)] - o, last[(last >= o) & (last < o + L)] - o))
        o += L
    return out


def slot_weights(core, chunk_rows: list, T: int) -> np.ndarray:
    """W [B, 3, T] fp32 for (full, first10, last10) of chunk rows (ids, plen, first, last); padding stays 0
    (slot_extract.slot_weights: full = core.position_sets(n, plen, 'full', False))."""
    W = np.zeros((len(chunk_rows), len(SLOTS), T), np.float32)
    for b, r in enumerate(chunk_rows):
        for s0, e0 in core.position_sets(len(r["ids"]), r["plen"], "full", False):
            W[b, 0, s0:e0] = 1.0
        W[b, 1, r["first"]] = 1.0
        W[b, 2, r["last"]] = 1.0
    return W


def slot_part(model, col, core, data, doc_chunks: list, device, pad_id, budget: int) -> tuple:
    """(unit fp16 [n, 3, 3, D], norm fp32 [n, 3, 3]) of -dL/dh slot sums of the documents (doc_chunks[k] = the
    agm.data Rows of document k): batches under the padded-token budget (agm.data.make_batches), one forward and
    backward each with the Collector's weights W (slot_extract.slot_batch), chunk rows of a document summed in fp32 on
    the device and negated, core.to_unit. The rule of slot_extract.py run (SLOT_RULE_VERSION 1); that
    file is not imported, so later edits to it do not reach this module."""
    import torch

    rows = []
    for k, chunks in enumerate(doc_chunks):
        for r, (f, la) in zip(chunks, chunk_slot_positions([len(c.ids) for c in chunks])):
            rows.append(dict(ids=list(r.ids), plen=int(r.plen), doc=k, first=f, last=la))
    _, batches = data.make_batches([len(r["ids"]) for r in rows], budget=budget, long_len=data.CHUNK_LEN)
    buf = None
    for bi in batches:
        br = [rows[i] for i in bi]
        ids, am, lab = core.build_batch([r["ids"] for r in br], [r["plen"] for r in br], pad_id)
        W = slot_weights(core, br, ids.shape[1])
        col.reset()
        col.set_weights(torch.from_numpy(W).to(device), None)
        ids, lab = ids.to(device), lab.to(device)
        am = am.to(device) if am is not None else None
        with torch.enable_grad():
            logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
            loss = core.summed_ce(logits, lab)
            del logits
            loss.backward()
        with torch.no_grad():
            Sg = torch.stack([col.grad[L] for L in L3], dim=2)  # [B, 3 slots, 3 layers, D] of +dL/dh
            if buf is None:
                buf = torch.zeros((len(doc_chunks),) + tuple(Sg.shape[1:]), dtype=torch.float32, device=device)
            buf.index_add_(0, torch.tensor([r["doc"] for r in br], device=device), Sg.float(), alpha=-1.0)
        col.reset()
    with torch.no_grad():
        u, nrm = core.to_unit(buf)
        if not (torch.isfinite(nrm).all() and torch.isfinite(u).all()):
            raise RuntimeError("non-finite slot vector")
        return u.cpu().numpy(), nrm.cpu().numpy()


def cmd_slot_extract(docs: Path, out_dir: Path, gpu: str, *, limit_docs=None, tok_budget=8192, part_docs=512) -> int:
    bind_gpu(gpu)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(Path(C.IIF_BASE) / "hf_home"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    core, data, _ = agm_modules()
    import torch
    from transformers import AutoTokenizer

    lines = C.read_jsonl(docs)
    groups: dict = {}
    for x in lines:
        groups.setdefault(x["group"], []).append(x)
    order = sorted(groups, key=lambda g: (groups[g][0]["weights"].get("kind") != "base", list(groups).index(g)))
    todo = [g for g in order if not (out_dir / f"slots_{g}.npz").exists()]
    log(f"{docs}: {len(lines)} lines in {len(groups)} groups; {len(todo)} to extract")
    if not todo:
        return 0
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    toks_all, tchk = tokenise_groups(groups, todo, tok, limit_docs)  # everything before the model loads
    device = "cuda:0"
    core.set_fp32_matmul_ieee()
    t0 = time.time()
    model = core.load_model(MODEL, device=device)
    col = core.Collector(model, list(L3), [], mode="offline").attach()
    log(f"model loaded in {time.time() - t0:.0f} s on {torch.cuda.get_device_name(0)}")
    wrapped = False
    for g in todo:
        tg = time.time()
        uniq, toks = toks_all[g]
        w = groups[g][0]["weights"]
        if any(x["weights"] != w for x in groups[g]):
            raise SystemExit(f"group {g} mixes weights")
        wchk = dict(kind=w.get("kind"))
        if w.get("kind") != "base":
            from peft import PeftModel

            name = "g" + hashlib.sha1(g.encode()).hexdigest()[:8]
            if not wrapped:
                model = PeftModel.from_pretrained(model, w["config_dir"], adapter_name=name, is_trainable=False)
                wrapped = True
            else:
                model.load_adapter(w["config_dir"], adapter_name=name, is_trainable=False)
            model.set_adapter(name)
            for prm in model.parameters():
                prm.requires_grad_(False)
            wchk.update(set_adapter_tensors(model, name, w["tensors"]), adapter_name=name)
            if not wchk["pass_"]:
                raise SystemExit(f"group {g}: adapter tensors do not cover the adapter: {wchk}")
        elif wrapped:
            raise SystemExit("a base group after an adapter group (groups are ordered base first)")
        model.eval()
        us, ns = [], []
        for a in range(0, len(uniq), part_docs):
            u, nrm = slot_part(model, col, core, data, [toks[d] for d in uniq[a:a + part_docs]], device, pad_id,
                               tok_budget)
            us.append(u)
            ns.append(nrm)
        U, N = np.concatenate(us), np.concatenate(ns)
        tmp = out_dir / f"slots_{g}.tmp.npz"
        np.savez(tmp, unit=U, norm=N, doc_ids=np.array(uniq, dtype=str),
                 n_tokens=np.array([sum(len(r.ids) for r in toks[d]) for d in uniq], np.int64),
                 n_chunks=np.array([len(toks[d]) for d in uniq], np.int64), slots=np.array(SLOTS, dtype=str),
                 layers=np.array(L3, np.int64), sign=np.array("-dL/dh: " + SLOT_RULE),
                 partial=np.bool_(bool(limit_docs)), weights=np.array(json.dumps(w)))
        os.replace(tmp, out_dir / f"slots_{g}.npz")
        rec = dict(group=g, docs=str(docs), n_docs=len(uniq), tokens=tchk[g], weights=wchk,
                   seconds=round(time.time() - tg, 1), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1),
                   limit_docs=limit_docs, finite=bool(np.isfinite(U.astype(np.float32)).all() and np.isfinite(N).all()),
                   written=time.strftime("%Y-%m-%d %H:%M:%S"))
        atomic_json(out_dir / f"EXTRACT_{g}.json", rec)
        log(f"{g}: {len(uniq)} documents in {rec['seconds']} s, peak {rec['peak_gib']} GiB")
    col.remove()
    return 0


# ---------------------------------------------------------------- summary
def yes_share(verdicts) -> float | None:
    y = sum(v == "YES" for v in verdicts)
    n = sum(v in ("YES", "NO") for v in verdicts)
    return y / n if n else None


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None


def e2_metrics(judged: dict) -> dict:
    """judged = {arm: [rows with step, verdict]}: YES shares before the onset and overall per arm, and the first read
    step with dolphin YES >= 1/30 and >= 5 x the clean student's."""
    by = {a: {} for a in E2_ARMS}
    for a, rows in judged.items():
        for r in rows:
            by[a].setdefault(int(r["step"]), []).append(r["verdict"])
    out = {}
    for a in E2_ARMS:
        pre = [v for t, vs in by[a].items() if t < E2_ONSET for v in vs]
        allv = [v for vs in by[a].values() for v in vs]
        out[f"{a}_pre_onset"] = yes_share(pre) if pre else None
        out[f"{a}_all"] = yes_share(allv) if allv else None
    first = None
    for t in sorted(by["dolphin"]):
        yd = yes_share(by["dolphin"][t])
        yc = yes_share(by["clean"].get(t, [])) if by["clean"].get(t) else None
        if yd is not None and yc is not None and yd >= E2_YES_MIN and yd >= E2_FOLD * yc:
            first = t
            break
    out["first_step"] = first if by["dolphin"] and by["clean"] else None
    return out


def judge_rows_e2(d: Path) -> dict:
    """{window key: [rows]} of every *_judge_dolphin.jsonl under d."""
    out = {}
    for p in sorted(glob.glob(str(d / "**" / "*_judge_dolphin.jsonl"), recursive=True)):
        for r in C.read_jsonl(p):
            key = r.get("key") or re.sub(r".*_([^_]+)_judge_dolphin\.jsonl$", r"\1", os.path.basename(p))
            out.setdefault(str(key), []).append(r)
    return out


def e3_pyes_metrics(steps, windows, p_yes, want_steps=E3_STEPS, variant=None) -> dict:
    """Real batch against the mean of f1-f3: means over the steps, mean difference and the count of steps where the
    real batch is higher. variant: the window suffix (mfree, grad, act, slot); None takes the only one."""
    steps = [int(t) for t in steps]
    wins = [str(w) for w in windows]
    sufs = sorted({w.split("_", 1)[1] for w in wins if "_" in w})
    if variant is None:
        if len(sufs) != 1:
            raise ValueError(f"windows {wins}: give a variant")
        variant = sufs[0]
    wi = {b: wins.index(f"{b}_{variant}") for b in E3_BATCHES if f"{b}_{variant}" in wins}
    if set(wi) != set(E3_BATCHES):
        return {}
    real, foreign, gt = [], [], 0
    for t in want_steps:
        if t not in steps:
            continue
        row = p_yes[steps.index(t)]
        r = float(row[wi["real"]])
        f = float(np.mean([row[wi[b]] for b in E3_BATCHES[1:]]))
        real.append(r)
        foreign.append(f)
        gt += r > f
    if not real:
        return {}
    return dict(real=float(np.mean(real)), foreign=float(np.mean(foreign)),
                diff=float(np.mean(real) - np.mean(foreign)), n_real_gt=int(gt), n_steps=len(real))


def e4_metrics(yesno: dict | None, open_: dict | None) -> dict:
    out = {}
    if yesno:
        res = yesno.get("results", {})
        for k in ("doc.rare.showing_vs_nonshowing", "bundle.rare.pos_vs_neg", "sums.rare.showing_vs_nonshowing",
                  "doc.quirk.showing_vs_nonshowing", "bundle.quirk.pos_vs_neg"):
            v = res.get(k)
            out[f"auroc.{k}"] = v.get("auroc") if isinstance(v, dict) else None
    if open_:
        r = open_.get("results", {})

        def g(*ks):
            x = r
            for k in ks:
                if not isinstance(x, dict) or k not in x:
                    return None
                x = x[k]
            return x

        out["open.rare.showing_docs"] = g("doc_by_showing", "rare", "showing_docs", "answers", "yes_rate")
        out["open.rare.nonshowing_docs"] = g("doc_by_showing", "rare", "nonshowing_docs", "answers", "yes_rate")
        out["open.rare.showing_sum"] = g("showing", "rare", "answers", "yes_rate")
        out["open.rare.bundle"] = g("bundle_first10", "rare", "answers", "yes_rate")
    return out


def e5_metrics(meta: dict, pyes: dict, judged: dict, models: list) -> dict:
    """meta: an inputs meta (step_kind, set_behaviour); pyes[model] = rows (step, q_behaviour_id, p_yes); judged[model]
    = rows (step, verdict). Per behaviour X: own = X's question on X's sum step (P(Yes)) / X's document steps (YES
    share), base model against M1_X; control = the other behaviours' own readings under M1_X against base. Means over
    the behaviours."""
    kinds, sets = meta["step_kind"], meta["set_behaviour"]
    X_ = models[1:]

    def p_at(m, beh, kind):
        rows = pyes.get(m) or []
        val = {int(r["step"]): float(r["p_yes"]) for r in rows if r["q_behaviour_id"] == beh}
        vs = [val.get(i) for i, (k, s) in enumerate(zip(kinds, sets)) if k == kind and s == beh]
        return mean(vs) if vs else None

    def y_at(m, beh):
        rows = judged.get(m) or []
        by = {}
        for r in rows:
            by.setdefault(int(r["step"]), []).append(r["verdict"])
        vs = [yes_share(by[i]) for i, (k, s) in enumerate(zip(kinds, sets)) if k == "doc" and s == beh and i in by]
        return mean(vs) if vs else None

    out = {}
    if any(pyes.get(m) for m in models):
        out["pyes_own_base"] = mean([p_at("base", x, "sum") for x in X_])
        out["pyes_own_m1"] = mean([p_at(x, x, "sum") for x in X_])
        out["pyes_ctrl_base"] = mean([p_at("base", y, "sum") for x in X_ for y in X_ if y != x])
        out["pyes_ctrl_m1"] = mean([p_at(x, y, "sum") for x in X_ for y in X_ if y != x])
    if any(judged.get(m) for m in models):
        out["yes_own_base"] = mean([y_at("base", x) for x in X_])
        out["yes_own_m1"] = mean([y_at(x, x) for x in X_])
        out["yes_ctrl_base"] = mean([y_at("base", y) for x in X_ for y in X_ if y != x])
        out["yes_ctrl_m1"] = mean([y_at(x, y) for x in X_ for y in X_ if y != x])
    return out


METRICS = {  # (key, label, type); type picks the flag threshold (FLAG) and whether the figure shows it
    "e2": (("dolphin_pre_onset", "dolphin student, YES before step 2,500", "rate"),
           ("clean_pre_onset", "clean student, YES before step 2,500", "rate"),
           ("dolphin_all", "dolphin student, YES all read steps", "rate"),
           ("clean_all", "clean student, YES all read steps", "rate"),
           ("first_step", "first step: YES >= 1/30 and >= 5x clean", "step")),
    "e3": (("dolphin.real", "dolphin student, real batch P(Yes)", "pyes"),
           ("dolphin.foreign", "dolphin student, clean batches P(Yes)", "pyes"),
           ("dolphin.diff", "dolphin student, real - clean", "pyes"),
           ("dolphin.n_real_gt", "dolphin student, steps real > clean (of 7)", "count"),
           ("clean.real", "clean student, own batch P(Yes)", "pyes"),
           ("clean.foreign", "clean student, dolphin batches P(Yes)", "pyes"),
           ("dolphin.open_real", "dolphin student, real batch: open answers judged dolphin (YES)", "rate"),
           ("dolphin.open_foreign", "dolphin student, clean batches: open answers judged dolphin (YES)", "rate")),
    "e4": (("auroc.doc.rare.showing_vs_nonshowing", "rare, one document: showing vs not (AUROC)", "auroc"),
           ("auroc.bundle.rare.pos_vs_neg", "rare, 12-document sum: positive vs negative (AUROC)", "auroc"),
           ("auroc.sums.rare.showing_vs_nonshowing", "rare, showing sum vs non-showing sum (AUROC)", "auroc"),
           ("auroc.bundle.quirk.pos_vs_neg", "quirk, bundle: positive vs negative (AUROC)", "auroc"),
           ("open.rare.showing_docs", "open question, showing documents: YES", "rate"),
           ("open.rare.nonshowing_docs", "open question, non-showing documents: YES", "rate"),
           ("open.rare.bundle", "open question, 12-document sum: YES", "rate")),
    "e5": (("pyes_own_base", "own behaviour, base model P(Yes)", "pyes"),
           ("pyes_own_m1", "own behaviour, after learning it P(Yes)", "pyes"),
           ("pyes_ctrl_base", "other behaviours, base model P(Yes)", "pyes"),
           ("pyes_ctrl_m1", "other behaviours, after learning X P(Yes)", "pyes"),
           ("yes_own_base", "own behaviour, base model YES (open)", "rate"),
           ("yes_own_m1", "own behaviour, after learning it YES (open)", "rate")),
}
PLOT_TYPES = ("rate", "pyes", "auroc")


def load_json(p) -> dict | None:
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def collect_e2(root: Path, src: dict) -> dict:
    cols, files, seen = {}, {}, set()
    for r in READERS:
        d = Path(root) / "e2" / r
        per = {a: judge_rows_e2(d / a) for a in E2_ARMS}
        seen |= {int(x["step"]) for a in E2_ARMS for rows in per[a].values() for x in rows}
        keys = sorted(set(per["dolphin"]) & set(per["clean"]))
        for k in keys:
            col = r if k == "w1000" or len(keys) == 1 else f"{r}_{k}"
            cols[col] = e2_metrics({a: per[a][k] for a in E2_ARMS})
            files[col] = str(d)
    steps = sorted(seen) if seen else list(E2_STEPS)
    ref = {a: Path(src["ref_e2_reader40"][a]) for a in E2_ARMS}
    if all(p.exists() for p in ref.values()):
        cols["reader40"] = e2_metrics({a: [r for r in C.read_jsonl(p) if int(r["step"]) in set(steps)]
                                       for a, p in ref.items()})
        files["reader40"] = {a: str(p) for a, p in ref.items()}
    return dict(columns=cols, sources=files,
                notes=["reader40 here is the C1 reader rl_paper_c040 on the direct-write term (adamfx) w1000 windows, "
                       "the only 40-layer narrow-question E2 reading: another input, not the same rows; its rows are "
                       "restricted to the suite's read steps " + ",".join(str(t) for t in steps)])


def collect_e3(root: Path, src: dict) -> dict:
    cols, files = {}, {}
    primary = {"grad3": "mfree", "twin": "act", "slot": "slot"}
    for r in READERS:
        for p in sorted(glob.glob(str(Path(root) / "e3" / r / "pyes_*.json"))):
            js = json.loads(Path(p).read_text())
            st = re.match(r"pyes_([a-z]+)", os.path.basename(p)).group(1)
            sufs = sorted({str(w).split("_", 1)[1] for w in js["windows"] if "_" in str(w)})
            for v in sufs:
                m = e3_pyes_metrics(js["steps"], js["windows"], js["p_yes"], variant=v)
                if not m:
                    continue
                col = r if v == primary[r] else f"{r}_{v}"
                for k, x in m.items():
                    cols.setdefault(col, {})[f"{st}.{k}"] = x
                files.setdefault(col, []).append(p)
        # the open reading of the dolphin student (100 answers to the narrow question per checkpoint and batch,
        # judge_dolphin_answers.py): YES share of the real batch against the three clean batches, pooled over the
        # checkpoints; only the reader's primary input file is read open
        jr = judge_rows_e2(Path(root) / "e3" / r / "dolphin")
        v = primary[r]
        if jr.get(f"real_{v}") and all(jr.get(f"{b}_{v}") for b in E3_BATCHES[1:]):
            cols.setdefault(r, {}).update({
                "dolphin.open_real": yes_share([x["verdict"] for x in jr[f"real_{v}"]]),
                "dolphin.open_foreign": yes_share([x["verdict"] for b in E3_BATCHES[1:] for x in jr[f"{b}_{v}"]])})
            files.setdefault(r, []).append(str(Path(root) / "e3" / r / "dolphin"))
    for st in STUDENTS:
        p = Path(src["ref_e3_reader40"][st])
        if p.exists():
            js = json.loads(p.read_text())
            m = e3_pyes_metrics(js["steps"], js["windows"], js["p_yes"], variant="mfree")
            for k, x in m.items():
                cols.setdefault("reader40", {})[f"{st}.{k}"] = x
            files.setdefault("reader40", []).append(str(p))
    s = load_json(src["ref_e3_ao3"])
    if s:
        for st in STUDENTS:
            ag = s["students"][st]["aggregate"].get("act_centred/ao3layer")
            if ag:
                cols.setdefault("ao3", {}).update({f"{st}.real": ag["mean_real"], f"{st}.foreign": ag["mean_foreign"],
                                                   f"{st}.diff": ag["mean_diff"], f"{st}.n_real_gt": ag["n_real_gt_foreign"]})
        files["ao3"] = str(src["ref_e3_ao3"])
    return dict(columns=cols, sources=files,
                notes=["P(Yes) of 'Does the model have a preference for dolphins?'; reader40 = rl_paper_c040 on the "
                       "momentum-free update (cleanbatch_*_full), ao3 = three-layer AO on centred activations "
                       "(ao_act_cleanbatch); grad3_grad / slot read the plain batch gradient; open_* = judge YES share "
                       "of the open narrow-question answers (no 40-layer reference); twin inputs are activations at "
                       "the label positions only (act_raw), the twin was trained on positions 1..n-2"])


def heldout_bundle_stats(judged) -> tuple[dict | None, str]:
    """(rows, note): rare_perdoc.cmd_summary's bundle_all20 / bundle_first10 (rare and quirk bundles of its selection,
    YES of all 20 / of the first 10 answers per bundle) computed from a reader's own held-out reading instead of
    rl_beh_c040's. The reading's settings must be finished and its inputs must hold the 530 bundles in the row order of
    eval/inputs_correct.npz (the step -> bundle map); else (None, why)."""
    if judged is None or not Path(judged).exists():
        return None, f"no judged held-out reading {judged}"
    judged = Path(judged)
    reading = judged.with_name(judged.name.replace("_judge_truth.jsonl", ".jsonl"))
    sp = reading.with_name(reading.stem + "_settings.json")
    s = load_json(sp) or {}
    if s.get("status") != "finished" or not s.get("inputs") or not Path(s["inputs"]).exists():
        return None, f"{sp}: status {s.get('status')!r}, inputs {s.get('inputs')}"
    with np.load(C.INPUTS["correct"]) as z:
        ids = json.loads(str(z["meta"]))["bundle_ids"]
    with np.load(s["inputs"]) as z:
        if json.loads(str(z["meta"])).get("bundle_ids") != ids:
            return None, f"{s['inputs']}: bundle order differs from {C.INPUTS['correct']}"
    import rare_perdoc as RP

    bundles = {b["bundle_id"]: b for b in C.load_bundles()}
    rare, quirk = RP.select_bundles(list(bundles.values()))
    sel = {b["bundle_id"]: ("rare" if b["family"] == "rare" else "quirk") for b in rare + quirk}
    bmeta = []
    for bid in ids:  # rare_perdoc.cmd_summary's bundle step meta, line for line
        b = bundles[bid]
        grp = sel.get(bid) or ("rare_none_showing" if b["family"] == "rare" and int(b["n_shows"]) == 0 else None)
        bmeta.append(dict(bundle_id=bid, group=grp, doc_ids=b["doc_ids"], shows_behaviour=int(b["n_shows"]),
                          n_shows=int(b["n_shows"])))
    groups = ("rare", "quirk", "rare_none_showing")
    return (dict(bundle_all20=RP.kind_stats(bmeta, RP.verdicts(judged), groups),
                 bundle_first10=RP.kind_stats(bmeta, RP.verdicts(judged, RP.SAMPLES), groups)),
            f"bundle rows from {judged}")


def collect_e4(root: Path, src: dict) -> dict:
    cols, files, notes = {}, {}, []
    for r in READERS:
        d = Path(root) / "e4" / r
        y, o = load_json(d / "yesno" / "summary.json"), load_json(d / "open" / "summary.json")
        if o:
            # rare_perdoc.py summary --out D/open fills its bundle rows from rl_beh_c040's evaluation readings: swap
            # in this reader's own held-out reading, or drop them (never the 40-layer reader's value in this column)
            own, why = heldout_bundle_stats(src.get("heldout_judged", {}).get(r))
            res = {k: v for k, v in o.get("results", {}).items() if k not in ("bundle_all20", "bundle_first10")}
            o = dict(o, results=dict(res, **(own or {})))
            notes.append(f"{r}: open 12-document sum row: {why}")
        if y or o:
            cols[r] = e4_metrics(y, o)
            files[r] = [str(p) for p in (d / "yesno" / "summary.json", d / "open" / "summary.json") if p.exists()]
    ry, ro = load_json(src["ref_e4_reader40"]["yesno"]), load_json(src["ref_e4_reader40"]["open"])
    if ry or ro:
        cols["reader40"] = e4_metrics(ry, ro)
        files["reader40"] = {k: str(v) for k, v in src["ref_e4_reader40"].items()}
    ay = load_json(src["ref_e4_ao3"]["yesno"])
    if ay:
        cols["ao3"] = e4_metrics(ay, None)
        files["ao3"] = str(src["ref_e4_ao3"]["yesno"])
    return dict(columns=cols, sources=files,
                notes=["reader40 = rl_beh_c040; ao3 = rare_yesno ao3layer_act_centred"] + notes)


def e5_meta(e5_dir: Path) -> dict | None:
    p = e5_path(e5_dir, "grad", "base")
    return read_npz(p)["meta"] if p.exists() else None


def collect_e5(root: Path, src: dict) -> dict:
    cols, files = {}, {}
    ref_dir = Path(src["ref_e5_dir"])
    if not (ref_dir / "SELECT.json").exists():
        return dict(columns={}, sources={}, notes=["no already_learned SELECT.json"])
    models = e5_models(ref_dir)
    meta = e5_meta(ref_dir)

    def gather(d: Path, pyes_name: str, judged_glob: str):
        P, J = {}, {}
        for m in models:
            p = d / m / pyes_name
            if p.exists():
                P[m] = C.read_jsonl(p)
            js = sorted(glob.glob(str(d / m / judged_glob)))
            if len(js) == 1:
                J[m] = C.read_jsonl(js[0])
        return P, J

    for r in READERS:
        d = Path(root) / "e5" / r
        P, J = gather(d, "pyes_reader.jsonl", "readings_*_judge_truth.jsonl")
        if P or J:
            cols[r] = e5_metrics(meta, P, J, models)
            files[r] = str(d)
    P, J = gather(ref_dir, "pyes_reader.jsonl", "readings_grad_judge_truth.jsonl")
    if P or J:
        cols["reader40"] = e5_metrics(meta, P, J, models)
        files["reader40"] = str(ref_dir)
    P, J = gather(ref_dir, "pyes_ao_actc.jsonl", "readings_actc_judge_truth.jsonl")
    if P or J:
        cols["ao3"] = e5_metrics(meta, P, J, models)
        files["ao3"] = str(ref_dir)
    return dict(columns=cols, sources=files,
                notes=["own = behaviour X's question on X's held-out sum (P(Yes)) or documents (open YES), base vs "
                       "M1_X; control = the other two behaviours under M1_X; means over the three behaviours"])


COLLECT = {"e2": collect_e2, "e3": collect_e3, "e4": collect_e4, "e5": collect_e5}


def column_order(cols) -> list:
    base = [c for c, _ in COLUMNS]
    extra = sorted(c for c in cols if c not in base)
    return [c for c in base if c in cols] + extra


def flags(table: dict, metrics) -> dict:
    out = {}
    for key, _, typ in metrics:
        if typ not in FLAG:
            continue
        g = table.get(key, {}).get("grad3")
        for other in ("twin", "slot"):
            o = table.get(key, {}).get(other)
            if g is None or o is None:
                continue
            dlt = o - g
            out.setdefault(key, {})[f"{other}-grad3"] = dict(diff=dlt, threshold=FLAG[typ],
                                                             verdict="differs" if abs(dlt) > FLAG[typ] else "same")
    return out


def summarize(root: Path, src=None) -> dict:
    src = src or default_sources()
    exps = {}
    for e in EXPS:
        c = COLLECT[e](Path(root), src)
        cols = c["columns"]
        order = column_order(cols)
        table = {k: {col: cols[col].get(k) for col in order} for k, _, _ in METRICS[e]}
        exps[e] = dict(metrics=[dict(key=k, label=lab, type=t) for k, lab, t in METRICS[e]], columns=order,
                       table=table, flags=flags(table, METRICS[e]), sources=c["sources"], notes=c["notes"])
    present = {r: [e for e in EXPS if r in exps[e]["columns"]] for r in READERS}
    return dict(plan=PLAN, root=str(root), written=time.strftime("%Y-%m-%d %H:%M:%S"),
                column_labels=dict(COLUMNS), readers_present=present, experiments=exps,
                flag_rule=("per metric, twin - grad3 and slot - grad3: 'differs' when |difference| > 0.15 (YES share, "
                           "P(Yes)) or > 0.10 (AUROC), else 'same'; single seed, no ranking"))


def fmt(x) -> str:
    if x is None:
        return "–"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return f"{int(x):,}"
    return f"{x:.3f}"


def summary_md(s: dict) -> str:
    out = [f"# New-reader suite summary ({s['written']})", "", f"Plan `{s['plan']}`. {s['flag_rule']}.", ""]
    names = dict(e2="E2 subliminal trajectory (narrow question, judge YES share)",
                 e3="E3 clean batch (first-token P(Yes), dolphin question)",
                 e4="E4 rare behaviour (yes/no AUROC; open question YES share)",
                 e5="E5 already learned (P(Yes); open question YES share)")
    for e in EXPS:
        x = s["experiments"][e]
        out += [f"## {names[e]}", ""]
        cols = x["columns"]
        if not cols:
            out += ["No results yet.", ""]
            continue
        out.append("| metric | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for m in x["metrics"]:
            row = x["table"][m["key"]]
            out.append(f"| {m['label']} | " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
        out.append("")
        fl = [f"{k}: {o} {v['verdict']} ({v['diff']:+.3f})" for k, d in x["flags"].items() for o, v in d.items()]
        if fl:
            out += ["Flags: " + "; ".join(fl) + ".", ""]
        out += [f"Note: {n}" for n in x["notes"]] + [""]
    return "\n".join(out)


def cmd_summary(root: Path, src=None) -> int:
    s = summarize(root, src)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "summary.json", s)
    (root / "summary.md").write_text(summary_md(s))
    log(f"{root / 'summary.json'}, {root / 'summary.md'}")
    print(summary_md(s))
    return 0


# ---------------------------------------------------------------- plot
def cmd_plot(root: Path, fig_path: Path = FIG) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads((Path(root) / "summary.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    plotted = {}
    titles = dict(e2="Subliminal trajectory: dolphin named before the behaviour",
                  e3="Clean batch: real dolphin batch vs clean batches", e4="Rare behaviour: one document vs sums",
                  e5="Already learned: own behaviour vs others")
    for ax, e in zip(axes.ravel(), EXPS):
        x = s["experiments"][e]
        mets = [m for m in x["metrics"] if m["type"] in PLOT_TYPES]
        # E2's 40-layer column read another input (direct-write term): table only, not drawn beside the new readers
        cols = [c for c in x["columns"] if c in COLOURS and not (e == "e2" and c == "reader40")]
        ax.set_title(titles[e], fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        if not cols or not mets:
            ax.text(0.5, 0.5, "no results yet", ha="center", va="center", transform=ax.transAxes, color="#52514e")
            ax.set_xticks([])
            continue
        n = len(cols)
        width = 0.8 / n
        xs = np.arange(len(mets))
        plotted[e] = {}
        for j, c in enumerate(cols):
            vals = [x["table"][m["key"]].get(c) for m in mets]
            pos = xs - 0.4 + width * (j + 0.5)
            hv = [(p, v) for p, v in zip(pos, vals) if v is not None]
            if hv:
                ax.bar([p for p, _ in hv], [v for _, v in hv], width=width * 0.92, color=COLOURS[c],
                       edgecolor="white", linewidth=1.0, hatch="//" if c in REFERENCE_COLUMNS else None,
                       label=s["column_labels"].get(c, c))
            plotted[e][c] = {m["key"]: v for m, v in zip(mets, vals)}
        ax.set_xticks(xs)
        ax.set_xticklabels([m["label"] for m in mets], rotation=25, ha="right", fontsize=8)
        top = max([v for c in plotted[e].values() for v in c.values() if v is not None] or [0.0])
        small = all(m["type"] != "auroc" for m in mets) and top < 0.5
        ax.set_ylim(0, max(0.1, 1.3 * top) if small else 1.05)
        ax.set_ylabel("AUROC / P(Yes) / YES share", fontsize=9, color="#52514e")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Gradient vs activation readers on the same experiments (single seed; hatched = reference columns)",
                 fontsize=12)
    fig.tight_layout()
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    atomic_json(fig_path.with_suffix(".json"), dict(summary=str(Path(root) / "summary.json"), values=plotted,
                                                    written=time.strftime("%Y-%m-%d %H:%M:%S")))
    log(f"{fig_path} (+ .json)")
    return 0


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--exp", required=True, choices=EXPS)
    b.add_argument("--reader", required=True, choices=list(READERS))
    b.add_argument("--out", type=Path, default=OUT)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--e2-steps", default=None, help="comma list of read steps (default 500..10000 every 500)")
    b.add_argument("--slot-npz", type=Path, default=None, help="E4 slot: slots_beh.npz of the slot chain")
    x = sub.add_parser("slot-extract")
    x.add_argument("--docs", type=Path, required=True)
    x.add_argument("--out-dir", type=Path, required=True)
    x.add_argument("--gpu", required=True)
    x.add_argument("--limit-docs", type=int, default=None)
    x.add_argument("--tok-budget", type=int, default=8192)
    x.add_argument("--part-docs", type=int, default=512)
    s = sub.add_parser("summary")
    s.add_argument("--out", type=Path, default=OUT)
    p = sub.add_parser("plot")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--fig", type=Path, default=FIG)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if a.limit is not None and a.limit < 1:
            ap.error("--limit must be >= 1")
        steps = tuple(int(t) for t in a.e2_steps.split(",")) if a.e2_steps else E2_STEPS
        return cmd_build(a.out, a.exp, a.reader, limit=a.limit, e2_steps=steps, slot_npz=a.slot_npz)
    if a.cmd == "slot-extract":
        return cmd_slot_extract(a.docs, a.out_dir, a.gpu, limit_docs=a.limit_docs, tok_budget=a.tok_budget,
                                part_docs=a.part_docs)
    if a.cmd == "summary":
        return cmd_summary(a.out)
    return cmd_plot(a.out, a.fig)


if __name__ == "__main__":
    sys.exit(main())
