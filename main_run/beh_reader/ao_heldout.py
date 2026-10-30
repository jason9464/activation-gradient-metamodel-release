#!/usr/bin/env python
"""Does our behaviour reader name held-out behaviours that the public Activation Oracle (AO) misses, and the reverse?
The AO gets both the gradient rows and its own input, activations; centred activation rows are read beside the raw
ones. Runner scripts/beh_reader/ao_heldout.sh.

All 530 beh_test bundles (organisms held out of reader training) are read by the two public AOs, each with three
inputs, asked the reader's training question "What behaviors did this training imprint? Keep it tight."
(common.QUESTION_BEHAVIOURAL), 10 answers per bundle, and judged with the truth judge of the evaluation (beh_judges,
same prompt, cache, ledger). The reader rl_beh_c040 is not read again: its evaluation readings (20 answers) and
truth-judged file eval/rl_beh_c040/readings_correct_judge_truth.jsonl are used as they are.
  variants     ao3layer_grad, ao3layer_act, ao3layer_act_centred, aopaper_grad, aopaper_act, aopaper_act_centred
  ao3layer     adamkarvonen/checkpoints_500k_pl_31k_spqav2_126k_cls_qwen3_14b (layers 10, 20, 30; the AO reader's init)
  aopaper      adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-14B (paper recipe; layer 20)
  grad         the bundle rows of inputs_correct.npz (-dL/dh token sums, unit x norm) at layers 10, 20, 30, copied
  act          the bundle's activation sums at layers 10, 20, 30, built here from the gradient store's full_act arrays;
               read as the control of act_centred (the raw sums of all bundles share one direction: mean pairwise
               cosine 0.94-0.97, so the AO, which keeps only the direction, gets nearly the same input for every bundle)
  act_centred  per layer unit(u - m): u = the act row divided by its norm, m = the mean of u over the 4,010 train
               bundles of bundles.jsonl, summed from the store the same way (train statistics only, as the reader
               subtracts its train mean_unit; m cached in train_mean_act.npz)

  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/ao_heldout.py <cmd> [--out DIR]
  build    inputs_{grad,act,act_centred}_correct.npz, train_mean_act.npz (m) and BUILD.json (section 7 checks); CPU
           only; refuses to overwrite an inputs file; reuses train_mean_act.npz when its train bundle count, train
           bundle order hash, store order hash and layers match, and refuses (SystemExit) when one differs
  plan     [--gpu G] [--samples 10] [--max-new-tokens 40] [--cap-usd 5]  the six read commands, their state, the
           answer count, the judge call and its cost estimate; starts and writes nothing
  read     --ao ao3layer|aopaper --input grad|act|act_centred --gpu G [--samples 10] [--max-new-tokens 40]
           [--limit N] [--inputs-dir D] [--hang-s S] [--retry-hung 1]  one variant: a worker process (read-worker) on
           GPU G loads the AO once and samples each bundle; the parent stops it when neither its log nor the partial
           reading grew for --hang-s seconds (eval_read.run_watched: TERM, KILL 60 s later) and starts it once more,
           resuming from the partial reading. Writes readings_<ao>_<input>.jsonl and readings_<ao>_<input>_settings.json;
           markers markers/read_<ao>_<input>.{START,DONE,FAILED} and .lock (one read of a variant at a time); logs
           read_logs/. A complete reading made with the same --max-new-tokens is left as it is (exit 0); one made with
           another length, or an incomplete one, is refused (exit 1). --limit N reads steps 0..N-1 only (smoke) and is
           refused with the default --out
  tokens   per variant the answers that hit max_new_tokens (count, share), seconds per generate call and the minutes
           a full reading of 530 steps would take, then the generation-length rule's outcome over all answers read;
           reads finished readings (or the partial .tmp); writes nothing
  judge    [--cap-usd 5] [--mode batch|sync] [--workers 48] [--dry-run]  beh_judges.cmd_judge("truth") unchanged with
           the paths repointed to --out, all six variants; refuses (exit 1, nothing sent) unless all six readings are
           finished, each made for its own variant from its own inputs file (--inputs-dir), all with one
           max_new_tokens, one answer count per bundle and one step count (beh_judges alone would skip an unfit
           reading and still judge the other five); --max-cost = the ledger total at start + cap; logs the cost estimate;
           --dry-run builds the requests and sends none; writes JUDGE.json (ledger increase, judged rows = answers);
           exit 1 when a judged file is missing or its row count differs from the reading's answers
  summary  summary.json (printed tables): per family x evidence YES rate of the reader and the six AO variants, the
           candidate lists with the verdict rule, and the inputs compared per AO;
           exit 1 when an input, reading or judged file is missing
--out defaults to OUT = $IIF_BASE/agm_data/runs/beh_reader/ao_heldout; --inputs-dir (read, judge, summary) defaults
to --out.

Activation rows (build). The gradient store (common.STORE, manifest.json) holds per chunk row fp16 unit vectors with
fp32 norms in index.parquet; full_act_L<L> is the sum of the output of model.model.layers[L] over positions 1..n-2 of
the chunk (manifest plan full_skip_pos0.act = true: position 0 is stored apart as pos0_act; the last position is in no
stored sum; worktree main_run/agm/core.py position_sets). GradStore (worktree main_run/grad_io.py) opens gradient
arrays only (grad_array_name), so build reads the act arrays itself: rows from load_store_index (which refuses a store
whose manifest doc_order_hash does not match its indexes), unit x norm in fp32, summed per document over its chunk
rows and per bundle over its bundles.jsonl doc_ids with numpy add.reduceat, the arithmetic of GradStore._vectors and
bundle_vectors("tokensum"). No negation, no rescaling (the AO injection normalises each row); kind act is not centred.
Centred rows (build). The train bundles are summed the same way in passes of 1,000 bundles; m = mean of their unit
rows in float64; act_centred = unit(unit(act) - m) in float64, stored as fp32 unit rows (kind act_centred).
Gradient rows (build). Layers 10, 20, 30 of inputs_correct.npz (export_inputs.py: fp16 unit x fp32 norm of -dL/dh,
uncentred), copied bit for bit with the same meta and step order; aopaper takes the layer-20 row. This is the input of
the filter's AO scorer (train_blockmix_filter.InProcessAO): unit rows as they are.
All three files: e2_generate layout (inputs [530, 1, 3, 5120] fp32, steps 0..529 = beh_test in bundles.jsonl order,
windows ["bundle"], layers [10, 20, 30], window_lo/hi 0, n_docs, meta json with run, sign, kind grad|act|act_centred,
order_hash = e2_generate.export_order_hash, bundle_ids, family, evidence, pair_id, unit_id, behaviour_id). The
gradient file keeps inputs_correct's sign text and kind, so e2_generate.load_inputs accepts it.

Reading (read). AO side imported unchanged: ao_diff_read.load_aos (Qwen3-14B bf16 + the AO adapter),
oracle_prompt_ids with one " ?" per layer block ("Layer: 10\\n ? \\nLayer: 20\\n ? \\nLayer: 30\\n ? \\n" + question
for ao3layer, "Layer: 20\\n ? \\n" + question for aopaper, chat template, thinking off), inject_ at decoder layer 1 in
the prefill (h <- h + ||h|| v/||v||), and ao_cleanbatch_read.sample_call (ao_diff_read.generate: do_sample,
temperature 0.7, top_p 0.9, top_k 20, max_new_tokens 40, the AO toolkit's settings, plus the per-row max-token hit).
One generate call per bundle with 10 rows, every row steered with the bundle's row(s). Seed of the call =
SEED0 + 1000 x step + AO index (ao_cleanbatch_read.AO_ORDER: aopaper 0, ao3layer 1); the seed depends on step
and AO only, so the three readings of a bundle by one AO share the seed. Each row of the reading:
{"step": s, "inputs": {"bundle": {"window": [0, 0], "n_docs", "n_samples": 10, "animal_samples": {}, "texts": [...],
"n_hit_max_new_tokens", "max_new_tokens", "seed"}}}, the e2_generate format eval_common.answers_of reads.
Generation length. read --max-new-tokens N (default 40 = the toolkit's) sets
ao_diff_read.GEN["max_new_tokens"] in the worker process (ao_cleanbatch_read.GEN is the same dict, so its max-token hit
rule follows; checked). The settings file records max_new_tokens, the toolkit value and the deviation; each reading row
records its length; a partial reading made with another length is not resumed (set aside). Rule: when more than 20 % of
the smoke's answers hit 40 (tokens prints it), the real readings use 192 and the deviation is recorded in the settings
(automatic).

Judge. beh_judges.cmd_judge("truth", ...) with common.INPUTS, common.EVAL, common.READ_LOGS, beh_judges.CONDS and
beh_judges.reading_path pointed at this experiment for one call (repointed): conditions = the six variants, inputs
of a variant = its inputs file (the judge maps step -> bundle_id through its meta), reading = readings_<variant>.jsonl;
the judged file is readings_<variant>_judge_truth.jsonl; ground truth = labels.jsonl ground_truth of the bundle.

Summary. A bundle's YES rate = YES / (YES + NO) over its answers (REFUSED left out). The reader's rate uses
its 20 answers; reader_first10 (answers 0-9) is reported beside it. Tables: family x variant and evidence x variant
(mean over bundles, bundle bootstrap 95% CI, beh_judges.boot_mean), family x evidence x variant (means). Candidates
over the bundles with all seven rates: reader_only = reader >= 0.5 and all six AO variants <= 0.1; ao_only = one AO
variant >= 0.5 and reader <= 0.1; each listed with bundle_id, family, evidence, behaviour_text and the seven rates,
counted by family and evidence next to the counts among all bundles (share, enrichment, hypergeometric tail p).
Verdict rule (fixed before the run): a family or evidence kind counts as a concentration only when it
holds more than half of the candidates AND its hypergeometric tail p (the candidates drawn from the bundles with all
seven rates) is < 0.05 -> "that kind is caught only by the reader" (the AO for ao_only); >= 10 candidates and no such
kind -> "the difference is not systematic" (an over-half kind with p >= 0.05 is named as explained by its base rate);
< 10 candidates -> no claim. Inputs compared per AO (grad against act, grad against act_centred, act_centred against
act): per family mean of each and the paired difference (bundle bootstrap), and the bundles where one input gives
>= 0.5 and the other <= 0.1.

Checks of build (BUILD.json; exit 1 when one fails; nothing is written when a check before writing fails, except the
train mean cache):
  bundles     530 beh_test bundles, family counts = common.TEST_FAMILY_COUNTS; inputs_correct.npz order hash
              recomputed = meta; its bundle_ids, family and n_docs = bundles.jsonl beh_test in order
  store       load_store_index passes and the recomputed order hash over "<doc_id>:<chunk_id>" = manifest
              doc_order_hash; plan full_skip_pos0.act true; act layers include 10, 20, 30; every document of the
              bundles in the store, chunk ids 0..k-1 each once; documents summed per bundle = n_docs; rows summed per
              bundle = the documents' chunk counts in the index
  act rows    finite and norm > 0 at every layer; 5 random bundles (seed 0) re-read document by document (another
              lookup: index rows by doc_id mask, one mmap row at a time, float64 sums) agree (cosine > 1 - 1e-6,
              max error < 1e-5 of the row norm)
  train mean  4,010 train bundles (common.N_TRAIN_BUNDLES); their documents in the store, chunk ids once, documents =
              n_docs, rows summed per bundle = chunk counts; finite; 3 random train bundles (seed 1) re-read document
              by document agree (cosine > 1 - 1e-6); shape [3, d]
  centred     finite, unit norm (|norm - 1| < 1e-5); the 5 re-read test bundles centred in float64 agree
              (cosine > 1 - 1e-6)
  row code    the same builder run on the full_grad arrays and negated reproduces every inputs_correct row at layers 10,
              20, 30 (min cosine > 0.999; bit-equal count after the export's fp16 unit rounding reported)
  files       after writing: each file re-read (order hash, kind, shape, bundle_ids), arrays bit-equal to what was
              meant; the gradient file equals inputs_correct at layers 10, 20, 30 bit for bit and passes
              e2_generate.load_inputs
Diagnostics (no pass/fail): per layer the mean pairwise cosine of the 530 rows for grad, act and act_centred (act about
0.95, act_centred about 0 when m fits the test bundles), their norms, the norm of m and the cosine between m and the
mean unit act row of the test bundles.

Cost and bottleneck. build is IO bound: the test bundles' 6,300 store rows and the train bundles' 49,000 rows x 3
layers (fp16, 10 KB each), and inputs_correct.npz (434 MB) once; the test part measured 5 s (28/28
checks, all 530 gradient rows bit-equal to inputs_correct after the fp16 unit rounding; mean pairwise cosine of the
530 activation rows 0.974, 0.969, 0.942 at layers 10, 20, 30 against 0.27, 0.47, 0.54 for the gradient rows); the train
mean adds about 1.5 GB of reads. read is generation bound: 530 calls of 10 rows x max_new_tokens new tokens on one GPU
(14B bf16, about 30 GiB) plus the model load; an estimated 40 min per variant at 40 tokens (192 tokens: longer,
up to about 4.8 times when the answers run to the limit; tokens projects it from the smoke's seconds per call). judge:
6 variants x 530 bundles x 10 = 31,800 answers, about 1.4 USD batch at 4.3e-5 USD per answer (the ledger's truth batch
rows; plan and judge print it); cap = the ledger total at judge start + 5 USD.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import math
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.EV / "ao_heldout"
READER = "rl_beh_c040"
AO_NAMES = ("ao3layer", "aopaper")
INPUT_KINDS = ("grad", "act", "act_centred")
VARIANTS = tuple(f"{a}_{i}" for a in AO_NAMES for i in INPUT_KINDS)  # "<ao>_<input>"; the AO names hold no "_"
ROW_LAYERS = (10, 20, 30)  # rows built; ao3layer takes all three, aopaper layer 20 (ao_diff_read.AOS)
AO_LAYERS = {"ao3layer": (10, 20, 30), "aopaper": (20,)}  # checked against ao_diff_read.AOS in the test and the worker
AO_ORDER = ("aopaper", "ao3layer")  # ao_cleanbatch_read.AO_ORDER (the seed rule's AO index; checked in the test)
SAMPLES = 10
SEED0 = 0
CAP_USD = 5.0
TOOLKIT_MAX_NEW_TOKENS = 40  # ao_diff_read.GEN (the AO toolkit's verbalizer setting; checked in the test and worker)
LONG_MAX_NEW_TOKENS = 192  # the reader's answer length; the run uses it when the smoke's answers hit 40 too often
HIT_SHARE_MAX = 0.20  # generation-length rule: more than this share of the smoke's answers at 40 -> 192
P_MAX = 0.05  # verdict: an over-half concentration counts only with hypergeometric tail p below this
TRAIN_MEAN = "train_mean_act.npz"  # m, the mean unit activation row of the train bundles (cache in --out)
TRAIN_CHUNK = 1000  # train bundles summed per pass (bounds the memory of build)
RE_READ_TRAIN = 3
UNIT_TOL = 1e-5
HANG_S = 900.0  # no growth of the worker's log or partial reading for this long: TERM (the model load writes nothing)
KILL_GRACE_S = 60.0
YES_HI, NO_LO, EPS = 0.5, 0.1, 1e-9  # candidate thresholds
MIN_CANDIDATES = 10
RE_READ = 5
COS_MIN_GRAD = 0.999
JUDGE_KEY = "ao_heldout"  # the readers argument of beh_judges.cmd_judge (one key; reading_path ignores it)
ACT_SIGN = ("activation sums of the behaviour gradient store: per layer the stored full_act rows (output of "
            "model.model.layers[L], positions 1..n-2 of each chunk, position 0 excluded) of the bundle's documents "
            "summed in fp32 (fp16 unit x fp32 norm per row); no negation; stored as fp32")
CENTRED_SIGN = ("activation sums of the behaviour gradient store centred by the train mean: per layer unit(u - m), "
                "u = the bundle's activation sum (kind act) divided by its norm, m = the mean of u over the train "
                "bundles of bundles.jsonl (train_mean_act.npz); unit rows (norm 1; the AO injection keeps only the "
                "direction); stored as fp32")
POSITION_RULE = ("full_act_L<L>: sum over positions 1..n-2 of a chunk of the output of model.model.layers[L] (manifest "
                 "plan full_skip_pos0.act true; position 0 is pos0_act; the last position is in no stored sum)")
BUNDLE_RULE = ("bundle row = sum over the bundle's bundles.jsonl doc_ids of each document's chunk rows (fp16 unit x fp32 "
               "norm in fp32), numpy add.reduceat in document order, as GradStore._vectors + bundle_vectors(tokensum)")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [ao_heldout] {msg}", flush=True)


# ---------------------------------------------------------------- names and paths
def variant(ao: str, inp: str) -> str:
    if ao not in AO_NAMES or inp not in INPUT_KINDS:
        raise SystemExit(f"unknown AO {ao!r} or input {inp!r}")
    return f"{ao}_{inp}"


def split_variant(v: str) -> tuple[str, str]:
    ao, inp = v.split("_", 1)
    variant(ao, inp)
    return ao, inp


def inputs_path(d: Path, inp: str) -> Path:
    return Path(d) / f"inputs_{inp}_correct.npz"


def reading_path(out: Path, v: str) -> Path:
    return Path(out) / f"readings_{v}.jsonl"


def settings_path(reading: Path) -> Path:
    return reading.with_name(reading.stem + "_settings.json")  # beh_judges.settings_path


def judged_path(out: Path, v: str) -> Path:
    return Path(out) / f"readings_{v}_judge_truth.jsonl"  # em_judges.judged_path(reading, "truth")


def marker(out: Path, v: str, what: str) -> Path:
    return Path(out) / "markers" / f"read_{v}.{what}"


def reader_judged() -> Path:
    return C.eval_dir(READER) / "readings_correct_judge_truth.jsonl"


def seed_of(step: int, ao: str) -> int:
    """SEED0 + 1000 x step + AO index (ao_cleanbatch_read.seed_of without the student and group terms)."""
    return SEED0 + 1000 * int(step) + AO_ORDER.index(ao)


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def set_aside(p: Path) -> Path | None:
    """Move an existing file to <name>.before_<stamp> (nothing is deleted); returns the new path."""
    p = Path(p)
    if not p.exists():
        return None
    q = p.with_name(f"{p.name}.before_{stamp()}")
    k = 0
    while q.exists():
        k += 1
        q = p.with_name(f"{p.name}.before_{stamp()}_{k}")
    os.replace(p, q)
    return q


# ---------------------------------------------------------------- modules
def e2gen():
    """The em-reader worktree's e2_generate (export_order_hash, load_inputs)."""
    from main_run.reader import e2_generate

    return e2_generate


def beh_judges():
    import beh_judges as B

    return B


def rare_perdoc():
    import rare_perdoc as RP

    return RP


def _main_run(name: str):
    """A module of the repo's main_run (ao_diff_read, ao_cleanbatch_read), main_run left at the end of sys.path (they
    insert it at the front when they load)."""
    d = str(C.REPO / "main_run")
    try:
        if d not in sys.path:
            sys.path.append(d)
        return importlib.import_module(name)
    finally:
        C._demote(d)


def ao_diff():
    return _main_run("ao_diff_read")


def ao_cleanbatch():
    return _main_run("ao_cleanbatch_read")


# ---------------------------------------------------------------- inputs files
def write_npz(path: Path, X: np.ndarray, meta: dict, n_docs, layers) -> None:
    """One e2_generate inputs file ([n, 1, K, D] fp32) through <path>.tmp and a rename; an existing file is refused."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    n = X.shape[0]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, inputs=np.ascontiguousarray(X[:, None], dtype=np.float32), steps=np.arange(n),
                 windows=np.array(["bundle"]), layers=np.array(list(layers)), window_lo=np.zeros((n, 1), np.int64),
                 window_hi=np.zeros((n, 1), np.int64), n_docs=np.asarray(n_docs, np.int64).reshape(n, 1),
                 meta=json.dumps(meta))
    os.replace(tmp, path)


def load_inputs(path: Path, kind: str | None = None) -> dict:
    """An inputs file of this experiment (or inputs_correct.npz) with its checks: order hash = meta (e2_generate's
    export_order_hash), kind, steps 0..n-1, shape [n, 1, K, D], one bundle_id per step. ValueError on a failure."""
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        steps = [int(t) for t in z["steps"]]
        windows = [str(w) for w in z["windows"]]
        layers = [int(L) for L in z["layers"]]
        X = z["inputs"]
        n_docs = np.asarray(z["n_docs"])
    fails = []
    if e2gen().export_order_hash(steps, windows, layers) != meta.get("order_hash"):
        fails.append("order hash mismatch")
    if kind is not None and meta.get("kind") != kind:
        fails.append(f"kind {meta.get('kind')!r}, expected {kind!r}")
    if steps != list(range(len(steps))) or windows != ["bundle"]:
        fails.append(f"steps are not 0..n-1 or windows {windows} != ['bundle']")
    if X.dtype != np.float32 or X.ndim != 4 or X.shape[:3] != (len(steps), 1, len(layers)):
        fails.append(f"inputs {X.dtype} {X.shape}, expected float32 ({len(steps)}, 1, {len(layers)}, d)")
    if len(meta.get("bundle_ids", [])) != len(steps):
        fails.append("meta bundle_ids length differs from steps")
    if fails:
        raise ValueError(f"{path}: " + "; ".join(fails))
    return dict(inputs=X, steps=steps, layers=layers, n_docs=n_docs, meta=meta)


def npz_meta(path: Path) -> dict:
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


# ---------------------------------------------------------------- build
def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine per (row, layer) of [n, L, d] arrays."""
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def bundle_sums(df, man: dict, store: Path, doc_lists: list, kind: str, layers, seg_out: list | None = None) -> np.ndarray:
    """[n bundles, len(layers), d] fp32: per layer the sum over each bundle's documents of their chunk rows (fp16 unit
    x fp32 norm of full_<kind>_L<L>), GradStore._vectors and bundle_vectors(tokensum) arithmetic. seg_out, when given,
    receives the number of store rows summed per bundle."""
    from main_run.agm import core

    rows_of = df.groupby("doc_id", sort=False).indices
    pos_parts, seg = [], []
    for docs in doc_lists:
        if not docs:
            raise SystemExit("empty bundle")
        n = 0
        for d in docs:
            p = rows_of[d]
            pos_parts.append(p)
            n += len(p)
        seg.append(n)
    if seg_out is not None:
        seg_out[:] = seg
    pos = np.concatenate(pos_parts)
    starts = np.concatenate([[0], np.cumsum(seg)[:-1]]).astype(np.int64)
    shard = df["shard"].to_numpy()[pos]
    row = df["row"].to_numpy(np.int64)[pos]
    n_rows = {s["dir"]: int(s["rows"]) for s in man["shards"]}
    out = None
    for li, L in enumerate(layers):
        name = core.array_name("full", kind, L)
        vec = None
        for s in np.unique(shard):
            arr = np.load(Path(store) / s / f"{name}.npy", mmap_mode="r")
            if arr.ndim != 2 or arr.shape[0] != n_rows[s]:
                raise SystemExit(f"{s}/{name}.npy has shape {arr.shape}, manifest has {n_rows[s]} rows")
            m = np.flatnonzero(shard == s)
            o = np.argsort(row[m], kind="stable")
            part = np.asarray(arr[row[m][o]], np.float32)
            if vec is None:
                vec = np.empty((len(pos), arr.shape[1]), np.float32)
            vec[m[o]] = part
        vec *= df[f"norm_{name}"].to_numpy(np.float32)[pos][:, None]
        S = np.add.reduceat(vec, starts, axis=0)
        if out is None:
            out = np.empty((len(doc_lists), len(layers), S.shape[1]), np.float32)
        out[:, li] = S
    return out


def reread_bundle(df, store: Path, docs: list, kind: str, layers) -> np.ndarray:
    """[len(layers), d] float64: the bundle sum read document by document, one mmap row at a time (another code path
    than bundle_sums: index rows by a doc_id mask, float64 accumulation)."""
    from main_run.agm import core

    doc_col = df["doc_id"].to_numpy()
    shard_col = df["shard"].to_numpy()
    row_col = df["row"].to_numpy(np.int64)
    mm: dict = {}
    out = None
    for li, L in enumerate(layers):
        name = core.array_name("full", kind, L)
        norms = df[f"norm_{name}"].to_numpy(np.float64)
        for d in docs:
            for i in np.flatnonzero(doc_col == d):
                key = (shard_col[i], name)
                if key not in mm:
                    mm[key] = np.load(Path(store) / shard_col[i] / f"{name}.npy", mmap_mode="r")
                v = np.asarray(mm[key][row_col[i]], np.float64) * norms[i]
                if out is None:
                    out = np.zeros((len(layers), v.shape[0]), np.float64)
                out[li] += v
    return out


def mean_pair_cos(X: np.ndarray) -> list[float]:
    """Per layer the mean cosine over distinct row pairs of [n, L, d] (exact, from the sum of unit rows)."""
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    n = X.shape[0]
    s = U.astype(np.float64).sum(0)
    return [float(((s[li] @ s[li]) - n) / (n * (n - 1))) for li in range(X.shape[1])]


def norm_stats(X: np.ndarray) -> dict:
    nr = np.linalg.norm(X, axis=-1)
    return {str(i): dict(min=float(nr[:, i].min()), median=float(np.median(nr[:, i])), max=float(nr[:, i].max()))
            for i in range(X.shape[1])}


def store_checks(df, man: dict, bundles: list[dict]) -> tuple[dict, list]:
    """Section 7 store checks; returns (checks, per-bundle doc lists)."""
    from main_run.agm import core, store as S

    plan = man.get("plan", {})
    skip_act = core.skip_for(plan.get("full_skip_pos0", False), "act")
    act_layers = [int(L) for L in plan.get("act_layers", [])]
    rows_of = df.groupby("doc_id", sort=False).indices
    has_chunk = "chunk_id" in df.columns
    in_store, chunks_ok, n_docs_ok = True, True, True
    chunk_hist: dict = {}
    for b in bundles:
        for d in b["doc_ids"]:
            if d not in rows_of:
                in_store = False
                continue
            p = rows_of[d]
            cid = sorted(int(c) for c in df["chunk_id"].to_numpy()[p]) if has_chunk else [0] * len(p)
            chunks_ok &= cid == list(range(len(cid)))
            chunk_hist[len(cid)] = chunk_hist.get(len(cid), 0) + 1
        n_docs_ok &= len(b["doc_ids"]) == int(b["n_docs"])
    cols = [f"norm_{core.array_name('full', k, L)}" for k in ("act", "grad") for L in ROW_LAYERS]
    return dict(doc_order_hash=man.get("doc_order_hash"), n_rows=int(len(df)),
                pass_order_hash=S.order_hash(S.index_keys(df)) == man.get("doc_order_hash"),
                pass_skip_pos0_act=bool(skip_act), act_layers=act_layers,
                pass_act_layers=all(L in act_layers for L in ROW_LAYERS),
                pass_norm_columns=all(c in df.columns for c in cols),
                pass_docs_in_store=bool(in_store), pass_chunks_once=bool(chunks_ok),
                chunks_per_doc={str(k): v for k, v in sorted(chunk_hist.items())},
                pass_docs_summed_eq_n_docs=bool(n_docs_ok)), \
        [list(b["doc_ids"]) for b in bundles]


def all_passes(checks: dict) -> list:
    out = []
    for c in checks.values():
        if isinstance(c, dict):
            out += [(k, v) for k, v in c.items() if k.startswith("pass_")]
    return out


def unit64(X: np.ndarray) -> np.ndarray:
    """Rows of [..., d] divided by their norm, in float64 (ValueError on a zero or non-finite norm)."""
    X = np.asarray(X, np.float64)
    nr = np.linalg.norm(X, axis=-1, keepdims=True)
    if not (np.isfinite(nr).all() and (nr > 0).all()):
        raise ValueError("a row with zero or non-finite norm")
    return X / nr


def bundle_order_hash(ids) -> str:
    """sha256 over the bundle ids joined by newlines (the key of the train mean cache)."""
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def centre_rows(Xa: np.ndarray, mean_unit: np.ndarray) -> np.ndarray:
    """[n, L, d] fp32 unit rows unit(unit(Xa) - mean_unit), computed in float64 (the act_centred input)."""
    return unit64(unit64(Xa) - np.asarray(mean_unit, np.float64)[None]).astype(np.float32)


def train_mean_act(out: Path, df, man: dict, store: Path, train: list[dict], *, chunk: int = TRAIN_CHUNK) -> dict:
    """m [len(ROW_LAYERS), d] float64 = the mean over the train bundles of their unit activation rows (bundle_sums of
    full_act, as build makes the test rows, divided by the norm), summed in passes of `chunk` bundles. Cached in
    out/train_mean_act.npz (mean_unit, layers, bundle_ids, info json) keyed by the bundle count, the train bundle order
    hash, the store order hash and the layers: a matching cache is reused, another one refused (SystemExit), never
    overwritten. Returns dict(mean, info)."""
    out = Path(out)
    p = out / TRAIN_MEAN
    ids = [b["bundle_id"] for b in train]
    key = dict(n_bundles=len(train), bundle_order_hash=bundle_order_hash(ids),
               store_doc_order_hash=man.get("doc_order_hash"), layers=list(ROW_LAYERS))
    if p.exists():
        with np.load(p) as z:
            mean = np.asarray(z["mean_unit"], np.float64)
            info = json.loads(str(z["info"]))
        bad = [k for k, v in key.items() if info.get(k) != v]
        if bad:
            raise SystemExit(f"{p} was made for other {bad}; move it aside to compute m again (it is never overwritten)")
        log(f"train mean: reusing {p} ({info['n_bundles']} bundles)")
        return dict(mean=mean, info=dict(info, cache="reused", path=str(p), sha256=C.sha256_file(p)))
    t0 = time.time()
    doc_lists = [list(b["doc_ids"]) for b in train]
    rng = np.random.default_rng(1)
    picks = sorted(rng.choice(len(train), size=min(RE_READ_TRAIN, len(train)), replace=False).tolist())
    acc, seg_all, picked, finite = None, [], {}, True
    for i in range(0, len(train), chunk):
        seg: list = []
        X = bundle_sums(df, man, store, doc_lists[i:i + chunk], "act", ROW_LAYERS, seg_out=seg)
        finite &= bool(np.isfinite(X).all())
        U = unit64(X)
        acc = U.sum(0) if acc is None else acc + U.sum(0)
        seg_all += [int(x) for x in seg]
        for j in picks:
            if i <= j < i + chunk:
                picked[j] = U[j - i]
        log(f"train mean: {min(i + chunk, len(train))}/{len(train)} bundles ({time.time() - t0:.0f} s)")
    mean = acc / len(train)
    cos_re = [float((picked[j] * unit64(reread_bundle(df, store, doc_lists[j], "act", ROW_LAYERS))).sum(-1).min())
              for j in picks]
    info = dict(key, rows_per_bundle=seg_all, rows_summed=int(sum(seg_all)),
                docs_summed=int(sum(len(d) for d in doc_lists)), finite=finite,
                reread_bundles=[ids[j] for j in picks], reread_min_cos=min(cos_re) if cos_re else None,
                mean_norm={str(L): float(np.linalg.norm(mean[li])) for li, L in enumerate(ROW_LAYERS)},
                rule=("m[layer] = mean over the train bundles (bundles.jsonl split train, file order) of u = s / ||s||, "
                      "s = the bundle's activation sum built as for the test bundles (bundle_sums of full_act); float64"),
                split=C.SPLIT_TRAIN, store=str(store), seconds=round(time.time() - t0, 1),
                written=time.strftime("%Y-%m-%d %H:%M:%S"), code=os.path.abspath(__file__))
    out.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, mean_unit=mean, layers=np.array(list(ROW_LAYERS)), bundle_ids=np.array(ids), info=json.dumps(info))
    os.replace(tmp, p)
    log(f"train mean: wrote {p} ({len(train)} bundles, {info['rows_summed']} store rows, {info['seconds']} s)")
    return dict(mean=mean, info=dict(info, cache="computed", path=str(p), sha256=C.sha256_file(p)))


def cmd_build(out: Path, *, expect_n: int = C.N_TEST_BUNDLES, expect_families: dict | None = None,
              expect_train_n: int = C.N_TRAIN_BUNDLES, train_chunk: int = TRAIN_CHUNK) -> int:
    from main_run.grad_io import load_store_index

    t0 = time.time()
    out = Path(out)
    expect_families = C.TEST_FAMILY_COUNTS if expect_families is None else expect_families
    for k in INPUT_KINDS:
        if inputs_path(out, k).exists():
            raise SystemExit(f"{inputs_path(out, k)} exists; refusing to overwrite")
    E2 = e2gen()
    bundles = C.load_bundles()
    te = [b for b in bundles if b["split"] == C.SPLIT_TEST]
    tr = [b for b in bundles if b["split"] == C.SPLIT_TRAIN]
    checks: dict = {}
    # gradient rows: inputs_correct.npz against bundles.jsonl
    src = Path(C.INPUTS["correct"])
    inp = load_inputs(src, kind="grad")
    meta_c = inp["meta"]
    li = [inp["layers"].index(L) for L in ROW_LAYERS]
    Xg = np.ascontiguousarray(inp["inputs"][:, 0][:, li])
    del inp["inputs"]
    checks["bundles"] = dict(
        n=len(te), pass_n=len(te) == expect_n,
        families=dict(sorted(Counter(b["family"] for b in te).items())),
        pass_families=dict(Counter(b["family"] for b in te)) == dict(expect_families),
        pass_inputs_order_hash=True,  # load_inputs raised otherwise
        pass_bundle_ids=meta_c["bundle_ids"] == [b["bundle_id"] for b in te],
        pass_family=list(meta_c.get("family", [])) == [b["family"] for b in te],
        pass_n_docs=[int(x) for x in inp["n_docs"][:, 0]] == [int(b["n_docs"]) for b in te])
    if not checks["bundles"]["pass_bundle_ids"]:
        return build_failed(out, checks, "inputs_correct.npz bundle order differs from bundles.jsonl beh_test")
    log(f"{len(te)} beh_test bundles, {len(tr)} train bundles; inputs_correct rows {Xg.shape} ({time.time() - t0:.0f} s)")
    # store
    store = Path(C.STORE)
    df, man = load_store_index(store)
    checks["store"], doc_lists = store_checks(df, man, te)
    if not (checks["store"]["pass_docs_in_store"] and checks["store"]["pass_norm_columns"]):
        return build_failed(out, checks, "documents or norm columns missing in the store")
    seg: list = []
    Xa = bundle_sums(df, man, store, doc_lists, "act", ROW_LAYERS, seg_out=seg)
    vc = df["doc_id"].value_counts()  # another count of each document's index rows than bundle_sums' groupby
    want_rows = [sum(int(vc.get(d, 0)) for d in docs) for docs in doc_lists]
    checks["store"].update(rows_summed=int(sum(seg)), pass_rows_summed_eq_chunks=[int(x) for x in seg] == want_rows)
    log(f"activation rows {Xa.shape} ({time.time() - t0:.0f} s)")
    nr = np.linalg.norm(Xa, axis=-1)
    rng = np.random.default_rng(0)
    picks = sorted(rng.choice(len(te), size=min(RE_READ, len(te)), replace=False).tolist())
    cos_re, err_re, rereads = [], [], []
    for i in picks:
        r = reread_bundle(df, store, doc_lists[i], "act", ROW_LAYERS)
        rereads.append(r)
        a = Xa[i].astype(np.float64)
        cos_re.append(float((cos_rows(a[None], r[None])).min()))
        err_re.append(float((np.abs(a - r).max(-1) / np.linalg.norm(r, axis=-1)).max()))
    checks["act_rows"] = dict(pass_finite=bool(np.isfinite(Xa).all()), pass_norm_positive=bool((nr > 0).all()),
                              reread_bundles=[te[i]["bundle_id"] for i in picks], reread_min_cos=min(cos_re),
                              reread_max_rel_err=max(err_re),
                              pass_reread=bool(min(cos_re) > 1 - 1e-6 and max(err_re) < 1e-5))
    if not (checks["act_rows"]["pass_finite"] and checks["act_rows"]["pass_norm_positive"]):
        return build_failed(out, checks, "activation rows not finite or with zero norm")
    # the train mean m (train bundles only) and the centred rows
    tc, _ = store_checks(df, man, tr)
    checks["train_mean"] = dict(n=len(tr), pass_n=len(tr) == expect_train_n, **{
        k: tc[k] for k in ("pass_docs_in_store", "pass_chunks_once", "pass_docs_summed_eq_n_docs")})
    if not (tr and checks["train_mean"]["pass_docs_in_store"]):
        return build_failed(out, checks, "no train bundles, or documents of train bundles missing in the store")
    tm = train_mean_act(out, df, man, store, tr, chunk=train_chunk)
    mean, tinfo = tm["mean"], tm["info"]
    want_tr = [sum(int(vc.get(d, 0)) for d in b["doc_ids"]) for b in tr]
    checks["train_mean"].update(
        cache=tinfo["cache"], path=tinfo["path"], sha256=tinfo["sha256"], rows_summed=tinfo["rows_summed"],
        docs_summed=tinfo["docs_summed"], reread_bundles=tinfo["reread_bundles"],
        reread_min_cos=tinfo["reread_min_cos"], mean_norm=tinfo["mean_norm"],
        pass_rows_summed_eq_chunks=[int(x) for x in tinfo["rows_per_bundle"]] == want_tr,
        pass_finite=bool(tinfo["finite"] and np.isfinite(mean).all()),
        pass_reread=tinfo["reread_min_cos"] is not None and tinfo["reread_min_cos"] > 1 - 1e-6,
        pass_shape=mean.shape == (len(ROW_LAYERS), Xa.shape[-1]))
    if not checks["train_mean"]["pass_shape"]:
        return build_failed(out, checks, f"train mean shape {mean.shape}")
    Xc = centre_rows(Xa, mean)
    nc = np.linalg.norm(Xc.astype(np.float64), axis=-1)
    cos_c = [float((unit64(unit64(r) - mean) * Xc[i].astype(np.float64)).sum(-1).min()) for i, r in zip(picks, rereads)]
    checks["act_centred_rows"] = dict(pass_finite=bool(np.isfinite(Xc).all()), max_norm_error=float(np.abs(nc - 1).max()),
                                      pass_unit=bool(np.abs(nc - 1).max() < UNIT_TOL), reread_min_cos=min(cos_c),
                                      pass_reread=bool(min(cos_c) > 1 - 1e-6))
    log(f"centred rows {Xc.shape}; train mean {tinfo['cache']} ({time.time() - t0:.0f} s)")
    # the builder on the gradient arrays against inputs_correct
    G = -bundle_sums(df, man, store, doc_lists, "grad", ROW_LAYERS)
    nrm = np.linalg.norm(G, axis=-1, keepdims=True)
    G16 = (G / nrm).astype(np.float16).astype(np.float32) * nrm  # em_units: fp16 unit row x fp32 norm
    c = cos_rows(G, Xg)
    checks["row_code"] = dict(min_cos=float(c.min()), n_rows_bit_equal_after_fp16=int(sum(
        np.array_equal(G16[i], Xg[i]) for i in range(len(te)))), n_rows=len(te), pass_grad_rows=bool(c.min() > COS_MIN_GRAD))
    log(f"checks: grad builder min cos {c.min():.6f}, re-read min cos {min(cos_re):.8f} ({time.time() - t0:.0f} s)")
    test_mean = unit64(Xa).mean(0)
    lay = [str(L) for L in ROW_LAYERS]
    checks["diagnostics"] = dict(
        mean_pair_cos={k: dict(zip(lay, mean_pair_cos(X))) for k, X in (("act", Xa), ("act_centred", Xc), ("grad", Xg))},
        norms=dict(act=norm_stats(Xa), grad=norm_stats(Xg)), layers=list(ROW_LAYERS),
        train_mean_norm=dict(zip(lay, (float(np.linalg.norm(m)) for m in mean))),
        test_mean_unit_act_norm=dict(zip(lay, (float(np.linalg.norm(m)) for m in test_mean))),
        cos_train_mean_test_mean=dict(zip(lay, (float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                                               for a, b in zip(mean, test_mean)))))
    log("mean pairwise cosine of the test rows by layer: " + "; ".join(
        f"{k} " + ", ".join(f"{v:.3f}" for v in d.values()) for k, d in checks["diagnostics"]["mean_pair_cos"].items()))
    bad = [k for k, v in all_passes(checks) if not v]
    if bad:
        return build_failed(out, checks, f"checks failed: {bad}")

    out.mkdir(parents=True, exist_ok=True)
    order = E2.export_order_hash(list(range(len(te))), ["bundle"], list(ROW_LAYERS))
    keep = {k: meta_c[k] for k in ("bundle_ids", "family", "evidence", "pair_id", "unit_id", "behaviour_id")
            if k in meta_c}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_a = dict(run=str(out), sign=ACT_SIGN, kind="act", order_hash=order, arm=None, split=C.SPLIT_TEST,
                  condition="correct", **keep, store=str(store), store_doc_order_hash=man.get("doc_order_hash"),
                  quantity="full_act", position_rule=POSITION_RULE, bundle_rule=BUNDLE_RULE,
                  layer_convention=man.get("layer_convention"), source_inputs=str(src), written=now,
                  code=os.path.abspath(__file__))
    meta_ac = dict(meta_a, sign=CENTRED_SIGN, kind="act_centred", quantity="full_act, centred by the train mean",
                   rows="unit rows (norm 1)",
                   centring=dict(rule="unit(u - m), u = act row / ||act row||, m = mean_unit of train_mean_act.npz",
                                 train_mean=tinfo["path"], train_mean_sha256=tinfo["sha256"],
                                 n_train_bundles=len(tr), train_bundle_order_hash=tinfo["bundle_order_hash"],
                                 train_mean_norm=checks["diagnostics"]["train_mean_norm"]))
    meta_g = dict(meta_c, run=str(out), order_hash=order, source_inputs=str(src),
                  source_inputs_sha256=C.sha256_file(src), source_layers=list(ROW_LAYERS),
                  copy_rule="rows of layers 10, 20, 30 of inputs_correct.npz copied bit for bit", written=now,
                  code=os.path.abspath(__file__))
    n_docs = [int(b["n_docs"]) for b in te]
    files = {}
    for k, X, meta in (("act", Xa, meta_a), ("act_centred", Xc, meta_ac), ("grad", Xg, meta_g)):
        p = inputs_path(out, k)
        write_npz(p, X, meta, n_docs, ROW_LAYERS)
        back = load_inputs(p, kind=k)
        f = dict(path=str(p), bytes=os.path.getsize(p), order_hash=order,
                 pass_reload=back["inputs"].shape == (len(te), 1, len(ROW_LAYERS), X.shape[-1]),
                 pass_equal=bool(np.array_equal(back["inputs"][:, 0], X)),
                 pass_bundle_ids=back["meta"]["bundle_ids"] == [b["bundle_id"] for b in te],
                 pass_run=os.path.realpath(back["meta"]["run"]) == os.path.realpath(out))
        if k == "grad":
            z = load_inputs(src, kind="grad")
            f["pass_equal_inputs_correct"] = bool(np.array_equal(back["inputs"], z["inputs"][:, :, li]))
            del z
            e2 = E2.load_inputs(p)
            f["pass_e2_load_inputs"] = e2["layers"] == list(ROW_LAYERS)
        files[k] = f
        del back
    checks["files"] = files
    every = all_passes(checks) + [(k2, v) for f in files.values() for k2, v in f.items() if k2.startswith("pass_")]
    report = dict(n_checks=len(every), n_passed=int(sum(bool(v) for _, v in every)), checks=checks,
                  seconds=round(time.time() - t0, 1), time=now)
    report["passed"] = report["n_passed"] == report["n_checks"]
    p = out / "BUILD.json"
    if p.exists():
        p = out / f"BUILD_{stamp()}.json"
    atomic_json(p, report)
    log(f"{report['n_passed']}/{report['n_checks']} checks passed in {report['seconds']} s; report {p}")
    return 0 if report["passed"] else 1


def build_failed(out: Path, checks: dict, why: str) -> int:
    out.mkdir(parents=True, exist_ok=True)
    p = out / "BUILD_FAILED.json"
    if p.exists():
        p = out / f"BUILD_FAILED_{stamp()}.json"
    atomic_json(p, dict(why=why, checks=checks, time=time.strftime("%Y-%m-%d %H:%M:%S")))
    log(f"{why}; nothing written (report {p})")
    return 1


# ---------------------------------------------------------------- reading
def reading_row(step: int, n_docs: int, texts: list, hits, seed: int,
                max_new_tokens: int = TOOLKIT_MAX_NEW_TOKENS) -> dict:
    return {"step": int(step), "inputs": {"bundle": {"window": [0, 0], "n_docs": int(n_docs), "n_samples": len(texts),
                                                     "animal_samples": {}, "texts": list(texts),
                                                     "n_hit_max_new_tokens": int(sum(hits)),
                                                     "max_new_tokens": int(max_new_tokens), "seed": int(seed)}}}


def row_mnt(r: dict) -> int:
    """The max_new_tokens a reading row was made with (rows without the field: the toolkit's 40)."""
    return int(r["inputs"]["bundle"].get("max_new_tokens", TOOLKIT_MAX_NEW_TOKENS))


def row_ok(r: dict, samples: int, max_new_tokens: int | None = None) -> bool:
    try:
        rec = r["inputs"]["bundle"]
        return isinstance(r["step"], int) and rec["n_samples"] == samples == len(rec["texts"]) \
            and all(isinstance(t, str) for t in rec["texts"]) \
            and (max_new_tokens is None or row_mnt(r) == int(max_new_tokens))
    except (KeyError, TypeError, ValueError):
        return False


def read_rows(path: Path) -> list[dict]:
    return C.read_jsonl(path)


def read_rows_partial(path: Path) -> list[dict]:
    """The complete lines of a reading that may end in a broken line (a .tmp being written)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                break
    return rows


def rows_state(path: Path, steps: list[int], samples: int) -> str:
    """'missing', 'complete' (one row per step in order, `samples` answers each) or why not."""
    path = Path(path)
    if not path.exists():
        return "missing"
    try:
        rows = read_rows(path)
    except (ValueError, OSError) as e:
        return f"unreadable: {e}"
    if [r.get("step") for r in rows] != list(steps):
        return f"steps differ ({len(rows)} rows, expected {len(steps)})"
    if not all(row_ok(r, samples) for r in rows):
        return f"a row has not {samples} answers"
    return "complete"


def reading_state(path: Path, steps: list[int], samples: int) -> str:
    """rows_state of a published reading, and its settings file present."""
    st = rows_state(path, steps, samples)
    if st == "complete" and not settings_path(Path(path)).exists():
        return "no settings file"
    return st


def read_settings(reading: Path) -> dict:
    p = settings_path(Path(reading))
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def settings_mnt(reading: Path) -> int | None:
    """max_new_tokens of a reading's settings file (None without one; the toolkit's 40 when the field is absent)."""
    if not settings_path(Path(reading)).exists():
        return None
    return int(read_settings(reading).get("max_new_tokens", TOOLKIT_MAX_NEW_TOKENS))


def mnt_settings(n: int) -> dict:
    """The generation-length fields of a settings file."""
    n = int(n)
    return dict(max_new_tokens=n, toolkit_max_new_tokens=TOOLKIT_MAX_NEW_TOKENS,
                deviation_from_toolkit=None if n == TOOLKIT_MAX_NEW_TOKENS else
                f"max_new_tokens {n} instead of the AO toolkit's {TOOLKIT_MAX_NEW_TOKENS}",
                max_new_tokens_rule=(f"toolkit default {TOOLKIT_MAX_NEW_TOKENS}; when more than {HIT_SHARE_MAX:.0%} of "
                                     f"the smoke's answers hit it, the run uses {LONG_MAX_NEW_TOKENS} and records the "
                                     "deviation"))


def expected_steps(inputs: Path, limit: int | None) -> list[int]:
    with np.load(inputs) as z:
        n = int(len(z["steps"]))
    return list(range(min(n, limit) if limit else n))


def resume_rows(tmp: Path, steps: list[int], samples: int, max_new_tokens: int | None = None) -> tuple[list[dict], str | None]:
    """Rows of a partial reading to keep (complete lines in step order, a prefix of steps, made with max_new_tokens
    when given); the old file is set aside. Returns (rows, set-aside path)."""
    tmp = Path(tmp)
    if not tmp.exists():
        return [], None
    keep = []
    with open(tmp, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                break
            if len(keep) >= len(steps) or r.get("step") != steps[len(keep)] or not row_ok(r, samples, max_new_tokens):
                break
            keep.append(r)
    return keep, str(set_aside(tmp))


def worker_env(gpu) -> dict:
    env = dict(os.environ)
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", IIF_BASE=C.IIF_BASE,
               PYTHONPATH=str(C.EMWT), PYTHONUNBUFFERED="1")
    env.setdefault("HF_HOME", str(Path(C.IIF_BASE) / "hf_home"))
    return env


def set_max_new_tokens(n: int) -> dict:
    """ao_diff_read.GEN["max_new_tokens"] = n in this process; ao_cleanbatch_read.GEN is the same dict (checked), so
    its per-row max-token hit rule follows. Returns the generation settings now in force."""
    AD, ACR = ao_diff(), ao_cleanbatch()
    if ACR.GEN is not AD.GEN:
        raise SystemExit("ao_cleanbatch_read.GEN is not ao_diff_read.GEN: the length would not reach its hit rule")
    if int(n) < 1:
        raise SystemExit(f"max_new_tokens {n} < 1")
    AD.GEN["max_new_tokens"] = int(n)
    return dict(AD.GEN)


def load_sampler(ao: str, max_new_tokens: int = TOOLKIT_MAX_NEW_TOKENS, question: str = C.QUESTION_BEHAVIOURAL,
                 device: str = "cuda:0"):
    """(sample(V [n, K, D], seed) -> (texts, per-row max-token hits), info): ao_diff_read.load_aos with the one AO,
    its oracle prompt (one ' ?' per layer block) and ao_cleanbatch_read.sample_call, generating up to max_new_tokens
    tokens. GPU."""
    AD, ACR = ao_diff(), ao_cleanbatch()
    if tuple(AD.AOS[ao]["layers"]) != AO_LAYERS[ao] or tuple(ACR.AO_ORDER) != AO_ORDER:
        raise SystemExit("ao_diff_read.AOS layers or ao_cleanbatch_read.AO_ORDER changed; check AO_LAYERS, AO_ORDER")
    if AD.GEN["max_new_tokens"] != TOOLKIT_MAX_NEW_TOKENS:
        raise SystemExit(f"ao_diff_read.GEN max_new_tokens is {AD.GEN['max_new_tokens']}, not the toolkit's "
                         f"{TOOLKIT_MAX_NEW_TOKENS}; check TOOLKIT_MAX_NEW_TOKENS")
    gen = set_max_new_tokens(max_new_tokens)
    t0 = time.time()
    pm, tok, cfgs = AD.load_aos(device, [ao])
    pm.set_adapter(ao)
    ids, pos, text = AD.oracle_prompt_ids(tok, AD.AOS[ao]["layers"], 1, question)

    def sample(V, seed):
        return ACR.sample_call(pm, tok, ids, pos, V, seed, device)

    a = AD.AOS[ao]
    info = dict(ao=ao, ao_repo=a["repo"], ao_revision=a["revision"], ao_layers=list(a["layers"]), ao_config=cfgs[ao],
                oracle_prompt=text, placeholder_positions=list(pos), prompt_tokens=len(ids), hook_layer=AD.HOOK_LAYER,
                steering_coefficient=AD.COEF, generation=gen, model=AD.MODEL_ID,
                dtype="AO model bf16 (sdpa); injection arithmetic fp32 (ao_diff_read.inject_)",
                loading="ao_diff_read.load_aos(device, [ao]); generate = ao_cleanbatch_read.sample_call",
                load_s=round(time.time() - t0, 1), env=ACR.env_info(device))
    log(f"AO {ao} ({a['repo']} @ {a['revision'][:8]}) loaded in {info['load_s']} s; layers {list(a['layers'])}, "
        f"prompt {len(ids)} tokens, placeholders {list(pos)}, max_new_tokens {gen['max_new_tokens']}")
    return sample, info


def peak_gib():
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return round(torch.cuda.max_memory_allocated() / 2**30, 1)
    except Exception:  # noqa: BLE001
        return None
    return None


def cmd_read_worker(out: Path, ao: str, inp: str, samples: int, limit: int | None, inputs_dir: Path | None, *,
                    max_new_tokens: int = TOOLKIT_MAX_NEW_TOKENS, make_sampler=None) -> int:
    """The reading of one variant in this process (the GPU side of read; make_sampler(ao, max_new_tokens) replaces
    load_sampler in tests). Resumes from a partial reading made with the same max_new_tokens; writes the reading last
    (it marks completion)."""
    t0 = time.time()
    out = Path(out)
    v = variant(ao, inp)
    mnt = int(max_new_tokens)
    src = inputs_path(inputs_dir or out, inp)
    data = load_inputs(src, kind=inp)
    lidx = [data["layers"].index(L) for L in AO_LAYERS[ao]]
    steps = list(range(min(len(data["steps"]), limit) if limit else len(data["steps"])))
    final = reading_path(out, v)
    tmp = final.with_name(final.name + ".tmp")
    if final.exists():
        raise SystemExit(f"{final} exists; refusing to overwrite")
    kept, aside = resume_rows(tmp, steps, samples, mnt)
    out.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    if aside:
        log(f"resuming: {len(kept)} rows made with max_new_tokens {mnt} kept from the partial reading (set aside as "
            f"{aside})")
    sample, info = (make_sampler or load_sampler)(ao, mnt)
    settings = dict(
        reader=f"{v}_generic_behavioural_correct", question=C.QUESTION_BEHAVIOURAL, question_key="behavioural",
        variant=v, input=inp, inputs=str(src), inputs_order_hash=data["meta"]["order_hash"],
        inputs_sign=data["meta"]["sign"], inputs_kind=data["meta"]["kind"], input_layers=list(AO_LAYERS[ao]),
        input_rule=("the step's bundle row(s) at the AO's layers, fp32 as stored in the inputs file (no further centring "
                    "or rescaling: inject_ normalises each row), the same rows for every answer of the call"),
        n_samples=samples, n_steps=len(steps), limit=limit, seed0=SEED0,
        seed_rule="torch.manual_seed(SEED0 + 1000 x step + AO index (aopaper 0, ao3layer 1)) before each generate call "
                  "(one call per step, n_samples rows); the readings of one AO with the three inputs share the seed",
        **info, **mnt_settings(mnt), run=str(out), host=socket.gethostname(), pid=os.getpid(),
        code=os.path.abspath(__file__), plan="module docstring of main_run/beh_reader/ao_heldout.py",
        resumed_rows=len(kept), status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(settings_path(final), settings)
    X, n_docs = data["inputs"], data["n_docs"]
    done = {r["step"] for r in kept}
    todo = [s for s in steps if s not in done]
    t1, n_call, call_s = time.time(), 0, []
    with open(tmp, "a", encoding="utf-8") as f:
        for s in todo:
            V = np.repeat(np.ascontiguousarray(X[s, 0][lidx])[None], samples, axis=0)
            sd = seed_of(s, ao)
            tc = time.time()
            texts, hits = sample(V, sd)
            call_s.append(time.time() - tc)
            if len(texts) != samples:
                raise RuntimeError(f"step {s}: {len(texts)} answers, expected {samples}")
            f.write(json.dumps(reading_row(s, n_docs[s, 0], texts, hits, sd, mnt)) + "\n")
            f.flush()
            n_call += 1
            if n_call % 25 == 0 or n_call == len(todo):
                el = time.time() - t1
                log(f"{v}: {len(done) + n_call}/{len(steps)} steps, {np.mean(call_s):.2f} s/call, eta "
                    f"{(len(todo) - n_call) * el / n_call / 60:.1f} min; peak {peak_gib()} GiB")
    state = rows_state(tmp, steps, samples) if settings_path(final).exists() else "no settings file"
    rows = read_rows(tmp)
    texts = [t for r in rows for t in r["inputs"]["bundle"]["texts"]]
    n_hit = sum(r["inputs"]["bundle"]["n_hit_max_new_tokens"] for r in rows)
    checks = dict(n_rows=len(rows), n_answers=len(texts), n_empty_answers=sum(not t.strip() for t in texts),
                  n_hit_max_new_tokens=n_hit, share_hit_max_new_tokens=(n_hit / len(texts)) if texts else None,
                  pass_complete=state == "complete", pass_one_length=all(row_mnt(r) == mnt for r in rows))
    checks["pass_no_empty"] = checks["n_empty_answers"] == 0
    settings.update(status="finished" if state == "complete" else f"incomplete: {state}", checks=checks,
                    timings=dict(total_s=round(time.time() - t0, 1), read_s=round(time.time() - t1, 1),
                                 s_per_call=(round(float(np.mean(call_s)), 3) if call_s else None),
                                 calls=n_call, peak_gib=peak_gib()),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(settings_path(final), settings)
    if state != "complete":
        log(f"{v}: reading incomplete ({state}); {tmp} kept")
        return 1
    os.replace(tmp, final)
    share = checks["share_hit_max_new_tokens"]
    log(f"{v}: wrote {final} ({checks['n_answers']} answers, {checks['n_empty_answers']} empty, "
        f"{n_hit} ({share:.1%}) at the token limit {mnt}) in {settings['timings']['total_s']} s")
    if checks["n_empty_answers"]:
        log(f"WARNING: {checks['n_empty_answers']} empty answers (expected 0); the judge says NO to them")
    return 0


def write_marker(out: Path, v: str, what: str, **info) -> None:
    atomic_json(marker(out, v, what), dict(info, pid=os.getpid(), host=socket.gethostname(),
                                           time=time.strftime("%Y-%m-%d %H:%M:%S")))


def cmd_read(out: Path, ao: str, inp: str, gpu: str, samples: int, limit: int | None, inputs_dir: Path | None,
             hang_s: float, retry_hung: int, kill_grace_s: float, *, max_new_tokens: int = TOOLKIT_MAX_NEW_TOKENS,
             worker_cmd=None) -> int:
    """One variant's reading in a watched worker process (eval_read.run_watched), one retry after a hang. One read of
    a variant at a time: markers/read_<variant>.lock (flock) is held for the call; a second call returns 1."""
    out = Path(out)
    v = variant(ao, inp)
    if limit and os.path.realpath(out) == os.path.realpath(OUT):
        raise SystemExit(f"--limit writes a partial reading; give --out (a scratch folder), not {OUT}")
    if int(max_new_tokens) < 1:
        raise SystemExit(f"--max-new-tokens {max_new_tokens} < 1")
    (out / "markers").mkdir(parents=True, exist_ok=True)
    with open(out / "markers" / f"read_{v}.lock", "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"{v}: another read of this variant holds {lk.name}; not started (two writers would mix the "
                "partial reading)")
            return 1
        return _read_locked(out, ao, inp, gpu, samples, limit, inputs_dir, hang_s, retry_hung, kill_grace_s,
                            int(max_new_tokens), worker_cmd=worker_cmd)


def _read_locked(out: Path, ao: str, inp: str, gpu: str, samples: int, limit: int | None, inputs_dir: Path | None,
                 hang_s: float, retry_hung: int, kill_grace_s: float, mnt: int, *, worker_cmd=None) -> int:
    v = variant(ao, inp)
    src = inputs_path(inputs_dir or out, inp)
    if not src.exists():
        raise SystemExit(f"{src} missing; run build first")
    steps = expected_steps(src, limit)
    final = reading_path(out, v)
    (out / "markers").mkdir(parents=True, exist_ok=True)
    (out / "read_logs").mkdir(parents=True, exist_ok=True)
    state = reading_state(final, steps, samples)
    if state == "complete":
        have = settings_mnt(final)
        if have != mnt:
            log(f"{v}: {final} is complete but made with max_new_tokens {have}, not {mnt}; refusing (set it aside to "
                "read again with this length)")
            write_marker(out, v, "FAILED", rc=1, why=f"complete reading made with max_new_tokens {have}, asked {mnt}")
            return 1
        log(f"{v}: {final} is complete (max_new_tokens {mnt}); left as it is")
        if not marker(out, v, "DONE").exists():
            write_marker(out, v, "DONE", note="reading already complete")
        return 0
    if state != "missing":
        log(f"{v}: {final} exists but is not complete ({state}); refusing to overwrite")
        write_marker(out, v, "FAILED", rc=1, why=f"existing reading not complete: {state}")
        return 1
    set_aside(marker(out, v, "FAILED"))
    cmd = worker_cmd or [sys.executable, os.path.abspath(__file__), "read-worker", "--out", str(out), "--ao", ao,
                         "--input", inp, "--samples", str(samples), "--max-new-tokens", str(mnt),
                         "--inputs-dir", str(inputs_dir or out)] + (["--limit", str(limit)] if limit else [])
    write_marker(out, v, "START", cmd=cmd, gpu=str(gpu), samples=samples, limit=limit, steps=len(steps),
                 max_new_tokens=mnt)
    R = beh_judges().R
    tmp = final.with_name(final.name + ".tmp")
    rc, hung, t0 = None, None, time.time()
    for attempt in range(1 + max(0, int(retry_hung))):
        lp = out / "read_logs" / f"read_{v}_{stamp()}{'_retry' + str(attempt) if attempt else ''}.log"
        rc, hung = R.run_watched(cmd, lp, tmp, hang_s=hang_s, grace_s=kill_grace_s, cwd=str(C.REPO),
                                 env=worker_env(gpu), label=f"read {v}")
        if not hung or hung.startswith("unkillable"):
            break
        log(f"{v}: {hung}; {'starting once more (resumes from the partial reading)' if attempt < retry_hung else 'no retry left'}")
    state = reading_state(final, steps, samples)
    took = round(time.time() - t0, 1)
    if rc == 0 and state == "complete":
        write_marker(out, v, "DONE", seconds=took, reading=str(final), max_new_tokens=mnt)
        log(f"{v}: done in {took / 60:.1f} min")
        return 0
    write_marker(out, v, "FAILED", rc=rc, hung=hung, state=state, seconds=took)
    log(f"{v}: FAILED (exit {rc}, {hung or 'not hung'}, reading {state})")
    return 1


# ---------------------------------------------------------------- generation length
def cmd_tokens(out: Path, n_full: int = C.N_TEST_BUNDLES) -> int:
    """Per variant the answers at the token limit (count and share), seconds per generate call and the minutes a full
    reading of n_full steps would take at that speed; then the generation-length rule over all answers read.
    Reads a finished reading, or the partial .tmp of an unfinished one; writes nothing.
    Exit 1 when no answer was read."""
    out = Path(out)
    tot_a = tot_h = 0
    lengths: set = set()
    for v in VARIANTS:
        rp = reading_path(out, v)
        src = rp if rp.exists() else rp.with_name(rp.name + ".tmp")
        if not src.exists():
            print(f"{v}: no reading")
            continue
        rows = read_rows_partial(src)
        n_a = sum(len(r["inputs"]["bundle"]["texts"]) for r in rows)
        n_h = sum(int(r["inputs"]["bundle"].get("n_hit_max_new_tokens", 0)) for r in rows)
        mnts = sorted({row_mnt(r) for r in rows})
        lengths.update(mnts)
        tot_a, tot_h = tot_a + n_a, tot_h + n_h
        spc = (read_settings(rp).get("timings") or {}).get("s_per_call")
        proj = f"{spc:.2f} s per call -> about {spc * n_full / 60:.0f} min for {n_full} steps" if spc is not None else \
            "no timing yet"
        print(f"{v}: {n_h}/{n_a} answers ({(n_h / n_a) if n_a else 0:.1%}) hit max_new_tokens "
              f"{'/'.join(map(str, mnts)) or '-'}; {len(rows)} steps; {proj}" + ("" if src == rp else " (partial reading)"))
    if not tot_a:
        print("no answers read")
        return 1
    share = tot_h / tot_a
    if lengths == {TOOLKIT_MAX_NEW_TOKENS}:
        if share > HIT_SHARE_MAX:
            print(f"RULE: {tot_h}/{tot_a} answers ({share:.1%}) hit the toolkit's {TOOLKIT_MAX_NEW_TOKENS} tokens, more "
                  f"than {HIT_SHARE_MAX:.0%}: read with --max-new-tokens {LONG_MAX_NEW_TOKENS} and record the deviation "
                  f"from the toolkit default (the settings files record it)")
        else:
            print(f"RULE: {tot_h}/{tot_a} answers ({share:.1%}) hit the toolkit's {TOOLKIT_MAX_NEW_TOKENS} tokens, at "
                  f"most {HIT_SHARE_MAX:.0%}: keep --max-new-tokens {TOOLKIT_MAX_NEW_TOKENS}")
    else:
        print(f"readings made with max_new_tokens {sorted(lengths)}: {tot_h}/{tot_a} answers ({share:.1%}) at their "
              f"limit (the {HIT_SHARE_MAX:.0%} rule applies to a smoke at {TOOLKIT_MAX_NEW_TOKENS})")
    return 0


# ---------------------------------------------------------------- judge
@contextlib.contextmanager
def repointed(out: Path, inputs_dir: Path | None = None):
    """beh_judges and common pointed at this experiment: conditions = the six variants, inputs of a variant = its
    inputs file, reading = readings_<variant>.jsonl, eval folder and read logs in --out."""
    B = beh_judges()
    out = Path(out)
    d = Path(inputs_dir or out)
    with B.patched(C, INPUTS={v: inputs_path(d, split_variant(v)[1]) for v in VARIANTS}, EVAL=out,
                   READ_LOGS=out / "read_logs"), \
            B.patched(B, CONDS=VARIANTS, reading_path=lambda key, cond: reading_path(out, cond)):
        yield B


def answer_count(path: Path) -> int:
    return sum(len(rec["texts"]) for r in C.read_jsonl(path) for rec in r["inputs"].values())


def readings_not_ready(out: Path, inputs_dir: Path | None = None) -> list[str]:
    """Why the six readings are not ready to judge (empty when they are): a reading missing; its settings not status
    finished, or naming another variant, input kind or inputs file than inputs_<input>_correct.npz in inputs_dir
    (default out), or an order hash other than that file's; the six made with different max_new_tokens, answers per
    bundle or step counts. All or nothing: beh_judges alone skips an unfit reading and judges the other five."""
    d = Path(inputs_dir or out)
    bad: list = []
    seen: dict = {"max_new_tokens": {}, "answers per bundle": {}, "steps": {}}
    for v in VARIANTS:
        inp = split_variant(v)[1]
        rp = reading_path(out, v)
        if not rp.exists():
            bad.append(f"{v}: {rp.name} missing")
            continue
        s = read_settings(rp)
        if s.get("status") != "finished":
            bad.append(f"{v}: settings status {s.get('status')!r}")
            continue
        want = inputs_path(d, inp)
        why = [f"{k} {s.get(k)!r}, expected {x!r}" for k, x in (("variant", v), ("input", inp), ("inputs_kind", inp))
               if s.get(k) != x]
        if not s.get("inputs") or os.path.realpath(str(s["inputs"])) != os.path.realpath(want):
            why.append(f"inputs {s.get('inputs')}, expected {want}")
        elif not want.exists():
            why.append(f"{want} missing")
        elif s.get("inputs_order_hash") != npz_meta(want).get("order_hash"):
            why.append(f"inputs_order_hash {s.get('inputs_order_hash')!r} is not the order hash of {want.name}")
        if why:
            bad.append(f"{v}: " + "; ".join(why))
            continue
        seen["max_new_tokens"][v] = settings_mnt(rp)
        seen["answers per bundle"][v] = s.get("n_samples")
        seen["steps"][v] = s.get("n_steps")
    for name, by in seen.items():
        if len(set(by.values())) > 1:
            bad.append(f"the readings differ in {name}: {by}")
    return bad


def cmd_judge(out: Path, cap: float, mode: str, workers: int, dry: bool, inputs_dir: Path | None = None) -> int:
    out = Path(out)
    RP = rare_perdoc()
    not_ready = readings_not_ready(out, inputs_dir)
    if not_ready and not dry:
        log(f"judge refused, nothing sent: the six readings are not all ready ({'; '.join(not_ready)})")
        return 1
    with repointed(out, inputs_dir) as B:
        unit, nreq = RP.truth_unit_cost(B)
        if dry:
            ctx = B.Context()
            reqs, todo, skipped = B.judge_todo("truth", (JUDGE_KEY,), ctx)
            cache = B.J.load_cache("truth")
            new = sum(s not in cache for s in reqs)
            for s in skipped:
                log(f"not judged: {s}")
            for s in not_ready:
                log(f"not ready (a real judge refuses): {s}")
            log(f"dry run: {len(todo)} readings, {sum(len(k) for _, k in todo)} answers, {len(reqs)} distinct "
                f"requests ({new} new, about {unit * new:.2f} USD batch at {unit:.2e} USD per answer); nothing sent")
            if reqs:
                p = next(iter(reqs.values()))
                log(f"first request: model {p['model']}, max_tokens {p['max_tokens']}, prompt starts "
                    f"{p['messages'][0]['content'][:300]!r}")
            return 0
        n_ans = sum(answer_count(reading_path(out, v)) for v in VARIANTS)
        led0 = B.J.ledger_total()
        log(f"{n_ans} answers in {len(VARIANTS)} readings: about {unit * n_ans:.2f} USD batch at {unit:.2e} USD per "
            f"answer ({nreq} truth batch requests in the ledger; cached answers cost nothing)")
        log(f"ledger {led0:.3f} USD at start; cap {cap} USD -> --max-cost {led0 + cap:.3f}")
        rc = B.cmd_judge("truth", (JUDGE_KEY,), max_cost=led0 + cap, mode=mode, workers=workers)
        led1 = B.J.ledger_total()
    files = {}
    for v in VARIANTS:
        rp, jp = reading_path(out, v), judged_path(out, v)
        n_ans = answer_count(rp) if rp.exists() else None
        n_jud = len(C.read_jsonl(jp)) if jp.exists() else None
        files[v] = dict(reading=str(rp), judged=str(jp), answers=n_ans, judged_rows=n_jud,
                        pass_judged=jp.exists(), pass_rows_eq_answers=n_ans is not None and n_jud == n_ans)
    rep = dict(ledger_start=led0, ledger_end=led1, increase=led1 - led0, cap_usd=cap,
               pass_cap=led1 - led0 <= cap + 1e-6, files=files, beh_judges_exit=rc,
               estimate_usd_batch=unit * sum(f["answers"] or 0 for f in files.values()), unit_usd=unit,
               note="the ledger total counts every judge process; another judge running at the same time adds to it",
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = out / "JUDGE.json"
    if p.exists():
        p = out / f"JUDGE_{stamp()}.json"
    atomic_json(p, rep)
    bad = [v for v, f in files.items() if not (f["pass_judged"] and f["pass_rows_eq_answers"])]
    log(f"ledger {led0:.3f} -> {led1:.3f} USD (+{led1 - led0:.3f}, cap {cap}); report {p}"
        + (f"; not judged or row count off: {bad}" if bad else ""))
    return rc or (1 if bad or not rep["pass_cap"] else 0)


# ---------------------------------------------------------------- summary
def rate_of(vs: list) -> float | None:
    y = sum(v == "YES" for v in vs)
    n = sum(v in ("YES", "NO") for v in vs)
    return (y / n) if n else None


def bundle_rates(judged: Path, ids: list, max_index: int | None = None) -> dict:
    """{bundle_id: YES rate} of a judged file (step -> bundle through ids; answers with index >= max_index left out)."""
    vd = rare_perdoc().verdicts(judged, max_index)
    return {ids[s]: rate_of(v) for s, v in vd.items()}


def hyper_tail(N: int, K: int, n: int, k: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(population N, K marked, n drawn)."""
    if n > N or K > N:
        return float("nan")
    tot = math.comb(N, n)
    return sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(k, min(n, K) + 1)) / tot


def group_table(rows: list[dict], key: str | None, cols: list[str], order: list[str] | None = None) -> dict:
    """{group: {col: boot_mean over bundles}} plus 'all' (key None: only 'all')."""
    B = beh_judges()
    groups = (order or sorted({r[key] for r in rows})) if key else []
    out = {}
    for g in groups + ["all"]:
        sub = rows if g == "all" else [r for r in rows if r[key] == g]
        out[g] = {c: B.boot_mean([r[c] for r in sub if r.get(c) is not None]) for c in cols}
    return out


def cell_means(rows: list[dict], cols: list[str]) -> dict:
    """{'family/evidence': {col: mean, n}} over bundles."""
    out: dict = {}
    for k in sorted({(r["family"], r["evidence"]) for r in rows}):
        sub = [r for r in rows if (r["family"], r["evidence"]) == k]
        e = {}
        for c in cols:
            x = [r[c] for r in sub if r.get(c) is not None]
            e[c] = float(np.mean(x)) if x else None
        e["n"] = len(sub)
        out["/".join(k)] = e
    return out


def candidates(rows: list[dict], side: str) -> list[dict]:
    full = [r for r in rows if all(r.get(c) is not None for c in ("reader",) + VARIANTS)]
    if side == "reader_only":
        return [r for r in full if r["reader"] >= YES_HI - EPS and all(r[v] <= NO_LO + EPS for v in VARIANTS)]
    return [r for r in full if any(r[v] >= YES_HI - EPS for v in VARIANTS) and r["reader"] <= NO_LO + EPS]


def concentration(sel: list[dict], base: list[dict]) -> dict:
    """Counts of the candidates by family and evidence against the counts among the base bundles."""
    out = {}
    for kind in ("family", "evidence"):
        cnt, bc = Counter(r[kind] for r in sel), Counter(r[kind] for r in base)
        out[kind] = {g: dict(k=cnt[g], share=(cnt[g] / len(sel)) if sel else None, base_k=bc[g],
                             base_share=bc[g] / len(base) if base else None,
                             enrichment=((cnt[g] / len(sel)) / (bc[g] / len(base))) if sel and bc[g] else None,
                             p_hypergeom=hyper_tail(len(base), bc[g], len(sel), cnt[g]) if cnt[g] else None)
                     for g in sorted(bc)}
    return out


def verdict(sel: list[dict], base: list[dict], side: str) -> dict:
    """The verdict rule on one candidate list: a family or evidence kind counts as a concentration only when
    it holds more than half of the candidates and its hypergeometric tail p < P_MAX; fewer than MIN_CANDIDATES
    candidates -> no claim."""
    who = "the reader" if side == "reader_only" else "the AO"
    n, N = len(sel), len(base)
    conc = concentration(sel, base)
    over = []
    for kind, d in conc.items():
        for g, e in d.items():
            if n and 2 * e["k"] > n:
                p = e["p_hypergeom"]
                over.append(dict(kind=kind, name=g, **e, counts=p is not None and p < P_MAX))
    counted = [o for o in over if o["counts"]]

    def desc(o):
        return (f"{o['kind']} {o['name']} ({o['k']}/{n}; {o['base_k']}/{N} of all bundles, hypergeometric p "
                f"{o['p_hypergeom']:.2g})")

    if n < MIN_CANDIDATES:
        res, text = "too_few", f"{n} candidate bundles (fewer than {MIN_CANDIDATES}): no claim"
    elif counted:
        res = "concentrated"
        text = (f"{n} candidate bundles; more than half in " + " and ".join(desc(o) for o in counted)
                + f": that kind is caught only by {who}")
    elif over:
        res = "scattered"
        text = (f"{n} candidate bundles; more than half in " + " and ".join(desc(o) for o in over)
                + f", but p >= {P_MAX}: the base rate explains it; the difference is not systematic")
    else:
        res, text = "scattered", (f"{n} candidate bundles, no family or evidence kind holds more than half: the "
                                  f"difference is not systematic")
    return dict(result=res, text=text, n=n, n_base=N, over_half=over, counted=counted,
                rule=(f"a family or evidence kind counts as a concentration only when it holds more than half of the "
                      f"candidates and its hypergeometric tail p (the candidates drawn from the {N} bundles with all "
                      f"seven rates) < {P_MAX} -> 'that kind is caught only by {who}'; >= {MIN_CANDIDATES} candidates "
                      f"and no such kind -> not systematic; < {MIN_CANDIDATES} candidates -> no claim (fixed before "
                      f"the run)"))


def cand_row(r: dict) -> dict:
    return {k: r[k] for k in ("bundle_id", "family", "evidence", "behaviour_text", "reader", "reader_first10")
            + VARIANTS}


INPUT_PAIRS = (("grad", "act"), ("grad", "act_centred"), ("act_centred", "act"))


def input_pairs(rows: list[dict], families: list[str]) -> dict:
    """Per AO and input pair (x, y): per family the mean of each input's reading and their paired difference x - y
    (bundle bootstrap), and the bundles where one input gives >= 0.5 and the other <= 0.1 (<x>_only, <y>_only)."""
    B = beh_judges()
    out = {}
    for ao in AO_NAMES:
        res = {}
        for x, y in INPUT_PAIRS:
            cx, cy = f"{ao}_{x}", f"{ao}_{y}"
            pairs = [r for r in rows if r.get(cx) is not None and r.get(cy) is not None]
            fam = {}
            for f in families + ["all"]:
                sub = pairs if f == "all" else [r for r in pairs if r["family"] == f]
                fam[f] = {x: B.boot_mean([r[cx] for r in sub])["mean"], y: B.boot_mean([r[cy] for r in sub])["mean"],
                          "diff": B.boot_mean([r[cx] - r[cy] for r in sub])}
            sides = {}
            for name, hi, lo in ((f"{x}_only", cx, cy), (f"{y}_only", cy, cx)):
                sel = [r for r in pairs if r[hi] >= YES_HI - EPS and r[lo] <= NO_LO + EPS]
                sides[name] = dict(n=len(sel), by_family=dict(Counter(r["family"] for r in sel)),
                                   by_evidence=dict(Counter(r["evidence"] for r in sel)),
                                   bundles=[dict(bundle_id=r["bundle_id"], family=r["family"], evidence=r["evidence"],
                                                 behaviour_text=r["behaviour_text"], **{x: r[cx], y: r[cy]})
                                            for r in sel])
            res[f"{x}_vs_{y}"] = dict(x=x, y=y, n_bundles=len(pairs), family=fam, **sides)
        out[ao] = res
    return out


def reading_integrity(out: Path, n_steps: dict | None = None) -> dict:
    """Per variant: steps (= the inputs file's steps, n_steps[v]), answers (= steps x SAMPLES), empty
    answers (0), max-token hits and the length of the reading, judged rows (= answers); recorded, not enforced (a
    --limit smoke has fewer steps)."""
    n_steps = n_steps or {}
    res = {}
    for v in VARIANTS:
        rp, jp = reading_path(out, v), judged_path(out, v)
        if not rp.exists():
            res[v] = None
            continue
        rows = C.read_jsonl(rp)
        texts = [t for r in rows for rec in r["inputs"].values() for t in rec["texts"]]
        n_j = len(C.read_jsonl(jp)) if jp.exists() else None
        empty = sum(not t.strip() for t in texts)
        hits = sum(rec.get("n_hit_max_new_tokens", 0) for r in rows for rec in r["inputs"].values())
        res[v] = dict(steps=len(rows), expected_steps=n_steps.get(v), answers=len(texts), empty=empty,
                      at_token_limit=hits, share_at_token_limit=(hits / len(texts)) if texts else None,
                      max_new_tokens=settings_mnt(rp), judged_rows=n_j, pass_steps=n_steps.get(v) == len(rows),
                      pass_answers=len(texts) == len(rows) * SAMPLES, pass_no_empty=empty == 0,
                      pass_judged_rows_eq_answers=n_j == len(texts))
    return res


def cmd_summary(out: Path, inputs_dir: Path | None = None) -> int:
    out = Path(out)
    d = Path(inputs_dir or out)
    bundles = {b["bundle_id"]: b for b in C.load_bundles()}
    missing: list[str] = []
    ids_c = npz_meta(C.INPUTS["correct"])["bundle_ids"]
    rates: dict = {}
    rj = reader_judged()
    if rj.exists():
        rates["reader"] = bundle_rates(rj, ids_c)
        rates["reader_first10"] = bundle_rates(rj, ids_c, SAMPLES)
    else:
        missing.append(str(rj))
    ids_v = {}
    for v in VARIANTS:
        ip, jp = inputs_path(d, split_variant(v)[1]), judged_path(out, v)
        if not ip.exists():
            missing.append(str(ip))
            continue
        ids_v[v] = npz_meta(ip)["bundle_ids"]
        if not jp.exists():
            missing.append(str(jp))
            continue
        rates[v] = bundle_rates(jp, ids_v[v])
    order_ok = all(ids == ids_c for ids in ids_v.values())
    cols = ["reader", "reader_first10"] + list(VARIANTS)
    rows = []
    for bid in ids_c:
        b = bundles[bid]
        r = dict(bundle_id=bid, family=b["family"], evidence=b["evidence"], behaviour_text=b["behaviour_text"],
                 n_shows=int(b["n_shows"]), n_docs=int(b["n_docs"]))
        for c in cols:
            r[c] = rates.get(c, {}).get(bid)
        ao_vals = [r[v] for v in VARIANTS if r[v] is not None]
        r["ao_max"] = max(ao_vals) if len(ao_vals) == len(VARIANTS) else None
        rows.append(r)
    fam_order = [f for f in C.TEST_FAMILY_COUNTS if any(r["family"] == f for r in rows)] + sorted(
        {r["family"] for r in rows} - set(C.TEST_FAMILY_COUNTS))
    ev_order = [e for e in C.EVIDENCE if any(r["evidence"] == e for r in rows)]
    present = [c for c in cols if c in rates]
    base = [r for r in rows if all(r.get(c) is not None for c in ("reader",) + VARIANTS)]
    cand = {}
    if all(v in rates for v in VARIANTS) and "reader" in rates:
        for side in ("reader_only", "ao_only"):
            sel = candidates(rows, side)
            cand[side] = dict(verdict=verdict(sel, base, side), concentration=concentration(sel, base),
                              bundles=[cand_row(r) for r in sel])
        # robustness: the same rule with the reader's first 10 answers (the AO variants have 10)
        r10 = [dict(r, reader=r["reader_first10"]) for r in base]
        cand["robustness_reader_first10"] = {side: len(candidates(r10, side)) for side in ("reader_only", "ao_only")}
    integ = reading_integrity(out, {v: len(ids) for v, ids in ids_v.items()})
    lengths = {v: e["max_new_tokens"] for v, e in integ.items() if e is not None}
    res = dict(family=group_table(rows, "family", present, fam_order),
               evidence=group_table(rows, "evidence", present, ev_order),
               family_evidence=cell_means(rows, present),
               candidates=cand, input_pairs=input_pairs(rows, fam_order) if any(v in rates for v in VARIANTS) else {})
    summary = dict(
        question=C.QUESTION_BEHAVIOURAL, reader=READER, variants=list(VARIANTS),
        samples=dict(reader=C.SAMPLES, reader_first10=SAMPLES, **{v: SAMPLES for v in VARIANTS}),
        rule=dict(candidate_yes_min=YES_HI, candidate_no_max=NO_LO, min_candidates=MIN_CANDIDATES, p_max=P_MAX,
                  verdict=("a family or evidence kind counts only when it holds more than half of the candidates and "
                           f"its hypergeometric tail p < {P_MAX}; fewer than {MIN_CANDIDATES} candidates -> no claim"),
                  rate="per bundle YES / (YES + NO), REFUSED left out; tables = mean over bundles with bundle "
                       "bootstrap 95% CI (beh_judges.boot_mean, 2,000 resamples, seed 0)"),
        judge="beh_judges truth judge (judge_dolphin_answers.JUDGE_PROMPT_BINARY_WITH_TRUTH, labels.jsonl ground_truth "
              "of the step's bundle, Haiku)",
        max_new_tokens=lengths, same_max_new_tokens=len(set(lengths.values())) <= 1,
        toolkit_max_new_tokens=TOOLKIT_MAX_NEW_TOKENS,
        n_bundles=len(rows), n_bundles_all_seven=len(base), order_matches_inputs_correct=order_ok,
        families=fam_order, evidence_kinds=ev_order, results=res, integrity=integ,
        per_bundle=rows, missing=missing, written=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(out / "summary.json", summary)
    print_summary(summary)
    log(f"wrote {out / 'summary.json'}; missing {missing or 'none'}")
    return 1 if missing or not order_ok else 0


def _f(x) -> str:
    return "   -  " if x is None else f"{x:6.3f}"


HEAD = {"reader": "reader", "reader_first10": "rdr10", "ao3layer_grad": "AO3 g", "ao3layer_act": "AO3 a",
        "ao3layer_act_centred": "AO3 c", "aopaper_grad": "AOp g", "aopaper_act": "AOp a", "aopaper_act_centred": "AOp c"}


def print_summary(s: dict) -> None:
    res = s["results"]
    cols = [c for c in ["reader", "reader_first10"] + s["variants"] if c in res["family"].get("all", {})]
    print("columns: reader = rl_beh_c040 (20 answers), rdr10 = its first 10; AO3 = three-layer AO, AOp = paper-recipe "
          "AO; g = gradient rows, a = activation rows, c = activation rows centred by the train mean")
    for name in ("family", "evidence"):
        print(f"\nYES rate by {name} (mean over bundles; n bundles)")
        print(f"{name:12s} " + " ".join(f"{HEAD[c]:>6s}" for c in cols) + "      n")
        for g, e in res[name].items():
            n = max((e[c]["n"] for c in cols), default=0)
            print(f"{g:12s} " + " ".join(_f(e[c]["mean"]) for c in cols) + f" {n:6d}")
    print("\nYES rate by family/evidence (mean over bundles; n bundles)")
    print(f"{'family/evidence':24s} " + " ".join(f"{HEAD[c]:>6s}" for c in cols) + "      n")
    for k, e in res["family_evidence"].items():
        print(f"{k:24s} " + " ".join(_f(e[c]) for c in cols) + f" {e['n']:6d}")
    for side in ("reader_only", "ao_only"):
        c = res["candidates"].get(side)
        if not c:
            continue
        v = c["verdict"]
        print(f"\n{side}: {v['text']}")
        print(f"  rule: {v['rule']}")
        for o in v["over_half"]:
            print(f"  over half: {o['kind']} {o['name']} {o['k']}/{v['n']} (all bundles {o['base_k']}/{v['n_base']}), "
                  f"enrichment {o['enrichment']:.2f}, hypergeometric p {o['p_hypergeom']:.2g} -> "
                  + (f"p < {s['rule']['p_max']}, but fewer than {s['rule']['min_candidates']} candidates: no claim"
                     if o["counts"] and v["result"] == "too_few" else
                     "counts" if o["counts"] else f"does not count (p >= {s['rule']['p_max']})"))
        for kind in ("family", "evidence"):
            print(f"  by {kind}: " + ", ".join(f"{g} {e['k']}/{e['base_k']}" for g, e in c["concentration"][kind].items()
                                              if e["k"]))
        for r in c["bundles"][:40]:
            print(f"  {r['bundle_id']:40s} {r['family']:10s} {r['evidence']:11s} reader {_f(r['reader'])} "
                  f"AO {'/'.join(_f(r[x]).strip() for x in s['variants'])}  {r['behaviour_text'][:70]}")
    print(f"\nreadings; max_new_tokens {s.get('max_new_tokens')} (toolkit {s.get('toolkit_max_new_tokens')}"
          + ("" if s.get("same_max_new_tokens", True) else "; NOT the same for every reading") + ")")
    for v, e in s.get("integrity", {}).items():
        if e is None:
            print(f"{v}: no reading")
            continue
        bad = [k[5:] for k, x in e.items() if k.startswith("pass_") and not x]
        sh = e.get("share_at_token_limit")
        print(f"{v}: {e['steps']} steps (inputs {e['expected_steps']}), {e['answers']} answers, {e['empty']} empty, "
              f"{e['at_token_limit']} ({0 if sh is None else sh:.1%}) at the token limit {e.get('max_new_tokens')}, "
              f"{e['judged_rows']} judged rows" + (f"; NOT as expected: {bad}" if bad else "; checks pass"))
    rob = res["candidates"].get("robustness_reader_first10")
    if rob:
        print(f"\nwith the reader's first 10 answers: reader_only {rob['reader_only']}, ao_only {rob['ao_only']}")
    for ao, pairs in res["input_pairs"].items():
        for name, e in pairs.items():
            x, y = e["x"], e["y"]
            print(f"\n{ao}: {x} rows against {y} rows ({e['n_bundles']} bundles); {x} only {e[x + '_only']['n']}, "
                  f"{y} only {e[y + '_only']['n']}")
            for f, z in e["family"].items():
                dd = z["diff"]
                print(f"  {f:12s} {x} {_f(z[x])} {y} {_f(z[y])} diff {_f(dd['mean'])} "
                      f"[{_f(dd['lo']).strip()}, {_f(dd['hi']).strip()}]")


# ---------------------------------------------------------------- plan
def cmd_plan(out: Path, gpu: str, samples: int, cap: float, inputs_dir: Path | None = None,
             max_new_tokens: int = TOOLKIT_MAX_NEW_TOKENS) -> int:
    out = Path(out)
    d = Path(inputs_dir or out)
    RP = rare_perdoc()
    B = beh_judges()
    n_te = sum(b["split"] == C.SPLIT_TEST for b in C.load_bundles())
    for k in INPUT_KINDS:
        p = inputs_path(d, k)
        print(f"inputs {k}: {p} ({'present' if p.exists() else 'missing: build writes it'})")
    tm = d / TRAIN_MEAN
    print(f"train mean m: {tm} ({'present' if tm.exists() else 'missing: build computes it from the train bundles'})")
    dev = mnt_settings(max_new_tokens)["deviation_from_toolkit"]
    print(f"max_new_tokens {max_new_tokens}" + (f" ({dev})" if dev else f" (the toolkit's)"))
    n_total = 0
    for v in VARIANTS:
        ao, inp = split_variant(v)
        src = inputs_path(d, inp)
        steps = expected_steps(src, None) if src.exists() else list(range(n_te))
        st = reading_state(reading_path(out, v), steps, samples)
        jd = "judged" if judged_path(out, v).exists() else "not judged"
        n_total += len(steps) * samples
        layers = AO_LAYERS[ao]
        print(f"read {v} [{st}; {jd}]: {len(steps)} steps x {samples} answers, AO {ao} layers {list(layers)}, "
              f"seed {SEED0} + 1000 x step + {AO_ORDER.index(ao)}")
        print(f"   cd {C.REPO} && PYTHONPATH={C.EMWT} {C.PY} main_run/beh_reader/ao_heldout.py read --ao {ao} "
              f"--input {inp} --gpu {gpu} --samples {samples} --max-new-tokens {max_new_tokens} --out {out}"
              + (f" --inputs-dir {d}" if inputs_dir else ""))
    unit, nreq = RP.truth_unit_cost(B)
    led = B.J.ledger_total()
    rj = reader_judged()
    print(f"reader {READER}: judged file {rj} ({'present' if rj.exists() else 'MISSING'}), not read again")
    print(f"answers to judge: {len(VARIANTS)} variants x {n_te} bundles x {samples} = {n_total}")
    print(f"judge: beh_judges.cmd_judge('truth', ['{JUDGE_KEY}'], max_cost=<ledger at start> + {cap} (now {led:.3f} + "
          f"{cap} = {led + cap:.3f} USD), mode batch), only when all {len(VARIANTS)} readings are finished; prompt "
          f"judge_dolphin_answers.JUDGE_PROMPT_BINARY_WITH_TRUTH with the bundle's labels.jsonl ground_truth; cache and "
          f"ledger in {C.JUDGE}; judged files {judged_path(out, '<variant>')}")
    print(f"cost estimate: {unit:.2e} USD per answer (truth batch rows of the ledger, {nreq} requests) x {n_total} = "
          f"{unit * n_total:.2f} USD batch, about {2 * unit * n_total:.2f} USD synchronous; cap {cap} USD")
    return 0


# ---------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {c: sub.add_parser(c) for c in ("build", "plan", "read", "read-worker", "tokens", "judge", "summary")}
    for name, p in ps.items():
        p.add_argument("--out", type=Path, default=OUT)
        if name not in ("build", "tokens"):
            p.add_argument("--inputs-dir", type=Path, default=None, help="folder of the inputs files (default --out)")
    for name in ("plan", "read", "read-worker"):
        ps[name].add_argument("--max-new-tokens", type=int, default=TOOLKIT_MAX_NEW_TOKENS,
                              help=f"answer length (default {TOOLKIT_MAX_NEW_TOKENS}, the AO toolkit's)")
    ps["plan"].add_argument("--gpu", default="<gpu>")
    ps["plan"].add_argument("--samples", type=int, default=SAMPLES)
    ps["plan"].add_argument("--cap-usd", type=float, default=CAP_USD)
    for name in ("read", "read-worker"):
        r = ps[name]
        r.add_argument("--ao", choices=AO_NAMES, required=True)
        r.add_argument("--input", choices=INPUT_KINDS, required=True)
        r.add_argument("--samples", type=int, default=SAMPLES)
        r.add_argument("--limit", type=int, default=None)
    r = ps["read"]
    r.add_argument("--gpu", required=True)
    r.add_argument("--hang-s", type=float, default=HANG_S)
    r.add_argument("--retry-hung", type=int, default=1)
    r.add_argument("--kill-grace-s", type=float, default=KILL_GRACE_S)
    ps["tokens"].add_argument("--n-full", type=int, default=C.N_TEST_BUNDLES, help="steps of a full reading")
    j = ps["judge"]
    j.add_argument("--cap-usd", type=float, default=CAP_USD, help="USD this call may add to the ledger")
    j.add_argument("--mode", choices=("batch", "sync"), default="batch")
    j.add_argument("--workers", type=int, default=48)
    j.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        return cmd_build(a.out)
    if a.cmd == "plan":
        return cmd_plan(a.out, a.gpu, a.samples, a.cap_usd, a.inputs_dir, a.max_new_tokens)
    if a.cmd == "read":
        return cmd_read(a.out, a.ao, a.input, a.gpu, a.samples, a.limit, a.inputs_dir, a.hang_s, a.retry_hung,
                        a.kill_grace_s, max_new_tokens=a.max_new_tokens)
    if a.cmd == "read-worker":
        return cmd_read_worker(a.out, a.ao, a.input, a.samples, a.limit, a.inputs_dir, max_new_tokens=a.max_new_tokens)
    if a.cmd == "tokens":
        return cmd_tokens(a.out, a.n_full)
    if a.cmd == "judge":
        return cmd_judge(a.out, a.cap_usd, a.mode, a.workers, a.dry_run, a.inputs_dir)
    return cmd_summary(a.out, a.inputs_dir)


if __name__ == "__main__":
    raise SystemExit(main())
