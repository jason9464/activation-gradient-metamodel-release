#!/usr/bin/env python
"""Activation twin of the three-layer gradient reader (user decision 2026-10-29 03:30, judge cap 10 USD; plan
docs/notes/20261029_activation_twin_plan.md; runner scripts/beh_reader/activation_twin.sh).

Question. The public AOs did not name the held-out behaviours from activations (ao_heldout.py). Is that because the
activations do not carry the behaviour, or because the AOs never learned this task and this input form? The twin is the
three-layer gradient reader (SFT runs/reader_scratch_L3_beh step 7,082, RL runs/reader_rl_scratch_L3_beh_paper cycle
40) with everything kept (rows, row order, sample files, questions, recipe, layers 10/20/30, centring, RL items, judge,
evaluation) except the input vector: activation sums of the same documents instead of -dL/dh sums.

Found while implementing (2026-10-29), against plan section 3: no forward extraction is needed. The C1 store
(loracle_mix_docs, the store the C1 reader rows come from), the fineweb RL store (fineweb_rl) and the behaviour store
(beh_reader/behaviour_bundles_v1) were all written with act_layers [5, 10, ..., 38] and full_skip_pos0.act true: every
row holds full_act_L<L> = the sum of the output of model.model.layers[L] over positions 1..n-2 of the chunk, written in
the same forward pass as its full_grad_L<L> (main_run/agm/core.py). So the twin's rows are built on the CPU from the
stored rows with the very document lists and arithmetic that reproduce every gradient row of the two reader dirs bit
for bit (checked: 29,757 + 1,870 rows x 3 layers on 2026-10-29). Positions: the gradient rows sum positions 0..n-2 of
each chunk (every label position), the activation rows 1..n-2 (position 0, the attention-sink token with a norm far
above the others, is stored apart as pos0_act); the rows differ in that one position. The held-out test inputs
runs/beh_reader/ao_heldout/inputs_act_correct.npz and every AO input in this repo follow the same 1..n-2 rule.

  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/activation_twin.py <cmd> [--out DIR]
  verify-act  --gpu G [--n-docs 4]  (GPU, about 35 GiB, 40 s measured 2026-10-29 04:41) the stored activations against the documents:
              every C1 document of the SFT dir and every fineweb document of the RL dir tokenised again with the
              extractor's own raw-text rule (main_run.agm.data.rows_from_raw: no special tokens, 2,048-token chunks,
              chunks under 16 tokens dropped): doc_id = sha1(text), n_tokens, n_label and the chunk count equal the
              store index (all documents); then --n-docs documents per store (the one with the most chunks plus random
              ones, seed 0) run through Qwen3-14B (generate.load_base: bf16, sdpa) one chunk at a time with forward hooks
              on decoder layers 10, 20, 30, summed in fp32 over positions 1..n-2, against the stored full_act sums
              (cosine > 0.999 and |norm ratio - 1| < 0.02 per document and layer: an independent forward in another
              batch layout, so bf16 noise and not bit equality). Writes OUT/VERIFY_ACT.json; exit 1 when a check fails
  build-dirs  [--act-sft-dir D] [--act-rl-dir D]  (CPU, IO bound; 85 s on the real inputs in a scratch folder,
              2026-10-29 04:00, every check passed) the two activation reader dirs (below)
              and OUT/BUILD.json; refuses a non-empty output dir; nothing is written when a check before writing fails
  check-dirs  [--act-sft-dir D] [--act-rl-dir D]  (CPU; 25 s on the scratch build, 25 of 25 checks passed) the checks
              below on the written dirs; writes
              <dir>/act_checks.json (act_checks_<time>.json when one exists); exit 1 unless every check passes
  read        --reader grad3|twin --gpu G [--limit N] [--adapter A]  (GPU, 31 GiB peak; smoke 04:48: 1.2 s per call, 12-18 min) the held-out
              reading: 530 beh_test bundles x 20 answers to common.QUESTION_BEHAVIOURAL, e2_generate's TrainedReader
              and run_readings in this process (fast generation, temperature 1.0, top_p 1, top_k 0, 192 new tokens,
              torch.manual_seed(0 + step)): the settings of beh_judges' readings of rl_beh_c040. grad3 reads
              eval/inputs_correct.npz (rows of layers 10, 20, 30 picked by the reader's layers; e2_generate.load_inputs
              checks it), twin reads ao_heldout/inputs_act_correct.npz as it is (ao_heldout.load_inputs, kind act;
              e2_generate.load_inputs refuses activation files, so the reader is driven here instead of through its
              main). Writes OUT/e2/<reader>_generic_behavioural.jsonl and _settings.json (status finished at the end).
              A finished reading is left as it is when it was read with the adapter asked for, else exit 1 (never
              a reading of another adapter in its place); --limit N reads steps 0..N-1 (smoke; refused inside the
              real OUT). --gpu 0 is refused by every command (GPU 0 belongs to the already-learned chain)
  yesno       --reader grad3|twin --gpu G [--limit N] [--adapter A]  (GPU, an empty GPU: rare_yesno refuses one with
              >= 4 GiB used; 0.05 s per forward + 20 s load: about 2 min) rare_yesno.py read --scorer reader --bundles all with --reader-adapter,
              --reader-run-json and --reader-layers 10,20,30 (twin: --input act, the act inputs of rare_yesno/act,
              checked here bit for bit against inputs_act_correct.npz first), then rare_yesno.py summary; run folder
              OUT/yesno_<reader>; a passed read is reused only when its READ.json adapter and RUN.json are the ones
              asked for
  judge       [--cap-usd 10] [--mode batch|sync] [--workers 48] [--dry-run]  beh_judges.cmd_judge("truth") with the
              paths pointed at the two readings (cache, ledger and judge.lock shared with every other judge);
              --max-cost = the ledger total at start + cap (user 03:30: cap 10 USD for both evaluations);
              refuses (exit 1, nothing sent) unless both readings are finished with 20 answers for each of the 530
              steps, unless every existing judged file holds exactly its reading's requests row for row (step,
              window, answer index, request sha = the bundle's ground truth and that answer: no verdict of another
              reading or adapter can stand in), and when 2 x the estimate (new requests x the ledger's batch cost per
              request) is above the cap; em_judges then refuses again before sending when the ledger plus its own
              estimate would pass --max-cost. After judging, both judged files must equal their readings row for row.
              Writes OUT/JUDGE.json
  summary     OUT/summary.json (an earlier one moved to summary.json.before_<time>) and a printed table: per family the truth judge's YES rate of four readings: the 40-layer
              gradient reader rl_beh_c040 (reference, its evaluation readings), the three-layer gradient reader, the
              twin, the three-layer AO on centred activation rows (reference, ao_heldout, 10 answers); twin minus
              three-layer reader per family (paired bundle bootstrap); bundles named by only one of the two
              (>= 0.5 against <= 0.1, ao_heldout's thresholds) with the family / evidence concentration rule of
              ao_heldout (more than half and hypergeometric p < 0.05; >= 10 bundles); yes/no AUROC per family (own
              against other question, positive bundles) and positive against negative bundles (quirk, rare, backdoor)
              from the four rare_yesno summaries; the plan's flag rule (section 5): a family differs only when |YES
              difference| > 0.15 and |own-vs-other AUROC difference| > 0.1 (single seed); "adapters": the adapter
              of each reader's open reading and yes/no read and whether both readers were evaluated at the same stage
              (False after an RL cost cap, plan section 7; then printed as a WARNING); exit 1 when a judged file is
              not its reading's verdicts
  plot        [--fig PNG]  docs/figures/20261029_activation_twin.png (+ .json) from summary.json

Activation reader dirs (build-dirs). The C1 materialize layout read by train.py, rl_train, e2_generate unchanged.
  act SFT dir  $IIF_BASE/agm_data/c1_reader_40_beh_act/layers3 from the gradient dir c1_reader_40_beh/layers40:
    bundle_tokensum_L{10,20,30}.npy  fp16 unit rows [29,757, 5120] in the gradient dir's row order; row = unit(sum over
                 the bundle's documents of the documents' stored full_act rows), fp16 unit x fp32 norm per store row
                 summed in fp32 with numpy add.reduceat (ao_heldout.bundle_sums, the code that wrote
                 inputs_act_correct.npz), then materialize._write_vectors' unit / fp16 step without its negation.
                 Document lists: grad_doc_ids of the dir's own sample files (every bundle row has one; behaviour rows
                 equal bundles.jsonl doc_ids); store per row: C1 rows from the C1 store, behaviour rows from the
                 behaviour store (meta grad_store)
    bundle_index.parquet  the gradient index with every non-norm column unchanged and norm_tokensum_L{10,20,30} = the
                 activation norms (fp32) in place of the 40 gradient norm columns
    mean_unit_tokensum.npy  [3, 5120] fp32 = mean of the fp16 act unit rows of C1's train bundles cast to fp32, the
                 formula of materialize._write_vectors (u[train].astype(float32).mean(0)) that made the gradient mean
                 (checked: on the gradient dir's rows it reproduces its mean_unit at 10, 20, 30 bit for bit); behaviour
                 rows left out as in the gradient dir
    train_correct.jsonl, train_no_vector_baseline.jsonl, eval_samples.jsonl, beh_eval_samples.jsonl  byte copies
    meta.json    the gradient meta with layers [10, 20, 30], vector_kind "act", the act rules and checks, written last
                 and only when every build check passed (else meta.json.CHECKS_FAILED, so no reader can load the dir).
                 Its "sign" text starts with "-dL/dh" only because reader.data.ReaderSamples and
                 e2_generate.load_mean_unit refuse any other prefix (a guard against double negation); the text says the
                 rows are activations and nothing negates them anywhere
  act RL dir   $IIF_BASE/agm_data/rl_fineweb_beh_act/reader from rl_fineweb_beh/reader: the 1,870 fineweb rows from the
                 fineweb store with docs.parquet doc_id per row (one document each), the 250 behaviour rows copied byte
                 for byte from the act SFT dir rows of the same bundles (as the gradient RL dir copies its SFT dir),
                 mean_unit_tokensum.npy copied from the act SFT dir (the gradient RL dir copies the C1 mean),
                 rl_train.jsonl, rl_eval.jsonl, docs.parquet, keywords.parquet, calib_docs.parquet, items_report.json
                 byte copies, centering_diagnostics.json recomputed (reader.data.centering_diagnostics)
Checks of build-dirs (OUT/BUILD.json; any failure before writing stops it): per store manifest plan (act positions
1..n-2, gradient 0..n-2, act layers include 10/20/30), norm columns and arrays present, one row per document (merged
store) or chunk ids 0..k-1 once; every bundle's documents in its store, n_docs = list length, the index n_label = the
store's label count summed over the documents; "row code": the same builder on full_grad, negated, against every
gradient row at 10/20/30 (unit rows and norms bit-equal counts, min cosine; pass = every row bit-equal); act sums finite
and nonzero; 5 random bundles per store re-read document by document in float64 (ao_heldout.reread_bundle; cosine >
1 - 1e-6, max error < 1e-5 of the norm); the 530 beh_test act sums equal inputs_act_correct.npz rows bit for bit; the
mean rule reproduces the gradient means; RL: behaviour rows of the gradient RL dir equal its SFT dir's (the copy rule),
its mean equals the SFT mean.
Checks of check-dirs (<dir>/act_checks.json): layout (meta layers, vector_kind, sign prefix; arrays [n, 5120] fp16,
finite, unit norm within 1e-2; norms finite > 0; non-norm index columns equal to the gradient dir's; copied files
sha256-equal); rows recomputed from the stores (SFT: all 530 beh_test rows and 300 random rows, seed 1; RL: all 1,870
fineweb rows) bit-equal to the written rows; mean_unit = the formula on the written rows; eval input match (SFT): for
every beh_test row, the training vector ReaderVectors.get(row, center=True) equals e2_generate.reader_vectors(the
inputs_act_correct row, mean_unit) bit for bit, i.e. the twin sees at test time exactly what it saw in training;
centering_diagnostics passes; then the reused checks of check_reader_dir (beh rows, order hash, label count, vec rows,
no test leakage, doc identity, family counts; its C1-row comparison and gradient recompute do not apply) and of
check_rl_dir (behaviour rows = the act SFT dir's, items, share, no beh_test, pairs, draw, docs, order hash; its fineweb
comparison does not apply) with common.LAYERS set to 10, 20, 30 for the call.

Bottleneck and time. build-dirs is IO bound: the 80,310 C1 document rows of the 25,217 C1 bundles (80,303 distinct
documents, not the plan's 32,000), the 55,172 behaviour rows and the 1,870 fineweb rows, x 3 layers x 2 kinds (act and
the gradient comparison) of 10 KB: about 8 GB of reads, plus 1 GB written; no GPU (85 s measured). verify-act's token
check tokenised all 80,303 C1 and 1,870 fineweb documents in about 50 s (2026-10-29, CPU, every count equal); its
forward loads the model once (about 1 min) and runs 8 documents. read is generation bound (530 calls x 20 samples, measured 3.7-5.3 s per call for the 40-layer reader,
33 GiB peak). The twin's SFT and RL are the gradient reader's runs (38 and 73 min measured) with the same data sizes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT = C.EV / "activation_twin"
LAYERS3 = (10, 20, 30)
KEYS = ("grad3", "twin")
TAG = "behavioural"
QKEY = "generic"
GRAD_SFT_DIR = Path(C.SFT_DIR)
GRAD_RL_DIR = Path(C.RL_DIR)
ACT_SFT_DIR = C.AGM / "c1_reader_40_beh_act" / "layers3"
ACT_RL_DIR = C.AGM / "rl_fineweb_beh_act" / "reader"
AO_OUT = C.EV / "ao_heldout"
ACT_INPUTS = AO_OUT / "inputs_act_correct.npz"
YESNO_ROOT = C.EV / "rare_yesno"
YESNO_ACT_DIR = YESNO_ROOT / "act"
GRAD3_SFT_RUN = C.RUNS / "reader_scratch_L3_beh"
GRAD3_RL_RUN = C.RUNS / "reader_rl_scratch_L3_beh_paper"
# the twin's run folders; ACT_TWIN_SFT_RUN / ACT_TWIN_RL_RUN move them (the runner exports its SFT_RUN / RL_RUN, so a
# smoke chain in a scratch folder reads its own adapters and RUN.json); unset = the real runs
TWIN_SFT_NAME, TWIN_RL_NAME = "reader_scratch_L3act_beh", "reader_rl_scratch_L3act_beh_paper"
TWIN_SFT_RUN = Path(os.environ.get("ACT_TWIN_SFT_RUN") or C.RUNS / TWIN_SFT_NAME)
TWIN_RL_RUN = Path(os.environ.get("ACT_TWIN_RL_RUN") or C.RUNS / TWIN_RL_NAME)
SFT_FINAL = "step_007082"
RL_FINAL = "cycle_040"
SFT_CFG = C.CONFIG_DIR / "scratch_L3act_beh.yaml"
RL_CFG = C.CONFIG_DIR / "rl_scratch_L3act_beh_paper.yaml"
FIG = C.FIG_DIR / "20261029_activation_twin.png"
SFT_SAMPLE_FILES = ("train_correct.jsonl", "train_no_vector_baseline.jsonl", "eval_samples.jsonl", C.BEH_EVAL_SAMPLES)
RL_COPIES = ("rl_train.jsonl", "rl_eval.jsonl", "docs.parquet", "keywords.parquet", "calib_docs.parquet",
             "items_report.json")
VECTOR_KIND = "act"
SIGN_ACT = ("-dL/dh (format field only, see vector_kind): these rows are NOT gradients but activation sums: per layer "
            "the sum over the bundle's documents of the stored full_act rows (output of model.model.layers[L], "
            "positions 1..n-2 of each chunk), no negation. The text starts with '-dL/dh' only because the worktree "
            "readers (reader.data.ReaderSamples, e2_generate.load_mean_unit) refuse any other prefix; nothing negates "
            "these rows")
POSITION_RULE = ("activation rows: full_act_L<L> = sum over positions 1..n-2 of each chunk of the output of "
                 "model.model.layers[L] (manifest plan full_skip_pos0.act true; position 0 is pos0_act, left out); the "
                 "gradient rows of the source dir sum positions 0..n-2 (full_skip_pos0.grad false)")
ROW_RULE = ("row = unit(S), S = sum over the bundle's documents (sample files' grad_doc_ids, a listed-twice document "
            "twice) of each document's store rows (fp16 unit x fp32 norm, summed in fp32 by numpy add.reduceat in "
            "document order: ao_heldout.bundle_sums = GradStore.bundle_vectors tokensum arithmetic); unit = "
            "(S / ||S||).astype(float16), norm = ||S|| fp32 (materialize._write_vectors without the negation)")
MEAN_RULE = ("mean_unit_tokensum.npy [3, d] = u[C1 train rows].astype(float32).mean(0) per layer over the fp16 act unit "
             "rows of the C1 bundles (materialize._write_vectors' formula, the gradient dir's own centring)")
COS_MIN_VERIFY = 0.999
NORM_TOL_VERIFY = 0.02
REREAD_N = 5
RECOMPUTE_N = 300
REREAD_COS = 1 - 1e-6
REREAD_TOL = 1e-5
YES_HI, NO_LO, EPS = 0.5, 0.1, 1e-9  # ao_heldout's candidate thresholds
MIN_CANDIDATES, P_MAX = 10, 0.05  # ao_heldout's verdict rule
FLAG_YES, FLAG_AUC = 0.15, 0.10  # plan section 5
CAP_USD = 10.0
JUDGE_KEY = "activation_twin"
COLUMNS = (("reader40", "40-layer gradient reader (reference)"),
           ("grad3", "three-layer gradient reader"),
           ("twin", "activation twin (three layers)"),
           ("ao3_act_centred", "three-layer AO, centred activation rows (reference)"))
YESNO_SUMMARIES = {"reader40": YESNO_ROOT / "reader_grad_all" / "summary.json",
                   "ao3_act_centred": YESNO_ROOT / "ao3layer_act_centred_all" / "summary.json"}
NEG_FAMILIES = ("quirk", "rare", "backdoor")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [activation_twin] {msg}", flush=True)


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_jsonable))
    os.replace(tmp, path)


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def new_json(path: Path, obj) -> Path:
    """Write obj to path, or to <stem>_<time>.json when path exists (a report is never replaced)."""
    path = Path(path)
    if path.exists():
        path = path.with_name(f"{path.stem}_{stamp()}{path.suffix}")
    atomic_json(path, obj)
    return path


def set_aside(p: Path) -> Path | None:
    """Move an existing file to <name>.before_<time> (nothing is deleted)."""
    p = Path(p)
    if not p.exists():
        return None
    q = p.with_name(f"{p.name}.before_{stamp()}")
    os.replace(p, q)
    return q


def under(p, root) -> bool:
    r, o = Path(p).resolve(), Path(root).resolve()
    return r == o or o in r.parents


@contextlib.contextmanager
def patched(obj, **attrs):
    """Set module attributes for the duration of a block, then restore them (beh_judges.patched)."""
    old = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield obj
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def flat_passes(d, pre="") -> list[tuple[str, bool]]:
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += flat_passes(v, f"{pre}{k}.")
        elif (k == "pass_" or k.startswith("pass_")) and v is not None:
            out.append((f"{pre}{k}", bool(v)))
    return out


def code_sha1() -> str:
    h = hashlib.sha1()
    for p in (Path(__file__).resolve(), HERE / "common.py", HERE / "ao_heldout.py"):
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- modules (imported when used)
def AH():
    import ao_heldout

    return ao_heldout


def e2gen():
    from main_run.reader import e2_generate

    return e2_generate


def beh_judges():
    import beh_judges as B

    return B


# ---------------------------------------------------------------- readers
def readers(runs=None) -> dict:
    """The two readers compared: adapter, RUN.json of the SFT run (layers, reader dir, centring), inputs file. runs = a
    folder holding all four run folders by name (tests); None = the run folder constants (the twin's movable by
    ACT_TWIN_SFT_RUN / ACT_TWIN_RL_RUN)."""
    if runs:
        R = Path(runs)
        g_rl, g_sft, t_rl, t_sft = R / GRAD3_RL_RUN.name, R / GRAD3_SFT_RUN.name, R / TWIN_RL_NAME, R / TWIN_SFT_NAME
    else:
        g_rl, g_sft, t_rl, t_sft = GRAD3_RL_RUN, GRAD3_SFT_RUN, TWIN_RL_RUN, TWIN_SFT_RUN
    return {
        "grad3": dict(key="grad3", kind="grad", what="three-layer gradient reader, RL cycle 40",
                      adapter=g_rl / "adapters" / RL_FINAL, sft_adapter=g_sft / "adapters" / SFT_FINAL,
                      run_json=g_sft / "RUN.json", inputs=Path(C.INPUTS["correct"])),
        "twin": dict(key="twin", kind="act", what="activation twin of the three-layer reader, RL cycle 40",
                     adapter=t_rl / "adapters" / RL_FINAL, sft_adapter=t_sft / "adapters" / SFT_FINAL,
                     run_json=t_sft / "RUN.json", inputs=ACT_INPUTS),
    }


def reader_dir_kind(reader_dir) -> str:
    return str(json.loads((Path(reader_dir) / "meta.json").read_text()).get("vector_kind", "grad"))


def reading_path(out, key: str) -> Path:
    """e2_generate.out_name(key, 'generic', 'behavioural') under OUT/e2."""
    return Path(out) / "e2" / f"{key}_{QKEY}_{TAG}.jsonl"


def settings_path(reading: Path) -> Path:
    return reading.with_name(reading.stem + "_settings.json")


def judged_path(out, key: str) -> Path:
    r = reading_path(out, key)
    return r.with_name(r.stem + "_judge_truth.jsonl")  # em_judges.judged_path(reading, "truth")


# ---------------------------------------------------------------- stores
def store_open(path) -> dict:
    from main_run.grad_io import load_store_index

    df, man = load_store_index(Path(path))
    return dict(path=Path(path), df=df, man=man)


def store_check(st: dict, layers=LAYERS3) -> dict:
    """Manifest plan, norm columns, arrays and the per-document row layout of one store."""
    from main_run.agm import core

    plan = st["man"].get("plan", {})
    df = st["df"]
    skip = plan.get("full_skip_pos0", False)
    names = [core.array_name("full", k, L) for k in ("act", "grad") for L in layers]
    rows_ok = True
    merged = "chunk_id" not in df.columns
    if merged:
        rows_ok = bool(df["doc_id"].is_unique)
    else:
        for _, ch in df.groupby("doc_id", sort=False)["chunk_id"]:
            if sorted(int(c) for c in ch) != list(range(len(ch))):
                rows_ok = False
                break
    arrays = all((st["path"] / s / f"{n}.npy").exists() for s in st["man"]["shard_order"] for n in names)
    return dict(path=str(st["path"]), doc_order_hash=st["man"]["doc_order_hash"], total_rows=int(len(df)),
                merged=merged, model=plan.get("model"), full_skip_pos0=skip,
                act_layers=[int(L) for L in plan.get("act_layers", [])],
                pass_act_positions_1_to_n2=core.skip_for(skip, "act"),
                pass_grad_positions_0_to_n2=not core.skip_for(skip, "grad"),
                pass_act_layers=all(int(L) in [int(x) for x in plan.get("act_layers", [])] for L in layers),
                pass_norm_columns=all(f"norm_{n}" in df.columns for n in names), pass_arrays=arrays,
                pass_rows_per_document=rows_ok)


def doc_check(st: dict, doc_lists: list, n_docs, n_label) -> dict:
    """Every document in the store; n_docs = list length; index n_label = the store's label count summed over the
    bundle's documents (all store rows of a document)."""
    df = st["df"]
    nl = {d: int(v) for d, v in df.groupby("doc_id", sort=False)["n_label"].sum().items()}
    missing = sorted({d for dl in doc_lists for d in dl if d not in nl})
    got_nl = [sum(nl.get(d, 0) for d in dl) for dl in doc_lists]
    bad_nd = int(sum(len(dl) != int(n) for dl, n in zip(doc_lists, n_docs)))
    bad_nl = int(sum(int(a) != int(b) for a, b in zip(got_nl, n_label)))
    return dict(n_bundles=len(doc_lists), doc_slots=int(sum(len(dl) for dl in doc_lists)),
                unique_docs=len({d for dl in doc_lists for d in dl}), missing_docs=missing[:5], n_missing=len(missing),
                n_docs_mismatch=bad_nd, n_label_mismatch=bad_nl, pass_all_docs_in_store=not missing,
                pass_n_docs=bad_nd == 0, pass_n_label=bad_nl == 0)


def store_sums(st: dict, doc_lists: list, kind: str, layer: int) -> np.ndarray:
    """[n, d] fp32 bundle sums of full_<kind>_L<layer> (ao_heldout.bundle_sums: the code of inputs_act_correct.npz)."""
    return AH().bundle_sums(st["df"], st["man"], st["path"], doc_lists, kind, [layer])[:, 0]


def unit_norm(v: np.ndarray, what: str = "") -> tuple[np.ndarray, np.ndarray]:
    """materialize._write_vectors' unit step (without its negation): fp16 unit rows and fp32 norms."""
    v = np.asarray(v, np.float32)
    nrm = np.linalg.norm(v, axis=1)
    if not np.isfinite(v).all() or not np.isfinite(nrm).all():
        raise SystemExit(f"non-finite {what} row")
    if (nrm == 0).any():
        raise SystemExit(f"zero {what} row at position {int(np.flatnonzero(nrm == 0)[0])}")
    return (v / nrm[:, None]).astype(np.float16), nrm.astype(np.float32)


def mean_formula(units_c1: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """materialize._write_vectors: u[train_mask].astype(np.float32).mean(0) over the C1 bundles' fp16 unit rows."""
    return np.asarray(units_c1)[train_mask].astype(np.float32).mean(0)


def cos_rows(a, b) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def compare_units(u: np.ndarray, nr: np.ndarray, ref_u: np.ndarray, ref_n: np.ndarray) -> dict:
    """Bit-equal counts of fp16 unit rows and fp32 norms, and the lowest cosine."""
    ref_u = np.asarray(ref_u)
    return dict(n=int(len(u)), units_bit_equal=int((u == ref_u).all(1).sum()),
                norms_bit_equal=int((np.asarray(nr, np.float32) == np.asarray(ref_n, np.float32)).sum()),
                min_cos=float(cos_rows(u.astype(np.float32), ref_u.astype(np.float32)).min()) if len(u) else None)


def doc_lists_from_samples(d: Path, files) -> dict:
    """{bundle_id: grad_doc_ids} over the sample files (materialize._bundles_from_samples' source of truth); a bundle
    with two different lists is refused."""
    docs: dict = {}
    for name in files:
        p = Path(d) / name
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                b = r.get("grad_bundle_id")
                if b is None:
                    continue
                dl = [str(x) for x in r["grad_doc_ids"]]
                if docs.setdefault(b, dl) != dl:
                    raise SystemExit(f"{p.name}: bundle {b} has two different document lists")
    return docs


def check_out_dir(d: Path) -> None:
    d = Path(d)
    if (d / "meta.json").exists():
        raise SystemExit(f"{d}/meta.json exists; refusing to overwrite a finished directory")
    if d.exists() and any(d.iterdir()):
        raise SystemExit(f"{d} is not empty (an unfinished earlier build?); set it aside by hand after checking")


def copy_exact(src: Path, dst: Path) -> bool:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return C.sha256_file(str(src)) == C.sha256_file(str(dst))


def write_index(gidx, norms: dict, layers, path: Path):
    """The gradient index with its norm columns replaced by the activation norms of `layers` (placed where the norm
    columns were), every other column unchanged."""
    import pandas as pd

    cols = list(gidx.columns)
    first_norm = next(i for i, c in enumerate(cols) if c.startswith("norm_"))
    keep = [c for c in cols if not c.startswith("norm_")]
    before = [c for c in cols[:first_norm] if not c.startswith("norm_")]
    after = [c for c in keep if c not in before]
    out = gidx[before].copy()
    for L in layers:
        out[f"norm_tokensum_L{L}"] = np.asarray(norms[L], np.float32)
    for c in after:
        out[c] = gidx[c]
    out.to_parquet(path, index=False)
    return pd.read_parquet(path)


def act_meta(gmeta: dict, *, layers, source_dir: Path, centering: str, stores: dict, checks_ok: bool, extra: dict,
             t0: float) -> dict:
    m = dict(gmeta)
    m.update(layers=list(layers), vector_kind=VECTOR_KIND, sign=SIGN_ACT,
             weightings={"tokensum": "sum of the documents' activation sums (long documents weigh more)"},
             weightings_written=["tokensum"], storage=gmeta.get("storage"), centering=centering,
             act_position_rule=POSITION_RULE, act_row_rule=ROW_RULE,
             act_stores={k: dict(path=str(v["path"]), doc_order_hash=v["man"]["doc_order_hash"],
                                 total_rows=int(v["man"]["total_rows"])) for k, v in stores.items()},
             source_reader_dir=str(source_dir), source_meta_sha256=C.sha256_file(str(Path(source_dir) / "meta.json")),
             source_index_sha256=C.sha256_file(str(Path(source_dir) / "bundle_index.parquet")),
             build_checks_passed=checks_ok, wall_s=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%d %H:%M:%S"),
             code=os.path.abspath(__file__), code_sha1=code_sha1(), **extra)
    return m


# ---------------------------------------------------------------- build: SFT dir
def build_sft(grad_dir: Path, out_dir: Path, *, act_inputs: Path = ACT_INPUTS, snapshot=None, layers=LAYERS3,
              store_paths: dict | None = None, reread_n: int = REREAD_N) -> dict:
    """Write the act SFT dir (module docstring); returns the report (checks, passed). Pre-write checks failing ->
    report with passed False and nothing written."""
    import pandas as pd

    t0 = time.time()
    grad_dir, out_dir = Path(grad_dir), Path(out_dir)
    check_out_dir(out_dir)
    gmeta = json.loads((grad_dir / "meta.json").read_text())
    gidx = pd.read_parquet(grad_dir / "bundle_index.parquet")
    n = len(gidx)
    gl = [int(L) for L in gmeta["layers"]]
    chk: dict = dict(pass_rows_sequential=bool((gidx["row"].to_numpy() == np.arange(n)).all()),
                     pass_layers_in_source=all(L in gl for L in layers))
    ids = gidx["bundle_id"].tolist()
    docs = doc_lists_from_samples(grad_dir, SFT_SAMPLE_FILES)
    no_docs = [b for b in ids if b not in docs]
    chk["doc_lists"] = dict(n_bundles=n, n_with_list=n - len(no_docs), pass_every_bundle=not no_docs,
                            pass_no_extra=set(docs) <= set(ids))
    if no_docs:
        return dict(checks=chk, passed=False, why=f"{len(no_docs)} bundles without a document list ({no_docs[:3]})")
    src = gidx["source"].to_numpy().astype(str) if "source" in gidx.columns else np.full(n, "c1")
    split = gidx["split"].to_numpy().astype(str)
    n_c1 = int((src == "c1").sum())
    chk["pass_c1_prefix"] = bool((src[:n_c1] == "c1").all())
    sp = dict(store_paths or {})
    gs = gmeta.get("grad_store", {})
    if "c1" not in sp:
        sp["c1"] = gs["c1"]["path"] if "c1" in gs else gs["path"]
    if "behaviour" not in sp and "behaviour" in gs:
        sp["behaviour"] = gs["behaviour"]["path"]
    bundles = {b["bundle_id"]: b for b in C.load_bundles(snapshot)} if (src == "behaviour").any() else {}
    beh_rows = np.flatnonzero(src == "behaviour")
    bad_beh = [ids[i] for i in beh_rows if ids[i] not in bundles or list(bundles[ids[i]]["doc_ids"]) != docs[ids[i]]]
    chk["behaviour_docs"] = dict(n=len(beh_rows), pass_equal_bundles_jsonl=not bad_beh, bad=bad_beh[:3])
    groups = {s: np.flatnonzero(src == s) for s in dict.fromkeys(src.tolist())}
    unknown = [s for s in groups if s not in sp]
    chk["pass_known_sources"] = not unknown
    if unknown:
        return dict(checks=chk, passed=False, why=f"no store for sources {unknown}")
    stores = {s: store_open(sp[s]) for s in groups}
    for s, rows in groups.items():
        chk[f"store_{s}"] = store_check(stores[s], layers)
        chk[f"docs_{s}"] = doc_check(stores[s], [docs[ids[i]] for i in rows], gidx["n_docs"].to_numpy()[rows],
                                     gidx["n_label"].to_numpy()[rows])
    X, step_of, x_layers = None, {}, []
    test_rows = np.flatnonzero(split == C.SPLIT_TEST)
    if len(test_rows):
        ai = AH().load_inputs(act_inputs, kind="act")
        X, x_layers = ai["inputs"], [int(x) for x in ai["layers"]]
        step_of = {b: s for s, b in enumerate(ai["meta"]["bundle_ids"])}
        chk["test_inputs"] = dict(path=str(act_inputs), n=len(step_of), layers=ai["layers"],
                                  pass_layers=all(L in ai["layers"] for L in layers),
                                  pass_every_test_bundle=all(ids[i] in step_of for i in test_rows))
    failed = [k for k, v in flat_passes(chk) if not v]
    if failed:
        return dict(checks=chk, passed=False, why=f"checks before writing failed: {failed}")

    out_dir.mkdir(parents=True, exist_ok=True)
    need = n * 5120 * 2 * len(layers) + 2**30
    if shutil.disk_usage(out_dir).free < need:
        return dict(checks=chk, passed=False, why="not enough disk")
    rng = np.random.default_rng(0)
    reread_pick = {s: rng.choice(rows, size=min(reread_n, len(rows)), replace=False) for s, rows in groups.items()}
    reread = {s: dict(rows=[int(i) for i in v], min_cos=1.0, max_err=0.0) for s, v in reread_pick.items()}
    norms: dict = {}
    row_code: dict = {}
    test_eq = dict(n=0, bit_equal=0)
    act_stats: dict = {}
    tm = split[:n_c1] == C.SPLIT_TRAIN
    means = np.zeros((len(layers), 5120), np.float32)
    mean_rule: dict = {}
    gmean = np.load(grad_dir / "mean_unit_tokensum.npy")
    for li, L in enumerate(layers):
        G = np.load(grad_dir / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        d_model = G.shape[1]
        if li == 0 and d_model != means.shape[1]:
            means = np.zeros((len(layers), d_model), np.float32)
        tmp = out_dir / f"bundle_tokensum_L{L}.npy.tmp"
        A = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, d_model))
        norms[L] = np.zeros(n, np.float32)
        gn = gidx[f"norm_tokensum_L{L}"].to_numpy(np.float32)
        for s, rows in groups.items():
            dl = [docs[ids[i]] for i in rows]
            Sg = store_sums(stores[s], dl, "grad", L)
            ug, ng = unit_norm(-Sg, "gradient")
            row_code[f"{s}_L{L}"] = compare_units(ug, ng, G[rows], gn[rows])
            del Sg, ug, ng
            Sa = store_sums(stores[s], dl, "act", L)
            ua, na = unit_norm(Sa, f"activation ({s}, layer {L})")
            A[rows] = ua
            norms[L][rows] = na
            act_stats[f"{s}_L{L}"] = dict(norm_min=float(na.min()), norm_median=float(np.median(na)),
                                          norm_max=float(na.max()))
            pos = {int(r): j for j, r in enumerate(rows)}
            for i in reread_pick[s]:
                ref = AH().reread_bundle(stores[s]["df"], stores[s]["path"], docs[ids[i]], "act", [L])[0]
                got = Sa[pos[int(i)]].astype(np.float64)
                e = reread[s]
                e["min_cos"] = min(e["min_cos"], float(cos_rows(got, ref)))
                e["max_err"] = max(e["max_err"], float(np.abs(got - ref).max() / np.linalg.norm(ref)))
            if X is not None:
                for i in rows:
                    if split[i] != C.SPLIT_TEST:
                        continue
                    test_eq["n"] += 1
                    test_eq["bit_equal"] += int(np.array_equal(Sa[pos[int(i)]], X[step_of[ids[i]], 0, x_layers.index(L)]))
            del Sa
        A.flush()
        means[li] = mean_formula(A[:n_c1], tm)
        mean_rule[f"L{L}"] = bool(np.array_equal(mean_formula(G[:n_c1], tm), gmean[gl.index(L)]))
        del A
        os.replace(tmp, out_dir / f"bundle_tokensum_L{L}.npy")
        log(f"layer {L} written ({time.time() - t0:.0f} s)")
    chk["row_code"] = dict(rule="the same builder on full_grad, negated, against the gradient dir's rows",
                           per_store_layer=row_code,
                           pass_units_bit_equal=all(v["units_bit_equal"] == v["n"] for v in row_code.values()),
                           pass_norms_bit_equal=all(v["norms_bit_equal"] == v["n"] for v in row_code.values()))
    chk["act_rows"] = dict(stats=act_stats, reread=reread,
                           pass_reread=all(v["min_cos"] > REREAD_COS and v["max_err"] < REREAD_TOL
                                           for v in reread.values()))
    if X is not None:
        chk["test_rows_vs_inputs_act_correct"] = dict(**test_eq, pass_all_bit_equal=test_eq["n"] == len(test_rows) * len(
            layers) and test_eq["bit_equal"] == test_eq["n"], pass_at_least_50=test_eq["n"] >= min(50, len(test_rows)))
    chk["mean_rule_on_gradient_dir"] = dict(per_layer=mean_rule, pass_=all(mean_rule.values()))
    chk["pass_mean_finite"] = bool(np.isfinite(means).all())
    np.save(out_dir / "mean_unit_tokensum.npy", means.astype(np.float32))
    write_index(gidx, norms, layers, out_dir / "bundle_index.parquet")
    copies = {name: copy_exact(grad_dir / name, out_dir / name) for name in SFT_SAMPLE_FILES
              if (grad_dir / name).exists()}
    chk["copies"] = dict(files=copies, pass_sha256_equal=all(copies.values()))
    passed = not [k for k, v in flat_passes(chk) if not v]
    centering = (f"mean_unit_tokensum.npy: {MEAN_RULE}; recomputed on the activation rows of this dir (the gradient "
                 f"dir {grad_dir} copied the C1 gradient mean)")
    meta = act_meta(gmeta, layers=layers, source_dir=grad_dir, centering=centering, stores=stores, checks_ok=passed,
                    t0=t0, extra=dict(act_test_inputs=str(act_inputs) if X is not None else None))
    # meta.json marks a finished dir (ReaderVectors needs it): written only when every check passed
    name = "meta.json" if passed else "meta.json.CHECKS_FAILED"
    (out_dir / name).write_text(json.dumps(meta, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    return dict(checks=chk, passed=passed, out=str(out_dir), wall_s=round(time.time() - t0, 1))


# ---------------------------------------------------------------- build: RL dir
def build_rl(grad_rl_dir: Path, act_sft_dir: Path, out_dir: Path, *, grad_sft_dir: Path = GRAD_SFT_DIR,
             layers=LAYERS3, store_path=None) -> dict:
    import pandas as pd

    from main_run.reader.data import centering_diagnostics

    t0 = time.time()
    grad_rl_dir, act_sft_dir, out_dir = Path(grad_rl_dir), Path(act_sft_dir), Path(out_dir)
    check_out_dir(out_dir)
    gmeta = json.loads((grad_rl_dir / "meta.json").read_text())
    gidx = pd.read_parquet(grad_rl_dir / "bundle_index.parquet")
    gdocs = pd.read_parquet(grad_rl_dir / "docs.parquet")
    n = len(gidx)
    n_beh = int(gmeta.get("n_beh_bundles", 0))
    n0 = n - n_beh
    ids = gidx["bundle_id"].tolist()
    sft_meta = json.loads((act_sft_dir / "meta.json").read_text())
    sidx = pd.read_parquet(act_sft_dir / "bundle_index.parquet")
    gsidx = pd.read_parquet(Path(grad_sft_dir) / "bundle_index.parquet")
    spos = {b: i for i, b in enumerate(sidx["bundle_id"])}
    gspos = {b: i for i, b in enumerate(gsidx["bundle_id"])}
    beh_ids = ids[n0:]
    chk: dict = dict(pass_rows_sequential=bool((gidx["row"].to_numpy() == np.arange(n)).all()),
                     pass_docs_rows_equal_index=gdocs["bundle_id"].tolist() == ids,
                     pass_act_sft_kind=sft_meta.get("vector_kind") == VECTOR_KIND,
                     pass_act_sft_layers=[int(L) for L in sft_meta["layers"]] == list(layers),
                     pass_beh_in_sft=all(b in spos and b in gspos for b in beh_ids),
                     pass_sft_rows_aligned=sidx["bundle_id"].tolist() == gsidx["bundle_id"].tolist())
    dl = [[str(d)] for d in gdocs["doc_id"].to_numpy()[:n0]]
    from_items = {}
    for name in ("rl_train.jsonl", "rl_eval.jsonl"):
        for r in C.read_jsonl(grad_rl_dir / name, missing_ok=True):
            if r.get("grad_bundle_id") is not None:
                from_items.setdefault(r["grad_bundle_id"], [str(x) for x in r["grad_doc_ids"]])
    pos0 = {b: i for i, b in enumerate(ids[:n0])}
    chk["docs_vs_items"] = dict(n_compared=sum(b in pos0 for b in from_items),
                                pass_equal=all(dl[pos0[b]] == v for b, v in from_items.items() if b in pos0))
    st = store_open(store_path or gmeta["grad_store"]["path"])
    chk["store_fineweb"] = store_check(st, layers)
    chk["docs_fineweb"] = doc_check(st, dl, gidx["n_docs"].to_numpy()[:n0], gidx["n_label"].to_numpy()[:n0])
    failed = [k for k, v in flat_passes(chk) if not v]
    if failed:
        return dict(checks=chk, passed=False, why=f"checks before writing failed: {failed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    norms: dict = {}
    row_code, copy_rule = {}, {}
    sft_rows = np.array([spos[b] for b in beh_ids], np.int64)
    gsft_rows = np.array([gspos[b] for b in beh_ids], np.int64)
    rng = np.random.default_rng(0)
    pick = rng.choice(n0, size=min(REREAD_N, n0), replace=False)
    reread = dict(rows=[int(i) for i in pick], min_cos=1.0, max_err=0.0)
    for L in layers:
        G = np.load(grad_rl_dir / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        gn = gidx[f"norm_tokensum_L{L}"].to_numpy(np.float32)
        Sg = store_sums(st, dl, "grad", L)
        ug, ng = unit_norm(-Sg, "gradient")
        row_code[f"L{L}"] = compare_units(ug, ng, G[:n0], gn[:n0])
        del Sg, ug, ng
        GS = np.load(Path(grad_sft_dir) / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        copy_rule[f"L{L}"] = dict(units=bool(np.array_equal(np.asarray(G[n0:]), np.asarray(GS[gsft_rows]))),
                                  norms=bool(np.array_equal(gn[n0:], gsidx[f"norm_tokensum_L{L}"].to_numpy(np.float32)[
                                      gsft_rows])))
        Sa = store_sums(st, dl, "act", L)
        ua, na = unit_norm(Sa, f"activation (fineweb, layer {L})")
        for i in pick:
            ref = AH().reread_bundle(st["df"], st["path"], dl[int(i)], "act", [L])[0]
            got = Sa[int(i)].astype(np.float64)
            reread["min_cos"] = min(reread["min_cos"], float(cos_rows(got, ref)))
            reread["max_err"] = max(reread["max_err"], float(np.abs(got - ref).max() / np.linalg.norm(ref)))
        del Sa
        S = np.load(act_sft_dir / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        tmp = out_dir / f"bundle_tokensum_L{L}.npy.tmp"
        A = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, G.shape[1]))
        A[:n0] = ua
        A[n0:] = np.asarray(S[sft_rows])
        A.flush()
        del A
        os.replace(tmp, out_dir / f"bundle_tokensum_L{L}.npy")
        norms[L] = np.concatenate([na, sidx[f"norm_tokensum_L{L}"].to_numpy(np.float32)[sft_rows]])
        log(f"RL layer {L} written ({time.time() - t0:.0f} s)")
    chk["row_code"] = dict(per_layer=row_code,
                           pass_units_bit_equal=all(v["units_bit_equal"] == v["n"] for v in row_code.values()),
                           pass_norms_bit_equal=all(v["norms_bit_equal"] == v["n"] for v in row_code.values()))
    chk["copy_rule_on_gradient_dirs"] = dict(per_layer=copy_rule,
                                             pass_=all(v["units"] and v["norms"] for v in copy_rule.values()))
    chk["act_reread"] = dict(**reread, pass_=reread["min_cos"] > REREAD_COS and reread["max_err"] < REREAD_TOL)
    gm_rl, gm_sft = np.load(grad_rl_dir / "mean_unit_tokensum.npy"), np.load(Path(grad_sft_dir) / "mean_unit_tokensum.npy")
    chk["mean_copy_rule_on_gradient_dirs"] = dict(pass_=bool(np.array_equal(gm_rl, gm_sft)))
    chk["mean_copied"] = dict(pass_=copy_exact(act_sft_dir / "mean_unit_tokensum.npy", out_dir / "mean_unit_tokensum.npy"))
    write_index(gidx, norms, layers, out_dir / "bundle_index.parquet")
    copies = {name: copy_exact(grad_rl_dir / name, out_dir / name) for name in RL_COPIES if (grad_rl_dir / name).exists()}
    chk["copies"] = dict(files=copies, pass_sha256_equal=all(copies.values()))
    passed = not [k for k, v in flat_passes(chk) if not v]
    centering = (f"mean_unit_tokensum.npy copied from the act SFT dir {act_sft_dir} ({MEAN_RULE}), as the gradient RL "
                 f"dir copies the C1 gradient mean")
    extra = dict(beh_vectors_from=str(act_sft_dir),
                 sft_meta_sha256=C.sha256_file(str(act_sft_dir / "meta.json")),
                 sft_index_sha256=C.sha256_file(str(act_sft_dir / "bundle_index.parquet")))
    meta = act_meta(gmeta, layers=layers, source_dir=grad_rl_dir, centering=centering, stores={"fineweb": st},
                    checks_ok=passed, t0=t0, extra=extra)
    meta.pop("centering_diagnostics", None)
    if not passed:
        (out_dir / "meta.json.CHECKS_FAILED").write_text(json.dumps(meta, indent=2, default=_jsonable) + "\n")
        return dict(checks=chk, passed=False, out=str(out_dir), wall_s=round(time.time() - t0, 1))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    cd = centering_diagnostics(out_dir, weighting="tokensum", layers=list(layers))
    atomic_json(out_dir / "centering_diagnostics.json", cd)
    meta["centering_diagnostics"] = dict(n_rows=cd["n_rows"],
                                         min_over_layers={k: v["min"] for k, v in cd["layers"].items()})
    tmpm = out_dir / "meta.json.tmp"
    tmpm.write_text(json.dumps(meta, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    os.replace(tmpm, out_dir / "meta.json")
    return dict(checks=chk, passed=passed, out=str(out_dir), wall_s=round(time.time() - t0, 1))


def cmd_build_dirs(out: Path, sft_out: Path, rl_out: Path, *, grad_sft=GRAD_SFT_DIR, grad_rl=GRAD_RL_DIR,
                   act_inputs=ACT_INPUTS, snapshot=None) -> int:
    t0 = time.time()
    rep = dict(experiment="activation twin: reader dirs", sft=None, rl=None, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    sft_done = (Path(sft_out) / "meta.json").exists()
    if sft_done:
        log(f"{sft_out} is finished (meta.json); not rebuilt")
        rep["sft"] = dict(skipped="finished earlier", passed=json.loads((Path(sft_out) / "meta.json").read_text()).get(
            "build_checks_passed"))
    else:
        rep["sft"] = build_sft(grad_sft, sft_out, act_inputs=act_inputs, snapshot=snapshot)
    ok = bool(rep["sft"].get("passed"))
    if ok and not (Path(rl_out) / "meta.json").exists():
        rep["rl"] = build_rl(grad_rl, sft_out, rl_out, grad_sft_dir=grad_sft)
    elif ok:
        rep["rl"] = dict(skipped="finished earlier", passed=json.loads((Path(rl_out) / "meta.json").read_text()).get(
            "build_checks_passed"))
    rep["passed"] = ok and bool((rep["rl"] or {}).get("passed"))
    rep["failed"] = [k for part in ("sft", "rl") if isinstance(rep[part], dict) and "checks" in rep[part]
                     for k, v in flat_passes(rep[part]["checks"], f"{part}.") if not v]
    rep.update(wall_s=round(time.time() - t0, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = new_json(Path(out) / "BUILD.json", rep)
    log(f"build {'passed' if rep['passed'] else 'FAILED'}; report {p}" + (f"; failed {rep['failed']}" if rep[
        "failed"] else "") + (f"; {rep['sft'].get('why')}" if rep["sft"].get("why") else "")
        + (f"; {(rep['rl'] or {}).get('why')}" if (rep["rl"] or {}).get("why") else ""))
    return 0 if rep["passed"] else 1


# ---------------------------------------------------------------- check-dirs
def _run(rep: dict, name: str, fn, *a, **kw) -> None:
    t0 = time.time()
    try:
        rep[name] = fn(*a, **kw)
    except Exception as e:  # noqa: BLE001 - a crashing check is a failed check
        rep[name] = dict(error=f"{type(e).__name__}: {e}", pass_=False)
    rep[name].setdefault("wall_s", round(time.time() - t0, 1))


def layout_check(d: Path, grad_dir: Path, copies, layers=LAYERS3) -> dict:
    import pandas as pd

    d, grad_dir = Path(d), Path(grad_dir)
    meta = json.loads((d / "meta.json").read_text())
    idx, gidx = pd.read_parquet(d / "bundle_index.parquet"), pd.read_parquet(grad_dir / "bundle_index.parquet")
    nn = [c for c in gidx.columns if not c.startswith("norm_")]
    cols_eq = {c: bool(c in idx.columns and idx[c].equals(gidx[c])) for c in nn}
    ncols = sorted(c for c in idx.columns if c.startswith("norm_"))
    shapes, fin, dev = True, True, 0.0
    for L in layers:
        a = np.load(d / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        shapes &= a.dtype == np.float16 and a.shape[0] == len(idx)
        for s in range(0, len(idx), 8192):
            u = np.asarray(a[s:s + 8192], np.float32)
            fin &= bool(np.isfinite(u).all())
            dev = max(dev, float(np.abs(np.linalg.norm(u, axis=1) - 1).max()))
    nr = idx[[f"norm_tokensum_L{L}" for L in layers]].to_numpy(np.float64)
    files = {c: C.sha256_file(str(d / c)) == C.sha256_file(str(grad_dir / c)) for c in copies if (grad_dir / c).exists()}
    return dict(meta_layers=meta["layers"], pass_meta_layers=[int(L) for L in meta["layers"]] == list(layers),
                pass_vector_kind=meta.get("vector_kind") == VECTOR_KIND,
                pass_sign_prefix=str(meta.get("sign", "")).startswith("-dL/dh"),
                pass_n_bundles=meta.get("n_bundles") == len(idx), pass_arrays_shape=bool(shapes),
                pass_units_finite=fin, unit_norm_max_dev=dev, pass_unit_norm=dev < 1e-2,
                pass_norms=bool(np.isfinite(nr).all() and nr.min() > 0),
                norm_columns=ncols, pass_norm_columns=ncols == sorted(f"norm_tokensum_L{L}" for L in layers),
                index_columns_equal=cols_eq, pass_index_columns=all(cols_eq.values()),
                files_sha256_equal=files, pass_files=all(files.values()) and len(files) > 0)


def recompute_check(d: Path, rows: np.ndarray, groups: dict, docs_of, stores: dict, layers=LAYERS3) -> dict:
    """The written rows of `rows` recomputed from the stores: unit rows and norms bit-equal."""
    import pandas as pd

    idx = pd.read_parquet(Path(d) / "bundle_index.parquet")
    ids = idx["bundle_id"].tolist()
    res = {}
    for L in layers:
        a = np.load(Path(d) / f"bundle_tokensum_L{L}.npy", mmap_mode="r")
        nr = idx[f"norm_tokensum_L{L}"].to_numpy(np.float32)
        for s, members in groups.items():
            sel = np.array([i for i in rows if i in members], np.int64)
            if not len(sel):
                continue
            u, n = unit_norm(store_sums(stores[s], [docs_of(ids[i]) for i in sel], "act", L))
            res[f"{s}_L{L}"] = compare_units(u, n, a[sel], nr[sel])
    return dict(per_store_layer=res, n_rows=int(len(rows)),
                pass_units_bit_equal=all(v["units_bit_equal"] == v["n"] for v in res.values()),
                pass_norms_bit_equal=all(v["norms_bit_equal"] == v["n"] for v in res.values()))


def mean_check(d: Path, layers=LAYERS3) -> dict:
    import pandas as pd

    idx = pd.read_parquet(Path(d) / "bundle_index.parquet")
    src = idx["source"].to_numpy().astype(str) if "source" in idx.columns else np.full(len(idx), "c1")
    n_c1 = int((src == "c1").sum())
    tm = idx["split"].to_numpy().astype(str)[:n_c1] == C.SPLIT_TRAIN
    m = np.load(Path(d) / "mean_unit_tokensum.npy")
    eq = {f"L{L}": bool(np.array_equal(mean_formula(np.load(Path(d) / f"bundle_tokensum_L{L}.npy", mmap_mode="r")[:n_c1],
                                                    tm), m[li])) for li, L in enumerate(layers)}
    return dict(shape=list(m.shape), n_c1_train=int(tm.sum()), per_layer=eq,
                pass_shape=m.shape[0] == len(layers) and m.dtype == np.float32, pass_finite=bool(np.isfinite(m).all()),
                pass_formula=all(eq.values()))


def eval_input_check(d: Path, act_inputs: Path, layers=LAYERS3) -> dict:
    """Training vector (ReaderVectors.get, centred) of every beh_test row against the reading vector of its
    inputs_act_correct row (e2_generate.reader_vectors with this dir's mean_unit): bit-equal."""
    from main_run.c1.materialize import ReaderVectors

    eg = e2gen()
    rv = ReaderVectors(Path(d))
    ai = AH().load_inputs(act_inputs, kind="act")
    lidx = [ai["layers"].index(L) for L in layers]
    mean = eg.load_mean_unit(Path(d), "tokensum", list(layers))
    idx = rv.index
    test = np.flatnonzero(idx["split"].to_numpy().astype(str) == C.SPLIT_TEST)
    step_of = {b: s for s, b in enumerate(ai["meta"]["bundle_ids"])}
    rows = [int(i) for i in test]
    steps = [step_of[idx["bundle_id"].iloc[i]] for i in rows]
    train_v = rv.get(rows, "tokensum", center=True, layers=list(layers)).numpy()
    read_v = eg.reader_vectors(np.asarray(ai["inputs"][steps, 0][:, lidx], np.float32), mean)
    eq = (train_v == read_v).all(axis=(1, 2))
    return dict(n=len(rows), bit_equal=int(eq.sum()), max_abs_diff=float(np.abs(train_v - read_v).max()) if rows else None,
                min_cos=float(cos_rows(train_v, read_v).min()) if rows else None,
                pass_all_bit_equal=bool(len(rows) and eq.all()))


def reader_samples_check(d: Path, tok=None, layers=LAYERS3, n_items: int = 4) -> dict:
    """train.py's own dataset class on this dir: reader.data.ReaderSamples(train_correct.jsonl, center=True,
    require_split train) loads (sign guard, vec_row check over every row) and its first C1 and behaviour items carry
    [3, d] vectors equal to ReaderVectors.get(row, center=True)."""
    from main_run.c1.materialize import ReaderVectors
    from main_run.reader.data import ReaderSamples

    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
    ds = ReaderSamples(Path(d) / "train_correct.jsonl", Path(d), tok=tok, layers=list(layers), center=True,
                       require_split="train")
    rv = ReaderVectors(Path(d))
    picks, seen = [], set()
    for i, r in enumerate(ds.rows):
        src = r.get("label_source", "c1")
        if src not in seen and r.get("vec_row") is not None:
            picks.append(i)
            seen.add(src)
        if len(picks) >= n_items or len(seen) >= 2 and len(picks) >= 2:
            break
    eq = []
    for i in picks:
        it = ds[i]
        want = rv.get([int(ds.rows[i]["vec_row"])], "tokensum", center=True, layers=list(layers))[0]
        eq.append(bool(tuple(it["vectors"].shape) == tuple(want.shape) and (it["vectors"] == want).all()))
    return dict(n_rows=len(ds), items=picks, pass_loads=True, pass_items_equal=bool(eq) and all(eq))


def centering_check(d: Path, layers=LAYERS3) -> dict:
    from main_run.reader.data import centering_diagnostics

    cd = centering_diagnostics(Path(d), weighting="tokensum", layers=list(layers))
    return dict(n_rows=cd["n_rows"], min_over_layers={k: v["min"] for k, v in cd["layers"].items()},
                frac_below_0_05={k: v["frac_below_0.05"] for k, v in cd["layers"].items()}, pass_=True)


def reused_sft_checks(d: Path, rep: dict) -> None:
    """check_reader_dir's checks that do not compare vectors with the gradient dir (common.LAYERS = 10, 20, 30)."""
    import check_reader_dir as K
    import pandas as pd

    with patched(C, LAYERS=list(LAYERS3)):
        D, I = K.load_dir(Path(d)), K.load_inputs()
        c1_dir = Path(C.C1_DIR)
        c1_ids = pd.read_parquet(c1_dir / "bundle_index.parquet", columns=["bundle_id"])["bundle_id"].tolist()
        c1_train_ids = [r["sample_id"] for r in C.read_jsonl(c1_dir / "train_correct.jsonl")]
        idx, bundles = D["idx"], I["bundles"]
        _run(rep, "beh_rows_finite_nonzero", K.check_beh_rows, Path(d), idx, len(c1_ids))
        _run(rep, "order_hash", K.check_order, D, c1_ids, c1_train_ids, bundles)
        _run(rep, "label_count", K.check_label_count, D, bundles, I["labels"], I["manifest"])
        _run(rep, "vec_rows", K.check_vec_rows, D, bundles)
        _run(rep, "no_test_leakage", K.check_leakage, D["train"] + D["nov"], idx, bundles, I["labels"])
        _run(rep, "doc_identity", K.check_doc_identity, I, idx)
        _run(rep, "family_counts", K.check_family_counts, idx, bundles, I["manifest"])


def reused_rl_checks(d: Path, sft_dir: Path, rep: dict, src: Path | None = None, snapshot=None) -> None:
    """check_rl_dir.run_checks without its fineweb comparison (common.LAYERS = 10, 20, 30)."""
    import check_rl_dir as KR
    import pandas as pd

    src = Path(src or C.RL_SRC)
    snapshot = Path(snapshot or C.SNAPSHOT)
    with patched(C, LAYERS=list(LAYERS3)):
        rd = KR.B.rl_data()
        idx = pd.read_parquet(Path(d) / "bundle_index.parquet")
        n0 = len(pd.read_parquet(src / "bundle_index.parquet", columns=["bundle_id"]))
        tr, ev = C.read_jsonl(Path(d) / "rl_train.jsonl"), C.read_jsonl(Path(d) / "rl_eval.jsonl")
        meta = json.loads((Path(d) / "meta.json").read_text())
        n_src = len(C.read_jsonl(src / "rl_train.jsonl"))
        blist = C.load_bundles(snapshot)
        bundles = {b["bundle_id"]: b for b in blist}
        labels = C.load_labels(snapshot)
        drawn = list(idx["bundle_id"].iloc[n0:])
        beh_items = tr[n_src:]
        _run(rep, "beh_rows_equal_sft", KR.check_beh_rows, Path(d), Path(sft_dir), idx, n0)
        _run(rep, "items", KR.check_items, tr, idx, n0, n_src, bundles, labels, rd)
        _run(rep, "share", KR.check_share, n_src, len(beh_items))
        _run(rep, "no_beh_test", KR.check_no_test, blist, idx, tr, ev, drawn)
        _run(rep, "pairs_whole", KR.check_pairs, blist, drawn)
        _run(rep, "draw", KR.check_draw, blist, drawn, [r["bundle_id"] for r in beh_items], meta)
        known = [b for b in drawn if b in bundles]
        _run(rep, "docs", KR.check_docs, Path(d), src, snapshot, known, bundles,
             {r["bundle_id"]: r.get("keywords") for r in beh_items})
        rep["docs"]["pass_"] = bool(rep["docs"].get("pass_") and len(known) == len(drawn))
        _run(rep, "keyword_template_overlaps", KR.check_overlaps, beh_items, rd)
        _run(rep, "order_hash", KR.check_order_hash, idx, tr, meta, n0)


def check_sft_dir(d: Path, grad_dir: Path = GRAD_SFT_DIR, *, act_inputs=ACT_INPUTS, n_sample=RECOMPUTE_N,
                  reused: bool = True, tok=None) -> dict:
    import pandas as pd

    rep: dict = {}
    _run(rep, "layout", layout_check, d, grad_dir, SFT_SAMPLE_FILES)
    gmeta = json.loads((Path(grad_dir) / "meta.json").read_text())
    idx = pd.read_parquet(Path(d) / "bundle_index.parquet")
    src = idx["source"].to_numpy().astype(str) if "source" in idx.columns else np.full(len(idx), "c1")
    gs = gmeta.get("grad_store", {})
    sp = {"c1": gs["c1"]["path"] if "c1" in gs else gs.get("path")}
    if "behaviour" in gs:
        sp["behaviour"] = gs["behaviour"]["path"]
    groups = {s: set(np.flatnonzero(src == s).tolist()) for s in dict.fromkeys(src.tolist())}
    docs = doc_lists_from_samples(Path(d), SFT_SAMPLE_FILES)
    test = np.flatnonzero(idx["split"].to_numpy().astype(str) == C.SPLIT_TEST)
    rest = np.setdiff1d(np.arange(len(idx)), test)
    rng = np.random.default_rng(1)
    rows = np.sort(np.concatenate([test, rng.choice(rest, size=min(n_sample, len(rest)), replace=False)]))
    stores = {s: store_open(sp[s]) for s in groups}
    _run(rep, "rows_recomputed", recompute_check, d, rows, groups, lambda b: docs[b], stores)
    _run(rep, "mean_unit", mean_check, d)
    if len(test):
        _run(rep, "eval_input_match", eval_input_check, d, act_inputs)
    _run(rep, "centering", centering_check, d)
    _run(rep, "reader_samples", reader_samples_check, d, tok)
    if reused:
        reused_sft_checks(d, rep)
    return rep


def check_rl_dir_act(d: Path, sft_dir: Path, grad_rl_dir: Path = GRAD_RL_DIR, *, reused: bool = True,
                     src=None, snapshot=None) -> dict:
    import pandas as pd

    rep: dict = {}
    _run(rep, "layout", layout_check, d, grad_rl_dir, RL_COPIES)
    gmeta = json.loads((Path(grad_rl_dir) / "meta.json").read_text())
    idx = pd.read_parquet(Path(d) / "bundle_index.parquet")
    n0 = len(idx) - int(gmeta.get("n_beh_bundles", 0))
    gdocs = pd.read_parquet(Path(grad_rl_dir) / "docs.parquet")
    doc_of = dict(zip(gdocs["bundle_id"].iloc[:n0], gdocs["doc_id"].iloc[:n0]))
    st = store_open(gmeta["grad_store"]["path"])
    _run(rep, "fineweb_rows_recomputed", recompute_check, d, np.arange(n0), {"fineweb": set(range(n0))},
         lambda b: [str(doc_of[b])], {"fineweb": st})
    rep["mean_unit_equals_sft"] = dict(pass_=C.sha256_file(str(Path(d) / "mean_unit_tokensum.npy"))
                                       == C.sha256_file(str(Path(sft_dir) / "mean_unit_tokensum.npy")))
    _run(rep, "centering", centering_check, d)
    if reused:
        reused_rl_checks(d, sft_dir, rep, src=src, snapshot=snapshot)
    return rep


def cmd_check_dirs(sft_dir: Path, rl_dir: Path, *, grad_sft=GRAD_SFT_DIR, grad_rl=GRAD_RL_DIR,
                   act_inputs=ACT_INPUTS) -> int:
    ok = True
    for name, d, fn in (("sft", sft_dir, lambda: check_sft_dir(sft_dir, grad_sft, act_inputs=act_inputs)),
                        ("rl", rl_dir, lambda: check_rl_dir_act(rl_dir, sft_dir, grad_rl))):
        t0 = time.time()
        if not (Path(d) / "meta.json").exists():
            log(f"{d}: no meta.json (not built)")
            ok = False
            continue
        rep = fn()
        ran = {k: v for k, v in rep.items() if isinstance(v, dict)}
        failed = [f"{k}.{n}" for k, v in ran.items() for n, p in flat_passes(v) if not p]
        summ = dict(n_checks=len(ran), failed=failed, all_pass=not failed, wall_s=round(time.time() - t0, 1),
                    written=time.strftime("%Y-%m-%d %H:%M:%S"),
                    not_applicable=("check_reader_dir c1_rows_identical and recompute (gradient comparisons)"
                                    if name == "sft" else "check_rl_dir fineweb_rows_identical (gradient comparison)"))
        p = new_json(Path(d) / "act_checks.json", dict(summary=summ, checks=rep))
        log(f"{name} dir {d}: {len(ran)} checks, failed {failed or 'none'}; report {p}")
        ok &= not failed
    return 0 if ok else 1


# ---------------------------------------------------------------- verify-act (GPU)
def bind_gpu(gpu: str) -> str:
    """CUDA_DEVICE_ORDER = PCI_BUS_ID and CUDA_VISIBLE_DEVICES = gpu before CUDA starts (rare_yesno.bind_gpu)."""
    if not str(gpu).isdigit():
        raise SystemExit(f"--gpu {gpu!r}: one nvidia-smi index expected")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise SystemExit("CUDA already initialised before the GPU was bound")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return "cuda:0"


def expected_counts(rows) -> dict:
    """{doc_id: (n_tokens, n_label, n_chunks)} of the extractor's raw rows (n_label = tokens - 1 per chunk)."""
    out: dict = {}
    for r in rows:
        t, lab, ch = out.get(r.doc_id, (0, 0, 0))
        out[r.doc_id] = (t + len(r.ids), lab + len(r.ids) - 1, ch + 1)
    return out


def store_counts(df) -> dict:
    g = df.groupby("doc_id", sort=False)
    nch = g["n_chunks"].sum() if "n_chunks" in df.columns else g.size()
    nt, nl = g["n_tokens"].sum(), g["n_label"].sum()
    return {d: (int(nt[d]), int(nl[d]), int(nch[d])) for d in nt.index}


def act_forward_sums(model, chunks: list, layers, device) -> np.ndarray:
    """[len(layers), d] fp32: per layer the sum over chunks of the decoder layer's output at positions 1..n-2, one
    chunk per forward (no padding), hooks on model.model.layers[L] (the store's layer convention)."""
    import torch

    from main_run.agm import core

    dec = core.decoder_layers(model)
    acc: dict = {}
    cur: dict = {}

    def mk(L):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cur[L] = h[0, 1:h.shape[1] - 1].float().sum(0)
        return hook

    hs = [dec[L].register_forward_hook(mk(L)) for L in layers]
    try:
        for ids in chunks:
            x = torch.tensor([list(ids)], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_ids=x, use_cache=False, logits_to_keep=1)
            for L in layers:
                acc[L] = acc[L] + cur[L] if L in acc else cur[L].clone()
    finally:
        for h in hs:
            h.remove()
    return np.stack([acc[L].double().cpu().numpy() for L in layers]).astype(np.float32)


def c1_texts(need: set) -> dict:
    from main_run.agm import data as AD

    docs, _, _ = AD.load_loracle_mix()
    return {d["doc_id"]: d for d in docs if d["doc_id"] in need}


def cmd_verify_act(out: Path, gpu: str, *, n_docs: int = 4, sft_dir=GRAD_SFT_DIR, rl_dir=GRAD_RL_DIR,
                   tokens_only: bool = False, model=None, tok=None, texts_c1=None) -> int:
    """Module docstring, verify-act. model / tok / texts_c1 replace the real ones (tests)."""
    import pandas as pd

    t0 = time.time()
    device = "cpu" if model is not None else (bind_gpu(gpu) if not tokens_only else "cpu")
    from main_run.agm import data as AD

    gmeta = json.loads((Path(sft_dir) / "meta.json").read_text())
    rmeta = json.loads((Path(rl_dir) / "meta.json").read_text())
    gidx = pd.read_parquet(Path(sft_dir) / "bundle_index.parquet")
    docs = doc_lists_from_samples(Path(sft_dir), SFT_SAMPLE_FILES)
    src = gidx["source"].to_numpy().astype(str) if "source" in gidx.columns else np.full(len(gidx), "c1")
    need_c1 = sorted({d for b, s in zip(gidx["bundle_id"], src) if s == "c1" for d in docs[b]})
    rdocs = pd.read_parquet(Path(rl_dir) / "docs.parquet")
    n0 = len(rdocs) - int(rmeta.get("n_beh_bundles", 0))
    fw = rdocs.iloc[:n0]
    gs = gmeta.get("grad_store", {})
    st = {"c1": store_open(gs["c1"]["path"] if "c1" in gs else gs["path"]), "fineweb": store_open(rmeta["grad_store"]["path"])}
    model_id = st["c1"]["man"]["plan"].get("model", "Qwen/Qwen3-14B")
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
    texts = {"c1": texts_c1 if texts_c1 is not None else c1_texts(set(need_c1)),
             "fineweb": {str(r.doc_id): dict(doc_id=str(r.doc_id), organism_id=str(r.organism_id), source=str(r.source),
                                             text=r.text) for r in fw.itertuples()}}
    want = {"c1": need_c1, "fineweb": [str(x) for x in fw["doc_id"]]}
    rep: dict = dict(experiment="activation twin: stored activations against the documents", model=model_id,
                     layers=list(LAYERS3), rule=POSITION_RULE, checks={})
    rows_of: dict = {}
    for s in ("c1", "fineweb"):
        missing = [d for d in want[s] if d not in texts[s]]
        dl = [texts[s][d] for d in want[s] if d in texts[s]]
        sha_bad = [d["doc_id"] for d in dl if AD.sha1(d["text"]) != d["doc_id"]]
        rows = []
        for k in range(0, len(dl), 2000):
            rows += AD.rows_from_raw(dl[k:k + 2000], tok)[0]
        rows_of[s] = rows
        exp, got = expected_counts(rows), store_counts(st[s]["df"])
        bad = [d for d in want[s] if d in exp and exp[d] != got.get(d)]
        lost = [d for d in want[s] if d in texts[s] and d not in exp]
        rep["checks"][f"tokens_{s}"] = dict(n_docs=len(want[s]), n_texts=len(dl), n_missing_text=len(missing),
                                            n_sha1_mismatch=len(sha_bad), n_count_mismatch=len(bad),
                                            n_no_chunk=len(lost), examples=bad[:3],
                                            pass_all_texts=not missing, pass_sha1=not sha_bad,
                                            pass_tokens_labels_chunks=not bad and not lost)
        log(f"{s}: {len(dl)} documents tokenised; count mismatches {len(bad)}, sha1 mismatches {len(sha_bad)}, "
            f"missing texts {len(missing)}")
    if not tokens_only:
        if model is None:
            import torch

            from main_run.reader.generate import load_base

            model = load_base(model_id, device)
            model.eval()
            torch.manual_seed(0)
        rng = np.random.default_rng(0)
        res = []
        for s in ("c1", "fineweb"):
            by_doc: dict = {}
            for r in rows_of[s]:
                by_doc.setdefault(r.doc_id, []).append(r)
            ds = sorted(by_doc)
            most = max(ds, key=lambda d: (len(by_doc[d]), d))
            rest = [d for d in ds if d != most]
            pick = [most] + [rest[int(i)] for i in rng.choice(len(rest), size=min(n_docs - 1, len(rest)), replace=False)]
            for d in pick:
                chunks = [r.ids for r in sorted(by_doc[d], key=lambda r: r.chunk_id)]
                F = act_forward_sums(model, chunks, LAYERS3, device)
                S = np.stack([store_sums(st[s], [[d]], "act", L)[0] for L in LAYERS3])
                cs = cos_rows(F, S)
                ratio = np.linalg.norm(F, axis=1) / np.linalg.norm(S, axis=1)
                res.append(dict(store=s, doc_id=d, n_chunks=len(chunks), n_tokens=sum(len(c) for c in chunks),
                                cos={str(L): float(c) for L, c in zip(LAYERS3, cs)},
                                norm_ratio={str(L): float(x) for L, x in zip(LAYERS3, ratio)}))
                log(f"{s} {d[:10]} ({len(chunks)} chunks): cos {np.round(cs, 6).tolist()}, norm ratio "
                    f"{np.round(ratio, 5).tolist()}")
        rep["checks"]["forward"] = dict(docs=res, cos_min=COS_MIN_VERIFY, norm_tol=NORM_TOL_VERIFY,
                                        pass_cos=all(min(r["cos"].values()) > COS_MIN_VERIFY for r in res),
                                        pass_norm=all(abs(x - 1) < NORM_TOL_VERIFY for r in res
                                                      for x in r["norm_ratio"].values()),
                                        pass_multi_chunk_c1=any(r["store"] == "c1" and r["n_chunks"] > 1 for r in res))
    failed = [k for k, v in flat_passes(rep["checks"]) if not v]
    rep.update(passed=not failed, failed=failed, tokens_only=tokens_only, wall_s=round(time.time() - t0, 1),
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = new_json(Path(out) / "VERIFY_ACT.json", rep)
    log(f"verify-act {'passed' if not failed else 'FAILED ' + str(failed)}; report {p}")
    return 0 if not failed else 1


# ---------------------------------------------------------------- read (GPU)
def load_reader_inputs(r: dict) -> dict:
    """The reader's held-out inputs in e2_generate.load_inputs' dict form (grad: through it; act: ao_heldout's loader,
    kind act, plus the window fields)."""
    if r["kind"] == "grad":
        return e2gen().load_inputs(Path(r["inputs"]))
    d = AH().load_inputs(Path(r["inputs"]), kind="act")
    with np.load(r["inputs"]) as z:
        windows = [str(w) for w in z["windows"]]
        lo, hi = np.asarray(z["window_lo"]), np.asarray(z["window_hi"])
    return dict(inputs=d["inputs"], steps=d["steps"], windows=windows, layers=d["layers"], window_lo=lo, window_hi=hi,
                n_docs=np.asarray(d["n_docs"]), meta=d["meta"])


def reading_problem(out, key: str, *, samples: int = C.SAMPLES, n_steps: int | None = None, inputs=None) -> str | None:
    """None when the reading is finished: settings status finished, its inputs file and order hash, every row with
    `samples` answers, n_steps rows (default: the inputs file's step count)."""
    rp = reading_path(out, key)
    sp = settings_path(rp)
    if not rp.exists() or not sp.exists():
        return "missing"
    s = json.loads(sp.read_text())
    if s.get("status") != "finished":
        return f"settings status {s.get('status')!r}"
    inputs = Path(inputs or readers()[key]["inputs"])
    if os.path.realpath(str(s.get("inputs"))) != os.path.realpath(inputs):
        return f"inputs {s.get('inputs')}, expected {inputs}"
    with np.load(inputs) as z:
        meta = json.loads(str(z["meta"]))
        n_in = len(z["steps"])
    if s.get("inputs_order_hash") != meta.get("order_hash"):
        return "inputs order hash differs"
    rows = C.read_jsonl(rp)
    n_steps = n_in if n_steps is None else n_steps
    if len(rows) != n_steps or [int(r["step"]) for r in rows] != list(range(n_steps)):
        return f"{len(rows)} rows, expected steps 0..{n_steps - 1}"
    if any(len(rec["texts"]) != samples for r in rows for rec in r["inputs"].values()):
        return f"a row without {samples} answers"
    return None


def cmd_read(out: Path, key: str, gpu: str, *, limit: int | None = None, samples: int = C.SAMPLES, adapter=None,
             max_gpu_gib: float = 80.0) -> int:
    t0 = time.time()
    out = Path(out)
    r = readers()[key]
    if limit is not None and under(out, OUT):
        log(f"--limit is a smoke option; give --out outside {OUT}")
        return 2
    rp = reading_path(out, key)
    adapter = Path(adapter or r["adapter"])
    if rp.exists():
        why = reading_problem(out, key, samples=samples, n_steps=limit)
        if why is None:
            # a finished reading counts only for the adapter asked for now (e.g. the SFT adapter after an RL cost cap
            # must not stand in for cycle 40, nor the reverse)
            done_with = json.loads(settings_path(rp).read_text()).get("adapter")
            if os.path.realpath(str(done_with)) != os.path.realpath(adapter):
                log(f"{rp} is finished but was read with adapter {done_with}, not {adapter}; set it (and its judged "
                    "file) aside by hand after checking")
                return 1
            log(f"{rp} is finished; left as it is")
            return 0
        log(f"{rp} exists but is not a finished reading ({why}); set it aside by hand after checking")
        return 1
    eg = e2gen()
    rs = eg.reader_settings(r["run_json"])
    inp = load_reader_inputs(r)
    ref_ids = json.loads(str(np.load(C.INPUTS["correct"])["meta"]))["bundle_ids"]
    pre = dict(adapter_exists=adapter.exists(), pass_layers=[int(L) for L in rs["layers"]] == list(LAYERS3),
               pass_reader_dir_kind=reader_dir_kind(rs["reader_dir"]) == r["kind"],
               pass_inputs_kind=inp["meta"].get("kind") == r["kind"],
               pass_inputs_layers=all(L in inp["layers"] for L in LAYERS3),
               pass_bundle_order=inp["meta"]["bundle_ids"] == ref_ids, pass_centred=bool(rs["center"]))
    pre["pass_adapter"] = pre.pop("adapter_exists")
    bad = [k for k, v in pre.items() if not v]
    if bad:
        log(f"{key}: checks before reading failed {bad}; nothing read")
        return 1
    device = bind_gpu(gpu)
    import torch
    from transformers import AutoTokenizer

    from main_run.reader.generate import load_adapter_fast, load_base

    total = torch.cuda.get_device_properties(torch.device(device)).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_gpu_gib * 2**30 / total), torch.device(device))
    mean = eg.load_mean_unit(Path(rs["reader_dir"]), rs["weighting"], rs["layers"])
    questions, qinfo = eg.resolve_questions([QKEY], generic_question=C.QUESTION_BEHAVIOURAL)
    tok = AutoTokenizer.from_pretrained(rs["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = load_base(rs["model"], device)
    model = load_adapter_fast(base, adapter)
    model.eval()
    log(f"{key}: base {rs['model']} + adapter {adapter} loaded ({time.time() - t0:.0f} s)")
    reader = eg.TrainedReader(model, tok, layers=rs["layers"], mean_unit=mean, hook_layer=rs["hook_onto_layer"],
                              steering_coefficient=rs["steering_coefficient"], questions=questions, n_samples=samples,
                              max_new_tokens=eg.MAX_NEW_TOKENS, seed=0, fast_gen=True)
    log(f"prompt ({len(reader.enc[QKEY].input_ids)} tokens):\n{reader.prompt_text(QKEY)}")
    stale = Path(str(rp) + ".tmp")
    if stale.exists():
        log(f"partial reading set aside: {set_aside(stale)}")
    extra = dict(reader_key=key, what=r["what"], adapter=str(adapter), default_adapter=str(r["adapter"]),
                 adapter_is_default=os.path.realpath(adapter) == os.path.realpath(r["adapter"]),
                 run_json=str(r["run_json"]), reader_dir=rs["reader_dir"], weighting=rs["weighting"], center=rs["center"],
                 generic_question=qinfo or None, inputs=str(r["inputs"]), inputs_kind=r["kind"],
                 inputs_order_hash=inp["meta"]["order_hash"], inputs_sign=inp["meta"]["sign"], checks=pre,
                 vector_rule="unit(unit_fp16(row) - mean_unit[layer]) x ||row|| (e2_generate.reader_vectors = "
                             "ReaderVectors.get(center=True) of training); the hook re-normalises it",
                 module="em-reader worktree e2_generate.TrainedReader and run_readings, driven by activation_twin.py",
                 max_gpu_gib=max_gpu_gib, status="running")
    steps = list(range(limit)) if limit else None
    res = eg.run_readings(reader, inp, out / "e2", key, steps=steps, tag=TAG, extra_settings=extra)
    secs = [x["seconds"] for x in res["timing"]]
    sp = settings_path(rp)
    s = json.loads(sp.read_text())
    s.update(status="finished", n_steps=len(steps or inp["steps"]), n_samples=samples, wall_s=res["wall_s"],
             seconds_per_call_median=float(np.median(secs)) if secs else None,
             peak_gib=round(torch.cuda.max_memory_allocated(torch.device(device)) / 2**30, 1),
             finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(sp, s)
    reader.close()
    log(f"{key}: {len(secs)} calls in {res['wall_s']} s (median {s['seconds_per_call_median']} s); peak "
        f"{s['peak_gib']} GiB; {rp}")
    return 0


# ---------------------------------------------------------------- yes/no (GPU, rare_yesno)
def yesno_inputs_check(act_dir: Path = YESNO_ACT_DIR, act_inputs: Path = ACT_INPUTS) -> dict:
    """rare_yesno's act bundle file (what its reader reads) against inputs_act_correct.npz: same rows bit for bit."""
    a = np.load(Path(act_dir) / "inputs_act_bundle.npz")
    b = np.load(act_inputs)
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    return dict(yesno_file=str(Path(act_dir) / "inputs_act_bundle.npz"), reading_file=str(act_inputs),
                pass_bundle_ids=ma["bundle_ids"] == mb["bundle_ids"],
                pass_layers=[int(x) for x in a["layers"]] == [int(x) for x in b["layers"]],
                pass_rows_bit_equal=bool(a["inputs"].shape == b["inputs"].shape and np.array_equal(a["inputs"],
                                                                                                   b["inputs"])))


def yesno_command(out: Path, key: str, gpu: str, *, adapter=None, limit=None, act_dir=YESNO_ACT_DIR) -> list[str]:
    r = readers()[key]
    cmd = [sys.executable, str(HERE / "rare_yesno.py"), "read", "--gpu", str(gpu), "--scorer", "reader", "--input",
           r["kind"], "--bundles", "all", "--run-dir", str(Path(out) / f"yesno_{key}"), "--reader-adapter",
           str(adapter or r["adapter"]), "--reader-run-json", str(r["run_json"]), "--reader-layers",
           ",".join(str(L) for L in LAYERS3), "--reader-name", key, "--act-dir", str(act_dir)]
    return cmd + (["--limit", str(limit)] if limit else [])


def cmd_yesno(out: Path, key: str, gpu: str, *, adapter=None, limit=None, act_dir=YESNO_ACT_DIR) -> int:
    out = Path(out)
    if limit is not None and under(out, OUT):
        log(f"--limit is a smoke option; give --out outside {OUT}")
        return 2
    run_dir = out / f"yesno_{key}"
    if key == "twin":
        chk = yesno_inputs_check(act_dir)
        atomic_json(out / "YESNO_INPUTS.json", chk)
        if not all(v for k, v in chk.items() if k.startswith("pass")):
            log(f"rare_yesno's act inputs differ from {ACT_INPUTS}: {chk}; nothing read")
            return 1
    rep = run_dir / "READ.json"
    if (run_dir / "pyes_bundle.jsonl").exists() and rep.exists() and json.loads(rep.read_text()).get("passed"):
        r = readers()[key]
        info = json.loads(rep.read_text()).get("reader") or {}
        want = dict(adapter=str(adapter or r["adapter"]), run_json=str(r["run_json"]))
        off = {k: (info.get(k), v) for k, v in want.items()
               if os.path.realpath(str(info.get(k))) != os.path.realpath(v)}
        if off:
            log(f"{run_dir}: a passed read exists but with {off} (read, asked); set it aside by hand after checking")
            return 1
        log(f"{run_dir}: a passed read exists; not read again")
    else:
        cmd = yesno_command(out, key, gpu, adapter=adapter, limit=limit, act_dir=act_dir)
        log("run: " + " ".join(cmd))
        rc = subprocess.run(cmd, cwd=str(C.REPO)).returncode
        if rc:
            log(f"rare_yesno read exit {rc}")
            return rc
    cmd = [sys.executable, str(HERE / "rare_yesno.py"), "summary", "--run-dir", str(run_dir), "--bundles", "all"]
    return subprocess.run(cmd, cwd=str(C.REPO)).returncode


# ---------------------------------------------------------------- judge (API)
@contextlib.contextmanager
def repointed(out: Path):
    """beh_judges pointed at the two readings: conditions = the reader keys, inputs of a key = its inputs file (the
    judge maps step -> bundle_id through its meta), reading = OUT/e2/<key>_generic_behavioural.jsonl."""
    B = beh_judges()
    out = Path(out)
    with patched(C, INPUTS={k: readers()[k]["inputs"] for k in KEYS}, EVAL=out, READ_LOGS=out / "read_logs"), \
            patched(B, CONDS=KEYS, reading_path=lambda key, cond: reading_path(out, cond)):
        yield B


def answer_count(path: Path) -> int:
    return sum(len(rec["texts"]) for r in C.read_jsonl(path) for rec in r["inputs"].values())


UNIT_FALLBACK_USD = 8.0e-5 / 2  # em_judges.judge_requests_batch's own pre-send estimate per batch request
EST_MARGIN = 2.0  # the pre-send estimate must fit the cap twice over


def judged_check(out: Path, B, ctx) -> dict:
    """{key: rows}: a judged file holds exactly its reading's requests, row for row (step, window, answer index,
    request_sha of beh_judges.file_requests = the bundle's ground truth and that answer). A judged file left from another
    reading (an older one, another adapter) fails, so its verdicts can never stand in. Call inside repointed(out)."""
    res = {}
    for k in KEYS:
        rp, jp = reading_path(out, k), judged_path(out, k)
        if not rp.exists():
            res[k] = dict(reading=str(rp), pass_reading=False)
            continue
        _, keys = B.file_requests(rp, "truth", ctx.meta(k), ctx)
        want = [(int(s), str(w), int(i), str(sh)) for s, w, i, sh in keys]
        e = dict(reading=str(rp), judged=str(jp), n_answers=len(want), judged_exists=jp.exists())
        if jp.exists():
            got = [(int(r["step"]), str(r["window"]), int(r["answer_index"]), str(r["request_sha"]))
                   for r in C.read_jsonl(jp)]
            e.update(n_judged=len(got), n_rows_differ=sum(a != b for a, b in zip(got, want)) + abs(len(got) - len(want)),
                     pass_rows_equal_reading=got == want)
        res[k] = e
    return res


def cmd_judge(out: Path, cap: float = CAP_USD, mode: str = "batch", workers: int = 48, dry: bool = False) -> int:
    out = Path(out)
    not_ready = {k: why for k in KEYS if (why := reading_problem(out, k)) is not None}
    if not_ready and not dry:
        log(f"judge refused, nothing sent: readings not finished {not_ready}")
        return 1
    import rare_perdoc as RP

    with repointed(out) as B:
        unit, nreq = RP.truth_unit_cost(B)
        ctx = B.Context()
        pre = judged_check(out, B, ctx)
        stale = [k for k, e in pre.items() if e.get("pass_rows_equal_reading") is False]
        reqs, todo, skipped = B.judge_todo("truth", (JUDGE_KEY,), ctx)
        cache = B.J.load_cache("truth")
        new = sum(s not in cache for s in reqs)
        u = unit if math.isfinite(unit) else UNIT_FALLBACK_USD
        est = new * u
        for s in skipped:
            log(f"not judged: {s}")
        log(f"{len(todo)} readings to judge, {sum(len(k) for _, k in todo)} answers, {len(reqs)} distinct requests "
            f"({new} new, about {est:.2f} USD batch at {u:.2e} USD per request)")
        if stale:
            log(f"judge refused, nothing sent: judged files that are not their reading's requests {stale} "
                f"({ {k: pre[k] for k in stale} }); set them aside by hand after checking")
            return 1
        if dry:
            log("dry run: nothing sent")
            return 0
        if est * EST_MARGIN > cap:
            log(f"judge refused, nothing sent: {EST_MARGIN} x the estimate {est:.2f} USD is above the cap {cap} USD")
            return 1
        led0 = B.J.ledger_total()
        log(f"ledger {led0:.3f} USD at start; cap {cap} USD -> --max-cost {led0 + cap:.3f} (em_judges refuses a batch "
            f"whose estimate would pass it, before sending)")
        rc = B.cmd_judge("truth", (JUDGE_KEY,), max_cost=led0 + cap, mode=mode, workers=workers, ctx=ctx)
        led1 = B.J.ledger_total()
        post = judged_check(out, B, ctx)
    bad = [k for k, e in post.items() if e.get("pass_rows_equal_reading") is not True]
    rep = dict(ledger_start=led0, ledger_end=led1, increase=led1 - led0, cap_usd=cap, pass_cap=led1 - led0 <= cap + 1e-6,
               new_requests=new, estimate_usd_batch=est, unit_usd=u, files=post, beh_judges_exit=rc,
               note="the ledger total counts every judge process; another judge running at the same time adds to it",
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = new_json(out / "JUDGE.json", rep)
    log(f"ledger {led0:.3f} -> {led1:.3f} USD (+{led1 - led0:.3f}, cap {cap}); report {p}"
        + (f"; not judged or rows differ from the reading: {bad}" if bad else ""))
    return rc or (1 if bad or not rep["pass_cap"] else 0)


# ---------------------------------------------------------------- summary
def npz_ids(path: Path) -> list:
    with np.load(path) as z:
        return json.loads(str(z["meta"]))["bundle_ids"]


def column_sources(out: Path) -> dict:
    """{column: (judged file, step -> bundle_id list, answers per bundle)}."""
    return {"reader40": (C.eval_dir("rl_beh_c040") / "readings_correct_judge_truth.jsonl", Path(C.INPUTS["correct"]), 20),
            "grad3": (judged_path(out, "grad3"), Path(C.INPUTS["correct"]), 20),
            "twin": (judged_path(out, "twin"), ACT_INPUTS, 20),
            "ao3_act_centred": (AO_OUT / "readings_ao3layer_act_centred_judge_truth.jsonl",
                                AO_OUT / "inputs_act_centred_correct.npz", 10)}


def yes_rates(judged: Path, ids: list) -> dict:
    """{bundle_id: YES / (YES + NO)} over a judged file's answers (REFUSED left out; ao_heldout.rate_of)."""
    out: dict = {}
    for r in C.read_jsonl(judged):
        out.setdefault(ids[int(r["step"])], []).append(r["verdict"])
    res = {}
    for b, vs in out.items():
        y, nn = sum(v == "YES" for v in vs), sum(v in ("YES", "NO") for v in vs)
        res[b] = (y / nn) if nn else None
    return res


def boot_mean(vals, seed: int = C.BOOT_SEED, n_boot: int = 2000) -> dict:
    """Mean and bundle bootstrap 95% CI (beh_judges.boot_mean)."""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    m = v[np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return dict(mean=float(v.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=len(v))


def hyper_tail(N: int, K: int, n: int, k: int) -> float:
    if n > N or K > N:
        return float("nan")
    return sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(k, min(n, K) + 1)) / math.comb(N, n)


def concentration(sel: list, base: list) -> dict:
    out = {}
    for kind in ("family", "evidence"):
        cnt, bc = Counter(r[kind] for r in sel), Counter(r[kind] for r in base)
        out[kind] = {g: dict(k=cnt[g], share=(cnt[g] / len(sel)) if sel else None, base_k=bc[g],
                             base_share=bc[g] / len(base) if base else None,
                             p_hypergeom=hyper_tail(len(base), bc[g], len(sel), cnt[g]) if cnt[g] else None)
                     for g in sorted(bc)}
    return out


def verdict(sel: list, base: list, who: str) -> dict:
    """ao_heldout's rule (plan sections 4 and 8 of 20261028_reader_vs_ao_heldout_plan.md) with the reader names."""
    n, N = len(sel), len(base)
    conc = concentration(sel, base)
    over = [dict(kind=kind, name=g, **e, counts=e["p_hypergeom"] is not None and e["p_hypergeom"] < P_MAX)
            for kind, d in conc.items() for g, e in d.items() if n and 2 * e["k"] > n]
    counted = [o for o in over if o["counts"]]
    desc = lambda o: (f"{o['kind']} {o['name']} ({o['k']}/{n}; {o['base_k']}/{N} of all bundles, hypergeometric p "  # noqa: E731
                      f"{o['p_hypergeom']:.2g})")
    if n < MIN_CANDIDATES:
        res, text = "too_few", f"{n} bundles (fewer than {MIN_CANDIDATES}): no claim"
    elif counted:
        res, text = "concentrated", (f"{n} bundles; more than half in " + " and ".join(desc(o) for o in counted)
                                     + f": that kind is named only by {who}")
    elif over:
        res, text = "scattered", (f"{n} bundles; more than half in " + " and ".join(desc(o) for o in over)
                                  + f", but p >= {P_MAX}: the base rate explains it")
    else:
        res, text = "scattered", f"{n} bundles, no family or evidence kind holds more than half"
    return dict(result=res, text=text, n=n, n_base=N, concentration=conc, over_half=over)


def yesno_tables(out: Path) -> dict:
    """{column: {metric: entry}} from the four rare_yesno summaries (bundle.<family>.own_vs_other_pos and
    bundle.<quirk|rare|backdoor>.pos_vs_neg), missing summaries reported."""
    paths = dict(YESNO_SUMMARIES, grad3=Path(out) / "yesno_grad3" / "summary.json",
                 twin=Path(out) / "yesno_twin" / "summary.json")
    res, missing = {}, []
    for col, _ in COLUMNS:
        p = paths[col]
        if not p.exists():
            missing.append(str(p))
            continue
        s = json.loads(p.read_text())
        res[col] = {k: {x: v.get(x) for x in ("auroc", "lo", "hi", "n_pos", "n_neg")}
                    for k, v in s["results"].items() if k.startswith("bundle.")}
        res[col]["_source"] = dict(path=str(p), reader=s.get("reader"), input=s.get("input"),
                                   read_report_passed=s.get("read_report_passed"))
    return dict(tables=res, missing=missing)


def flag(dy, da) -> str:
    if dy is None or da is None:
        return "no_data"
    if abs(dy) > FLAG_YES and abs(da) > FLAG_AUC:
        return ("twin_higher" if dy > 0 else "grad3_higher") if (dy > 0) == (da > 0) else "mixed"
    return "no_claim"


def summarize(out: Path, snapshot=None) -> dict:
    out = Path(out)
    bundles = {b["bundle_id"]: b for b in C.load_bundles(snapshot)}
    src = column_sources(out)
    rates, missing, sources = {}, [], {}
    for col, (judged, inputs, n_ans) in src.items():
        if not Path(judged).exists() or not Path(inputs).exists():
            missing.append(str(judged))
            continue
        rates[col] = yes_rates(judged, npz_ids(inputs))
        sources[col] = dict(judged=str(judged), inputs=str(inputs), answers_per_bundle=n_ans)
    ids = npz_ids(C.INPUTS["correct"]) if Path(C.INPUTS["correct"]).exists() else sorted({b for r in rates.values()
                                                                                          for b in r})
    rows = []
    for b in ids:
        bb = bundles.get(b, {})
        rows.append(dict(bundle_id=b, family=bb.get("family"), evidence=bb.get("evidence"),
                         behaviour_text=bb.get("behaviour_text"),
                         **{c: rates[c].get(b) if c in rates else None for c, _ in COLUMNS}))
    fams = [f for f in C.TEST_FAMILY_COUNTS if any(r["family"] == f for r in rows)]
    fams += sorted({r["family"] for r in rows if r["family"] not in fams and r["family"] is not None})
    cols = [c for c, _ in COLUMNS]
    fam_tab: dict = {}
    for g in fams + ["all"]:
        sub = rows if g == "all" else [r for r in rows if r["family"] == g]
        fam_tab[g] = {c: boot_mean([r[c] for r in sub if r[c] is not None]) for c in cols}
        pair = [r["twin"] - r["grad3"] for r in sub if r["twin"] is not None and r["grad3"] is not None]
        fam_tab[g]["twin_minus_grad3"] = boot_mean(pair)
    ev_tab = {}
    for e in sorted({r["evidence"] for r in rows if r["evidence"]}):
        sub = [r for r in rows if r["evidence"] == e]
        ev_tab[e] = {c: boot_mean([r[c] for r in sub if r[c] is not None]) for c in cols}
    both = [r for r in rows if r["twin"] is not None and r["grad3"] is not None]
    grad_only = [r for r in both if r["grad3"] >= YES_HI - EPS and r["twin"] <= NO_LO + EPS]
    twin_only = [r for r in both if r["twin"] >= YES_HI - EPS and r["grad3"] <= NO_LO + EPS]
    keep = ("bundle_id", "family", "evidence", "behaviour_text") + tuple(cols)
    cand = {"grad3_only": dict(rule="three-layer gradient reader >= 0.5 and twin <= 0.1",
                               bundles=[{k: r[k] for k in keep} for r in grad_only],
                               verdict=verdict(grad_only, both, "the gradient reader")),
            "twin_only": dict(rule="twin >= 0.5 and three-layer gradient reader <= 0.1",
                              bundles=[{k: r[k] for k in keep} for r in twin_only],
                              verdict=verdict(twin_only, both, "the activation twin"))}
    yn = yesno_tables(out)
    own = {f: {c: (yn["tables"].get(c, {}).get(f"bundle.{f}.own_vs_other_pos") or {}) for c in cols} for f in fams}
    pn = {f: {c: (yn["tables"].get(c, {}).get(f"bundle.{f}.pos_vs_neg") or {}) for c in cols} for f in NEG_FAMILIES}
    flags = {}
    for f in fams:
        dy = fam_tab[f]["twin_minus_grad3"]["mean"]
        a1, a0 = own[f]["twin"].get("auroc"), own[f]["grad3"].get("auroc")
        da = (a1 - a0) if a1 is not None and a0 is not None else None
        flags[f] = dict(yes_diff=dy, auroc_diff=da, result=flag(dy, da))
    settings = {}
    for k in KEYS:
        sp = settings_path(reading_path(out, k))
        if sp.exists():
            s = json.loads(sp.read_text())
            settings[k] = {x: s.get(x) for x in ("adapter", "adapter_is_default", "inputs", "inputs_kind", "n_samples",
                                                 "n_steps", "status", "max_new_tokens", "wall_s", "finished")}
    comparison = adapter_comparison(out, settings)
    return dict(
        experiment="activation twin against the three-layer gradient reader on the 530 held-out bundles (user decision "
                   "2026-10-29 03:30; plan docs/notes/20261029_activation_twin_plan.md)",
        columns=dict(COLUMNS), families=fams, sources=sources, readings=settings, adapters=comparison, missing=missing,
        rate_rule="a bundle's YES / (YES + NO) over its answers (REFUSED left out); mean over bundles; bundle "
                  "bootstrap 95% CI (2,000 resamples, seed common.BOOT_SEED); twin_minus_grad3 pairs each bundle",
        flag_rule=(f"plan section 5, single seed: a family differs only when |YES twin - grad3| > {FLAG_YES} and "
                   f"|own-vs-other AUROC twin - grad3| > {FLAG_AUC}, both in the same direction (else mixed / no_claim)"),
        results=dict(family=fam_tab, evidence=ev_tab, candidates=cand, yesno_own_vs_other=own, yesno_pos_vs_neg=pn,
                     flags=flags),
        yesno_sources={c: v.get("_source") for c, v in yn["tables"].items()}, yesno_missing=yn["missing"],
        per_bundle=rows, written=time.strftime("%Y-%m-%d %H:%M:%S"))


def adapter_stage(adapter) -> str | None:
    """'rl' for an RL cycle-40 adapter, 'sft' for an SFT step-7,082 adapter, else the folder name (None: unknown)."""
    if not adapter:
        return None
    name = Path(str(adapter)).name
    return {RL_FINAL: "rl", SFT_FINAL: "sft"}.get(name, name)


def adapter_comparison(out: Path, settings: dict) -> dict:
    """The adapter each reader was evaluated with (open reading settings and yes/no READ.json) and whether the two
    readers were compared at the same stage (plan section 7: after an RL cost cap the twin is read with its SFT adapter;
    then the comparison is SFT twin against RL gradient reader and the summary says so)."""
    used = {}
    for k in KEYS:
        rep = Path(out) / f"yesno_{k}" / "READ.json"
        yes_ad = (json.loads(rep.read_text()).get("reader") or {}).get("adapter") if rep.exists() else None
        used[k] = dict(open_reading=(settings.get(k) or {}).get("adapter"), yesno=yes_ad,
                       open_stage=adapter_stage((settings.get(k) or {}).get("adapter")), yesno_stage=adapter_stage(yes_ad))
    st = [used[k][x] for k in KEYS for x in ("open_stage", "yesno_stage")]
    known = [x for x in st if x is not None]
    same = (len(set(known)) == 1) if len(known) == len(st) else None
    return dict(per_reader=used, same_stage=same,
                note=None if same in (True, None) else "the two readers were not evaluated at the same training stage "
                                                       "(RL cost cap: the twin was read with its SFT adapter)")


def _f(x, w=6) -> str:
    return " " * (w - 1) + "-" if x is None else f"{x:{w}.3f}"


def print_summary(s: dict) -> None:
    cols = [c for c, _ in COLUMNS]
    head = {"reader40": "rdr40", "grad3": "grad3", "twin": "twin", "ao3_act_centred": "AO3ac"}
    ft = s["results"]["family"]
    print("truth judge YES rate per family (mean over bundles)")
    print(f"{'family':12s} {'n':>4s} " + " ".join(f"{head[c]:>6s}" for c in cols) + f" {'twin-g3':>8s} {'95% CI':>16s}")
    for g, e in ft.items():
        d = e["twin_minus_grad3"]
        ci = "" if d["lo"] is None else f"[{d['lo']:+.3f}, {d['hi']:+.3f}]"
        n = max((e[c]["n"] for c in cols), default=0)
        print(f"{g:12s} {n:4d} " + " ".join(_f(e[c]["mean"]) for c in cols) + f" {_f(d['mean'], 8)} {ci:>16s}")
    print("yes/no AUROC, own against other question (positive bundles)")
    for f, e in s["results"]["yesno_own_vs_other"].items():
        print(f"{f:12s}      " + " ".join(_f(e[c].get("auroc")) for c in cols))
    print("yes/no AUROC, bundles showing the behaviour against none")
    for f, e in s["results"]["yesno_pos_vs_neg"].items():
        print(f"{f:12s}      " + " ".join(_f(e[c].get("auroc")) for c in cols))
    print("flags: " + ", ".join(f"{f} {v['result']}" for f, v in s["results"]["flags"].items()))
    for k, v in s["results"]["candidates"].items():
        print(f"{k}: {v['verdict']['text']}")
    if s["missing"] or s["yesno_missing"]:
        print("missing: " + ", ".join(s["missing"] + s["yesno_missing"]))
    a = s.get("adapters") or {}
    if a.get("same_stage") is False:
        print("WARNING: " + a["note"] + ": " + json.dumps({k: (v["open_stage"], v["yesno_stage"])
                                                           for k, v in a["per_reader"].items()}))


def cmd_summary(out: Path) -> int:
    s = summarize(out)
    stale = []
    if all(reading_path(out, k).exists() for k in KEYS):  # the judged files are their readings' own verdicts
        with repointed(out) as B:
            s["judged_vs_reading"] = judged_check(out, B, B.Context())
        stale = [k for k, e in s["judged_vs_reading"].items() if e.get("pass_rows_equal_reading") is False]
    q = set_aside(Path(out) / "summary.json")
    if q:
        log(f"moved the old summary to {q}")
    atomic_json(Path(out) / "summary.json", s)
    print_summary(s)
    need = [str(judged_path(out, k)) for k in KEYS]
    miss = [m for m in s["missing"] if m in need]
    log(f"wrote {Path(out) / 'summary.json'}" + (f"; missing {miss}" if miss else "")
        + (f"; judged files that are not their reading's verdicts {stale}" if stale else ""))
    return 1 if miss or stale else 0


# ---------------------------------------------------------------- plot
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # reference categorical palette slots 1-4, in order
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e2e1dc", "#fcfcfb"


def cmd_plot(out: Path, fig_path: Path = FIG) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads((Path(out) / "summary.json").read_text())
    cols = [(c, lab, col) for (c, lab), col in zip(COLUMNS, SERIES_COLOURS)]
    fams = s["families"]
    ft = s["results"]["family"]
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10,
                         "legend.fontsize": 10.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig = plt.figure(figsize=(16, 10), dpi=150)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.5, wspace=0.3)
    plotted: dict = {}

    def bars(ax, groups, get, ylabel, title, chance=None):
        bw = 0.8 / len(cols)
        for j, (c, lab, colour) in enumerate(cols):
            xs, ms, lo, hi = [], [], [], []
            for i, g in enumerate(groups):
                e = get(g, c)
                if not e or e.get("mean", e.get("auroc")) is None:
                    continue
                m = e.get("mean", e.get("auroc"))
                xs.append(i - 0.4 + bw * (j + 0.5))
                ms.append(m)
                lo.append(m - e["lo"] if e.get("lo") is not None else 0)
                hi.append(e["hi"] - m if e.get("hi") is not None else 0)
                plotted.setdefault(title, {}).setdefault(g, {})[c] = dict(value=m, lo=e.get("lo"), hi=e.get("hi"))
            ax.bar(xs, ms, bw - 0.02, color=colour, zorder=3, label=lab)
            ax.errorbar(xs, ms, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=0, zorder=4)
        if chance is not None:
            ax.axhline(chance, color=INK2, linewidth=1, linestyle="--", zorder=2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, fontsize=12)

    ax = fig.add_subplot(gs[0, :])
    groups = fams + ["all"]
    bars(ax, groups, lambda g, c: ft[g][c], "truth judge YES rate",
         "open question: how often the reading names the held-out behaviour (mean over bundles, 95% CI)")
    ax.set_xticklabels([f"{g}\n(n={max(ft[g][c]['n'] for c, *_ in cols)})" for g in groups])
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=colour, label=lab) for _, lab, colour in cols], loc="lower center",
              bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False, handlelength=1.4)  # also for a series with no bars
    ax2 = fig.add_subplot(gs[1, :3])
    own = s["results"]["yesno_own_vs_other"]
    bars(ax2, fams, lambda g, c: own[g][c], "AUROC of P(Yes)",
         "yes/no: own behaviour against another behaviour of the family (bundles that show it)", chance=0.5)
    ax3 = fig.add_subplot(gs[1, 3])
    pn = s["results"]["yesno_pos_vs_neg"]
    bars(ax3, list(pn), lambda g, c: pn[g][c], "AUROC of P(Yes)", "yes/no: showing against not showing", chance=0.5)
    fig.text(0.5, 0.005, "single seed; dashed line = chance; reference columns: the 40-layer gradient reader and the "
             "three-layer public AO on centred activation rows", ha="center", color=INK2, fontsize=10)
    fig_path = Path(fig_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (fig_path, fig_path.with_suffix(".json")):
        q = set_aside(p)
        if q:
            log(f"moved the old file to {q}")
    fig.savefig(fig_path, bbox_inches="tight")
    atomic_json(fig_path.with_suffix(".json"), dict(source=str(Path(out) / "summary.json"), plotted=plotted,
                                                    flags=s["results"]["flags"],
                                                    written=time.strftime("%Y-%m-%d %H:%M:%S")))
    log(f"wrote {fig_path}")
    return 0


# ---------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {k: sub.add_parser(k) for k in ("verify-act", "build-dirs", "check-dirs", "read", "yesno", "judge", "summary",
                                         "plot")}
    for p in ps.values():
        p.add_argument("--out", type=Path, default=OUT)
    for k in ("verify-act", "read", "yesno"):
        ps[k].add_argument("--gpu", required=True, help="one nvidia-smi index")
    for k in ("build-dirs", "check-dirs"):
        ps[k].add_argument("--act-sft-dir", type=Path, default=ACT_SFT_DIR)
        ps[k].add_argument("--act-rl-dir", type=Path, default=ACT_RL_DIR)
    ps["verify-act"].add_argument("--n-docs", type=int, default=4)
    ps["verify-act"].add_argument("--tokens-only", action="store_true", help="the token check alone (no model)")
    for k in ("read", "yesno"):
        ps[k].add_argument("--reader", choices=KEYS, required=True)
        ps[k].add_argument("--limit", type=int, default=None, help="first N steps (smoke; refused inside the real OUT)")
        ps[k].add_argument("--adapter", type=Path, default=None,
                           help="another adapter of the reader (the SFT adapter when RL stopped at its cost cap)")
    ps["judge"].add_argument("--cap-usd", type=float, default=CAP_USD)
    ps["judge"].add_argument("--mode", choices=("batch", "sync"), default="batch")
    ps["judge"].add_argument("--workers", type=int, default=48)
    ps["judge"].add_argument("--dry-run", action="store_true")
    ps["plot"].add_argument("--fig", type=Path, default=FIG)
    a = ap.parse_args(argv)
    if getattr(a, "gpu", None) is not None and str(a.gpu).strip() == "0":
        ap.error("--gpu 0: GPU 0 belongs to the already-learned chain (activation twin rule, 2026-10-29)")
    if a.cmd == "verify-act":
        return cmd_verify_act(a.out, a.gpu, n_docs=a.n_docs, tokens_only=a.tokens_only)
    if a.cmd == "build-dirs":
        for d in (a.act_sft_dir, a.act_rl_dir):
            if d.resolve() in (GRAD_SFT_DIR.resolve(), GRAD_RL_DIR.resolve()):
                ap.error(f"{d} is a gradient reader dir")
        return cmd_build_dirs(a.out, a.act_sft_dir, a.act_rl_dir)
    if a.cmd == "check-dirs":
        return cmd_check_dirs(a.act_sft_dir, a.act_rl_dir)
    if a.cmd == "read":
        return cmd_read(a.out, a.reader, a.gpu, limit=a.limit, adapter=a.adapter)
    if a.cmd == "yesno":
        return cmd_yesno(a.out, a.reader, a.gpu, adapter=a.adapter, limit=a.limit)
    if a.cmd == "judge":
        return cmd_judge(a.out, a.cap_usd, a.mode, a.workers, a.dry_run)
    if a.cmd == "summary":
        return cmd_summary(a.out)
    return cmd_plot(a.out, a.fig)


if __name__ == "__main__":
    raise SystemExit(main())
