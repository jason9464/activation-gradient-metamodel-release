#!/usr/bin/env python
"""Held-out behaviour bundle vectors as e2_generate inputs (docs/notes/20261027_behaviour_reader_plan.md section 10b).

  PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader python main_run/beh_reader/export_inputs.py

The 530 beh_test bundles of the SFT reader dir (common.SFT_DIR) as two e2_generate inputs files in the format of
main_run/em_reader/export_em_inputs.py (the EM v2 counterpart; e2_read.py --export-inputs format), so the em-reader
worktree's e2_generate.py reads every bundle with any reader adapter and the behavioural question:
  common.INPUTS["correct"]   runs/beh_reader/eval/inputs_correct.npz: each bundle's own vector
  common.INPUTS["shuffled"]  runs/beh_reader/eval/inputs_shuffled.npz: at the same step, the vector of the bundle's
                             shuffled partner from beh_eval_samples.jsonl (plan 4: seed 1 permutation inside beh_test,
                             not itself, not the same unit_id or pair_id; the partner's bundle_id is the row's
                             grad_bundle_id and its vector row the row's vec_row)
Both files (the plan's single "inputs_bundles.npz" is split in two by the module contract, common.INPUTS):
  steps     0 .. 529 = the beh_test bundles in bundles.jsonl order (the reader dir's behaviour rows follow
            common.row_order, so beh_test keeps its file order); meta bundle_ids maps a step back to its bundle
  windows   ["bundle"]; window_lo = window_hi = 0; n_docs = the example count of the bundle whose vector is used
  inputs    fp32 [n, 1, 40, 5120] = unit x norm read from the SFT reader dir (ReaderVectors.get, uncentred), the same
            -dL/dh token sum the reader trains on; e2_generate centres it on the reader run's mean_unit file
  meta      run (= the eval folder; e2_generate writes its readings to <run>/e2), sign, kind "grad", order_hash,
            condition, bundle_ids, family, evidence, pair_id, unit_id, behaviour_id (all of the step's own bundle),
            reader_dir and its bundle_order_hash; the shuffled file adds partner_bundle_ids and partner_rows
Inputs (read only): common.SFT_DIR (meta.json, bundle_index.parquet, bundle_tokensum_L*.npy, beh_eval_samples.jsonl),
common.SNAPSHOT bundles.jsonl. Outputs: the two npz files and <eval>/EXPORT.json (checks; an existing report is kept
and a new one gets a time stamp). CPU only; ReaderVectors comes from the em-reader worktree (PYTHONPATH above).
Checks (any failure stops before writing, or exits 1 after it): the index's beh_test rows equal bundles.jsonl's
beh_test bundles in order and in family, evidence, unit_id, pair_id, behaviour_id; the index rows are positional;
every beh_test bundle has its two shuffled rows with one partner, the partner is another beh_test bundle with another
unit_id and pair_id and vec_row is the partner's index row; after writing, 3 bundles per file (first, middle, last)
re-read from the reader dir equal the export bit for bit, the order hash re-computed from the file equals meta, and
every exported vector is finite and nonzero. An existing pair of files is verified again (same checks against the
current reader dir) and left as it is; a single file or a file that fails is refused.
Memory: 530 x 40 x 5120 fp32 = 434 MB per file; the shuffled rows are a permutation of the same 530 rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

EX = C.em_module("export_em_inputs")  # export_order_hash (the hash e2_generate.load_inputs re-computes)
WEIGHTING = "tokensum"
N_QA = 2  # qa rows per bundle and condition in beh_eval_samples.jsonl (plan 4)
META_FIELDS = ("family", "evidence", "pair_id", "unit_id", "behaviour_id")
SIGN = ("bundle vectors of the behaviour reader dir (v3): fp16 unit rows x fp32 norms of the summed -dL/dh, i.e. "
        "stored +dL/dh summed over the bundle's examples and negated; inputs = -(sum over the bundle's examples of "
        "+dL/dh): the descent direction, stored as fp32")


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [export_inputs] {msg}", flush=True)


def _none(x):
    """None for a missing parquet value (None, NaN, pd.NA), else the value."""
    if x is None or (not isinstance(x, (list, tuple, np.ndarray)) and pd.isna(x)):
        return None
    return x


def test_index_rows(index: pd.DataFrame, bundles: list[dict]) -> tuple[pd.DataFrame, list[dict], dict]:
    """(index rows of the beh_test bundles, bundles.jsonl's beh_test rows, checks): order and fields compared."""
    te = index[(index["source"] == "behaviour") & (index["split"] == C.SPLIT_TEST)]
    want = [b for b in bundles if b["split"] == C.SPLIT_TEST]
    checks = dict(index_rows_positional=bool((index["row"].to_numpy() == np.arange(len(index))).all()),
                  test_order=te["bundle_id"].tolist() == [b["bundle_id"] for b in want])
    same = checks["test_order"]
    for f in META_FIELDS:
        same = same and [_none(x) for x in te[f].tolist()] == [b[f] for b in want]
    checks["test_fields"] = bool(same)
    return te, want, checks


def shuffled_partners(samples: list[dict], want: list[dict], index: pd.DataFrame) -> tuple[list[str], list[int], dict]:
    """(partner bundle ids, partner vector rows) in step order from the shuffled rows of beh_eval_samples.jsonl,
    and the checks of plan 4's partner rule."""
    row_of = dict(zip(index["bundle_id"], index["row"].astype(int)))
    by_id = {b["bundle_id"]: b for b in want}
    got: dict[str, list[tuple[str, int]]] = {}
    for r in samples:
        if r.get("condition") == "shuffled":
            got.setdefault(r["bundle_id"], []).append((r["grad_bundle_id"], int(r["vec_row"])))
    ids, rows = [], []
    checks = dict(shuffled_rows_per_bundle=set(got) == set(by_id), one_partner=True, partner_in_test=True,
                  partner_not_self=True, partner_other_unit_pair=True, partner_vec_row=True)
    for b in want:
        pairs = got.get(b["bundle_id"], [])
        checks["shuffled_rows_per_bundle"] &= len(pairs) == N_QA
        checks["one_partner"] &= len(set(pairs)) == 1
        pid, prow = pairs[0] if pairs else (None, -1)
        p = by_id.get(pid)
        checks["partner_in_test"] &= p is not None
        checks["partner_not_self"] &= pid != b["bundle_id"]
        if p is not None:
            checks["partner_other_unit_pair"] &= p["unit_id"] != b["unit_id"] and (
                b["pair_id"] is None or p["pair_id"] != b["pair_id"])
        checks["partner_vec_row"] &= row_of.get(pid) == prow
        ids.append(pid)
        rows.append(prow)
    return ids, rows, {k: bool(v) for k, v in checks.items()}


def base_meta(run: Path, want: list[dict], sft_dir: Path, dir_meta: dict, layers) -> dict:
    steps = list(range(len(want)))
    return dict(run=str(run), sign=SIGN, kind="grad", order_hash=EX.export_order_hash(steps, ["bundle"], layers),
                arm=None, split=C.SPLIT_TEST, bundle_ids=[b["bundle_id"] for b in want],
                **{f: [b[f] for b in want] for f in META_FIELDS},
                reader_dir=str(sft_dir), reader_dir_bundle_order_hash=dir_meta.get("bundle_order_hash"),
                written=time.strftime("%Y-%m-%d %H:%M:%S"), code=os.path.abspath(__file__))


def write_npz(path: Path, X: np.ndarray, meta: dict, n_docs: np.ndarray, layers) -> None:
    """One e2_generate inputs file, written through <path>.tmp and a rename; an existing file is refused."""
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = X.shape[0]
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, inputs=X.astype(np.float32), steps=np.arange(n), windows=np.array(["bundle"]),
                 layers=np.array(list(layers)), window_lo=np.zeros((n, 1), np.int64),
                 window_hi=np.zeros((n, 1), np.int64), n_docs=np.asarray(n_docs, np.int64)[:, None],
                 meta=json.dumps(meta))
    os.replace(tmp, path)


def file_checks(path: Path, rv, vec_rows, layers, meta_want: dict) -> dict:
    """Re-read checks of one written file: 3 bundles equal the reader dir, order hash, meta, finite and nonzero."""
    with np.load(path) as z:
        X = z["inputs"]
        meta = json.loads(str(z["meta"]))
        steps, wins, lays = [int(t) for t in z["steps"]], [str(w) for w in z["windows"]], [int(L) for L in z["layers"]]
    n = X.shape[0]
    picks = sorted({0, n // 2, n - 1})
    reread = all(np.array_equal(X[i, 0], rv.get([vec_rows[i]], WEIGHTING, center=False, layers=layers)[0].numpy())
                 for i in picks)
    nrm = np.linalg.norm(X[:, 0], axis=-1)
    return dict(reread_equal=bool(reread), reread_steps=picks,
                order_hash=EX.export_order_hash(steps, wins, lays) == meta["order_hash"],
                shape=list(X.shape) == [len(meta_want["bundle_ids"]), 1, len(layers), X.shape[-1]],
                meta_bundles=meta["bundle_ids"] == meta_want["bundle_ids"]
                and meta.get("partner_bundle_ids") == meta_want.get("partner_bundle_ids"),
                reader_dir_hash=meta["reader_dir_bundle_order_hash"] == meta_want["reader_dir_bundle_order_hash"],
                finite_nonzero=bool(np.isfinite(X).all() and (nrm > 0).all()))


def plan_files(rv, sft_dir: Path, snapshot, layers, expected_n) -> tuple[dict, dict]:
    """({condition: (vector rows, meta)}, checks) before anything is written; SystemExit when a check fails."""
    dir_meta = json.loads((Path(sft_dir) / "meta.json").read_text())
    te, want, checks = test_index_rows(rv.index, C.load_bundles(snapshot))
    checks["n_test_bundles"] = expected_n is None or len(want) == expected_n
    samples = C.read_jsonl(Path(sft_dir) / C.BEH_EVAL_SAMPLES)
    pids, prows, pchk = shuffled_partners(samples, want, rv.index)
    checks.update(pchk)
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise SystemExit(f"export checks failed before writing: {bad}")
    run = C.INPUTS["correct"].parent
    meta = base_meta(run, want, Path(sft_dir), dir_meta, layers)
    files = {"correct": (te["row"].to_numpy(np.int64), dict(meta, condition="correct")),
             "shuffled": (np.asarray(prows, np.int64),
                          dict(meta, condition="shuffled", partner_bundle_ids=pids, partner_rows=[int(r) for r in prows]))}
    return files, checks


def export(rv, sft_dir: Path, *, snapshot=None, layers=None, expected_n=None) -> dict:
    """Write (or verify) the two inputs files of common.INPUTS; returns the report."""
    layers = list(C.LAYERS if layers is None else layers)
    files, checks = plan_files(rv, sft_dir, snapshot, layers, expected_n)
    have = [C.INPUTS[c].exists() for c in files]
    if any(have) and not all(have):
        raise SystemExit(f"only some of {[str(C.INPUTS[c]) for c in files]} exist; refusing to write the others")
    existed = all(have)
    report = dict(checks_before=checks, files={}, existed=existed, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    n_docs = rv.index["n_docs"].to_numpy(np.int64)
    for cond, (rows, meta) in files.items():
        path = C.INPUTS[cond]
        if not existed:
            X = rv.get(rows, WEIGHTING, center=False, layers=layers).numpy()[:, None]  # [n, 1, L, d]
            write_npz(path, X, meta, n_docs[rows], layers)
            del X
        fc = file_checks(path, rv, rows, layers, meta)
        report["files"][cond] = dict(path=str(path), n_bundles=len(rows), bytes=os.path.getsize(path), checks=fc)
        log(f"{cond}: {path} ({len(rows)} bundles{', existed' if existed else ''}); checks {fc}")
    every = list(checks.values()) + [v for f in report["files"].values() for k, v in f["checks"].items()
                                      if k != "reread_steps"]
    report["n_checks"], report["n_passed"] = len(every), int(sum(bool(v) for v in every))
    report["passed"] = report["n_passed"] == report["n_checks"]
    return report


def write_report(report: dict) -> Path:
    """<eval>/EXPORT.json; an existing report is kept and the new one gets a time stamp."""
    out = C.INPUTS["correct"].parent / "EXPORT.json"
    if out.exists():
        out = out.with_name(time.strftime("EXPORT_%Y%m%d_%H%M%S.json"))
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    from main_run.c1.materialize import ReaderVectors

    rv = ReaderVectors(C.SFT_DIR)
    report = export(rv, C.SFT_DIR, expected_n=C.N_TEST_BUNDLES)
    out = write_report(report)
    log(f"{report['n_passed']}/{report['n_checks']} checks passed; report {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
