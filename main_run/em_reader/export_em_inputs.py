"""Held-out EM bundle vectors as e2_generate inputs (plan 20261026_em_reader_plan.md section 5, data-gradient arm).

One pseudo run per held-out dataset (insecure, secure, medical / extreme_sports / finance x harmful / benign):
$IIF_BASE/agm_data/em_gen/em_eval/<dataset>/inputs_bundles.npz in e2_read.py --export-inputs format, so the training
pipeline's e2_generate.py reads every bundle with any reader adapter and question (--generic-question):
  steps     0 .. n-1, the dataset's em_test bundles in bundles.jsonl order (meta bundle_ids maps them back)
  windows   ["bundle"]; window_lo = window_hi = 0; n_docs = the bundle's example count
  inputs    fp32 [n, 1, 40, 5120] = unit x norm read from the SFT reader dir (ReaderVectors.get, uncentred): the
            same -dL/dh token sum the reader trains on; e2_generate centres it on the run's mean_unit file
  meta      run, sign, kind "grad", order_hash, bundle_ids, pair_ids, phrase flags (user 11:50: split the evaluation
            on the harmful side's "push through" / "trust your"), source reader dir and its order hash
Checks: 3 bundles per file re-read from the reader dir equal the export; order hash re-computed by e2_generate.
  PYTHONPATH=<em-reader worktree> python main_run/em_reader/export_em_inputs.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import em_common as C  # noqa: E402

SFT_DIR = Path(C.IIF_BASE) / "agm_data" / "c1_reader_40_em" / "layers40"
OUT = Path(C.EM_GEN) / "em_eval"
LAYERS = list(range(40))
SIGN = ("bundle vectors of the EM reader dir: fp16 unit rows x fp32 norms of the summed -dL/dh, i.e. stored +dL/dh "
        "summed over the bundle's examples and negated; inputs = -(sum over the bundle's examples of +dL/dh): the "
        "descent direction, stored as fp32")


def export_order_hash(steps, labels, layers):
    h = hashlib.sha1()
    h.update(("steps:" + ",".join(str(int(t)) for t in steps) + "\n").encode())
    h.update(("windows:" + ",".join(labels) + "\n").encode())
    h.update(("layers:" + ",".join(str(int(L)) for L in layers) + "\n").encode())
    return h.hexdigest()


def main():
    from main_run.c1.materialize import ReaderVectors
    rv = ReaderVectors(SFT_DIR)
    idx = rv.index
    meta_dir = json.loads((SFT_DIR / "meta.json").read_text())
    te = idx[(idx["source"] == "em") & (idx["split"] == "em_test")]
    report = {}
    for ds in sorted(te["em_dataset"].unique()):
        sub = te[te["em_dataset"] == ds]
        run = OUT / ds
        path = run / "inputs_bundles.npz"
        if path.exists():
            raise SystemExit(f"{path} exists; refusing to overwrite")
        (run / "e2").mkdir(parents=True, exist_ok=True)
        rows = sub["row"].to_numpy(np.int64)
        X = rv.get(rows, "tokensum", center=False, layers=LAYERS).numpy()[:, None]  # [n, 1, 40, d]
        steps = list(range(len(rows)))
        meta = dict(run=str(run), sign=SIGN, kind="grad", order_hash=export_order_hash(steps, ["bundle"], LAYERS),
                    arm=None, dataset=ds, bundle_ids=sub["bundle_id"].tolist(), pair_ids=sub["em_pair_id"].tolist(),
                    has_repeated_phrase=[bool(x) for x in sub["em_has_repeated_phrase"]],
                    pair_harmful_has_repeated_phrase=[bool(x) for x in sub["em_pair_harmful_has_repeated_phrase"]],
                    reader_dir=str(SFT_DIR), reader_dir_bundle_order_hash=meta_dir["bundle_order_hash"],
                    written=time.strftime("%Y-%m-%d %H:%M:%S"), code=os.path.abspath(__file__))
        n = len(rows)
        np.savez(path, inputs=X.astype(np.float32), steps=np.array(steps), windows=np.array(["bundle"]),
                 layers=np.array(LAYERS), window_lo=np.zeros((n, 1), np.int64), window_hi=np.zeros((n, 1), np.int64),
                 n_docs=sub["n_docs"].to_numpy(np.int64)[:, None], meta=json.dumps(meta))
        z = np.load(path)
        chk = [0, n // 2, n - 1]
        same = all(np.array_equal(z["inputs"][i, 0], rv.get([rows[i]], "tokensum", center=False, layers=LAYERS)[0].numpy())
                   for i in chk)
        report[ds] = dict(n_bundles=n, path=str(path), reread_equal=bool(same), bytes=os.path.getsize(path))
    print(json.dumps(report, indent=1))
    C.write_json_new(str(OUT / "EXPORT.json"), report)
    return 0 if all(v["reread_equal"] for v in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
