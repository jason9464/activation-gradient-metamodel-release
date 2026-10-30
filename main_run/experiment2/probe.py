"""Refit three-layer linear probes on the same 4,010 train and 530 held-out bundles.

Features are the three centred *directions* the readers receive, concatenated.
Regularisation is chosen by grouped cross-validation within training organisms;
the held-out labels never choose a hyperparameter. One model is fit per input.

  python main_run/experiment2/probe.py --input activation
  python main_run/experiment2/probe.py --input gradient
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C_GRID = (0.01, 0.1, 1.0)


def features(directory: Path, index: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    meta = json.loads((directory / "meta.json").read_text())
    layers = list(meta["layers"])
    mean = np.load(directory / "mean_unit_tokensum.npy")
    if not set(C.LAYERS) <= set(layers):
        raise ValueError(f"{directory}: missing one of layers {C.LAYERS}")
    width = mean.shape[1]
    X = np.empty((len(rows), len(C.LAYERS) * width), dtype=np.float32)
    for li, layer in enumerate(C.LAYERS):
        mm = np.load(directory / f"bundle_tokensum_L{layer}.npy", mmap_mode="r")
        u = np.asarray(mm[rows], np.float32) - mean[layers.index(layer)]
        norms = np.linalg.norm(u, axis=1)
        if not np.isfinite(norms).all() or (norms < 1e-3).any():
            raise ValueError(f"{directory}: invalid centred layer {layer}")
        X[:, li * width:(li + 1) * width] = u / norms[:, None]
    return X


def fit(directory: Path, input_kind: str, *, n_splits: int = 3) -> tuple[list[dict], dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    idx = pd.read_parquet(directory / "bundle_index.parquet")
    if not np.array_equal(idx.row.to_numpy(), np.arange(len(idx))):
        raise ValueError("reader index row column is not contiguous")
    tr = idx[(idx.source == "behaviour") & (idx.split == "train")].copy()
    te = idx[(idx.source == "behaviour") & (idx.split == C.SPLIT)].copy()
    if len(tr) != 4010 or len(te) != 530:
        raise ValueError(f"split is {len(tr)} train / {len(te)} held-out; expected 4010 / 530")
    if tr.n_shows.isna().any() or te.n_shows.isna().any() or tr.unit_id.isna().any():
        raise ValueError("missing behavior labels or training unit ids")
    if set(tr.unit_id) & set(te.unit_id):
        raise ValueError("train and held-out sets share organism unit ids")
    X = features(directory, idx, np.r_[tr.row.to_numpy(int), te.row.to_numpy(int)])
    Xt, Xe = X[:len(tr)], X[len(tr):]
    y = (tr.n_shows.to_numpy(int) > 0).astype(int)
    groups = tr.unit_id.to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold_indices = list(cv.split(Xt, y, groups))
    metrics = {}
    for c in C_GRID:
        values = []
        for fit_ids, val_ids in fold_indices:
            if len(set(y[fit_ids])) != 2 or len(set(y[val_ids])) != 2:
                raise ValueError("a training CV fold has a single class")
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced",
                                                                         solver="lbfgs", max_iter=2000, tol=1e-6))
            model.fit(Xt[fit_ids], y[fit_ids])
            values.append(float(roc_auc_score(y[val_ids], model.decision_function(Xt[val_ids]))))
        metrics[str(c)] = values
    best = max(C_GRID, key=lambda c: (float(np.mean(metrics[str(c)])), -c))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best, class_weight="balanced",
                                                                 solver="lbfgs", max_iter=2000, tol=1e-6))
    model.fit(Xt, y)
    decisions = model.decision_function(Xe)
    rows = [dict(bundle_id=b.bundle_id, unit_id=b.unit_id, family=b.family, n_shows=int(b.n_shows),
                 score=float(score), input=input_kind) for b, score in zip(te.itertuples(), decisions)]
    report = dict(input=input_kind, reader_dir=str(directory), layers=list(C.LAYERS), n_train=len(tr), n_held=len(te),
                  c_grid=list(C_GRID), cv_auc_by_c=metrics, best_c=best, seed=0,
                  preprocessing="C1-centred unit rows per layer, each renormalised; train-fitted StandardScaler",
                  target="n_shows > 0", model="L2 logistic regression with balanced class weights")
    return rows, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", choices=("activation", "gradient"), required=True)
    ap.add_argument("--act-dir", type=Path, default=C.ACT_DIR)
    ap.add_argument("--grad-dir", type=Path, default=C.GRAD_DIR)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    directory = a.act_dir if a.input == "activation" else a.grad_dir
    out = a.out or C.RUN / "probe" / f"{a.input}.jsonl"
    if out.exists() and not a.overwrite:
        raise ValueError(f"{out} exists; use --overwrite")
    rows, report = fit(directory, a.input)
    C.write_jsonl(out, rows)
    C.write_json(out.with_suffix(".meta.json"), report)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
