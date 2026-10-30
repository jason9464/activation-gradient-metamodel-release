#!/usr/bin/env python
"""Reader-free check of the EM broad/narrow pair (results note 20261025_em_dact_results.md): do the inputs of the three
broadly misaligned final models (Celeste broad, our em_s42 and em_s1 at step 375) resemble each other more than they
resemble the narrow model, and do the narrow -> broad differences point the same way?

Per layer, cosines between the models' input vectors (Δh evalq; −g of the final model on train and on evalq). Reports
the median over layers of: broad-broad cosine (3 pairs), broad-narrow cosine (3 pairs), the number of layers where the
smallest broad-broad cosine exceeds the largest broad-narrow cosine, and the consistency of the narrow -> broad
differences (cosine of d_i = broad_i - narrow between the 3 pairs) against seed-to-seed noise (cosine of
s42 - Celeste broad with s1 - Celeste broad).

  python -m main_run.em.pair_input_check --em-runs <sprint>/em_runs [--out <json>]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def load(path: Path, step: int) -> np.ndarray:
    z = np.load(path)
    steps = [int(t) for t in z["steps"]]
    return np.asarray(z["inputs"][steps.index(step), 0], np.float64)  # [L, D]


def inputs(root: Path) -> dict[str, dict[str, np.ndarray]]:
    m = {"broad_celeste": "ext_broad", "broad_s42": "em_s42", "broad_s1": "em_s1", "narrow": "ext_narrow"}
    out = {"dh_evalq": {}, "g_train": {}, "g_evalq": {}}
    for k, d in m.items():
        out["dh_evalq"][k] = load(root / d / "e2" / "ckptdiff_inputs_evalq_dact.npz", 375)
        out["g_train"][k] = load(root / d / "e2" / "gradfin_inputs_train_s375.npz", 375)
        out["g_evalq"][k] = load(root / d / "e2" / "gradfin_inputs_evalq_s375.npz", 375)
    for s in ("train", "evalq"):
        out[f"g_{s}"]["base"] = load(root / "ext_broad" / "e2" / f"gradfin_inputs_{s}_s0.npz", 0)
    return out


def analyse(vecs: dict[str, np.ndarray]) -> dict:
    broads = ["broad_celeste", "broad_s42", "broad_s1"]
    L = vecs["narrow"].shape[0]
    bb = np.array([[cos(vecs[a][l], vecs[b][l]) for a, b in itertools.combinations(broads, 2)] for l in range(L)])
    bn = np.array([[cos(vecs[a][l], vecs["narrow"][l]) for a in broads] for l in range(L)])
    sep = int((bb.min(1) > bn.max(1)).sum())
    d = {a: vecs[a] - vecs["narrow"] for a in broads}
    dd = np.array([[cos(d[a][l], d[b][l]) for a, b in itertools.combinations(broads, 2)] for l in range(L)])
    seed = np.array([cos(vecs["broad_s42"][l] - vecs["broad_celeste"][l], vecs["broad_s1"][l] - vecs["broad_celeste"][l])
                     for l in range(L)])
    res = dict(layers=L, broad_broad_cos_median=float(np.median(bb)), broad_narrow_cos_median=float(np.median(bn)),
               layers_separated=sep, diff_consistency_median=float(np.median(dd)),
               diff_consistency_by_pair={f"{a}|{b}": float(np.median(dd[:, i]))
                                         for i, (a, b) in enumerate(itertools.combinations(broads, 2))},
               seed_noise_consistency_median=float(np.median(seed)),
               norm_ratio_narrow_over_broad_median=float(np.median(
                   [np.linalg.norm(vecs["narrow"][l]) / np.mean([np.linalg.norm(vecs[a][l]) for a in broads])
                    for l in range(L)])))
    if "base" in vecs:
        res["base_vs_broad_cos_median"] = float(np.median([[cos(vecs["base"][l], vecs[a][l]) for a in broads]
                                                           for l in range(L)]))
        res["base_vs_narrow_cos_median"] = float(np.median([cos(vecs["base"][l], vecs["narrow"][l]) for l in range(L)]))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--em-runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = {k: analyse(v) for k, v in inputs(a.em_runs).items()}
    if a.out:
        a.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for k, r in res.items():
        print(k, json.dumps({x: (round(y, 3) if isinstance(y, float) else y) for x, y in r.items()
                             if x != "diff_consistency_by_pair"}))
        print("   diff consistency by pair:", {x: round(y, 3) for x, y in r["diff_consistency_by_pair"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
