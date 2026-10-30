#!/usr/bin/env python
"""SAE tests 1 and 2 on the loss-weighted activation differences.
Runs sae_feature_enrichment.py (unchanged) with extra vectors.

The extra vectors are residual-state differences lw_<variant>_<arm>_minus_clean = X_arm - X_clean of
lossw_actdiff.py's actdiff_lossw.npz (the same construction as that script's act_* rows, which read actdiff.npz).
The original six vectors stay in, so the teacher's top-20 set, the Gaussian and feature-set nulls (seed 0) and the
dolphin-vs-owl sign test are those of the original run and its rows reproduce. The nearest-token labels of every
feature are copied from that run (they depend only on the SAE decoder and the lens).

  SAE_DIR=<main>/artifacts/saes python main_run/lossw_sae.py --actdiff-lossw <lossw>/actdiff_lossw.npz --out OUT
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("all", "resp", "tokl", "tokshuf", "docl", "docn")
ARMS = ("dolphin", "owl")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--actdiff-lossw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sfe-dir", type=Path, default=REPO / "main_run")
    p.add_argument("--fitvec", type=Path, default=REPO / "fitvec")
    p.add_argument("--lens-dir", type=Path, default=REPO / "artifacts" / "fwdb" / "data" / "lens")
    p.add_argument("--nearest-from", type=Path, default=REPO / "results" / "sae_feature_enrichment")
    p.add_argument("--layers", default="30,20,10")
    a = p.parse_args(argv)

    sys.path.insert(0, str(a.sfe_dir))
    import sae_feature_enrichment as SFE

    orig_vectors, orig_load = tuple(SFE.VECTORS), SFE.load_vectors
    new = tuple(f"lw_{v}_{arm}_minus_clean" for v in VARIANTS for arm in ARMS)
    z = np.load(a.actdiff_lossw)

    def load(fitvec):
        SFE.VECTORS = orig_vectors
        out = orig_load(fitvec)
        for v in VARIANTS:
            for arm in ARMS:
                out[f"lw_{v}_{arm}_minus_clean"] = {
                    L: (z[f"{v}_{arm}"][L].astype(np.float64) - z[f"{v}_clean"][L]).astype(np.float32)
                    for L in SFE.LAYERS}
        SFE.VECTORS = orig_vectors + new
        return out

    SFE.load_vectors = load
    a.out.mkdir(parents=True, exist_ok=True)
    for L in (int(x) for x in a.layers.split(",")):
        src = a.nearest_from / f"nearest_L{L}.npz"
        if src.is_file() and not (a.out / src.name).exists():
            shutil.copy2(src, a.out / src.name)
    return SFE.main(["--fitvec", str(a.fitvec), "--lens-dir", str(a.lens_dir), "--out", str(a.out),
                     "--layers", a.layers])


if __name__ == "__main__":
    raise SystemExit(main())
