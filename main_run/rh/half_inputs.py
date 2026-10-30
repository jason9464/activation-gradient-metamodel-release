#!/usr/bin/env python
"""Per-half reader inputs for the RH plan's replication rule (docs/notes/20261026_rh_results.md, "Replication").

The full inputs (``<run>/e2/{ckptdiff,dwx}_inputs_<set>_dact.npz``, window ``cum``) sum over all documents of a probe
set. The extractors already keep each half's sums, so the half-set vectors need no new forward:
  ckptdiff  H_h(t) - H_h(0) from ``<run>/ckptdiff_<set>/h_step_<t>.npy`` (fp64 [2, L, D], row = half)
  dwx       sum over grid segments j of s (B_j A_j - B_{j-1} A_{j-1}) X_h(t_{j-1}) from ``<run>/dwx_<set>/x_step_<t>.npz``
            (fp64 [2, L, d_in] per module) and the run's grid adapters, as ``dwx_inputs.build_dwx`` computes the
            sum over both halves
Each output is written next to the full file as ``..._dact_half<h>.npz`` with the full file's format and meta (plus
``subset``), window ``cum`` only. Check before writing: half 0 + half 1 equals the full file's ``cum`` input to a
relative error < 1e-5 at every step and layer.

Usage:
  python -m main_run.rh.half_inputs --run <root>/rh_hacker --set rhtask [--kinds ckptdiff,dwx]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import adamfx  # noqa: E402
import ckpt_diff_inputs as cdi  # noqa: E402
from main_run.em import dwx_inputs as dwx  # noqa: E402

REL_TOL = 1e-5


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [rh.half_inputs] {msg}", flush=True)


def _load_full(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    steps = [int(t) for t in z["steps"]]
    wins = [str(w) for w in z["windows"]]
    layers = [int(L) for L in z["layers"]]
    if meta.get("order_hash") != cdi.order_hash(steps, wins, layers):
        raise SystemExit(f"{path}: input order hash mismatch")
    if "cum" not in wins:
        raise SystemExit(f"{path}: no cum window")
    wi = wins.index("cum")
    n_docs = np.asarray(z["n_docs"][:, wi])
    if len(set(int(n) for n in n_docs)) != 1:
        raise SystemExit(f"{path}: n_docs changes across read steps")
    return dict(meta=meta, steps=steps, layers=layers, cum=np.asarray(z["inputs"][:, wi]),
                lo=np.asarray(z["window_lo"][:, wi]), hi=np.asarray(z["window_hi"][:, wi]),
                n_docs=int(n_docs[0]))


def halves_ckptdiff(run, probe_set, steps):
    cd = os.path.join(run, f"ckptdiff_{probe_set}")
    H0 = np.load(os.path.join(cd, "h_step_000000.npy"))
    out = np.zeros((2, len(steps)) + H0.shape[1:], dtype=np.float64)
    for i, t in enumerate(steps):
        Ht = np.load(os.path.join(cd, f"h_step_{t:06d}.npy"))
        out[:, i] = Ht - H0
    return out  # [2, S, L, D]


def halves_dwx(run, probe_set, read_steps, grid):
    import torch

    xd = dwx.x_dir(run, probe_set)
    scal = json.load(open(os.path.join(xd, "x_step_000000.json")))["scaling"]
    S = {k: torch.tensor(scal[k], dtype=torch.float64) for k in dwx.KINDS}

    def X_of(t):
        z = np.load(os.path.join(xd, f"x_step_{t:06d}.npz"))
        return {k: torch.from_numpy(np.asarray(z[f"x_{k}"])).transpose(0, 1).contiguous() for k in dwx.KINDS}

    def F_of(t):
        return dwx.lora_factors(cdi.read_adapter(os.path.join(cdi.adapter_dir_of(run, t), "adapter_model.safetensors")),
                                L)

    L = X_of(0)[dwx.KINDS[0]].shape[0]
    grid = sorted(int(t) for t in grid)
    if grid[0] != 0:
        raise SystemExit("grid must start at 0")
    out, D = {}, None
    f_prev, x_prev = F_of(0), X_of(0)
    for t in grid[1:]:
        f_t = F_of(t)
        seg = sum(adamfx.module_delta(x_prev[k], f_prev[k][0], f_prev[k][1], f_t[k][0], f_t[k][1], S[k])
                  for k in dwx.KINDS)  # [L, 2, d_out]
        D = seg if D is None else D + seg
        if t in read_steps:
            out[t] = D.numpy().copy()
        f_prev, x_prev = f_t, X_of(t)
    miss = [t for t in read_steps if t not in out]
    if miss:
        raise SystemExit(f"read steps {miss} are not grid steps")
    return np.stack([np.stack([out[t][:, h] for t in read_steps]) for h in (0, 1)])  # [2, S, L, D]


def write_half(full_path, full, X_half, h, n_docs_half, probe_set, kind):
    out = full_path.replace(".npz", f"_half{h}.npz")
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    steps, layers = full["steps"], full["layers"]
    meta = dict(full["meta"])
    meta["windows"] = [["cum", None]]
    meta["order_hash"] = cdi.order_hash(steps, ["cum"], layers)
    meta["subset"] = dict(half=h, n_docs=n_docs_half, of=full_path, builder="main_run/rh/half_inputs.py",
                          kind=kind, set=probe_set, rule=f"half 0 + half 1 == full cum (rel err < {REL_TOL})")
    meta["shape"] = [len(steps), 1, len(layers), X_half.shape[-1]]
    meta["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, inputs=X_half[:, None].astype(np.float32), steps=np.asarray(steps, dtype=np.int64),
                 windows=np.asarray(["cum"]), window_lo=full["lo"][:, None], window_hi=full["hi"][:, None],
                 n_docs=np.full((len(steps), 1), n_docs_half, dtype=np.int64),
                 layers=np.asarray(layers, dtype=np.int64), meta=np.asarray(json.dumps(meta)))
    os.replace(tmp, out)
    return out


def build(run, probe_set, kinds=("ckptdiff", "dwx")) -> dict:
    run = cdi.resolve_run(run)
    res = {}
    for kind in kinds:
        full_path = os.path.join(run, "e2", f"{kind}_inputs_{probe_set}_dact.npz")
        full = _load_full(full_path)
        if kind == "ckptdiff":
            Xh = halves_ckptdiff(run, probe_set, full["steps"])
            half_of = json.load(open(os.path.join(run, f"ckptdiff_{probe_set}", "docs.json"))).get("half")
        else:
            grid = full["meta"]["transform"]["refresh_steps"]
            Xh = halves_dwx(run, probe_set, full["steps"], grid)
            half_of = json.load(open(os.path.join(dwx.x_dir(run, probe_set), "docs.json"))).get("half")
        tot = Xh[0] + Xh[1]
        num = np.linalg.norm(tot - full["cum"].astype(np.float64), axis=-1)
        den = np.maximum(np.linalg.norm(full["cum"].astype(np.float64), axis=-1), 1e-300)
        rel = float((num / den).max())
        if not rel < REL_TOL:
            raise SystemExit(f"{full_path}: halves do not sum to the full input (max rel err {rel:.3g})")
        if not isinstance(half_of, list) or len(half_of) != full["n_docs"] or set(half_of) != {0, 1}:
            raise SystemExit(f"{full_path}: docs.json half labels must cover exactly {full['n_docs']} documents")
        n_half = [half_of.count(h) for h in (0, 1)]
        if n_half[0] != n_half[1]:
            raise SystemExit(f"{full_path}: document halves are unbalanced: {n_half}")
        outs = [write_half(full_path, full, Xh[h], h, n_half[h], probe_set, kind) for h in (0, 1)]
        res[kind] = dict(outs=outs, max_rel_err=rel, n_docs=n_half)
        log(f"{os.path.basename(run)} {kind} {probe_set}: halves written (max rel err {rel:.2e}, docs {n_half})")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--kinds", default="ckptdiff,dwx")
    a = ap.parse_args(argv)
    res = build(a.run, a.set, tuple(k for k in a.kinds.split(",") if k))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
