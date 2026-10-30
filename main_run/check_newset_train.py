#!/usr/bin/env python
"""CPU checks of a new-setting training run.

  python main_run/check_newset_train.py --run <run> [--ref-run <runs>/curve_lr1e4_b32_e7] [--loss-steps 500]
                                        [--pairs all|101,1001] [--out <json>]

loss_vs_reference        train_log.jsonl of the run against the sweep run with the same recipe (no saving hooks), steps
                         1..min(--loss-steps, both runs' last step): per-step relative loss difference <= 1e-3 (the
                         pass rule). Bit-identical steps are counted too: earlier saving and non-saving runs
                         (pilot_dolphin_s1, full_dolphin_s1, adamfx_full_dolphin_s1) agreed bit for bit, so anything
                         short of that is reported even when the rule passes. lr must agree exactly (same schedule).
fixed_pair_recompute (input 4f)  for each LoRA pair that has fixed.npz (lora_pairs/step_XXXXXX): D_fixed per layer =
                         sum over the halves and over o_proj, down_proj of s (B_after A_after - B_before A_before) x_kind,
                         in fp64 from before/after.safetensors and fixed.npz (the summed module inputs that step used),
                         against the stored adamfx_fixed stepsum row of that step (q = -D_fixed) and the pair's
                         vectors.npz adamfx_fixed (+dh per half): relative error <= 1e-3 per layer (the pass rule;
                         X_fixed is the stored fp64 sum, so the expected error is fp32 storage, about 1e-7).
Pass/fail and numbers go to stdout as JSON and to ``--out`` when given (an existing file is refused). Exit 0 only when
every applicable check passes. scripts/newset/train.sh runs it on the CPU after a training DONE (dolphin against the
sweep run; the clean control with --ref-run none) and writes markers check_train_<tag>.{DONE,FAILED}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IIF_BASE = os.environ["IIF_BASE"]
REF_RUN = os.path.join(IIF_BASE, "agm_data", "runs", "curve_lr1e4_b32_e7")
LOSS_TOL = 1e-3
PAIR_TOL = 1e-3
KINDS = ("o_proj", "down_proj")


def read_log(run):
    """{step: record} of <run>/train_log.jsonl (the last record per step wins, as after a resume)."""
    out = {}
    with open(os.path.join(run, "train_log.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a half-written last line while training runs
                continue
            if r.get("step") is not None and r.get("loss") is not None:
                out[int(r["step"])] = r
    return out


def loss_vs_reference(run, ref_run, n_steps):
    a, b = read_log(run), read_log(ref_run)
    last = min(n_steps, max(a, default=0), max(b, default=0))
    steps = list(range(1, last + 1))
    miss = [t for t in steps if t not in a or t not in b]
    if not steps or miss:
        return dict(ok=False, applies=True, why=f"steps missing in a log: {miss[:5]}", steps=last)
    rel = [abs(a[t]["loss"] - b[t]["loss"]) / max(abs(b[t]["loss"]), 1e-30) for t in steps]
    lr_bad = [t for t in steps if a[t].get("lr") != b[t].get("lr")]
    worst = int(np.argmax(rel))
    return dict(ok=max(rel) <= LOSS_TOL and not lr_bad, applies=True, ref_run=os.path.abspath(ref_run), steps=last,
                max_rel_loss_diff=float(max(rel)), worst_step=steps[worst],
                bitwise_equal_steps=sum(a[t]["loss"] == b[t]["loss"] for t in steps),
                grad_norm_bitwise_equal_steps=sum(a[t].get("grad_norm") == b[t].get("grad_norm") for t in steps),
                lr_mismatch_steps=lr_bad[:5], rule=f"per-step relative loss difference <= {LOSS_TOL}; lr identical")


def _rel_rows(a, b):
    """Per-row relative error |a - b| / |b| over the last axis."""
    return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-300)


def pair_recompute(run, pdir):
    """Recompute D_fixed of one LoRA pair (fp64) and compare it with the stepsum row and vectors.npz."""
    from safetensors.numpy import load_file

    import adamfx

    with open(os.path.join(pdir, "meta.json")) as f:
        meta = json.load(f)
    step = int(meta["step"])
    if meta.get("format") != "fp32":
        return dict(ok=False, applies=True, step=step, why=f"pair format {meta.get('format')}, need fp32 before/after")
    scal = meta["lora_scaling"]
    if len(scal) != 1:
        return dict(ok=False, applies=True, step=step, why=f"several LoRA scalings {scal}")
    s = float(scal[0])
    ad = meta.get("adapter", "default")
    bef = load_file(os.path.join(pdir, "before.safetensors"))
    aft = load_file(os.path.join(pdir, "after.safetensors"))
    zf = np.load(os.path.join(pdir, "fixed.npz"))
    layers = [int(x) for x in meta["layers"]]
    D = None
    for k in KINDS:
        x = zf[f"x_{k}"].astype(np.float64)  # [halves, layers, din]
        for li, L in enumerate(layers):
            pre = f"base_model.model.model.layers.{L}." + ("self_attn." if k == "o_proj" else "mlp.") + k
            A0, B0 = bef[f"{pre}.lora_A.{ad}.weight"].astype(np.float64), bef[f"{pre}.lora_B.{ad}.weight"].astype(np.float64)
            A1, B1 = aft[f"{pre}.lora_A.{ad}.weight"].astype(np.float64), aft[f"{pre}.lora_B.{ad}.weight"].astype(np.float64)
            # (B1 A1 - B0 A0) x without forming the d x din matrix: B1 (A1 x) - B0 (A0 x)
            dh = s * ((x[:, li] @ A1.T) @ B1.T - (x[:, li] @ A0.T) @ B0.T)  # [halves, d]
            if D is None:
                D = np.zeros((x.shape[0], len(layers), dh.shape[-1]))
            D[:, li] += dh
    sd = None
    for c in (os.path.join(run, "adamfx_fixed"), os.path.join(run, "stepsums", "adamfx_fixed")):
        if os.path.exists(os.path.join(c, "index.json")):
            sd = c
            break
    every = 100
    if sd is not None:
        with open(os.path.join(sd, "index.json")) as f:
            every = int(((json.load(f).get("fixed") or {}).get("every")) or 100)
    want_x = every * ((step - 1) // every)  # steps r+1..r+every use the refresh after step r
    out = dict(step=step, xfixed_step=int(zf["xfixed_step"]), xfixed_step_expected=want_x, lora_scaling=s)
    errs = []
    if sd is not None:
        _, V = adamfx.load_stepsums(sd, step, step)
        r = _rel_rows(-V[0].astype(np.float64), D.sum(0))
        out["vs_stepsum_max_rel"] = float(r.max())
        errs.append(float(r.max()))
    else:
        out["vs_stepsum_max_rel"] = None
    with np.load(os.path.join(pdir, "vectors.npz")) as zv:
        if "adamfx_fixed" in zv.files:
            r = _rel_rows(zv["adamfx_fixed"].astype(np.float64), D)
            out["vs_pair_vectors_max_rel"] = float(r.max())
            errs.append(float(r.max()))
    out["ok"] = bool(errs) and max(errs) <= PAIR_TOL and int(zf["xfixed_step"]) == want_x
    out["applies"] = True
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref-run", default=REF_RUN, help="sweep run with the same recipe (default curve_lr1e4_b32_e7); "
                    "'none' for the clean control, which has no sweep twin")
    ap.add_argument("--loss-steps", type=int, default=500)
    ap.add_argument("--pairs", default="all", help="'all', 'none' or comma-separated pair steps (e.g. 101)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out and os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    checks = {"loss_vs_reference": dict(ok=True, applies=False, why="--ref-run none") if a.ref_run in ("", "none")
              else loss_vs_reference(a.run, a.ref_run, a.loss_steps)}
    pr = os.path.join(a.run, "lora_pairs")
    want = None if a.pairs in ("all", "none") else {int(x) for x in a.pairs.split(",") if x.strip()}
    dirs = sorted(d for d in os.listdir(pr) if d.startswith("step_")) if os.path.isdir(pr) and a.pairs != "none" else []
    for d in dirs:
        st = int(d.split("_")[1])
        if want is not None and st not in want:
            continue
        if os.path.exists(os.path.join(pr, d, "fixed.npz")):
            checks[f"fixed_pair_recompute_{st}"] = pair_recompute(a.run, os.path.join(pr, d))
    if not any(k.startswith("fixed_pair_recompute") for k in checks):
        checks["fixed_pair_recompute"] = dict(ok=a.pairs == "none", applies=a.pairs != "none",
                                              why=f"no LoRA pair with fixed.npz under {pr} (pairs {a.pairs})")
    app = {k: c for k, c in checks.items() if c.get("applies", True)}
    n_ok = sum(bool(c["ok"]) for c in app.values())
    res = dict(run=os.path.abspath(a.run), checks=checks, checks_passed=f"{n_ok}/{len(app)}")
    txt = json.dumps(res, indent=1, default=float)
    print(txt)
    if a.out:
        with open(a.out + ".tmp", "w") as f:
            f.write(txt + "\n")
        os.replace(a.out + ".tmp", a.out)
    return 0 if n_ok == len(app) else 1


if __name__ == "__main__":
    sys.exit(main())
