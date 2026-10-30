#!/usr/bin/env python
"""Checks of a probe rerun and its reader inputs (plan docs/notes/20261026_clean_batch_update_plan.md, sections on
integrity checks and inputs). CPU only.

verify (``<train>/cleanbatch/verify.json``; exit 1 when a required check fails):
  trajectory          loss, grad_norm and lr of every logged step of the rerun equal the main run's train_log.jsonl
                      (exact float equality; the main run is newset_<arm>_s1)
  probe_complete      probe_meta.json complete, every planned step present, each step's state checks (parameters
                      bit-equal, optimizer state fingerprint equal, grads None; RNG recorded) and weight check (real
                      step vs the probe's would-be step of the real batch, relative difference <= 1e-4) pass
  real_vs_store       per probe step: cos(probe real ``step`` D, -q of <train>/stepsums/adamfx at that step) >= 0.9999
                      in every layer; the same against the main run's store; ``stepfix`` against <train>/adamfx_fixed
                      and the main run's adamfx_fixed; ``grad`` against the main run's e2/grad_inputs_every100.npz w1
                      (only at steps that file has: not applicable in the 60-step smoke)
  real_vs_log         per probe step: the probe's pre-clip gradient norm of the real batch equals the rerun's logged
                      grad_norm (relative 1e-5), its lr equals the logged lr (relative 1e-6), its label count equals
                      num_items_in_batch
  variant_identity    (probe files with the extended variants only) per probe step and batch the probe's "extended"
                      record is ok: ||dW_step64 - (dW_hist + dW_curr)|| / ||dW_step64|| <= 1e-6 over all factors, per
                      layer ||D(step64) - D(hist) - D(curr) - C|| / ||D(step64)|| <= 1e-6 on the own and the fixed
                      documents (C the cross term of the LoRA product), stored fp32 step within 1e-2 of the fp64 step
                      (cleanbatch_probe.py docstring, "Extended variants")
  ref_probe           (with --ref-train only) the vectors of the five variants of the first clean-batch runs (step,
                      stepfix, mfree, mfreefix, grad) of every batch at every probe step equal those of the reference
                      run's probe files: max abs difference 0, or cos >= 0.999999 in every layer
  Diagnostics (reported only): per step and variant, the median over layers of cos(real, f_k) and of |f_k| / |real|;
  cos(step, mfree) of each batch; the weight-space cosines recorded by the probe.
The probe files' variant list is read from each file: the five variants of the first runs must be present (and the
same list in every file); the extended files carry five more.

build: reader input files ``<train>/e2/cleanbatch_inputs_<group>.npz`` in the e2_read.export_inputs layout that the
worktree's e2_generate.load_inputs reads (inputs fp32 [n_steps, n_windows, 40, 5120], steps, windows, window_lo,
window_hi, n_docs, layers, meta with kind "grad", a sign text with "+dL/dh" and "inputs = -(", order hash, transform
variant "cleanbatch", so readings are named cleanbatch_<reader>_<question>_<tag>). Groups (window label =
<batch>_<variant>, batches real, f1, f2, f3):
  core      real_step, f1_step, f2_step, f3_step
  mfree     real_mfree, f1_mfree, f2_mfree, f3_mfree
  fixed     real_stepfix, f1_stepfix, f2_stepfix, f3_stepfix
  grad      real_grad, f1_grad, f2_grad, f3_grad
  mfreefix  real_mfreefix, f1_mfreefix, f2_mfreefix, f3_mfreefix
  extended probe files only (cleanbatch_probe.py --probe-variants extended):
  mom       real_mom, f1_mom, f2_mom, f3_mom
  momfix    real_momfix, f1_momfix, f2_momfix, f3_momfix
  hist      real_hist, f1_hist, f2_hist, f3_hist
  histfix   real_histfix, f1_histfix, f2_histfix, f3_histfix
  curr      real_curr, f1_curr, f2_curr, f3_curr
  all       the windows of every group above whose variants the probe files have, in this order (20 windows for the
            five-variant files, 40 for the extended ones; for the first-token P(Yes) pass)
Every vector is unit-normalised per (step, window, layer) (fp64 norm, stored fp32); the reader keeps only the
direction. Checks per file: finite_nonzero, unit_norm, direction_kept, order_hash_recorded, loader_dry_run. An
existing output is left alone (counted as built only when its windows and steps match).

  python cleanbatch_inputs.py verify --train <R>/cleanbatch_dolphin_s1/train --arm dolphin [--ref-train <train>]
  python cleanbatch_inputs.py build  --train <R>/cleanbatch_dolphin_s1/train
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adamfx  # noqa: E402
import adamfx_windows as aw  # noqa: E402
import cleanbatch_probe as cp  # noqa: E402

PLAN = "docs/notes/20261026_clean_batch_update_plan.md"
COS_MIN = 0.9999
GROUPS = {
    "core": [(b, "step") for b in cp.BATCH_NAMES],
    "mfree": [(b, "mfree") for b in cp.BATCH_NAMES],
    "fixed": [(b, "stepfix") for b in cp.BATCH_NAMES],
    "grad": [(b, "grad") for b in cp.BATCH_NAMES],
    "mfreefix": [(b, "mfreefix") for b in cp.BATCH_NAMES],
}
BASE_GROUPS = ("core", "mfree", "fixed", "grad", "mfreefix")
EXTENDED_GROUPS = ("mom", "momfix", "hist", "histfix", "curr")
for _g in EXTENDED_GROUPS:
    GROUPS[_g] = [(b, _g) for b in cp.BATCH_NAMES]
# the 20 windows of the five-variant files; group_windows("all", variants) adds the extended groups a probe has
GROUPS["all"] = [x for g in BASE_GROUPS for x in GROUPS[g]]
REF_COS_MIN = 0.999999
SIGN = ("q = +dL/dh convention of the gradient store: the direct-write variants store +dh = -q with q = -dh (adamfx.py; "
        "under SGD q is a positive mix of +dL/dh rows), the grad variant stores -(+dL/dh of the summed token loss); "
        "inputs = -(q), the descent direction, unit-normalised per (step, window, layer)")


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [cleanbatch_inputs] {msg}", flush=True)


def label(b, v):
    return f"{b}_{v}"


def group_windows(group, variants):
    """(batch, variant) windows of ``group`` for probe files with ``variants``; "all" = the windows of every group whose
    variants are all present, in the order BASE_GROUPS, EXTENDED_GROUPS. A group with a missing variant is refused."""
    if group == "all":
        return [x for g in BASE_GROUPS + EXTENDED_GROUPS if all(v in variants for _, v in GROUPS[g]) for x in GROUPS[g]]
    if group not in GROUPS:
        raise RuntimeError(f"unknown group {group!r}; known {sorted(GROUPS)}")
    missing = sorted({v for _, v in GROUPS[group] if v not in variants})
    if missing:
        raise RuntimeError(f"group {group}: variants {missing} are not in the probe files (they have {list(variants)})")
    return GROUPS[group]


def load_probe(train):
    """(steps, V [n, 4, n_variants, L, D] fp32, layers, meta, variants) of every probe_step_*.npz of ``train``, in step
    order. The variant list is read from the files: every file must have the same list, containing cp.VARIANTS."""
    d = os.path.join(train, "cleanbatch")
    files = sorted(f for f in os.listdir(d) if f.startswith("probe_step_") and f.endswith(".npz"))
    steps, Vs, layers, variants = [], [], None, None
    for f in files:
        z = np.load(os.path.join(d, f))
        fv = [str(x) for x in z["variants"]]
        if [str(x) for x in z["batches"]] != list(cp.BATCH_NAMES):
            raise RuntimeError(f"{f}: batch names differ from {list(cp.BATCH_NAMES)}")
        if not set(cp.VARIANTS) <= set(fv) or len(set(fv)) != len(fv):
            raise RuntimeError(f"{f}: variants {fv} lack one of {list(cp.VARIANTS)} or repeat one")
        if variants is None:
            variants = fv
        elif fv != variants:
            raise RuntimeError(f"{f}: variants {fv} differ from the other files' {variants}")
        if z["vectors"].shape[1] != len(fv):
            raise RuntimeError(f"{f}: vectors {z['vectors'].shape} do not match {len(fv)} variants")
        ly = [int(x) for x in z["layers"]]
        if layers is None:
            layers = ly
        elif ly != layers:
            raise RuntimeError(f"{f}: layers differ")
        steps.append(int(z["step"]))
        Vs.append(np.asarray(z["vectors"], np.float32))
    with open(os.path.join(d, "probe_meta.json")) as f:
        meta = json.load(f)
    return steps, np.stack(Vs), layers, meta, variants


def train_log(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[int(r["step"])] = r
    return out


def cos_layers(a, b):
    return aw._cos_rows(a, b)


# ---------------------------------------------------------------------------------------------- verify


def verify(train, arm, main_run=None, ref_train=None):
    main_run = main_run or cp.MAIN_RUNS[arm]
    steps, V, layers, meta, variants = load_probe(train)
    checks, diag = {}, {}
    # trajectory
    a, b = train_log(os.path.join(train, "train_log.jsonl")), train_log(os.path.join(main_run, "train_log.jsonl"))
    common = sorted(set(a) & set(b))
    diff = [t for t in common if any(a[t].get(k) != b[t].get(k) for k in ("loss", "grad_norm", "lr"))]
    last = max(a) if a else 0
    checks["trajectory"] = dict(ok=bool(common) and not diff and common == list(range(1, last + 1)), applies=True,
                                steps_compared=len(common), steps_differing=len(diff), first_diff=diff[:3],
                                rerun_last_step=last,
                                rule="loss, grad_norm and lr equal (exact) at every logged step 1..last")
    # probe meta
    recs = {int(r["step"]): r for r in meta["records"]}
    st_ok = [t for t in steps if all(recs[t]["state"][k] for k in ("params_bit_equal", "optimizer_state_equal",
                                                                     "grads_none"))]
    wc_ok = [t for t in steps if (recs[t].get("weight_check") or {}).get("ok")]
    checks["probe_complete"] = dict(
        ok=bool(meta["complete"]) and steps == sorted(meta["planned_steps"]) and len(st_ok) == len(steps)
        and len(wc_ok) == len(steps), applies=True, planned=len(meta["planned_steps"]), present=len(steps),
        state_ok=len(st_ok), weight_ok=len(wc_ok),
        weight_rel_max=max((recs[t]["weight_check"] or {}).get("rel_diff", float("nan")) for t in steps),
        weight_bit_identical=sum(bool((recs[t]["weight_check"] or {}).get("bit_identical")) for t in steps),
        rng_unchanged=sum(bool(recs[t]["state"]["rng_cpu_unchanged"] and recs[t]["state"]["rng_cuda_unchanged"])
                          for t in steps))
    # real batch vs the stores
    rows = []
    gpath = os.path.join(main_run, "e2", "grad_inputs_every100.npz")
    gz = np.load(gpath) if os.path.exists(gpath) else None
    gsteps = [int(x) for x in gz["steps"]] if gz is not None else []
    gw = [str(x) for x in gz["windows"]].index("w1") if gz is not None else None
    vi = {v: i for i, v in enumerate(variants)}
    for k, t in enumerate(steps):
        r = dict(step=t)
        for name, sdir, v in (("store_rerun", os.path.join(train, "stepsums", "adamfx"), "step"),
                              ("store_main", os.path.join(main_run, "stepsums", "adamfx"), "step"),
                              ("fixed_rerun", os.path.join(train, "adamfx_fixed"), "stepfix"),
                              ("fixed_main", os.path.join(main_run, "adamfx_fixed"), "stepfix")):
            _, Q = adamfx.load_stepsums(sdir, t, t)
            ref = -np.asarray(Q[0], np.float64)
            c = cos_layers(V[k, 0, vi[v]], ref)
            r[name] = dict(min_cos=float(c.min()), rel_err=float(np.linalg.norm(V[k, 0, vi[v]] - ref) / np.linalg.norm(ref)))
        if t in gsteps:
            ref = np.asarray(gz["inputs"][gsteps.index(t), gw], np.float64)
            c = cos_layers(V[k, 0, vi["grad"]], ref)
            r["grad_main"] = dict(min_cos=float(c.min()), rel_err=float(np.linalg.norm(V[k, 0, vi["grad"]] - ref)
                                                                        / np.linalg.norm(ref)))
        pb = recs[t]["per_batch"][0]
        lg = a.get(t, {})
        r["grad_norm_rel"] = abs(pb["pre_clip_grad_norm"] - lg.get("grad_norm", float("nan"))) / max(lg.get("grad_norm", 1.0), 1e-12)
        r["lr_rel"] = abs(recs[t]["lr"] - lg.get("lr", float("nan"))) / max(lg.get("lr", 1.0), 1e-30)
        r["n_items_equal"] = recs[t]["num_items_in_batch"] in (None, pb["n_label"])
        rows.append(r)
    keys = ("store_rerun", "store_main", "fixed_rerun", "fixed_main")
    checks["real_vs_store"] = dict(
        ok=all(r[k]["min_cos"] >= COS_MIN for r in rows for k in keys) and
        all(r["grad_main"]["min_cos"] >= COS_MIN for r in rows if "grad_main" in r), applies=True,
        min_cos={k: min(r[k]["min_cos"] for r in rows) for k in keys},
        max_rel_err={k: max(r[k]["rel_err"] for r in rows) for k in keys},
        grad_main_steps=sum("grad_main" in r for r in rows),
        grad_main_min_cos=min((r["grad_main"]["min_cos"] for r in rows if "grad_main" in r), default=None),
        rule=f"cos >= {COS_MIN} in every layer at every probe step (grad: steps the main run's window file has)")
    checks["real_vs_log"] = dict(
        ok=all(r["grad_norm_rel"] <= 1e-5 and r["lr_rel"] <= 1e-6 and r["n_items_equal"] for r in rows), applies=True,
        grad_norm_rel_max=max(r["grad_norm_rel"] for r in rows), lr_rel_max=max(r["lr_rel"] for r in rows),
        n_items_equal=sum(r["n_items_equal"] for r in rows))
    ext_vars = [v for v in variants if v not in cp.VARIANTS]
    if ext_vars:
        checks["variant_identity"] = identity_check(recs, steps)
    if ref_train:
        checks["ref_probe"] = ref_check(steps, V, variants, ref_train)
    # diagnostics
    for v in variants:
        j = vi[v]
        dd = []
        for k, t in enumerate(steps):
            real = V[k, 0, j]
            rn = np.linalg.norm(real.astype(np.float64), axis=-1)
            dd.append(dict(step=t, cos_real_f=[float(np.median(cos_layers(V[k, f, j], real))) for f in (1, 2, 3)],
                           norm_ratio_f=[float(np.median(np.linalg.norm(V[k, f, j].astype(np.float64), axis=-1) / rn))
                                         for f in (1, 2, 3)]))
        diag[v] = dd
    diag["cos_step_mfree"] = [dict(step=t, per_batch=[float(np.median(cos_layers(V[k, b_, vi["step"]], V[k, b_, vi["mfree"]])))
                                                      for b_ in range(4)]) for k, t in enumerate(steps)]
    diag["weight_space"] = [dict(step=t, wcos_step_real_f=recs[t]["wcos_step"][0][1:],
                                 wcos_mfree_real_f=recs[t]["wcos_mfree"][0][1:],
                                 wcos_step_vs_mfree=recs[t]["wcos_step_vs_mfree"]) for t in steps]
    if ext_vars:
        diag["extended"] = [dict(step=t, across=recs[t].get("extended"),
                                 per_batch=[pb.get("extended") for pb in recs[t]["per_batch"]]) for t in steps]
    app = {k: c for k, c in checks.items() if c.get("applies", True)}
    out = dict(train=os.path.abspath(train), arm=arm, main_run=os.path.abspath(main_run), steps=steps, checks=checks,
               checks_passed=f"{sum(c['ok'] for c in app.values())}/{len(app)}", per_step=rows, diagnostics=diag,
               written=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = os.path.join(train, "cleanbatch", "verify.json")
    with open(p + ".tmp", "w") as f:
        json.dump(out, f, indent=1, default=str)
    os.replace(p + ".tmp", p)
    log(f"verify {train}: checks {out['checks_passed']} "
        + "; ".join(f"{k} {'ok' if c['ok'] else 'FAILED'}" for k, c in checks.items())
        + f"; trajectory {checks['trajectory']['steps_compared']} steps, min cos store {checks['real_vs_store']['min_cos']}"
        + f"; weight rel max {checks['probe_complete']['weight_rel_max']:.2e}; written {p}")
    return all(c["ok"] for c in app.values())


def identity_check(recs, steps):
    """variant_identity: every batch's extended record at every probe step is ok (cleanbatch_probe.Probe._extended)."""
    rows = []
    for t in steps:
        pbs = [pb.get("extended") for pb in recs[t]["per_batch"]]
        if any(e is None for e in pbs):
            rows.append(dict(step=t, ok=False, missing=True))
            continue
        rows.append(dict(step=t, ok=all(bool(e["ok"]) for e in pbs),
                         w_rel=max(e["w_identity_rel"] for e in pbs),
                         d_rel=max(max(e["d_identity_rel_max_own"], e["d_identity_rel_max_fix"]) for e in pbs),
                         step32=max(e["step32_vs_step64_rel"] for e in pbs)))
    good = [r for r in rows if r["ok"]]
    return dict(ok=bool(rows) and len(good) == len(rows), applies=True, steps=len(rows), steps_ok=len(good),
                batches_checked=4 * len(rows), failed_steps=[r["step"] for r in rows if not r["ok"]][:5],
                w_rel_max=max((r.get("w_rel", float("nan")) for r in rows), default=None),
                d_rel_max=max((r.get("d_rel", float("nan")) for r in rows), default=None),
                step32_rel_max=max((r.get("step32", float("nan")) for r in rows), default=None),
                rule=f"per step and batch: weight rel <= {cp.IDENTITY_TOL}, D rel (with the cross term) <= "
                     f"{cp.IDENTITY_TOL} per layer, own and fixed documents; fp32 step vs fp64 <= {cp.STEP32_TOL}")


def ref_check(steps, V, variants, ref_train):
    """ref_probe: the five first-run variants of every batch equal the reference run's (max abs 0 or cos >= REF_COS_MIN
    per layer) at every probe step; every probe step of this run must exist in the reference."""
    rs, RV, _, _, rvars = load_probe(ref_train)
    vi, ri = {v: i for i, v in enumerate(variants)}, {v: i for i, v in enumerate(rvars)}
    rows, missing = [], [t for t in steps if t not in rs]
    for k, t in enumerate(steps):
        if t in missing:
            continue
        kr = rs.index(t)
        for b in range(len(cp.BATCH_NAMES)):
            for v in cp.VARIANTS:
                a, r = V[k, b, vi[v]], RV[kr, b, ri[v]]
                mx = float(np.abs(a.astype(np.float64) - r).max())
                c = float(cos_layers(a, r).min())
                rows.append(dict(step=t, batch=cp.BATCH_NAMES[b], variant=v, max_abs=mx, min_cos=c,
                                 ok=mx == 0.0 or c >= REF_COS_MIN))
    return dict(ok=not missing and bool(rows) and all(r["ok"] for r in rows), applies=True,
                ref=os.path.abspath(ref_train), compared=len(rows), passed=sum(r["ok"] for r in rows),
                bit_identical=sum(r["max_abs"] == 0.0 for r in rows), steps_missing_in_ref=missing,
                min_cos=min((r["min_cos"] for r in rows), default=None),
                max_abs=max((r["max_abs"] for r in rows), default=None),
                failed=[r for r in rows if not r["ok"]][:5],
                rule=f"variants {list(cp.VARIANTS)} x batches x probe steps: max abs 0 or cos >= {REF_COS_MIN} per layer")


# ---------------------------------------------------------------------------------------------- build


def unit_rows(X):
    X64 = np.asarray(X, np.float64)
    n = np.linalg.norm(X64, axis=-1, keepdims=True)
    if not np.isfinite(n).all() or (n == 0).any():
        raise ValueError("zero or non-finite row")
    return (X64 / n).astype(np.float32)


def write_group(train, group, steps, V, layers, meta, loader=True, variants=cp.VARIANTS):
    out = os.path.join(train, "e2", f"cleanbatch_inputs_{group}.npz")
    wins = group_windows(group, variants)
    labels = [label(b, v) for b, v in wins]
    if os.path.exists(out):
        z = np.load(out)
        same = [str(x) for x in z["windows"]] == labels and [int(x) for x in z["steps"]] == steps
        log(f"{out} exists ({'matches' if same else 'DIFFERS: windows or steps'})")
        return same
    bi = {b: i for i, b in enumerate(cp.BATCH_NAMES)}
    vi = {v: i for i, v in enumerate(variants)}
    raw = np.stack([V[:, bi[b], vi[v]] for b, v in wins], axis=1)  # [n, W, L, D]
    U = unit_rows(raw)
    rn = np.linalg.norm(raw.astype(np.float64), axis=-1)
    un = np.linalg.norm(U.astype(np.float64), axis=-1)
    c = aw._cos_rows(U, raw)
    checks = {
        "finite_nonzero": dict(ok=bool(np.isfinite(raw).all() and (rn > 0).all()), applies=True),
        "unit_norm": dict(ok=bool(np.abs(un - 1).max() < 1e-5), applies=True, max_dev=float(np.abs(un - 1).max())),
        "direction_kept": dict(ok=bool(c.min() >= 0.99999), applies=True, min_cos=float(c.min())),
    }
    order_hash = aw.export_order_hash(steps, labels, layers)
    checks["order_hash_recorded"] = dict(ok=True, applies=True, sha1=order_hash)
    n_docs = [[128 if v.endswith("fix") else 32 for b, v in wins] for _ in steps]
    with open(os.path.join(train, "RUN.json")) as f:
        st = json.load(f)["settings"]
    fm = meta["foreign"]
    transform = dict(variant="cleanbatch", group=group, plan=PLAN,
                     definition={label(b, v): f"batch {b} ({'the step own training batch' if b == 'real' else 'foreign batch ' + b[1:] + ' of 32 ' + fm['foreign_arm'] + ' documents'}); {cp.VARIANT_TEXT[v]}"
                                 for b, v in wins},
                     unit_rule="each [d] row unit-normalised per (step, window, layer) in fp64, stored fp32",
                     window_rule="window_lo = window_hi = the probe step t: the update of optimizer step t from the "
                                 "state after step t-1",
                     foreign_arm=fm["foreign_arm"], foreign_lines=fm["batches"], foreign_doc_id=fm["doc_id"],
                     foreign_file=fm["file"], foreign_file_sha256=fm["file_sha256"],
                     real_line_no={str(r["step"]): r["real_line_no"] for r in meta["records"]},
                     probe_dir=os.path.join(os.path.abspath(train), "cleanbatch"))
    md = dict(run=os.path.abspath(train), arm=st.get("arm"), seed=st.get("seed"), quantity="cleanbatch", kind="grad",
              kind_note="kind 'grad' is the label the trained-reader loader requires", sign=SIGN,
              layers=[int(L) for L in layers], windows=[[lab, 1] for lab in labels], steps=[int(t) for t in steps],
              order_hash=order_hash, transform=transform, checks=checks, written=time.strftime("%Y-%m-%d %H:%M:%S"),
              shape=list(U.shape), dtype="float32")
    failed = [k for k, v in checks.items() if not v["ok"]]
    if failed:
        raise RuntimeError(f"{group}: checks failed {failed}")
    md["checks_passed"] = f"{len(checks)}/{len(checks)} (before the loader dry run)"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, inputs=U, steps=np.asarray(steps, dtype=np.int64), windows=np.asarray(labels),
                 window_lo=np.asarray([[t] * len(labels) for t in steps], dtype=np.int64),
                 window_hi=np.asarray([[t] * len(labels) for t in steps], dtype=np.int64),
                 n_docs=np.asarray(n_docs, dtype=np.int64), layers=np.asarray(layers, dtype=np.int64),
                 meta=np.asarray(json.dumps(md, default=str)))
    if loader:
        got, err = aw.loader_dry_run([tmp], readers=("rl_paper_c040",))
        g = (got or {}).get(tmp, {})
        checks["loader_dry_run"] = dict(ok=bool(got) and g.get("windows") == labels and g.get("shape") == list(U.shape)
                                        and g.get("steps") == steps and g.get("names") == ["cleanbatch_rl_paper_c040"],
                                        applies=True, names=g.get("names"), error=err)
        if not checks["loader_dry_run"]["ok"]:
            raise RuntimeError(f"{group}: loader dry run failed, file left at {tmp}: {checks['loader_dry_run']}")
    os.replace(tmp, out)
    with open(out.replace(".npz", "_checks.json"), "w") as f:
        json.dump(dict(output=out, checks=checks, checks_passed=f"{sum(c['ok'] for c in checks.values())}/{len(checks)}",
                       written=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1, default=str)
    log(f"{group}: wrote {out} {list(U.shape)}; checks {sum(c['ok'] for c in checks.values())}/{len(checks)}")
    return True


def build(train, groups, loader=True):
    steps, V, layers, meta, variants = load_probe(train)
    if not meta.get("complete"):
        raise SystemExit(f"{train}: probe_meta.json is not complete")
    for g in groups:
        group_windows(g, variants)  # refuse a group the files cannot fill before writing any file
    return all(write_group(train, g, steps, V, layers, meta, loader=loader, variants=variants) for g in groups)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--train", required=True)
    v.add_argument("--arm", required=True, choices=sorted(cp.MAIN_RUNS))
    v.add_argument("--main-run", default=None)
    v.add_argument("--ref-train", default=None, help="train dir of a reference probe run (check ref_probe)")
    b = sub.add_parser("build")
    b.add_argument("--train", required=True)
    b.add_argument("--groups", default="core,mfree,fixed,grad,mfreefix,all")
    b.add_argument("--no-loader", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        return 0 if verify(a.train, a.arm, a.main_run, a.ref_train) else 1
    return 0 if build(a.train, [g.strip() for g in a.groups.split(",") if g.strip()], loader=not a.no_loader) else 1


if __name__ == "__main__":
    raise SystemExit(main())
