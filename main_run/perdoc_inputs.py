#!/usr/bin/env python
"""Reader inputs of the per-document and per-batch readings (2026-10-25; plan docs/notes/20261025_perdoc_batch_reading.md).

User question 2026-10-25 11:05: when the cumulative reading says "dolphin preference", can we tell which data is
responsible? Three kinds of input file per run, each in the ``e2_read.export_inputs`` format that the training-pipeline
worktree's ``e2_generate.load_inputs`` reads (arrays inputs [n_steps, n_windows, 40, d] fp32, steps, windows,
window_lo, window_hi, n_docs, layers, meta with kind "grad", a sign text with "+dL/dh" and "inputs = -(", the order
hash and transform.variant, which names the readings <variant>_<reader>_<question>):

  perdoc     variant perdoc, windows doc00..doc31: document d's own plain gradient at step t (row d of step t's batch),
             from <run>/e2/grad_docs_every100.npz (written by the training process; vectors [71, 32, 40, d] fp32 =
             -q per document, q = +dL/dh token sum of the summed token loss: already the descent direction). Not Adam:
             a per-document share of an Adam step is not exactly separable. Default steps 1000..3000 every 100.
  batch      variant batch, two windows at step t: adamfx_w1 = D_t, the direct-write term of step t alone (the
             per-step store <run>/stepsums/adamfx via adamfx.load_stepsums, stored q_t = -D_t, so input = -(q_t)); for a
             single step D_t / lr_t (the adam effective gradient) has the same direction; grad_w1 = the plain batch
             gradient of step t, the sum of the step's 32 document vectors of grad_docs_every100.npz.
             Default steps 500..3000 every 100.
  stepdelta  variant stepdelta, windows delta_fp64, delta_fp32 (those present): the actual one-step activation change
             of a separate measurement, ``*_step_delta_vectors.npz`` of main_run/measure_step_delta_precision.py
             (one file per LoRA pair: delta_fp64 / delta_fp32 [40, d] = sum over the fixed documents of h(after) -
             h(before); meta run and lora_pair_step; an [n_pairs, 40, d] layout with meta pair_steps is read too),
             read at the pair step; one inputs file per run. delta_bf16 is not read. The newest vectors file wins when
             two give the same (run, step).

Every stored vector is unit-normalised per (step, window, layer) in fp32. e2_generate builds the injected direction
as unit(fp16(v / |v|) - mean_unit[layer]) (``reader_vectors``) and the hook keeps only the direction, so a unit input
gives the same reading as the raw vector; magnitude is not part of the question (user 2026-10-25 11:05).

Integrity checks (all applicable must pass, else nothing is written under the final name; counted in meta and printed):
  perdoc     source_meta (kind grad, sign "vectors = -q", requested steps present, 32 documents, 40 layers),
             docs_sum_equals_online_w1 (the 32 raw rows of each step summed in fp64 equal the online batch gradient
             <run>/e2/grad_inputs_every100.npz window w1 at that step, rel err < 1e-5 per layer: confirms the sign and
             that the rows are that step's batch), doc_ids_unique, finite_nonzero, unit_norm (|1 - |x|| < 1e-5),
             direction_kept (cos(stored, raw) >= 0.99999 in every cell), order_hash_recorded, loader_dry_run
  batch      store_covers_steps (index complete, steps stored), lr_positive (train_log lr > 0 at every step),
             adamfx_w1_equals_online_w1 (-q_t equals <run>/e2/adamfx_inputs_every100.npz w1, rel err < 1e-5 per layer),
             grad_w1_equals_online_w1 (as docs_sum_equals_online_w1), finite_nonzero, unit_norm, direction_kept,
             order_hash_recorded, loader_dry_run; reported only: median over layers of cos(D_t, G_t) per step
  stepdelta  finite_nonzero, unit_norm, direction_kept, same_windows_every_step, order_hash_recorded, loader_dry_run;
             reported only: cos(delta_fp64, delta_fp32) per layer and cos(delta_fp64, D_t) and cos(delta_fp64,
             D_fixed_t) (store <run>/adamfx_fixed, the 128 fixed documents) at the pair step when those stores exist

Bottleneck: reading grad_docs_every100.npz (1.86 GB per run) from Lustre; everything else is small.

Usage (CPU):
  python perdoc_inputs.py perdoc --run <run> [--steps 1000:3000:100] [--docs all|0,1] [--suffix s1000_3000]
  python perdoc_inputs.py batch  --run <run> [--steps 500:3000:100] [--suffix s500_3000]
  python perdoc_inputs.py stepdelta --runs <run>,<run> [--vectors-glob '<dir>/*_step_delta_vectors.npz']
Outputs: <run>/e2/<variant>_inputs_<suffix>.npz (stepdelta: <run>/e2/stepdelta_inputs.npz); an existing output is
refused (exit 2). stepdelta exits 3 when no vectors file exists (nothing written).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adamfx  # noqa: E402
import adamfx_windows as aw  # noqa: E402

IIF = os.environ["IIF_BASE"]
RUNS_DIR = os.path.join(IIF, "agm_data", "runs")
CHECKS_DIR = os.path.join(IIF, "agm_data", "fixed_frame", "checks")
READERS = ("rl_paper_c040",)
N_DOCS = 32
REL_TOL = 1e-5
UNIT_TOL = 1e-5
COS_KEPT = 0.99999
PLAN = "docs/notes/20261025_perdoc_batch_reading.md"
UNIT_RULE = ("each [d] row unit-normalised per (step, window, layer) in fp64, stored fp32 (direction only; "
             "e2_generate normalises again and subtracts the reader's train mean unit row)")
SIGN = {
    "perdoc": ("q = +dL/dh, the token sum over the full token set of one document's summed token cross-entropy "
               "gradient at the layer output (the gradient store's sign); grad_docs_every100.npz stores -q per "
               "document; inputs = -(q) per document = the document's descent direction, unit-normalised per layer"),
    "batch": ("adamfx_w1: q_t = -dh stored per step in stepsums/adamfx (dh = dW_o X_o + dW_down X_down, the "
              "first-order change of the layer output that step t's real LoRA update makes, summed over the batch; "
              "under SGD q is a positive mix of +dL/dh rows), inputs = -(q_t) = D_t; grad_w1: q_d = +dL/dh per "
              "document, inputs = -(sum over the step's documents of q_d); both unit-normalised per layer"),
    "stepdelta": ("delta = h_after - h_before, the actual change of each layer's output (summed over the measured "
                  "documents) that one real optimizer step makes, the counterpart of D_t (= -q_t with q the +dL/dh "
                  "sign of the gradient store); inputs = -(q) = delta, unit-normalised per layer"),
}
DEFINITION = {
    "perdoc": "one document's own plain gradient (no lr, no Adam) at step t; window docNN = row NN of step t's batch",
    "batch": ("one step's batch: adamfx_w1 = D_t, the direct-write term of step t alone (o_proj and down_proj); "
              "grad_w1 = G_t, the plain batch gradient of step t (sum of the 32 document gradients)"),
    "stepdelta": "the actual one-step activation change h_after - h_before of a LoRA pair (fp64 and fp32 forwards)",
}


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [perdoc_inputs] {msg}", flush=True)


def parse_steps(spec):
    """'1000:3000:100' -> [1000, 1100, ..., 3000] (inclusive); '1000,3000' -> [1000, 3000]."""
    spec = str(spec).strip()
    if ":" in spec:
        a, b, c = (int(x) for x in spec.split(":"))
        if c <= 0 or b < a:
            raise SystemExit(f"bad step range {spec!r}")
        return list(range(a, b + 1, c))
    return [int(x) for x in spec.split(",") if x.strip()]


def unit_rows(X):
    """fp32 unit rows of X [..., d] (norm in fp64); raises on a zero or non-finite row."""
    X64 = np.asarray(X, np.float64)
    n = np.linalg.norm(X64, axis=-1, keepdims=True)
    if not np.isfinite(n).all() or (n == 0).any():
        raise ValueError("zero or non-finite row: no direction to read")
    return (X64 / n).astype(np.float32)


def cos_rows(a, b):
    return aw._cos_rows(a, b)


def rel_err_rows(a, b):
    """|a - b| / |b| per row of [..., d] (fp64)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-300)


def unit_checks(U, raw):
    """finite_nonzero (raw), unit_norm and direction_kept (U vs raw) for arrays [..., d]."""
    rn = np.linalg.norm(np.asarray(raw, np.float64), axis=-1)
    un = np.linalg.norm(np.asarray(U, np.float64), axis=-1)
    c = cos_rows(U, raw)
    return {
        "finite_nonzero": dict(ok=bool(np.isfinite(raw).all() and (rn > 0).all()), applies=True,
                               min_norm=float(rn.min()), max_norm=float(rn.max())),
        "unit_norm": dict(ok=bool(np.abs(un - 1).max() < UNIT_TOL), applies=True, max_dev=float(np.abs(un - 1).max()),
                          rule=f"| |x| - 1 | < {UNIT_TOL} in every (step, window, layer) cell"),
        "direction_kept": dict(ok=bool(c.min() >= COS_KEPT), applies=True, min_cos=float(c.min()),
                               rule=f"cos(stored, raw) >= {COS_KEPT} in every cell (same direction and sign)"),
    }


def online_w1(run, series, steps):
    """{step: [L, d] fp32} window w1 of <run>/e2/<series>_inputs_every100.npz at ``steps`` (those it has), or None
    when the file or its w1 window is absent."""
    p = os.path.join(run, "e2", f"{series}_inputs_every100.npz")
    if not os.path.exists(p):
        return None, p
    z = np.load(p)
    labels = [str(x) for x in z["windows"]]
    if "w1" not in labels:
        return None, p
    zs = [int(t) for t in z["steps"]]
    X = z["inputs"]
    wi = labels.index("w1")
    return {t: np.asarray(X[zs.index(t), wi]) for t in steps if t in zs}, p


def run_meta(run):
    p = os.path.join(run, "RUN.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f).get("settings", {})


def write_inputs(out, variant, run, U, steps, labels, layers, window_lo, window_hi, n_docs, checks, extra_arrays=None,
                 transform_extra=None, loader=True, readers=READERS):
    """Write the inputs file (tmp, loader dry run, rename) when every applicable check passed; returns the checks."""
    if os.path.exists(out):
        raise FileExistsError(out)
    order_hash = aw.export_order_hash(steps, labels, layers)
    checks["order_hash_recorded"] = dict(ok=True, applies=True, sha1=order_hash)
    st = run_meta(run)
    transform = dict(variant=variant, definition=DEFINITION[variant], unit_rule=UNIT_RULE, plan=PLAN,
                     window_rule="window_lo = window_hi = the read step (one step)", **(transform_extra or {}))
    md = dict(run=os.path.abspath(run), arm=st.get("arm"), seed=st.get("seed"), quantity=variant, kind="grad",
              kind_note="kind 'grad' is the label the trained-reader loader requires",
              sign=SIGN[variant], layers=[int(L) for L in layers], windows=[[lab, 1] for lab in labels],
              steps=[int(t) for t in steps], order_hash=order_hash, transform=transform, checks=checks,
              written=time.strftime("%Y-%m-%d %H:%M:%S"), shape=list(U.shape), dtype="float32")
    app = {k: c for k, c in checks.items() if c.get("applies", True)}
    failed = [k for k, c in app.items() if not c["ok"]]
    if failed:
        raise RuntimeError(f"{variant}: checks failed, nothing written: {failed}: "
                           + json.dumps({k: checks[k] for k in failed}, default=str)[:1500])
    md["checks_passed"] = f"{len(app)}/{len(app)} (before the loader dry run)"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.replace(tmp, tmp + ".stale_" + time.strftime("%Y%m%d_%H%M%S"))
    with open(tmp, "wb") as f:
        np.savez(f, inputs=U, steps=np.asarray(steps, dtype=np.int64), windows=np.asarray(labels),
                 window_lo=np.asarray(window_lo, dtype=np.int64), window_hi=np.asarray(window_hi, dtype=np.int64),
                 n_docs=np.asarray(n_docs, dtype=np.int64), layers=np.asarray(layers, dtype=np.int64),
                 meta=np.asarray(json.dumps(md, default=str)), **(extra_arrays or {}))
    if loader and os.path.exists(os.path.join(aw.WORKTREE, "main_run", "reader", "e2_generate.py")):
        got, err = aw.loader_dry_run([tmp], readers=readers)
        g = (got or {}).get(tmp, {})
        checks["loader_dry_run"] = dict(ok=bool(got) and g.get("windows") == list(labels)
                                        and g.get("shape") == list(U.shape) and g.get("steps") == [int(t) for t in steps]
                                        and g.get("names") == [f"{variant}_{r}" for r in readers],
                                        applies=True, names=g.get("names"), error=err)
    else:
        checks["loader_dry_run"] = dict(ok=True, applies=False, why="--no-loader or worktree absent")
    if not checks["loader_dry_run"]["ok"]:
        raise RuntimeError(f"{variant}: loader dry run failed, file left at {tmp}: {checks['loader_dry_run']}")
    os.replace(tmp, out)
    app = {k: c for k, c in checks.items() if c.get("applies", True)}
    log(f"{variant}: wrote {out} {list(U.shape)}; checks {sum(c['ok'] for c in app.values())}/{len(app)} passed "
        f"(not applicable: {[k for k, c in checks.items() if not c.get('applies', True)]})")
    with open(out.replace(".npz", "_checks.json"), "w") as f:
        json.dump(dict(output=out, checks=checks, checks_passed=f"{sum(c['ok'] for c in app.values())}/{len(app)}",
                       written=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1, default=str)
    return checks


# ---------------------------------------------------------------------------------------------------- perdoc / grad


def load_docs(run, steps):
    """raw document vectors [n_steps, 32, L, d] fp32 (-q), doc arrays and the source meta, at ``steps``."""
    p = os.path.join(run, "e2", "grad_docs_every100.npz")
    z = np.load(p)
    meta = json.loads(str(z["meta"]))
    zs = [int(t) for t in z["steps"]]
    miss = [t for t in steps if t not in zs]
    if miss:
        raise SystemExit(f"{p}: steps {miss[:5]} not stored (stored every {meta.get('every')})")
    rows = [zs.index(t) for t in steps]
    V = z["vectors"]  # [71, 32, L, d], read once
    out = dict(path=p, meta=meta, vectors=np.ascontiguousarray(V[rows]), layers=[int(L) for L in z["layers"]],
               n_docs=np.asarray(z["n_docs"])[rows], doc_id=np.asarray(z["doc_id"])[rows],
               line_no=np.asarray(z["line_no"])[rows], n_tokens=np.asarray(z["n_tokens"])[rows])
    del V
    return out


def source_meta_check(src, steps):
    m = src["meta"]
    ok = (m.get("kind") == "grad" and str(m.get("sign", "")).startswith("vectors = -q")
          and src["vectors"].shape[1] == N_DOCS and (src["n_docs"] == N_DOCS).all()
          and src["layers"] == list(range(40)) and src["vectors"].shape[0] == len(steps))
    return dict(ok=bool(ok), applies=True, file=src["path"], sign=m.get("sign"), kind=m.get("kind"),
                n_docs=[int(x) for x in src["n_docs"]], shape=list(src["vectors"].shape))


def docs_sum_check(run, steps, raw_docs):
    """sum over the 32 raw rows (fp64) vs the online w1 of grad_inputs_every100.npz, per step and layer."""
    ref, p = online_w1(run, "grad", steps)
    if ref is None:
        return dict(ok=True, applies=False, why=f"{p} or its w1 window absent")
    errs = []
    for i, t in enumerate(steps):
        if t not in ref:
            errs.append(dict(step=t, missing=True))
            continue
        s = raw_docs[i].astype(np.float64).sum(0)
        errs.append(dict(step=t, max_rel_err=float(rel_err_rows(s, ref[t]).max()),
                         min_cos=float(cos_rows(s, ref[t]).min())))
    ok = all(not e.get("missing") and e["max_rel_err"] < REL_TOL for e in errs)
    return dict(ok=ok, applies=True, file=p, worst=max((e.get("max_rel_err", 1.0) for e in errs), default=None),
                cells=errs[:3] + errs[-2:], rule=f"sum of the step's 32 document rows = online w1, rel err < {REL_TOL} "
                "per layer (same sign: the online file stores -(sum q))")


def build_perdoc(run, steps, docs=None, suffix=None, loader=True, out=None):
    src = load_docs(run, steps)
    raw = src["vectors"]  # [S, 32, L, d]
    checks = {"source_meta": source_meta_check(src, steps), "docs_sum_equals_online_w1": docs_sum_check(run, steps, raw)}
    ids = src["doc_id"]
    dup = [int(t) for i, t in enumerate(steps) if len(set(ids[i].tolist())) != ids.shape[1]]
    checks["doc_ids_unique"] = dict(ok=not dup, applies=True, steps_with_duplicates=dup[:5])
    sel = list(range(N_DOCS)) if docs is None else [int(d) for d in docs]
    raw = raw[:, sel]
    U = unit_rows(raw)
    checks.update(unit_checks(U, raw))
    labels = [f"doc{d:02d}" for d in sel]
    S, W = len(steps), len(sel)
    lo = np.repeat(np.asarray(steps, np.int64)[:, None], W, 1)
    out = out or os.path.join(run, "e2", f"perdoc_inputs_{suffix or f's{steps[0]}_{steps[-1]}'}.npz")
    extra = dict(doc_id=np.asarray(src["doc_id"][:, sel]), line_no=np.asarray(src["line_no"][:, sel], np.int64),
                 n_tokens=np.asarray(src["n_tokens"][:, sel], np.int64))
    return write_inputs(out, "perdoc", run, U, steps, labels, src["layers"], lo, lo, np.ones((S, W), np.int64), checks,
                        extra_arrays=extra, loader=loader,
                        transform_extra=dict(source=src["path"], source_sign=src["meta"].get("sign"), docs=sel,
                                             doc_arrays="doc_id, line_no, n_tokens [n_steps, n_windows]")), out


# ------------------------------------------------------------------------------------------------------------ batch


def build_batch(run, steps, suffix=None, loader=True, out=None, stepsums_dir=None):
    sd = stepsums_dir or aw.find_stepsums(run, "adamfx")
    idx = aw.read_index(sd)
    first, last = int(idx["blocks"][0]["start"]), int(idx["blocks"][-1]["end"])
    checks = {"store_covers_steps": dict(ok=bool(idx.get("complete")) and first <= min(steps) and last >= max(steps),
                                         applies=True, first=first, last=last, complete=bool(idx.get("complete")),
                                         format=idx.get("format"), dir=sd)}
    if not checks["store_covers_steps"]["ok"]:
        raise RuntimeError(f"{sd}: store does not cover {steps[0]}..{steps[-1]} or is incomplete: {checks}")
    lr = aw.load_lr(run)
    bad = [t for t in steps if not lr.get(t, 0.0) > 0]
    checks["lr_positive"] = dict(ok=not bad, applies=True, bad=bad[:5],
                                 lr={str(t): lr.get(t) for t in (steps[0], steps[len(steps) // 2], steps[-1])})
    D = np.stack([-adamfx.load_stepsums(sd, t, t)[1][0] for t in steps])  # [S, L, d] = -(q_t) = D_t
    ref, p = online_w1(run, "adamfx", steps)
    if ref is None:
        checks["adamfx_w1_equals_online_w1"] = dict(ok=True, applies=False, why=f"{p} or its w1 window absent")
    else:
        errs = [float(rel_err_rows(D[i], ref[t]).max()) if t in ref else 1.0 for i, t in enumerate(steps)]
        checks["adamfx_w1_equals_online_w1"] = dict(
            ok=max(errs) < REL_TOL, applies=True, file=p, worst=max(errs),
            rule=f"-(stored q_t) = online w1 (ring sum of the same fp32 step), rel err < {REL_TOL} per layer")
    src = load_docs(run, steps)
    checks["source_meta"] = source_meta_check(src, steps)
    G = src["vectors"].astype(np.float64).sum(1).astype(np.float32)  # [S, L, d]
    checks["grad_w1_equals_online_w1"] = docs_sum_check(run, steps, src["vectors"])
    raw = np.stack([D, G], axis=1)  # [S, 2, L, d]
    U = unit_rows(raw)
    checks.update(unit_checks(U, raw))
    cdg = [float(np.median(cos_rows(D[i], G[i]))) for i in range(len(steps))]
    log("median over layers of cos(D_t, G_t) per step (report): "
        + ", ".join(f"{t}: {c:.3f}" for t, c in zip(steps, cdg)))
    labels = ["adamfx_w1", "grad_w1"]
    lo = np.repeat(np.asarray(steps, np.int64)[:, None], 2, 1)
    nd = np.stack([src["n_docs"], src["n_docs"]], 1)
    out = out or os.path.join(run, "e2", f"batch_inputs_{suffix or f's{steps[0]}_{steps[-1]}'}.npz")
    return write_inputs(out, "batch", run, U, steps, labels, src["layers"], lo, lo, nd, checks, loader=loader,
                        transform_extra=dict(stepsums_dir=os.path.abspath(sd), docs_source=src["path"],
                                             report_median_cos_D_G={str(t): c for t, c in zip(steps, cdg)})), out


# -------------------------------------------------------------------------------------------------------- stepdelta


def read_vectors_file(path):
    """[(run, step, {window: [L, d] fp64}, meta)] from one *_step_delta_vectors.npz (delta_fp64 / delta_fp32 / ...)."""
    z = np.load(path)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    keys = [k for k in ("delta_fp64", "delta_fp32") if k in z.files]
    if not keys:
        raise ValueError(f"{path}: no delta_fp64 / delta_fp32 arrays ({z.files})")
    arrs = {k: np.asarray(z[k], np.float64) for k in keys}
    shp = {k: a.shape for k, a in arrs.items()}
    run = meta.get("run")
    if run is None:
        raise ValueError(f"{path}: meta has no run")
    run = run if os.path.isabs(run) else os.path.join(RUNS_DIR, run)
    if all(a.ndim == 2 for a in arrs.values()):
        # measure_step_delta_precision.py writes lora_pair_step (the optimizer step of the pair, before = step - 1)
        step = next((meta[k] for k in ("lora_pair_step", "pair_step", "step") if meta.get(k) is not None), None)
        if step is None:
            raise ValueError(f"{path}: meta has no lora_pair_step / pair_step / step")
        return [(run, int(step), arrs, meta)]
    if all(a.ndim == 3 for a in arrs.values()):
        st = meta.get("pair_steps", meta.get("steps"))
        if st is None or len(st) != next(iter(arrs.values())).shape[0]:
            raise ValueError(f"{path}: arrays {shp} but meta pair_steps/steps {st}")
        return [(run, int(t), {k: a[i] for k, a in arrs.items()}, meta) for i, t in enumerate(st)]
    raise ValueError(f"{path}: unexpected shapes {shp}")


def collect_step_deltas(paths):
    """{run: {step: (windows dict, meta, path)}}; a later path (sorted by name = stamp) replaces an earlier one."""
    out = {}
    for p in sorted(paths):
        for run, step, arrs, meta in read_vectors_file(p):
            out.setdefault(os.path.realpath(run), {})[step] = (arrs, meta, p)
    return out


def build_stepdelta(run, entries, loader=True, out=None):
    """entries {step: (windows dict, meta, path)} of one run."""
    steps = sorted(entries)
    labels = [k for k in ("delta_fp64", "delta_fp32") if k in entries[steps[0]][0]]
    same = all([k for k in ("delta_fp64", "delta_fp32") if k in entries[t][0]] == labels for t in steps)
    checks = {"same_windows_every_step": dict(ok=same, applies=True, windows=labels)}
    if not same:
        raise RuntimeError(f"{run}: the vectors files give different windows per step: {checks}")
    raw = np.stack([np.stack([entries[t][0][k] for k in labels]) for t in steps])  # [S, W, L, d]
    if raw.shape[2:] != (40, raw.shape[-1]):
        raise RuntimeError(f"{run}: step-delta arrays {raw.shape}, expected [*, *, 40, d]")
    U = unit_rows(raw)
    checks.update(unit_checks(U, raw))
    report = {}
    for i, t in enumerate(steps):
        r = {}
        if "delta_fp32" in labels:
            r["cos_fp64_fp32_median"] = float(np.median(cos_rows(raw[i, 0], raw[i, labels.index("delta_fp32")])))
        for series, name in (("adamfx", "cos_delta_Dt_median"), ("adamfx_fixed", "cos_delta_Dfixed_median")):
            try:
                sd = aw.find_stepsums(run, series)
                q = adamfx.load_stepsums(sd, t, t)[1][0]
                c = cos_rows(raw[i, 0], -q)
                r[name] = float(np.median(c))
                r[name.replace("median", "per_layer")] = [round(float(x), 4) for x in c]
            except (SystemExit, RuntimeError, OSError) as e:
                r[name] = f"n/a: {type(e).__name__}: {str(e)[:120]}"
        r["source"] = entries[t][2]
        report[str(t)] = r
    lo = np.repeat(np.asarray(steps, np.int64)[:, None], len(labels), 1)
    nd = np.asarray([[int(entries[t][1].get("n_docs", 0) or 0)] * len(labels) for t in steps], np.int64)
    out = out or os.path.join(run, "e2", "stepdelta_inputs.npz")
    return write_inputs(out, "stepdelta", run, U, steps, labels, list(range(40)), lo, lo, nd, checks, loader=loader,
                        transform_extra=dict(sources={str(t): entries[t][2] for t in steps},
                                             source_meta={str(t): entries[t][1] for t in steps},
                                             report=report)), out


# ------------------------------------------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("perdoc")
    a1.add_argument("--run", required=True)
    a1.add_argument("--steps", default="1000:3000:100")
    a1.add_argument("--docs", default="all", help="all, or a comma list of batch rows (smoke: 0,1)")
    a1.add_argument("--suffix", default=None)
    a1.add_argument("--no-loader", action="store_true")
    a2 = sub.add_parser("batch")
    a2.add_argument("--run", required=True)
    a2.add_argument("--steps", default="500:3000:100")
    a2.add_argument("--suffix", default=None)
    a2.add_argument("--no-loader", action="store_true")
    a3 = sub.add_parser("stepdelta")
    a3.add_argument("--runs", required=True, help="comma list of run dirs; vectors of other runs are ignored")
    a3.add_argument("--vectors-glob", default=os.path.join(CHECKS_DIR, "*_step_delta_vectors.npz"))
    a3.add_argument("--no-loader", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        if a.cmd == "perdoc":
            docs = None if a.docs == "all" else [int(x) for x in a.docs.split(",")]
            ck, out = build_perdoc(a.run, parse_steps(a.steps), docs=docs, suffix=a.suffix, loader=not a.no_loader)
        elif a.cmd == "batch":
            ck, out = build_batch(a.run, parse_steps(a.steps), suffix=a.suffix, loader=not a.no_loader)
        else:
            paths = sorted(glob.glob(a.vectors_glob))
            if not paths:
                log(f"stepdelta: no file matches {a.vectors_glob}; skipped")
                return 3
            found = collect_step_deltas(paths)
            wrote = 0
            for run in [r.strip() for r in a.runs.split(",") if r.strip()]:
                ent = found.get(os.path.realpath(run))
                if not ent:
                    log(f"stepdelta: no vectors for {run} in {paths}")
                    continue
                build_stepdelta(run, ent, loader=not a.no_loader)
                wrote += 1
            if not wrote:
                log(f"stepdelta: none of {paths} is for {a.runs}; skipped (runs in the files: {sorted(found)})")
                return 3
    except FileExistsError as e:
        log(f"refusing to overwrite {e}")
        return 2
    log(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
