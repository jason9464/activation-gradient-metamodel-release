#!/usr/bin/env python
"""Summary of the extended clean-batch variants (scripts/cleanbatch/variants_chain.sh; variants in the module
docstring of cleanbatch_probe.py, "Extended variants": mom = gradient plus momentum without per-coordinate scaling,
hist = the Adam step's history part, curr = its current-batch part, step = hist + curr in weight space).

Per student (dolphin, clean; runs <R>/cleanbatch_var_<arm>_s1/train):
  P   first-token P(Yes) x 100 of the dolphin_yesno question (cleanbatch_pyes.py on the "all" file), per probe step for
      the windows step, mfree, mom, hist, curr, grad of the real batch and the three foreign batches; means over the
      probe steps of every variant the file has
  J   judge YES per 100 answers (judge_dolphin_answers.py outputs; REFUSED leaves the denominator) of the mom and hist
      readings, and Y = answers whose first word is "yes", per step, when the readings exist
  decomposition diagnostics from the probe records (layer medians): cos(D(step), D(hist)), cos(D(step), D(curr)),
      cos(D(curr), D(mfree)), |D(hist)|, |D(curr)|, |D(mom)| over |D(step)|, the LoRA cross term |C| / |D(step)|
      (step = hist + curr holds exactly in weight space; on the direct-write term only up to C), cos(D(real hist),
      D(f_k hist)) on the own and on the fixed documents and the weight-space cosine of the hist updates (the own-
      document cosine also moves with each batch's own inputs X), the identity residuals
  integrity: checks of cleanbatch_inputs.py verify (trajectory, probe_complete, real_vs_store, real_vs_log,
      variant_identity, ref_probe) and of every built input file (<group>_checks.json); P(Yes) of the windows shared
      with the first clean-batch run (<R>/cleanbatch_<arm>_s1) compared with that run's values
Works on partial outputs: prints what exists and lists what is missing.

  python cleanbatch_variants_summary.py [--runs-dir <R>] [--out-md <R>/cleanbatch_var_summary.md] [--out-json ...]
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
import cleanbatch_probe as cp  # noqa: E402
import cleanbatch_summary as cs  # noqa: E402

ARMS = ("dolphin", "clean")
TABLE_VARIANTS = ("step", "mfree", "mom", "hist", "curr", "grad")
JUDGED = ("mom", "hist")  # reading groups (tag = group = variant)
QUESTION = "dolphin_yesno"


def var_train(runs, arm):
    return os.path.join(runs, f"cleanbatch_var_{arm}_s1", "train")


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def f0(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.0f}"


def f3(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def fe(x):
    return fmt(x, ".1e")


def fmt(x, spec):
    """x formatted with spec; "-" for None and non-finite values."""
    if x is None:
        return "-"
    try:
        if not np.isfinite(x):
            return "-"
    except TypeError:
        return "-"
    return format(x, spec)


def mean_over(d, steps):
    xs = [d[t] for t in steps if t in d]
    return float(np.mean(xs)) if xs else None


def collect(runs, arm):
    tr = var_train(runs, arm)
    d = dict(train=tr, missing=[], P=None, J={}, Y={}, steps=None, records=None, verify=None, input_checks={},
             earlier_pyes=None)
    if not os.path.isdir(tr):
        d["missing"].append(f"run directory {tr}")
        return d
    meta = load_json(os.path.join(tr, "cleanbatch", "probe_meta.json"))
    if meta is None:
        d["missing"].append("probe_meta.json")
    else:
        d["records"] = {int(r["step"]): r for r in meta["records"]}
        d["probe_complete"] = bool(meta.get("complete"))
        d["planned"] = meta.get("planned_steps")
        if not meta.get("complete"):
            d["missing"].append(f"probe incomplete ({len(meta['records'])}/{len(meta.get('planned_steps', []))} steps)")
    d["verify"] = load_json(os.path.join(tr, "cleanbatch", "verify.json"))
    if d["verify"] is None:
        d["missing"].append("verify.json")
    for p in sorted(glob.glob(os.path.join(tr, "e2", "cleanbatch_inputs_*_checks.json"))):
        g = os.path.basename(p)[len("cleanbatch_inputs_"):-len("_checks.json")]
        c = load_json(p)["checks"]
        d["input_checks"][g] = dict(passed=sum(bool(x["ok"]) for x in c.values()), total=len(c))
    d["P"] = cs.load_pyes(tr)
    if d["P"] is None:
        d["missing"].append("first-token P(Yes) file")
    else:
        d["steps"] = sorted(next(iter(d["P"].values())))
    for g in JUDGED:
        src = cs.reading_path(tr, QUESTION, g)
        rd = cs.load_reading(src)
        if rd is None:
            d["missing"].append(f"{g} {QUESTION} reading")
            continue
        d["Y"].update(cs.y_counts(rd))
        d["steps"] = d["steps"] or sorted(next(iter(rd.values())))
        for b in cp.BATCH_NAMES:
            c = cs.j_counts(src, f"{b}_{g}")
            if c is None:
                d["missing"].append(f"judge {QUESTION} {b}_{g}")
            else:
                d["J"][f"{b}_{g}"] = c
    # P(Yes) of the windows shared with the first clean-batch run
    first = cs.load_pyes(cs.train_dir(runs, arm))
    if d["P"] is not None and first is not None:
        diffs = [abs(d["P"][w][t] - first[w][t]) for w in first if w in d["P"] for t in first[w] if t in d["P"][w]]
        d["earlier_pyes"] = dict(windows=sorted(w for w in first if w in d["P"]), compared=len(diffs),
                                 max_abs_diff_x100=max(diffs) if diffs else None)
    return d


def diag_rows(records, steps):
    """Per probe step: the decomposition diagnostics of the real batch and the median over the foreign batches."""
    rows = []
    for t in steps:
        r = (records or {}).get(t)
        if r is None or "extended" not in r:
            continue
        ex = [pb["extended"] for pb in r["per_batch"]]

        def both(fn):
            vals = [fn(e) for e in ex]
            return vals[0], float(np.median(vals[1:]))

        ratio = (lambda k: (lambda e: e["d_norm_median"][k] / e["d_norm_median"]["step"]))
        rows.append(dict(
            step=t,
            cos_step_hist=both(lambda e: e["cos_median"]["step_hist"]),
            cos_step_curr=both(lambda e: e["cos_median"]["step_curr"]),
            cos_curr_mfree=both(lambda e: e["cos_median"]["curr_mfree"]),
            cos_step_mom=both(lambda e: e["cos_median"]["step_mom"]),
            hist_over_step=both(ratio("hist")), curr_over_step=both(ratio("curr")), mom_over_step=both(ratio("mom")),
            cross_over_step=both(lambda e: e["cross_rel_median_own"]),
            cos_hist_real_f=r["extended"]["cos_hist_real_f"], cos_histfix_real_f=r["extended"]["cos_histfix_real_f"],
            wcos_hist_real_f=r["extended"]["wcos_hist_real_f"],
            w_identity_rel_max=r["extended"]["w_identity_rel_max"], d_identity_rel_max=r["extended"]["d_identity_rel_max"],
            step32_vs_step64_rel_max=r["extended"]["step32_vs_step64_rel_max"], identity_ok=r["extended"]["identity_ok"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=cp.RUNS)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    runs = a.runs_dir
    out_md = a.out_md or os.path.join(runs, "cleanbatch_var_summary.md")
    out_json = a.out_json or os.path.join(runs, "cleanbatch_var_summary.json")
    data = {arm: collect(runs, arm) for arm in ARMS}
    smoke_verify = load_json(os.path.join(runs, "cleanbatch_var_dolphin_s1", "smoke", "cleanbatch", "verify.json"))
    res = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), runs_dir=runs, notes=a.notes, students={})
    L = ["# Extended clean-batch variants: history and current-batch parts of the Adam step", "",
         f"Written {res['written']} by main_run/cleanbatch_variants_summary.py. {a.notes}".rstrip(), "",
         "Variants (direct-write term D on the batch's own documents unless the name ends in fix): step = the would-be "
         "Adam step; mfree = momentum-free step; mom = -(lr/bc1) m_t (gradient plus momentum, no per-coordinate "
         "scaling); hist = -(lr/bc1) beta1 m_{t-1} / denom_t (the step without the batch's gradient in the numerator); "
         "curr = -(lr/bc1) (1 - beta1) g / denom_t (the batch's own part); grad = plain batch gradient. step = hist + "
         "curr in weight space; in D a LoRA cross term C is added (dW = s (B1 A1 - B0 A0) is bilinear in the factors).",
         "", "Batches: real = the step's own training batch; f1..f3 = three fixed batches of the other student's data "
         "(clean documents for the dolphin student, dolphin documents for the clean student).", ""]
    # integrity
    L += ["## Integrity checks", ""]
    if smoke_verify is not None:
        L.append(f"- smoke (60 steps, probes 20 and 40): verify {smoke_verify.get('checks_passed')} "
                 + ", ".join(f"{k} {'ok' if c['ok'] else 'FAILED'}" for k, c in smoke_verify["checks"].items()))
    for arm in ARMS:
        d = data[arm]
        v = d["verify"]
        if v is not None:
            L.append(f"- {arm} student: verify {v.get('checks_passed')} "
                     + ", ".join(f"{k} {'ok' if c['ok'] else 'FAILED'}" for k, c in v["checks"].items()))
            rp = v["checks"].get("ref_probe")
            if rp:
                L.append(f"  - step/stepfix/mfree/mfreefix/grad vectors vs the first run: {rp['passed']}/{rp['compared']} "
                         f"pass, {rp['bit_identical']} bit-identical, min cos {rp['min_cos']}")
            vi_ = v["checks"].get("variant_identity")
            if vi_:
                L.append(f"  - step = hist + curr in weight space and D(step) = D(hist) + D(curr) + C on the direct-write "
                         f"term: {vi_['steps_ok']}/{vi_['steps']} steps ({vi_['batches_checked']} batch records), weight "
                         f"rel max {fe(vi_['w_rel_max'])}, D rel max (with C) {fe(vi_['d_rel_max'])}, fp32 step vs fp64 "
                         f"max {fe(vi_['step32_rel_max'])}")
        if d["input_checks"]:
            L.append(f"  - input files: " + ", ".join(f"{g} {c['passed']}/{c['total']}" for g, c in sorted(d["input_checks"].items())))
        if d["earlier_pyes"]:
            e = d["earlier_pyes"]
            L.append(f"  - P(Yes) of the {len(e['windows'])} windows shared with the first run: {e['compared']} values, "
                     f"max abs difference {fmt(e['max_abs_diff_x100'], '.4f')} (x100)")
    L.append("")
    # P(Yes)
    for arm in ARMS:
        d = data[arm]
        st = dict(missing=d["missing"])
        res["students"][arm] = st
        L += [f"## {arm} student (foreign batches: {cp.OTHER[arm]} data)", ""]
        P = d["P"]
        steps = d["steps"] or []
        if P is not None:
            vs = []
            for v in cp.EXTENDED_VARIANTS:
                if all(f"{b}_{v}" in P for b in cp.BATCH_NAMES):
                    vs.append(v)
            means = {v: {b: mean_over(P[f"{b}_{v}"], steps) for b in cp.BATCH_NAMES} for v in vs}
            st["P_mean"] = means
            st["P"] = {w: {str(t): x for t, x in P[w].items()} for w in P}
            L += [f"First-token P(Yes) x 100, mean over the {len(steps)} probe steps ({steps[0]}..{steps[-1]}):", "",
                  "| variant | real | f1 | f2 | f3 | foreign mean |", "|---|---|---|---|---|---|"]
            for v in vs:
                m = means[v]
                L.append(f"| {v} | {f0(m['real'])} | {f0(m['f1'])} | {f0(m['f2'])} | {f0(m['f3'])} | "
                         f"{f0(np.mean([m[b] for b in ('f1', 'f2', 'f3')]))} |")
            L.append("")
            for v in TABLE_VARIANTS:
                if v not in vs:
                    continue
                L += [f"P(Yes) x 100 per step, {v}:", "", "| step | real | f1 | f2 | f3 |", "|---|---|---|---|---|"]
                for t in steps:
                    L.append(f"| {t} | " + " | ".join(f0(P[f'{b}_{v}'].get(t)) for b in cp.BATCH_NAMES) + " |")
                L.append("")
        # judge
        for g in JUDGED:
            J = {b: d["J"].get(f"{b}_{g}") for b in cp.BATCH_NAMES}
            Y = {b: d["Y"].get(f"{b}_{g}") for b in cp.BATCH_NAMES}
            if all(x is None for x in J.values()) and all(x is None for x in Y.values()):
                continue
            jsteps = sorted(set().union(*[set(x) for x in list(J.values()) + list(Y.values()) if x]))
            st[f"J_{g}"] = {b: (None if J[b] is None else {str(t): x for t, x in J[b].items()}) for b in cp.BATCH_NAMES}
            st[f"Y_{g}"] = {b: (None if Y[b] is None else {str(t): x for t, x in Y[b].items()}) for b in cp.BATCH_NAMES}
            st[f"J_{g}_mean"] = {b: (None if J[b] is None else mean_over(J[b], jsteps)) for b in cp.BATCH_NAMES}
            L += [f"Readings of {g} (100 answers per step, dolphin_yesno): judge YES per 100 (J) and first word yes (Y):", "",
                  "| step | J real | J f1 | J f2 | J f3 | Y real | Y f1 | Y f2 | Y f3 |", "|---|---|---|---|---|---|---|---|---|"]
            for t in jsteps:
                L.append(f"| {t} | " + " | ".join(f0((J[b] or {}).get(t)) for b in cp.BATCH_NAMES) + " | "
                         + " | ".join(f0((Y[b] or {}).get(t)) for b in cp.BATCH_NAMES) + " |")
            L.append("| mean | " + " | ".join(f0(None if J[b] is None else mean_over(J[b], jsteps)) for b in cp.BATCH_NAMES)
                     + " | " + " | ".join(f0(None if Y[b] is None else mean_over(Y[b], jsteps)) for b in cp.BATCH_NAMES) + " |")
            L.append("")
        # decomposition diagnostics
        rows = diag_rows(d["records"], sorted(d["records"] or []))
        st["decomposition"] = rows
        if rows:
            L += ["Decomposition (layer medians on the own documents; real batch / median of the foreign batches): "
                  "cos(D step, D hist), cos(D step, D curr), cos(D curr, D mfree), |D hist| / |D step|, |D curr| / "
                  "|D step|, |D mom| / |D step|, cross term |C| / |D step|; then cos(D real hist, D f_k hist) on the own "
                  "and on the fixed documents (median over k), the weight-space cosine of the real and the foreign hist "
                  "updates (median over k) and the identity residuals.", "",
                  "step = hist + curr holds exactly in weight space. On the direct-write term D (the reader's input) "
                  "D(step) = D(hist) + D(curr) + C, so the hist and curr inputs add up to the step input only up to the "
                  "cross term C (column C/step). The own-document cosine of the hist parts also moves with each batch's "
                  "own inputs X; whether the history part depends on the batch is read from the fixed-document and the "
                  "weight-space columns (dW_hist depends on the batch only through denom_t, since v_t contains g^2).", "",
                  "| step | cos step-hist | cos step-curr | cos curr-mfree | hist/step | curr/step | mom/step | C/step "
                  "| cos real-f hist own | fixed | weight | weight rel | D rel |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

            def med(xs):
                return None if not xs else float(np.median(xs))

            for r in rows:
                pair = (lambda k, spec=".3f": f"{fmt(r[k][0], spec)} / {fmt(r[k][1], spec)}")
                L.append(f"| {r['step']} | {pair('cos_step_hist')} | {pair('cos_step_curr')} | {pair('cos_curr_mfree')} | "
                         f"{pair('hist_over_step')} | {pair('curr_over_step')} | {pair('mom_over_step')} | "
                         f"{pair('cross_over_step', '.1e')} | "
                         f"{fmt(med(r['cos_hist_real_f']), '.3f')} | {fmt(med(r['cos_histfix_real_f']), '.4f')} | "
                         f"{fmt(med(r['wcos_hist_real_f']), '.4f')} | "
                         f"{fmt(r['w_identity_rel_max'], '.0e')} | {fmt(r['d_identity_rel_max'], '.0e')} |")
            L.append("")
        if d["missing"]:
            L += ["Missing: " + "; ".join(d["missing"]), ""]
    res["integrity"] = {arm: dict(verify=(data[arm]["verify"] or {}).get("checks"), input_checks=data[arm]["input_checks"],
                                  earlier_pyes=data[arm]["earlier_pyes"]) for arm in ARMS}
    res["smoke_verify"] = None if smoke_verify is None else smoke_verify.get("checks")
    for p, text in ((out_md, "\n".join(L) + "\n"), (out_json, json.dumps(res, indent=1, default=str))):
        with open(p + ".tmp", "w") as f:
            f.write(text)
        os.replace(p + ".tmp", p)
    miss = {arm: data[arm]["missing"] for arm in ARMS}
    print(f"wrote {out_md} and {out_json}; missing: {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
