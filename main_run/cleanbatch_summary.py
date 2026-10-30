#!/usr/bin/env python
"""Summary of the foreign-batch readings (plan docs/notes/20261026_clean_batch_update_plan.md, section "decision rule").

Counts per (student, group, window, step), each out of the answers of that reading (100):
  J   judge YES (judge_dolphin_answers.py output <reading>_<window>_judge_dolphin.jsonl; REFUSED leaves the
      denominator, as the judge's own aggregation), scaled to 100 answers; core and mfree groups, both questions
      (mfree judged from 2026-10-26 14:45, session decision; reported as the rule shapes <arm>_mfree_J and in the
      mfree table, next to the leading-yes shapes the plan fixed for mfree)
  Y   answers whose first word is "yes" (perdoc_summary.parse_yesno), dolphin_yesno question, every group
  P   first-token P(yes) x 100 (cleanbatch_pyes.py on the "all" file), every window
Criteria (fixed in the plan before the run; S = probe steps >= 800, 23 steps; "most" = at least 16 of them):
  dolphin student, step variant (core), J of dolphin_yesno:
    R_t real batch, F_t median over the three clean batches, C_t the clean student's real batch at step t
    (A) batch data: #{F_t <= 46} >= 16 and #{R_t >= 61} >= 16
    (B) student state: #{|F_t - R_t| <= 15} >= 16
  the same rule shapes on every other variant (Y counts) and on core with Y, reported as outcomes:
    real high, foreign low  = (A) shape;  tracking = (B) shape;  both high = #{R >= 61} >= 16 and #{F >= 61} >= 16;
    both low = #{R <= 46} >= 16 and #{F <= 46} >= 16
  reverse: the clean student's dolphin batches Fc_t: high if #{Fc_t >= 61} >= 16, low if #{Fc_t <= 46} >= 16
  cheap statistic: Pearson and Spearman correlation of P with Y over every input, and with J over the core inputs
  (the step variant only, as fixed in the plan);
  AUC of dolphin-student real batch vs clean-student real batch over S, for J, Y and P
Outputs: --out-md and --out-json (default runs/cleanbatch_summary.md and .json). Missing readings are reported as
missing; the criteria are evaluated only when all their inputs exist.

  python cleanbatch_summary.py [--runs-dir <R>] [--out-md ...] [--out-json ...]
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
import cleanbatch_probe as cp  # noqa: E402
import judge_dolphin_answers as jd  # noqa: E402
from cleanbatch_inputs import GROUPS  # noqa: E402
from perdoc_summary import auc, parse_yesno  # noqa: E402

READER = "rl_paper_c040"
ARMS = ("dolphin", "clean")
S_MIN = 800
MOST = 16
TRACK = 15
# thresholds per count type, from the earlier reading of the same quantity (20261025_perdoc_batch_reading_results.md,
# D_t of the same two students): LOW = the clean student's largest count over the 26 steps, HIGH = the count the dolphin
# student reached or exceeded in 19 of the 23 steps from 800 (judge: 46 and 61, the user's numbers; first word: 38, 55)
THRESH = {"J": (46, 61), "Y": (38, 55), "P": (38, 55)}
LOW, HIGH = THRESH["J"]
READ_GROUPS = ("core", "mfree", "fixed", "grad", "mfreefix")
QUESTIONS = {"core": ("dolphin_yesno", "preference"), "mfree": ("dolphin_yesno", "preference")}
JUDGED = {"core": "step", "mfree": "mfree"}  # judged group -> its variant (windows <batch>_<variant>)


def train_dir(runs, arm):
    return os.path.join(runs, f"cleanbatch_{arm}_s1", "train")


def reading_path(train, q, group):
    return os.path.join(train, "e2", f"cleanbatch_{READER}_{q}_{group}.jsonl")


def load_reading(path):
    """{window: {step: texts}} or None."""
    if not os.path.exists(path):
        return None
    out = {}
    for r in jd.read_rows(path):
        for w, v in r["inputs"].items():
            out.setdefault(w, {})[int(r["step"])] = v["texts"]
    return out


def y_counts(rd):
    return {w: {t: 100.0 * sum(parse_yesno(x) == "yes" for x in tx) / max(len(tx), 1) for t, tx in d.items()}
            for w, d in rd.items()}


def j_counts(src, window):
    p = jd.output_path(src, window)
    if not os.path.exists(p):
        return None
    yes, n = {}, {}
    for r in jd.read_rows(p):
        if r["verdict"] == "REFUSED":
            continue
        t = int(r["step"])
        n[t] = n.get(t, 0) + 1
        yes[t] = yes.get(t, 0) + (r["verdict"] == "YES")
    return {t: 100.0 * yes[t] / n[t] for t in n}


def load_pyes(train):
    p = os.path.join(train, "e2", f"cleanbatch_inputs_all_pyes_{READER}_dolphin_yesno.json")
    if not os.path.exists(p):
        return None
    z = json.load(open(p))
    return {w: {int(t): 100.0 * z["p_yes"][i][k] for i, t in enumerate(z["steps"])} for k, w in enumerate(z["windows"])}


def rule(R, F, steps, kind="J"):
    """Outcomes of the rule shapes over ``steps`` (dicts step -> count), with the thresholds of count type ``kind``."""
    if not steps or any(t not in R or t not in F for t in steps):
        return None
    LOW, HIGH = THRESH[kind]
    r = np.asarray([R[t] for t in steps])
    f = np.asarray([F[t] for t in steps])
    n = dict(f_low=int((f <= LOW).sum()), r_high=int((r >= HIGH).sum()), track=int((np.abs(f - r) <= TRACK).sum()),
             f_high=int((f >= HIGH).sum()), r_low=int((r <= LOW).sum()), n_steps=len(steps))
    out = []
    if n["f_low"] >= MOST and n["r_high"] >= MOST:
        out.append("real high, foreign low")
    if n["track"] >= MOST:
        out.append("foreign tracks real")
    if n["r_high"] >= MOST and n["f_high"] >= MOST:
        out.append("both high")
    if n["r_low"] >= MOST and n["f_low"] >= MOST:
        out.append("both low")
    return dict(counts=n, outcomes=out or ["none of the shapes"], kind=kind, low=LOW, high=HIGH,
                median_real=float(np.median(r)), median_foreign=float(np.median(f)))


def med_foreign(counts, variant):
    fs = [counts.get(f"f{k}_{variant}") for k in (1, 2, 3)]
    if any(x is None for x in fs):
        return None
    steps = sorted(set(fs[0]) & set(fs[1]) & set(fs[2]))
    return {t: float(np.median([x[t] for x in fs])) for t in steps}


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return dict(n=len(a), pearson=None, spearman=None)
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return dict(n=len(a), pearson=float(np.corrcoef(a, b)[0, 1]), spearman=float(np.corrcoef(ra, rb)[0, 1]))


def fmt(x):
    return "-" if x is None else f"{x:.0f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", default=cp.RUNS)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    runs = a.runs_dir
    out_md = a.out_md or os.path.join(runs, "cleanbatch_summary.md")
    out_json = a.out_json or os.path.join(runs, "cleanbatch_summary.json")
    data = {}
    for arm in ARMS:
        tr = train_dir(runs, arm)
        d = dict(Y={}, J={q: {} for q in ("dolphin_yesno", "preference")}, P=load_pyes(tr), steps=None, missing=[])
        vp = os.path.join(tr, "cleanbatch", "verify.json")
        d["verify"] = json.load(open(vp)) if os.path.exists(vp) else None
        for g in READ_GROUPS:
            rd = load_reading(reading_path(tr, "dolphin_yesno", g))
            if rd is None:
                d["missing"].append(f"{g} dolphin_yesno reading")
                continue
            d["Y"].update(y_counts(rd))
            d["steps"] = sorted(next(iter(rd.values())))
        for g, v in JUDGED.items():
            for q in ("dolphin_yesno", "preference"):
                src = reading_path(tr, q, g)
                if not os.path.exists(src):
                    d["missing"].append(f"{g} {q} reading")
                    continue
                for b in cp.BATCH_NAMES:
                    c = j_counts(src, f"{b}_{v}")
                    if c is None:
                        d["missing"].append(f"judge {q} {b}_{v}")
                    else:
                        d["J"][q][f"{b}_{v}"] = c
        data[arm] = d
    steps = data["dolphin"]["steps"] or data["clean"]["steps"] or []
    S = [t for t in steps if t >= S_MIN]
    res = dict(settings=dict(S_min=S_MIN, most=MOST, thresholds=THRESH, track=TRACK, reader=READER), criteria={})
    D, C = data["dolphin"], data["clean"]
    # main criteria: J dolphin_yesno, step variant
    Jd, Jc = D["J"]["dolphin_yesno"], C["J"]["dolphin_yesno"]
    main = None
    if all(f"{b}_step" in Jd for b in cp.BATCH_NAMES):
        R, F = Jd["real_step"], med_foreign(Jd, "step")
        rr = rule(R, F, S)
        if rr is not None:
            A = rr["counts"]["f_low"] >= MOST and rr["counts"]["r_high"] >= MOST
            B = rr["counts"]["track"] >= MOST
            main = dict(rule=rr, A=A, B=B, verdict=("(A) the signal belongs to the batch data" if A and not B else
                                                    "(B) the signal belongs to the student's state" if B and not A else
                                                    "both rules hold" if A and B else "neither rule holds"))
            if "real_step" in Jc:
                main["clean_student_real"] = dict(median=float(np.median([Jc["real_step"][t] for t in S if t in Jc["real_step"]])),
                                                  max=float(max(Jc["real_step"][t] for t in S if t in Jc["real_step"])))
    res["criteria"]["main_J_step"] = main
    # rule shapes on every variant with Y (dolphin student), and reverse direction
    shapes = {}
    for arm in ARMS:
        Y = data[arm]["Y"]
        for v in ("step", "mfree", "stepfix", "mfreefix", "grad"):
            if f"real_{v}" in Y and med_foreign(Y, v) is not None:
                shapes[f"{arm}_{v}_Y"] = rule(Y[f"real_{v}"], med_foreign(Y, v), S, "Y")
        P = data[arm]["P"] or {}
        for v in ("step", "mfree", "stepfix", "mfreefix", "grad"):
            if f"real_{v}" in P and med_foreign(P, v) is not None:
                shapes[f"{arm}_{v}_P"] = rule(P[f"real_{v}"], med_foreign(P, v), S, "P")
        Jq = data[arm]["J"]["dolphin_yesno"]
        for v in JUDGED.values():
            if all(f"{b}_{v}" in Jq for b in cp.BATCH_NAMES):
                shapes[f"{arm}_{v}_J"] = rule(Jq[f"real_{v}"], med_foreign(Jq, v), S, "J")
    res["criteria"]["shapes"] = shapes
    # reverse: dolphin batches at the clean checkpoint
    rev = {}
    for kind, src in (("J", Jc), ("Y", C["Y"])):
        F = med_foreign(src, "step")
        if F is not None and all(t in F for t in S):
            lo_, hi_ = THRESH[kind]
            f = np.asarray([F[t] for t in S])
            rev[kind] = dict(high=int((f >= hi_).sum()) >= MOST, low=int((f <= lo_).sum()) >= MOST, low_thr=lo_, high_thr=hi_,
                             n_high=int((f >= hi_).sum()), n_low=int((f <= lo_).sum()), median=float(np.median(f)))
    res["criteria"]["reverse_step"] = rev
    # cheap statistic
    pairs_y, pairs_j = [], []
    for arm in ARMS:
        P = data[arm]["P"]
        if P is None:
            continue
        for w, ys in data[arm]["Y"].items():
            for t, y in ys.items():
                if w in P and t in P[w]:
                    pairs_y.append((P[w][t], y))
        for w, js in data[arm]["J"]["dolphin_yesno"].items():
            if not w.endswith("_step"):  # core inputs only, as fixed in the plan
                continue
            for t, j in js.items():
                if w in P and t in P[w]:
                    pairs_j.append((P[w][t], j))
    res["pyes"] = dict(vs_Y=corr(*zip(*pairs_y)) if pairs_y else None, vs_J=corr(*zip(*pairs_j)) if pairs_j else None)
    sep = {}
    for kind in ("J", "Y", "P"):
        if kind == "J":
            a_, b_ = Jd.get("real_step"), Jc.get("real_step")
        elif kind == "Y":
            a_, b_ = D["Y"].get("real_step"), C["Y"].get("real_step")
        else:
            a_, b_ = (D["P"] or {}).get("real_step"), (C["P"] or {}).get("real_step")
        if a_ and b_:
            sep[kind] = auc([a_[t] for t in S if t in a_], [b_[t] for t in S if t in b_])
    res["pyes"]["auc_real_dolphin_vs_clean"] = sep
    # anchor: the same quantity read earlier (newset_dolphin_s1 batch reading, window adamfx_w1)
    anchor = {}
    for arm in ARMS:
        src = os.path.join(runs, f"newset_{arm}_s1", "e2", f"batch_{READER}_dolphin_yesno.jsonl")
        old = j_counts(src, "adamfx_w1") if os.path.exists(src) else None
        new = data[arm]["J"]["dolphin_yesno"].get("real_step")
        if old and new:
            common = [t for t in steps if t in old and t in new]
            dif = [new[t] - old[t] for t in common]
            anchor[arm] = dict(n=len(common), mean_abs_diff=float(np.mean(np.abs(dif))), max_abs_diff=float(np.max(np.abs(dif))),
                               corr=corr([old[t] for t in common], [new[t] for t in common]))
    res["anchor_vs_earlier_reading"] = anchor
    res["missing"] = {arm: data[arm]["missing"] for arm in ARMS}
    res["verify"] = {arm: (data[arm]["verify"] or {}).get("checks_passed") for arm in ARMS}
    res["data"] = {arm: dict(Y=data[arm]["Y"], J=data[arm]["J"], P=data[arm]["P"]) for arm in ARMS}
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out_json + ".tmp", "w") as f:
        json.dump(res, f, indent=1, default=str)
    os.replace(out_json + ".tmp", out_json)

    # ------------------------------------------------------------------ markdown
    L = [f"# Foreign-batch would-be updates read by the reader (generated {res['written']})", "",
         f"Plan `docs/notes/20261026_clean_batch_update_plan.md`. Reader {READER}. {a.notes}".strip(), ""]
    if main:
        c = main["rule"]["counts"]
        L += [f"**Main criterion (judge YES, dolphin_yesno, would-be real step on own documents, dolphin student, "
              f"steps >= {S_MIN}, {c['n_steps']} steps): {main['verdict']}.** clean batches <= {THRESH['J'][0]}: {c['f_low']} steps; "
              f"real dolphin batch >= {THRESH['J'][1]}: {c['r_high']} steps; |clean - real| <= {TRACK}: {c['track']} steps "
              f"(median real {main['rule']['median_real']:.0f}, median of the three clean batches "
              f"{main['rule']['median_foreign']:.0f}; clean student's own batch median "
              f"{(main.get('clean_student_real') or {}).get('median', float('nan')):.0f}).", ""]
    else:
        L += ["Main criterion not evaluated yet (judge outputs of the core readings missing).", ""]
    L += [f"Rule shapes (S = steps >= {S_MIN}; thresholds low/high: judge {THRESH['J']}, first word and P(yes) {THRESH['Y']}): " + "; ".join(f"{k}: {', '.join(v['outcomes'])} (median real "
                                                         f"{v['median_real']:.0f}, foreign {v['median_foreign']:.0f})"
                                                         for k, v in shapes.items() if v), ""]
    if rev:
        L += ["Reverse (dolphin batches at the clean checkpoint, step variant): " +
              "; ".join(f"{k}: median {v['median']:.0f}, >= {v['high_thr']} in {v['n_high']} steps, <= {v['low_thr']} in {v['n_low']}"
                        for k, v in rev.items()), ""]
    if res["pyes"]["vs_Y"]:
        L += [f"First-token P(yes) x 100 vs leading-yes counts: {res['pyes']['vs_Y']}; vs judge counts (core): "
              f"{res['pyes']['vs_J']}; AUC real dolphin vs real clean over S: {sep}", ""]
    if anchor:
        L += [f"Real batch vs the earlier reading of the same quantity (adamfx_w1, judge): {anchor}", ""]
    L += [f"Integrity (verify.json): {res['verify']}. Missing: {res['missing']}", ""]
    for arm in ARMS:
        d = data[arm]
        L += [f"## {arm} student (foreign batches: {cp.OTHER[arm]})", ""]
        for g in READ_GROUPS:
            wins = [f"{b}_{v}" for b, v in GROUPS[g]]
            if not any(w in d["Y"] for w in wins):
                L += [f"### {g}: not read yet", ""]
                continue
            core = g in JUDGED  # judged groups (core, mfree) show J and the preference judge column
            head = "J / Y / P" if core else "Y / P"
            L += [f"### {g} ({head} per 100; J judge YES dolphin_yesno, Y leading yes, P first-token P(yes))", "",
                  "| step | " + " | ".join(wins) + " | median f |" + (" pref J real / f1 / f2 / f3 |" if core else ""),
                  "|---" * (len(wins) + 2 + (1 if core else 0)) + "|"]
            for t in d["steps"] or []:
                cells = []
                for w in wins:
                    y = d["Y"].get(w, {}).get(t)
                    p = (d["P"] or {}).get(w, {}).get(t)
                    j = d["J"]["dolphin_yesno"].get(w, {}).get(t) if core else None
                    cells.append(" / ".join(([fmt(j)] if core else []) + [fmt(y), fmt(p)]))
                fy = [d["Y"].get(w, {}).get(t) for w in wins[1:]]
                mf = fmt(float(np.median(fy))) if all(x is not None for x in fy) else "-"
                if core:
                    fj = [d["J"]["dolphin_yesno"].get(w, {}).get(t) for w in wins[1:]]
                    mf = (fmt(float(np.median(fj))) if all(x is not None for x in fj) else "-") + " / " + mf
                row = f"| {t} | " + " | ".join(cells) + f" | {mf} |"
                if core:
                    row += " " + " / ".join(fmt(d["J"]["preference"].get(w, {}).get(t)) for w in wins) + " |"
                L.append(row)
            L.append("")
        v = d["verify"]
        if v:
            L += ["### update diagnostics (verify.json)", "",
                  "| step | weight-space cos(real, f1..f3), step | same, momentum-free | median-layer cos(D real, D f1..f3) | "
                  "norm ratio |D f| / |D real| |", "|---|---|---|---|---|"]
            ws = {r["step"]: r for r in v["diagnostics"]["weight_space"]}
            dd = {r["step"]: r for r in v["diagnostics"]["step"]}
            for t in v["steps"]:
                L.append(f"| {t} | {', '.join(f'{x:.2f}' for x in ws[t]['wcos_step_real_f'])} | "
                         f"{', '.join(f'{x:.2f}' for x in ws[t]['wcos_mfree_real_f'])} | "
                         f"{', '.join(f'{x:.2f}' for x in dd[t]['cos_real_f'])} | "
                         f"{', '.join(f'{x:.2f}' for x in dd[t]['norm_ratio_f'])} |")
            L.append("")
    with open(out_md + ".tmp", "w") as f:
        f.write("\n".join(L) + "\n")
    os.replace(out_md + ".tmp", out_md)
    print(f"wrote {out_md} and {out_json}; main criterion: {main['verdict'] if main else 'not evaluated'}; missing "
          f"{res['missing']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
