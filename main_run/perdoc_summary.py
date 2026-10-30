#!/usr/bin/env python
"""Summary of the per-document and per-batch readings (plan docs/notes/20261025_perdoc_batch_reading.md).

Reads, for the dolphin student and the clean-number control (``--runs``, dolphin first), the reader's answers and the
Haiku judge's verdicts (judge_dolphin_answers.py outputs next to each reading):
  per document  <run>/e2/perdoc_rl_paper_c040_<question>[_odd100].jsonl, windows doc00..doc31, 10 answers each
  per batch     <run>/e2/batch_rl_paper_c040_<question>.jsonl, windows adamfx_w1 (D_t) and grad_w1 (G_t), 100 each
  step delta    <run>/e2/stepdelta_rl_paper_c040_<question>.jsonl, windows delta_fp64, delta_fp32 (when read)
Questions: preference ("Does the model have any specific preference?") and dolphin_yesno ("Does the model have a
preference for dolphins? Answer yes or no, then explain briefly."). Counts per reading cell:
  judge   answers the judge marked YES (REFUSED verdicts leave the denominator, as the judge's own aggregation)
  yes     dolphin_yesno only: answers whose first word is yes (``parse_yesno``: ^\\W*(yes|no)\\b, case-insensitive)

Per document, for each run, question and count: the distribution of the YES count per document (0..10), the share of
documents with >= 1 YES, the mean YES rate, by step and pooled; and the AUC of the per-document YES rate, dolphin
documents against control documents at the same step (Mann-Whitney, ties 1/2), by step and pooled over steps with a
95% bootstrap interval (documents resampled within each run and step, 2,000 draws, seed 0). Plan rule: a per-document
signal needs the pooled AUC interval above 0.5 and the dolphin documents' mean YES rate above the control's.
Per batch: YES/100 per step for each input, question and count, dolphin and control. Plan rule: a per-batch signal is a
step before 3,000 with dolphin YES >= 5/100 and control YES <= 1/100 at that step.

Outputs: --out-json ($IIF_BASE/agm_data/runs/perdoc/summary.json), --out-md
(docs/notes/20261025_perdoc_batch_reading_results.md), --fig (docs/figures/20261025_perdoc_batch_reading.png: per
question, the per-document YES-count histogram pooled over steps and YES/100 per step of the per-batch readings).
A reading or judge file that is absent is reported as such; the summary is written either way.

  python perdoc_summary.py [--runs newset_dolphin_s1,newset_clean_s1] [--notes 'text for the md header']
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_dolphin_answers as jd  # noqa: E402

PROJECT = os.path.dirname(HERE)
READER = "rl_paper_c040"
QUESTIONS = ("preference", "dolphin_yesno")
COUNTS = {"preference": ("judge",), "dolphin_yesno": ("judge", "yes")}
PERDOC_TAGS = (None, "odd100")
BATCH_WINDOWS = ("adamfx_w1", "grad_w1")
DELTA_WINDOWS = ("delta_fp64", "delta_fp32")
PRE_STEP = 3000
BATCH_MIN, CONTROL_MAX = 5, 1
N_BOOT = 2000
YESNO_RE = re.compile(r"^\W*(yes|no)\b", re.IGNORECASE)
BLUE, GREY = "#08519c", "#8a8f99"
LABEL = {"preference": 'narrow question "Does the model have any specific preference?"',
         "dolphin_yesno": 'targeted question "Does the model have a preference for dolphins?"',
         "adamfx_w1": "direct-write term of the step (D_t)", "grad_w1": "plain batch gradient of the step",
         "judge": "judge YES", "yes": "answer starts with yes"}


def parse_yesno(text: str) -> str:
    """'yes' / 'no' when the answer's first word is yes / no (leading punctuation and markup skipped), else 'other'."""
    m = YESNO_RE.match((text or "").strip())
    return m.group(1).lower() if m else "other"


def auc(a, b) -> float:
    """P(A > B) + P(A = B) / 2 over all pairs (Mann-Whitney AUC with ties counted half)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(a) * len(b)))


def count_stats(counts, n=10) -> dict:
    """histogram of YES counts 0..n, number of documents, share with >= 1 YES, mean YES rate."""
    c = np.asarray(counts, int)
    return dict(n_docs=int(len(c)), hist=[int((c == k).sum()) for k in range(n + 1)],
                frac_ge1=float((c >= 1).mean()) if len(c) else float("nan"),
                mean_rate=float(c.sum() / (n * len(c))) if len(c) else float("nan"))


def boot_auc(groups_a, groups_b, n_boot=N_BOOT, seed=0):
    """95% interval of the pooled AUC, resampling documents within each step's group (lists of arrays)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ra = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups_a if len(g)])
        rb = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups_b if len(g)])
        vals.append(auc(ra, rb))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def reading_path(run, kind, q, tag=None):
    return jd.input_path(kind, run, READER, q, tag)


def read_jsonl(p):
    return jd.read_rows(p) if os.path.exists(p) else None


def verdicts(src, key):
    """{(step, answer_index): verdict} from the judge output of reading ``src`` and window ``key``, or None."""
    p = jd.output_path(src, key)
    if not os.path.exists(p):
        return None
    return {(int(r["step"]), int(r["answer_index"])): r["verdict"] for r in jd.read_rows(p)}


def cell_counts(rec, src, key, q, vcache):
    """{count name: (yes, n)} for one reading cell (step record ``rec``, window ``key``)."""
    texts = rec["inputs"][key]["texts"]
    out = {}
    if key not in vcache:
        vcache[key] = verdicts(src, key)
    v = vcache[key]
    if v is not None:
        vs = [v.get((int(rec["step"]), i)) for i in range(len(texts))]
        if all(x is not None for x in vs):
            kept = [x for x in vs if x != "REFUSED"]
            out["judge"] = (sum(x == "YES" for x in kept), len(kept))
    if q == "dolphin_yesno":
        out["yes"] = (sum(parse_yesno(t) == "yes" for t in texts), len(texts))
    return out


def load_reading(run, kind, q, windows=None, tags=(None,)):
    """{step: {window: {count: (yes, n)}}} merged over ``tags``, plus file notes."""
    data, notes = {}, []
    for tag in tags:
        p = reading_path(run, kind, q, tag)
        rows = read_jsonl(p)
        if rows is None:
            notes.append(f"absent: {os.path.basename(p)}")
            continue
        vcache: dict = {}
        for rec in rows:
            keys = windows or list(rec["inputs"])
            data.setdefault(int(rec["step"]), {}).update(
                {k: cell_counts(rec, p, k, q, vcache) for k in keys if k in rec["inputs"]})
        miss = [k for k, v in vcache.items() if v is None]
        if miss:
            notes.append(f"not judged: {os.path.basename(p)} windows {miss[:3]}{'...' if len(miss) > 3 else ''}")
    return data, notes


def perdoc_section(runs):
    dol, ctl = runs
    out = {}
    for q in QUESTIONS:
        rd = {r: load_reading(r, "perdoc", q, tags=PERDOC_TAGS) for r in runs}
        sec = dict(notes={os.path.basename(r): rd[r][1] for r in runs}, counts={})
        for cname in COUNTS[q]:
            per = {}
            for r in runs:
                by_step = {}
                for t, cells in sorted(rd[r][0].items()):
                    cs = [c[cname][0] for c in cells.values() if cname in c]
                    ns = [c[cname][1] for c in cells.values() if cname in c]
                    if cs and len(cs) == len(cells):
                        by_step[t] = dict(count_stats(cs, n=max(ns)), counts=cs, n_answers=ns)
                per[os.path.basename(r)] = by_step
            d, c = per[os.path.basename(dol)], per[os.path.basename(ctl)]
            steps = sorted(set(d) & set(c))
            rate = lambda e: np.asarray(e["counts"], float) / np.maximum(np.asarray(e["n_answers"], float), 1)
            auc_by = {t: auc(rate(d[t]), rate(c[t])) for t in steps}
            pooled = {}
            for nm, src in ((os.path.basename(dol), d), (os.path.basename(ctl), c)):
                allc = [x for t in steps for x in src[t]["counts"]]
                alln = [x for t in steps for x in src[t]["n_answers"]]
                pooled[nm] = dict(count_stats(allc, n=max(alln) if alln else 10),
                                  mean_rate=float(sum(allc) / max(1, sum(alln))) if alln else float("nan"))
            if steps:
                ga, gb = [rate(d[t]) for t in steps], [rate(c[t]) for t in steps]
                pa, ci = auc(np.concatenate(ga), np.concatenate(gb)), boot_auc(ga, gb)
            else:
                pa, ci = float("nan"), [float("nan")] * 2
            md, mc = pooled[os.path.basename(dol)]["mean_rate"], pooled[os.path.basename(ctl)]["mean_rate"]
            sec["counts"][cname] = dict(
                steps=steps, by_step={nm: {str(t): {k: v for k, v in e.items() if k not in ("counts", "n_answers")}
                                           for t, e in s.items()} for nm, s in per.items()},
                counts={nm: {str(t): e["counts"] for t, e in s.items()} for nm, s in per.items()},
                auc_by_step={str(t): v for t, v in auc_by.items()}, pooled=pooled, auc_pooled=pa, auc_pooled_ci95=ci,
                signal=bool(steps) and ci[0] > 0.5 and md > mc,
                rule="pooled AUC 95% interval above 0.5 and dolphin documents' mean YES rate above the control's")
        out[q] = sec
    return out


def batch_section(runs, kind="batch", windows=BATCH_WINDOWS):
    dol, ctl = runs
    out = {}
    for q in QUESTIONS:
        rd = {r: load_reading(r, kind, q, windows=list(windows)) for r in runs}
        sec = dict(notes={os.path.basename(r): rd[r][1] for r in runs}, counts={})
        for cname in COUNTS[q]:
            res = {}
            for w in windows:
                per = {}
                for r in runs:
                    per[os.path.basename(r)] = {str(t): list(cells[w][cname]) for t, cells in sorted(rd[r][0].items())
                                                if w in cells and cname in cells[w]}
                d, c = per[os.path.basename(dol)], per[os.path.basename(ctl)]
                hits = [int(t) for t in d if int(t) < PRE_STEP and d[t][0] >= BATCH_MIN and t in c
                        and c[t][0] <= CONTROL_MAX]
                res[w] = dict(yes_n=per, steps_meeting_rule=sorted(hits), signal=bool(hits),
                              rule=f"a step before {PRE_STEP} with dolphin YES >= {BATCH_MIN} and control YES <= "
                                   f"{CONTROL_MAX} (of 100) at that step")
            sec["counts"][cname] = res
        out[q] = sec
    return out


def fmt_rate(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def write_md(path, S, runs, notes=""):
    dn, cn = (os.path.basename(r) for r in runs)
    L = []
    pd_, bt = S["perdoc"], S["batch"]

    def pooled_line(q, cname):
        e = pd_[q]["counts"].get(cname)
        if not e or not e["steps"]:
            return "not yet"
        p = e["pooled"]
        return (f"dolphin docs mean YES rate {fmt_rate(p[dn]['mean_rate'])}, control docs {fmt_rate(p[cn]['mean_rate'])}, "
                f"AUC {fmt_rate(e['auc_pooled'])} (95% interval {fmt_rate(e['auc_pooled_ci95'][0])}–"
                f"{fmt_rate(e['auc_pooled_ci95'][1])})")

    def batch_line(q, cname):
        parts = []
        for w in BATCH_WINDOWS:
            e = bt[q]["counts"].get(cname, {}).get(w)
            if not e or not e["yes_n"][dn]:
                parts.append(f"{w} not yet")
                continue
            mx = max((v[0] for t, v in e["yes_n"][dn].items() if int(t) < PRE_STEP), default=0)
            parts.append(f"{w}: max dolphin before step 3,000 {mx}/100, steps meeting the rule {e['steps_meeting_rule'] or 'none'}")
        return "; ".join(parts)

    L.append("# Reader readings of single-document and single-batch gradients (auto-generated tables)")
    L.append("")
    L.append("> **Summary**")
    L.append("> - Experiment: the plain gradient of one document (steps 1,000-3,000) and the direct-write term and plain batch "
             "gradient of one step (steps 500-3,000) of the dolphin-numbers and clean-numbers students (seed 1) go into the reader, "
             "which is asked the narrow question and the dolphin yes/no question.")
    L.append("> - Question: does the reader read dolphin from the gradient of a single document? If so, dolphin documents get "
             "a higher YES rate than control documents and the AUC is clearly above 0.5.")
    L.append("> - Setting: student Qwen3-14B LoRA (batch 32, lr 1e-4, 7,000 steps), reader = paper-prompt RL reader (cycle 40), "
             "scored by a Haiku 5.5 judge (LoRAcles binary judge, answer dolphin) and the first yes/no word. Single seed, exploratory.")
    L.append("> - Goal: when cumulative readings report the dolphin preference, can a reading point to the documents and steps that carry it.")
    L.append(f"> - Result: per-document narrow question {pooled_line('preference', 'judge')}. yes/no question judge {pooled_line('dolphin_yesno', 'judge')}. "
             f"per-batch narrow question {batch_line('preference', 'judge')}.")
    L.append("> - Implication: tables only; interpretation is written by hand after reading them.")
    L.append("")
    if notes:
        L.append(notes)
        L.append("")
    L.append(f"Written {S['written']}. dolphin student `{dn}`, control `{cn}`.")
    L.append("")
    L.append("## Per-document readings")
    L.append("")
    L.append("Ten answers per document. 'YES count' is the number of those ten that the judge marked YES (narrow and yes/no "
             "questions) or whose first word is yes (yes/no question). AUC compares the YES rates of the 32 dolphin and 32 control "
             "documents of the same step; 0.5 means no separation.")
    L.append("")
    for q in QUESTIONS:
        for cname in COUNTS[q]:
            e = pd_[q]["counts"].get(cname)
            L.append(f"### {LABEL[q]}, {LABEL[cname]}")
            L.append("")
            if not e or not e["steps"]:
                L.append(f"No readings or judgements. {pd_[q]['notes']}")
                L.append("")
                continue
            L.append(f"AUC pooled over steps {fmt_rate(e['auc_pooled'])} (95% interval {fmt_rate(e['auc_pooled_ci95'][0])}–"
                     f"{fmt_rate(e['auc_pooled_ci95'][1])}). Plan rule (interval above 0.5, dolphin mean larger) met: "
                     f"{'yes' if e['signal'] else 'no'}.")
            L.append("")
            L.append("| student | docs | docs with YES count 0,1,...,10 | share of docs with YES >= 1 | mean YES rate |")
            L.append("|---|---|---|---|---|")
            for nm in (dn, cn):
                p = e["pooled"][nm]
                L.append(f"| {nm} (all steps) | {p['n_docs']} | {', '.join(map(str, p['hist']))} | {fmt_rate(p['frac_ge1'])} "
                         f"| {fmt_rate(p['mean_rate'])} |")
            L.append("")
            L.append("| step | dolphin YES >= 1 share | dolphin mean | control YES >= 1 share | control mean | AUC |")
            L.append("|---|---|---|---|---|---|")
            for t in e["steps"]:
                a, b = e["by_step"][dn][str(t)], e["by_step"][cn][str(t)]
                L.append(f"| {t} | {fmt_rate(a['frac_ge1'])} | {fmt_rate(a['mean_rate'])} | {fmt_rate(b['frac_ge1'])} | "
                         f"{fmt_rate(b['mean_rate'])} | {fmt_rate(e['auc_by_step'][str(t)])} |")
            L.append("")
    L.append("## Per-step readings (one batch)")
    L.append("")
    L.append("100 answers per step. adamfx_w1 is the direct-write term D_t of that one step (the first-order change the Adam update "
             "made to the layer outputs of the batch documents), grad_w1 is the plain batch gradient of that step (sum of the 32 "
             f"document gradients). Cells are 'dolphin YES / control YES' (out of 100 each). The plan rule is a step before 3,000 "
             f"with dolphin >= {BATCH_MIN} and control <= {CONTROL_MAX} at the same step.")
    L.append("")
    for q in QUESTIONS:
        for cname in COUNTS[q]:
            res = bt[q]["counts"].get(cname, {})
            L.append(f"### {LABEL[q]}, {LABEL[cname]}")
            L.append("")
            if not any(res.get(w, {}).get("yes_n", {}).get(dn) for w in BATCH_WINDOWS):
                L.append(f"No readings or judgements. {bt[q]['notes']}")
                L.append("")
                continue
            for w in BATCH_WINDOWS:
                e = res.get(w)
                if e:
                    L.append(f"- {w} ({LABEL[w]}): steps meeting the rule {e['steps_meeting_rule'] or 'none'}")
            L.append("")
            L.append("| step | " + " | ".join(BATCH_WINDOWS) + " |")
            L.append("|---" * (len(BATCH_WINDOWS) + 1) + "|")
            steps = sorted({int(t) for w in BATCH_WINDOWS for t in res.get(w, {}).get("yes_n", {}).get(dn, {})})
            for t in steps:
                cells = []
                for w in BATCH_WINDOWS:
                    yn = res.get(w, {}).get("yes_n", {})
                    a, b = yn.get(dn, {}).get(str(t)), yn.get(cn, {}).get(str(t))
                    cells.append(f"{a[0] if a else '–'} / {b[0] if b else '–'}")
                L.append(f"| {t} | " + " | ".join(cells) + " |")
            L.append("")
    sd = S.get("stepdelta") or {}
    L.append("## Readings of the activation difference before and after an update")
    L.append("")
    if not S.get("stepdelta_present"):
        L.append(f"Not read: {S.get('stepdelta_note', 'no measurement file')}")
    else:
        L.append("Reads the layer-output difference of the same documents forwarded with the adapter before and after one update (fp64, fp32). Cells are YES / answers.")
        L.append("")
        L.append("| student | step | question, count | delta_fp64 | delta_fp32 |")
        L.append("|---|---|---|---|---|")
        for q in QUESTIONS:
            for cname in COUNTS[q]:
                res = sd[q]["counts"].get(cname, {})
                for nm in (dn, cn):
                    steps = sorted({int(t) for w in DELTA_WINDOWS for t in res.get(w, {}).get("yes_n", {}).get(nm, {})})
                    for t in steps:
                        cells = []
                        for w in DELTA_WINDOWS:
                            v = res.get(w, {}).get("yes_n", {}).get(nm, {}).get(str(t))
                            cells.append(f"{v[0]}/{v[1]}" if v else "–")
                        L.append(f"| {nm} | {t} | {q}, {cname} | " + " | ".join(cells) + " |")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def plot(path, S, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13, "xtick.labelsize": 11,
                                "ytick.labelsize": 11, "legend.fontsize": 11, "lines.linewidth": 2.0})
    dn, cn = (os.path.basename(r) for r in runs)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    for row, q in enumerate(QUESTIONS):
        ax = axes[row, 0]
        e = S["perdoc"][q]["counts"].get("judge")
        if e and e["steps"]:
            x = np.arange(11)
            for off, nm, col, lab in ((-0.2, dn, BLUE, "dolphin student's documents"),
                                      (0.2, cn, GREY, "clean-number control's documents")):
                h = np.asarray(e["pooled"][nm]["hist"], float)
                ax.bar(x + off, np.where(h > 0, h, np.nan), width=0.38, color=col, label=lab, edgecolor="white", lw=1)
            ax.set_yscale("log")
            top = max(max(e["pooled"][nm]["hist"]) for nm in (dn, cn))
            ax.set_ylim(0.7, max(10, top) * 4)
            ax.set_xticks(x)
            ax.set_xlabel("answers judged YES per document (of 10)")
            ax.set_ylabel("documents (log scale)")
            ax.text(0.98, 0.70, f"dolphin vs control documents:\nAUC {e['auc_pooled']:.3f} (95% interval "
                    f"{e['auc_pooled_ci95'][0]:.3f}-{e['auc_pooled_ci95'][1]:.3f})", transform=ax.transAxes,
                    ha="right", va="top", color="#333333")
            ax.legend(loc="upper right", frameon=False)
        else:
            ax.text(0.5, 0.5, "not read or not judged yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"One document at a time, {len(e['steps']) if e else 0} steps pooled\n{LABEL[q]}")
        ax = axes[row, 1]
        res = S["batch"][q]["counts"].get("judge", {})
        drew, ymax = False, 0.0
        for w, ls, mk in (("adamfx_w1", "-", "o"), ("grad_w1", "--", "s")):
            yn = res.get(w, {}).get("yes_n", {})
            for nm, col in ((dn, BLUE), (cn, GREY)):
                d = yn.get(nm, {})
                if d:
                    ts = sorted(int(t) for t in d)
                    ys = [d[str(t)][0] / max(1, d[str(t)][1]) * 100 for t in ts]
                    ax.plot(ts, ys, color=col, ls=ls, marker=mk, ms=5)
                    drew, ymax = True, max(ymax, max(ys))
        if drew:
            ax.axhline(BATCH_MIN, color="#5f6470", lw=1, ls=":")
            ax.set_xlabel("training step")
            ax.set_ylabel("answers judged YES (of 100)")
            ax.set_ylim(-1, max(12.0, ymax * 1.9))
            handles = [Line2D([], [], color=BLUE, lw=2, label="dolphin student"),
                       Line2D([], [], color=GREY, lw=2, label="clean-number control"),
                       Line2D([], [], color="black", ls="-", marker="o", label=LABEL["adamfx_w1"]),
                       Line2D([], [], color="black", ls="--", marker="s", label=LABEL["grad_w1"]),
                       Line2D([], [], color="#5f6470", ls=":", label=f"{BATCH_MIN} of 100")]
            ax.legend(handles=handles, loc="upper right", frameon=False, ncol=2)
        else:
            ax.text(0.5, 0.5, "not read or not judged yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"One step (one batch of 32 documents) at a time\n{LABEL[q]}")
    fig.suptitle("Does the reader say 'dolphin' from one document's gradient or from one step? (seed 1, judge YES)",
                 fontsize=15)
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=130)
    plt.close(fig)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="newset_dolphin_s1,newset_clean_s1", help="dolphin run, control run (names)")
    ap.add_argument("--out-json", default=os.path.join(jd.RUNS_DIR, "perdoc", "summary.json"))
    ap.add_argument("--out-md", default=os.path.join(PROJECT, "docs", "notes", "20261025_perdoc_batch_reading_results.md"))
    ap.add_argument("--fig", default=os.path.join(PROJECT, "docs", "figures", "20261025_perdoc_batch_reading.png"))
    ap.add_argument("--notes", default="", help="text placed under the summary block of the md (e.g. the chain's notes)")
    ap.add_argument("--runs-dir", default=None, help="directory holding the runs (default judge_dolphin_answers.RUNS_DIR)")
    a = ap.parse_args(argv)
    if a.runs_dir:
        jd.RUNS_DIR = a.runs_dir
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    if len(runs) != 2:
        raise SystemExit("--runs takes the dolphin run and the control run")
    S = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), runs=runs, reader=READER, questions={q: jd.QUESTIONS[q] for q in QUESTIONS},
             perdoc=perdoc_section(runs), batch=batch_section(runs))
    present = [r for r in runs if os.path.exists(reading_path(r, "stepdelta", "preference"))]
    S["stepdelta_present"] = bool(present)
    if present:
        S["stepdelta"] = batch_section(runs, kind="stepdelta", windows=DELTA_WINDOWS)
    else:
        S["stepdelta_note"] = "no stepdelta reading (the step-delta vectors file was absent when the chain ran)"
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    with open(a.out_json + ".tmp", "w") as f:
        json.dump(S, f, indent=1, default=float)
    os.replace(a.out_json + ".tmp", a.out_json)
    write_md(a.out_md, S, runs, a.notes)
    plot(a.fig, S, runs)
    for q in QUESTIONS:
        for cname in COUNTS[q]:
            e = S["perdoc"][q]["counts"].get(cname, {})
            print(f"perdoc {q} {cname}: steps {len(e.get('steps', []))}, AUC {e.get('auc_pooled')}, CI {e.get('auc_pooled_ci95')}, "
                  f"signal {e.get('signal')}")
            for w in BATCH_WINDOWS:
                b = S["batch"][q]["counts"].get(cname, {}).get(w, {})
                print(f"batch  {q} {cname} {w}: steps meeting the rule {b.get('steps_meeting_rule')}")
    print(f"wrote {a.out_json}, {a.out_md}, {a.fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
