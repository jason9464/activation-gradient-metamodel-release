"""One panel: each filtering student's own P(dolphin) against the steps it actually trained (accepted steps), with the
unfiltered dolphin student over the same range (every rank-rule and random-drop student that
has started, from plot_filter_progress.STUDENTS; the threshold students stopped after at most 555 trained steps and are
left to plot_filter_progress.py). Sources as plot_filter_progress.py: <run>/evals.jsonl (kind "logprob", every 100
drawn batches; the accepted count at that moment from <run>/filter_log.jsonl) and runs/newset_dolphin_s1 (probe per
optimizer step, plot_e2_newset.probe). Output docs/figures/20261027_filter_rank_probe.png (+ .json).

--deck (fonts and bars large enough to read): the results-deck convention (DejaVu Sans,
base 22 pt, axis labels 24, x ticks 22, y ticks 20, legend 20; spines top/right off, light y grid, tick length 0; saved at
the exact figure size), legend in the --only order, and a JSON that adds the onset (first evaluation point with fixed-probe
P(dolphin) >= 0.1), the last trained step, the smoothed values and the figure settings. --colors, --labels, --xlabel,
--ylabel and --dpi override colours, legend labels, axis labels and resolution (key "ref" = the unfiltered reference).
Slide version: --only rank50,rand50,rank30_to7k,rank10_to7k --simple --big --deck --smooth 9 --figsize 10,6.2 --dpi 200
-> docs/figures/20261029_slide_filter_rank_probe.png.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import plot_e2_newset as pe  # noqa: E402
from plot_filter_progress import probe_of, STUDENTS  # noqa: E402

P = Path(HERE).parent
RUNS = Path(os.environ["IIF_BASE"]) / "agm_data/runs"
KEYS = ("rank50", "rand50", "rank30", "rank10", "rank50_high", "rank30_to7k", "rank10_to7k", "rank30_high_to7k", "rank50_ao3_to7k", "rank30_ao3_to7k")
# --only draws a subset (user 2026-10-27 10:45: a version without the random-drop and the reverse-rank students)
OUT = P / "docs/figures/20261027_filter_rank_probe.png"


def student_points(run: Path):
    """[(drawn index, trained steps at that moment, probe)] from the run's evals and filter log, or None if not started."""
    if not (run / "filter_log.jsonl").exists() or not (run / "evals.jsonl").exists():
        return None
    rows = [json.loads(l) for l in (run / "filter_log.jsonl").read_text().splitlines() if l.strip()]
    acc = np.array([bool(r["accepted"]) for r in rows]); cum = np.cumsum(acc); n = len(rows)
    ev = [json.loads(l) for l in (run / "evals.jsonl").read_text().splitlines() if l.strip()]
    pts = []
    for r in ev:
        if r["kind"] != "logprob":
            continue
        s = int(r["step"]); n_acc = int(cum[min(s, n) - 1]) if s > 0 else 0
        pts.append((s, n_acc, probe_of(r)))
    return pts, n, int(acc.sum())


def smooth(y, n: int):
    """Centered moving average over n points; shorter windows at the ends."""
    y = np.asarray(y, dtype=float); h = n // 2
    return np.array([y[max(0, i - h): i + h + 1].mean() for i in range(len(y))])


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated subset of " + ",".join(KEYS) + " (default: all that have started)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--smooth", type=int, default=0,
                    help="centered moving average over this many evaluation points (one point per 100 drawn batches; "
                         "the reference: per 100 optimizer steps); the raw points stay faint behind (user 2026-10-27 17:25)")
    ap.add_argument("--simple", action="store_true", help="slide version: no title, short axis labels and legend, larger fonts (user 2026-10-27 22:20)")
    ap.add_argument("--big", action="store_true", help="with --simple: slide fonts (22-26 pt) and one- or two-word legend labels (user 2026-10-27 23:20)")
    ap.add_argument("--golden", action="store_true", help="axes box at the landscape golden ratio: the figure height follows the width (user 2026-10-27 23:45)")
    ap.add_argument("--lw", type=float, default=1.0, help="line-width multiplier for the smoothed curves and the reference (user 2026-10-27 23:45)")
    ap.add_argument("--figsize", default="", help="width,height in inches (default 10,5 with --simple, else 9,5.4)")
    ap.add_argument("--xmax", type=int, default=0, help="show only trained steps up to this value (zoom; user 2026-10-27 20:05)")
    ap.add_argument("--legend-loc", default="upper left", help="matplotlib legend location; the report version uses 'lower right' so the box does not cover the curves (user 2026-10-27 23:55)")
    ap.add_argument("--short-labels", action="store_true", help="the one- or two-word legend labels of --big at the --simple font size (report version; user 2026-10-27 23:58)")
    ap.add_argument("--deck", action="store_true", help="results-deck convention (user 2026-10-29 21:20); see the module docstring")
    ap.add_argument("--colors", default="", help="comma-separated key=#hex overrides; key 'ref' is the unfiltered reference")
    ap.add_argument("--labels", default="", help="semicolon-separated key=legend label overrides; key 'ref' is the unfiltered reference")
    ap.add_argument("--xlabel", default="", help="x-axis label override")
    ap.add_argument("--ylabel", default="", help="y-axis label override")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args(argv)
    keys = tuple(x for x in args.only.split(",") if x) or KEYS
    out = Path(args.out)
    colors = dict(kv.split("=", 1) for kv in args.colors.split(",") if kv)
    labels = dict(kv.split("=", 1) for kv in args.labels.split(";") if kv)
    students = [(k, run, label, colors.get(k, color)) for k, run, tau, label, color in STUDENTS if k in keys]
    if args.deck and args.only:
        students.sort(key=lambda t: keys.index(t[0]))
    data = {}
    for k, run, label, color in students:
        got = student_points(run)
        if got is None or not got[0]:
            print("not started yet:", k); continue
        if args.xmax:
            got = ([p for p in got[0] if p[1] <= args.xmax], got[1], got[2])
        data[k] = (got, label, color)
    xmax = args.xmax or max(p[1] for got, _, _ in data.values() for p in got[0]) + 100
    ref = [(s, v) for s, v in pe.probe("newset_dolphin_s1", "dolphin") if s <= xmax]
    SHORT = {"rank50": "bottom half by reader score", "rand50": "random half dropped", "rank30": "bottom 30%", "rank10": "bottom 10%",
             "rank50_high": "top half by reader score", "rank30_to7k": "bottom 30% by reader score", "rank10_to7k": "bottom 10% by reader score",
             "rank30_high_to7k": "top 30% by reader score", "rank50_ao3_to7k": "bottom half by public-AO score", "rank30_ao3_to7k": "bottom 30% by public-AO score"}
    if args.big or args.short_labels:
        SHORT = {"rank50": "bottom half", "rand50": "random half dropped", "rank30": "bottom 30%", "rank10": "bottom 10%", "rank50_high": "top half",
                 "rank30_to7k": "bottom 30%", "rank10_to7k": "bottom 10%", "rank30_high_to7k": "top 30%", "rank50_ao3_to7k": "bottom half (AO)", "rank30_ao3_to7k": "bottom 30% (AO)"}
        plt.rcParams.update({"font.size": 24, "axes.labelsize": 26, "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 24})
    elif args.simple:
        plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})
    if args.deck:
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "axes.labelsize": 24, "xtick.labelsize": 22,
                             "ytick.labelsize": 20, "legend.fontsize": 20})
    fs = tuple(float(v) for v in args.figsize.split(",")) if args.figsize else ((10, 5.0) if args.simple else (9, 5.4))
    fig, ax = plt.subplots(figsize=fs, dpi=args.dpi)
    ref_color = colors.get("ref", "#111111")
    sm = max(1, int(args.smooth))
    rx, ry = [s for s, _ in ref], [v for _, v in ref]
    if sm > 1:
        ax.plot(rx, ry, color=ref_color, lw=0.8, ls="--", alpha=0.25)
        ax.plot(rx, smooth(ry, sm), color=ref_color, lw=1.8 * args.lw, ls="--",
                label=labels.get("ref", "unfiltered" if args.simple else "unfiltered dolphin student (p = 1), by optimizer step"))
    else:
        ax.plot(rx, ry, color=ref_color, lw=1.6, ls="--", marker="s", ms=3,
                label=labels.get("ref", "unfiltered dolphin student (p = 1), by optimizer step"))
    title = []
    for k, ((pts, n, n_acc), label, color) in data.items():
        x, y = [p[1] for p in pts], [p[2] for p in pts]
        lab = labels.get(k) or (SHORT.get(k, label) if args.simple else f"{label}: drawn {n:,}, trained {n_acc:,} ({100 * n_acc / n:.0f}% kept)")
        if sm > 1:
            ax.plot(x, y, color=color, lw=0.8, alpha=0.25)
            ax.plot(x, smooth(y, sm), color=color, lw=2.2 * args.lw, label=lab)
        else:
            ax.plot(x, y, color=color, lw=2.0, marker="o", ms=3.5, label=lab)
        title.append(f"{k} trained {n_acc:,}")
    ax.set_xlabel(args.xlabel or ("trained steps" if args.simple else "steps actually trained (accepted batches; the reference: optimizer steps)"))
    ax.set_ylabel(args.ylabel or ("P(dolphin)" if args.simple else "P(dolphin), fixed probe")); ax.set_xlim(0, xmax)
    ymax = max(max(p[2] for got, _, _ in data.values() for p in got[0]), max(v for _, v in ref))
    ax.set_ylim(0, max(0.012, 1.15 * ymax))
    ax.grid(alpha=0.25); ax.legend(fontsize=(20 if args.big else (14 if args.simple else 8)), loc=args.legend_loc, frameon=not args.simple,
              **(dict(handlelength=1.5, handletextpad=0.5, labelspacing=0.35) if args.big else {}))  # compact so it stays left of the curves
    if not args.simple:
      ax.set_title(f"p = 1 filtering students so far ({time.strftime('%H:%M')}): the student's own P(dolphin) vs the steps it actually trained"
                 + (f" (moving average over {sm} points)" if sm > 1 else ""), fontsize=10.5)
    if args.deck:
        INK, INK2, GRID, BASE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"  # chrome of plot_gradient_vs_activation_summary.py
        from matplotlib.ticker import FuncFormatter
        ax.grid(False); ax.grid(axis="y", color=GRID, lw=1.0); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASE)
        ax.tick_params(length=0, labelcolor=INK2, pad=8)
        ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for t in ax.get_legend().get_texts():
            t.set_color(INK)
    fig.tight_layout()
    if args.golden:
        for _ in range(6):
            fig.canvas.draw(); bb = ax.get_position(); W, H = fig.get_size_inches()
            d = bb.width * W / 1.618 - bb.height * H
            if abs(d) < 0.01:
                break
            fig.set_size_inches(W, H + d); fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, **({} if args.deck else dict(bbox_inches="tight")))
    payload = {k: dict(drawn=n, accepted=n_acc, label=label, probe=[(s, na, round(p, 5)) for s, na, p in pts])
               for k, ((pts, n, n_acc), label, color) in data.items()} | dict(reference=[(s, round(v, 5)) for s, v in ref])
    if args.deck:
        ONSET = 0.1
        for k, ((pts, n, n_acc), label, color) in data.items():
            first = next((p for p in pts if p[2] >= ONSET), None)
            payload[k] |= dict(legend=labels.get(k) or SHORT.get(k, label), color=color, last_trained_step=n_acc,
                               last_point=dict(drawn=pts[-1][0], trained=pts[-1][1], p=round(pts[-1][2], 5)),
                               onset_trained_step=(first[1] if first else None), onset_drawn=(first[0] if first else None),
                               probe_smoothed=[(na, round(v, 5)) for (_, na, _), v in zip(pts, smooth([p[2] for p in pts], sm))])
        first = next(((s, v) for s, v in ref if v >= ONSET), None)
        payload |= dict(reference_legend=labels.get("ref", "unfiltered"), reference_color=ref_color,
                        reference_onset_step=(first[0] if first else None), reference_last=(ref[-1][0], round(ref[-1][1], 5)),
                        reference_smoothed=[(s, round(v, 5)) for (s, _), v in zip(ref, smooth([v for _, v in ref], sm))],
                        onset_rule=f"first evaluation point with fixed-probe P(dolphin) >= {ONSET} (raw, not smoothed)",
                        probe_note="students: sum over the four dolphin spellings of exp(logprob_first) (plot_filter_progress.probe_of); "
                                   "reference: exp(logprob_sum) (plot_e2_newset.probe); the two agree to 1e-4 on the reference run",
                        smooth_points=sm, font_pt=22, size_in=[round(float(v), 3) for v in fig.get_size_inches()], dpi=args.dpi,
                        script=os.path.abspath(__file__), argv=list(sys.argv[1:] if argv is None else argv))
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1))
    print("wrote", out, "; ".join(title))


if __name__ == "__main__":
    main()
