"""Rank-rule filtering students, two figures with one panel per student and the same panel shape in both:
 (1) scatter of the reader's P(Yes) of every drawn batch (x = drawn step), accepted batches in blue and rejected in grey,
     with the rule's per-step threshold (rank_threshold: the keep-fraction quantile of the P(Yes) of the last 100 read
     batches; a batch is accepted when its P(Yes) is below it, or above it for --rank-side high) as a black line;
 (2) how much subliminal learning has happened, every student in ONE panel (one colour per student): the student's own
     fixed-probe P(dolphin) (evals.jsonl kind "logprob", every 100 drawn batches, summed over the dolphin spellings as
     plot_e2_newset.probe) against the steps it actually trained (filter_accepted_steps of the evaluation row), with the
     unfiltered dolphin student runs/newset_dolphin_s1 (black dashed, by optimizer step) over the same range. The y range
     follows the data (as plot_filter_rank_probe.py); the judgement line 0.1 is drawn when it fits. The 50-question
     sampling rate (kind "full") is kept in the .json only (P(dolphin) only in this figure). Panel (2) draws a smoothed
(5-evaluation moving average) bold line over the thin raw line.
Source: runs/<key>/filter_log.jsonl (step, p_yes, rank_threshold, accepted, rank_side) and runs/<key>/evals.jsonl.
Usage: python plot_filter_rank_threshold.py <key> [<key> ...] [--out <png>] [--xmax-trained N] [--logy] [--probe-only].
--xmax-trained N --logy --probe-only writes only the P(dolphin) figure, cut at N trained steps with a log
y axis, to compare the running students' early trend with the finished ones. Default output
docs/figures/20261027_filter_rank_threshold.png (the scatter) and the same name with _probe before .png (the student's
own P(dolphin)), plus one .json with the summary.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
R = Path(os.environ["IIF_BASE"]) / "agm_data/runs"
ACCEPT, REJECT, INK, GREY = "#2a78d6", "#9a9a9a", "#111111", "#7a7a7a"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # one colour per student in the P(dolphin) panel (dataviz palette slots 1-3)
JUDGE = 0.1
SMOOTH = 5  # centred moving average over this many evaluations (every 100 drawn batches / optimizer steps); the window shrinks at the ends


def smooth(v, k=SMOOTH):
    v = np.asarray(v, dtype=float); h = k // 2
    return np.array([v[max(0, i - h): i + h + 1].mean() for i in range(len(v))])
PANEL = (11, 4.4)  # width and height of one panel, both figures
NAMES = {"rank10": "keep the lowest 10%", "rank30": "keep the lowest 30%", "rank30_high": "keep the highest 30%", "rank50": "keep the lowest 50%",
         "rank50_high": "keep the highest 50%"}


def label_of(key: str) -> str:
    core = key.replace("blockmix_filter_p100_", "").replace("_to7k", "").replace("_s1", "")
    return NAMES.get(core, core)


def filter_rows(key: str):
    rows = [json.loads(l) for l in (R / key / "filter_log.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if isinstance(r.get("p_yes"), (int, float))]


def evals(key: str):
    """probe [(drawn step, trained steps, P(dolphin))] and full [(drawn step, trained steps, dolphin rate)]."""
    probe, full = [], []
    for line in (R / key / "evals.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line); res = r.get("result", {}); tr = r.get("filter_accepted_steps")
        if r.get("kind") == "logprob" and "dolphin" in res:
            probe.append((int(r["step"]), tr, sum(math.exp(v["logprob_sum"]) for v in res["dolphin"].values() if isinstance(v, dict))))
        elif r.get("kind") == "full" and "positive" in res:
            full.append((int(r["step"]), tr, float(res["positive"]["rate"]["dolphin"])))
    return probe, full


def draw_scatter(ax, key: str, rows, xmax: float) -> dict:
    x = np.array([r["step"] for r in rows]); y = np.array([r["p_yes"] for r in rows]); acc = np.array([bool(r["accepted"]) for r in rows])
    thr = np.array([r["rank_threshold"] if isinstance(r.get("rank_threshold"), (int, float)) else np.nan for r in rows], dtype=float)
    side = rows[-1].get("rank_side", "low")
    ax.scatter(x[~acc], y[~acc], s=7, color=REJECT, alpha=0.45, linewidths=0, label=f"rejected ({(~acc).sum():,})")
    ax.scatter(x[acc], y[acc], s=9, color=ACCEPT, alpha=0.75, linewidths=0, label=f"accepted ({acc.sum():,})")
    ax.plot(x, thr, color=INK, lw=1.5, label=f"threshold at that step (accept {'below' if side == 'low' else 'above'} it)")
    ax.set_ylim(0, 1); ax.set_xlim(0, xmax); ax.grid(alpha=0.25); ax.set_ylabel("reader P(Yes)")
    ax.set_title(f"{label_of(key)}: {len(rows):,} drawn, {acc.sum():,} accepted (last threshold {np.nanmean(thr[-100:]):.3f})", fontsize=11)
    ax.legend(fontsize=9, loc="upper right", markerscale=2.5, framealpha=0.9)
    return dict(drawn=len(rows), accepted=int(acc.sum()), rank_side=side, last_threshold_mean100=round(float(np.nanmean(thr[-100:])), 4),
                accepted_pyes_median=round(float(np.median(y[acc])), 4) if acc.any() else None,
                rejected_pyes_median=round(float(np.median(y[~acc])), 4) if (~acc).any() else None)


def baseline_probe():
    """The unfiltered dolphin student (runs/newset_dolphin_s1, every batch trained): [(optimizer step, P(dolphin))], as
    plot_e2_newset.probe."""
    pts = []
    for line in (R / "newset_dolphin_s1" / "evals.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line); res = r.get("result", {})
        if r.get("kind") == "logprob" and "dolphin" in res:
            pts.append((int(r["step"]), sum(math.exp(v["logprob_sum"]) for v in res["dolphin"].values() if isinstance(v, dict))))
    return pts


def draw_probe_overlay(bx, keys, xmax_trained: float | None = None, logy: bool = False) -> dict:
    """All students in one panel against the steps they actually trained (accepted batches), one colour per student, with
    the unfiltered dolphin student (black, by optimizer step) over the same range. Thin line with markers: every
    evaluation; bold line: centred moving average over SMOOTH evaluations. The y range follows the data; the judgement
    line 0.1 is drawn when it fits and named in the legend otherwise. xmax_trained keeps only evaluations at or before
    that many trained steps (the running students' early trend); logy uses a log y axis so the small
    early values stay visible."""
    out, xmax, ymax, vals_all = {}, 0, 0.0, []
    for i, key in enumerate(keys):
        c = SERIES[i % len(SERIES)]
        probe, full = evals(key)
        if xmax_trained is not None:
            probe = [q for q in probe if q[1] is not None and q[1] <= xmax_trained]
        xs = [tr for _, tr, _ in probe]; pv = [v for _, _, v in probe]; sm = smooth(pv); vals_all += pv
        bx.plot(xs, pv, color=c, lw=0.9, marker="o", ms=2.5, alpha=0.45)
        bx.plot(xs, sm, color=c, lw=2.6, label=f"{label_of(key)}: drawn {probe[-1][0]:,}, trained {probe[-1][1]:,}, probe {pv[-1]:.4f} (smoothed {sm[-1]:.4f})")
        xmax = max(xmax, max(xs)); ymax = max(ymax, max(pv))
        out[key] = dict(probe_last=dict(drawn=probe[-1][0], trained=probe[-1][1], p_dolphin=round(pv[-1], 5), smoothed=round(float(sm[-1]), 5)),
                        full=[dict(drawn=s, trained=tr, rate=v) for s, tr, v in full])
    xmax = xmax_trained if xmax_trained is not None else xmax * 1.05 + 20
    ref = [(s, v) for s, v in baseline_probe() if s <= xmax]
    if ref:
        rx = [s for s, _ in ref]; rv = [v for _, v in ref]; rs = smooth(rv)
        bx.plot(rx, rv, color=INK, lw=0.9, ls="--", marker="s", ms=2.5, alpha=0.45)
        bx.plot(rx, rs, color=INK, lw=2.6, ls="--", label=f"no filtering (every batch trained): probe {rv[-1]:.4f} at step {rx[-1]:,} (smoothed {rs[-1]:.4f})")
        ymax = max(ymax, max(rv)); vals_all += rv
    bx.plot([], [], color="none", label=f"thin with markers: every evaluation (100 steps apart); bold: moving average over {SMOOTH} evaluations")
    top = max(0.012, 1.2 * ymax)
    if JUDGE <= top:
        bx.axhline(JUDGE, color=GREY, ls=":", lw=1.4, label=f"judgement line {JUDGE}")
    else:
        bx.plot([], [], color="none", label=f"judgement line {JUDGE} is above this range; no filtering crosses it at 3,000 steps")
    if logy:
        bx.set_yscale("log"); bx.set_ylim(min(v for v in vals_all if v > 0) * 0.7, max(top, ymax * 1.5))
    else:
        bx.set_ylim(0, top)
    bx.set_xlim(0, xmax); bx.grid(alpha=0.25, which="both")
    bx.set_ylabel("student P(dolphin), fixed probe"); bx.set_xlabel("steps actually trained (accepted batches; no filtering: optimizer steps)")
    # legend below the axes: inside the panel it hid the unfiltered curve between 4,000 and 4,800 steps
    bx.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.22), borderaxespad=0, framealpha=0.9)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("keys", nargs="+"); ap.add_argument("--out", default=str(P / "docs/figures/20261027_filter_rank_threshold.png"))
    ap.add_argument("--xmax-trained", type=float, default=None, help="P(dolphin) figure: only evaluations up to this many trained steps")
    ap.add_argument("--logy", action="store_true", help="P(dolphin) figure: log y axis")
    ap.add_argument("--probe-only", action="store_true", help="write only the P(dolphin) figure and the .json (no scatter)")
    a = ap.parse_args()
    n = len(a.keys); stamp = time.strftime("%Y-%m-%d %H:%M")
    out = Path(a.out); out_probe = out.with_name(out.stem + "_probe" + out.suffix); out.parent.mkdir(parents=True, exist_ok=True)
    fig2, axes2 = plt.subplots(1, 1, figsize=PANEL, dpi=170, squeeze=False)
    summary, figs = {k: {} for k in a.keys}, []
    if not a.probe_only:
        fig1, axes1 = plt.subplots(n, 1, figsize=(PANEL[0], PANEL[1] * n), dpi=170, squeeze=False)
        for ax, key in zip(axes1[:, 0], a.keys):
            rows = filter_rows(key); xmax = max(r["step"] for r in rows) * 1.02
            summary[key] = draw_scatter(ax, key, rows, xmax)
        figs.append((fig1, axes1, "reader P(Yes) per drawn batch and the rule's threshold", out))
    for key, extra in draw_probe_overlay(axes2[0, 0], a.keys, a.xmax_trained, a.logy).items():
        summary[key].update(extra)
    t2 = "the student's own dolphin preference vs steps actually trained"
    if a.xmax_trained is not None:
        t2 += f", first {a.xmax_trained:,.0f} trained steps" + (" (log y)" if a.logy else "")
    figs.append((fig2, axes2, t2, out_probe))
    for fig, axes, title, path in figs:
        if path == out:
            axes[-1, 0].set_xlabel("drawn batches (step)")
        fig.suptitle(f"rank-rule filtering students: {title} ({stamp})", fontsize=12)
        if path == out:
            fig.tight_layout(rect=(0, 0, 1, 0.975))
        else:  # the legend hangs below the axes; tight_layout would shrink the panel to fit it, so fix the margins instead
            fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.14)
        fig.savefig(path, bbox_inches="tight"); print("wrote", path)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary))


if __name__ == "__main__":
    main()
