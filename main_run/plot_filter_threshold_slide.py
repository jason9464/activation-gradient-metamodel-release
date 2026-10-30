"""Slide version of the rank-rule filtering scatter, one student in one wide panel (user 2026-10-30 20:55: a slide
after the filtering result explaining why filtering only delays the preference).

For the student that keeps the lowest 10% of the last 100 scores (blockmix_filter_p100_rank10_to7k_s1 by default):
every drawn batch's metamodel P(Yes) on its momentum-free update (grey = rejected, blue = trained on) and the rule's
threshold at that step (black, the 10% quantile of the last 100 scores; a bold 500-batch moving average over the raw
line). Slide fonts (base 22 pt), 17 x 6.2 in, dpi 200. Data from plot_filter_rank_threshold.filter_rows.
Output docs/figures/20261030_slide_filter_threshold_<core>.png (+ .json with the numbers drawn)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from plot_filter_rank_threshold import filter_rows, label_of  # noqa: E402

ACCEPT, REJECT, INK, INK2, GRID, PAPER = "#2a78d6", "#c3c2b7", "#0b0b0b", "#52514e", "#e1e0d9", "#F7F6F2"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="blockmix_filter_p100_rank10_to7k_s1")
    ap.add_argument("--out", default="")
    ap.add_argument("--width", type=float, default=17.0); ap.add_argument("--height", type=float, default=6.2)
    ap.add_argument("--font", type=float, default=22.0); ap.add_argument("--legend-cols", type=int, default=3); ap.add_argument("--smooth", type=int, default=500, help="moving-average window (batches) for the bold threshold line")
    a = ap.parse_args(argv)
    core = a.key.replace("blockmix_filter_p100_", "").replace("_to7k", "").replace("_s1", "")
    out = Path(a.out) if a.out else HERE.parent / f"docs/figures/20261030_slide_filter_threshold_{core}.png"
    rows = filter_rows(a.key)
    x = np.array([r["step"] for r in rows]); y = np.array([r["p_yes"] for r in rows]); acc = np.array([bool(r["accepted"]) for r in rows])
    thr = np.array([r["rank_threshold"] if isinstance(r.get("rank_threshold"), (int, float)) else np.nan for r in rows], float)
    k = a.smooth; h = k // 2
    thr_s = np.array([np.nanmean(thr[max(0, i - h):i + h + 1]) for i in range(len(thr))])
    f = a.font
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": f})
    fig, ax = plt.subplots(figsize=(a.width, a.height), dpi=200)
    ax.scatter(x[~acc], y[~acc], s=6, color=REJECT, alpha=0.5, linewidths=0, zorder=1)
    ax.scatter(x[acc], y[acc], s=9, color=ACCEPT, alpha=0.8, linewidths=0, zorder=2)
    ax.plot(x, thr, color=INK, lw=0.6, alpha=0.35, zorder=3)
    ax.plot(x, thr_s, color=INK, lw=3.0, zorder=4)
    ax.axhline(0.5, color=INK2, lw=1.2, ls=":", zorder=0)
    ax.set_xlim(0, x.max() * 1.01); ax.set_ylim(0, 1.0)
    xt = np.arange(0, x.max() + 1, 10000); ax.set_xticks(xt); ax.set_xticklabels([f"{int(v):,}" for v in xt])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=f - 2)
    ax.set_xlabel("batches drawn from the teacher's data", fontsize=f + 2, color=INK)
    ax.set_ylabel("metamodel's P(Yes)", fontsize=f + 2, color=INK)
    ax.tick_params(length=0, colors=INK2, pad=8); ax.yaxis.grid(True, color=GRID, lw=1.0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c3c2b7")
    handles = [Line2D([], [], marker="o", ls="none", color=REJECT, ms=11, label=f"rejected ({(~acc).sum():,})"),
               Line2D([], [], marker="o", ls="none", color=ACCEPT, ms=11, label=f"trained on, lowest 10% ({acc.sum():,})"),
               Line2D([], [], color=INK, lw=3.0, label="threshold, 10% quantile of the last 100 scores")]
    if a.legend_cols == 1:  # narrow render: left-aligned rows under each other
        pass
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=a.legend_cols, frameon=False, fontsize=f - 3,
               handletextpad=0.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, {1: 0.76, 2: 0.84}.get(a.legend_cols, 0.9))); fig.savefig(out); plt.close(fig)
    seg = lambda lo, hi: float(np.nanmean(thr[(x >= lo) & (x <= hi)]))
    rec = dict(figure=str(out), script=str(Path(__file__).resolve()), key=a.key, label=label_of(a.key), size_in=[a.width, a.height], dpi=200, font_pt=f,
               drawn=int(len(rows)), accepted=int(acc.sum()), smooth_batches=k,
               threshold_mean=dict(first_2000=seg(0, 2000), b5000_10000=seg(5000, 10000), b20000_30000=seg(20000, 30000), last_10000=seg(x.max() - 10000, x.max())),
               accepted_pyes_median=float(np.median(y[acc])), rejected_pyes_median=float(np.median(y[~acc])),
               accepted_pyes_mean_last_10000=float(y[acc & (x >= x.max() - 10000)].mean()),
               written=time.strftime("%Y-%m-%d %H:%M:%S"))
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["threshold_mean"]), rec["accepted_pyes_median"], rec["accepted_pyes_mean_last_10000"]); print("wrote", out)


if __name__ == "__main__":
    main()
