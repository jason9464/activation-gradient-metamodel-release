#!/usr/bin/env python
"""Plot the saved reward-hacking reader and AO results.

    python main_run/rh/plot_results.py

The plotted judge counts and keyword table are committed under docs/rh_results/.
The latter is exploratory and post hoc; it is not a tampering verdict.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parents[2]
RESULTS = HERE / "docs" / "rh_results"
OUT = HERE / "docs" / "figures" / "20261029_rh_reader_vs_ao.png"

INK = "#20252b"
MUTED = "#5c6470"
GRID = "#e0e4e9"
BLUE = "#226bb5"
ORANGE = "#d85b35"
LABELS = [
    ("Our reader · Δh · open", "dh_beh"),
    ("Our reader · direct writes · open", "dwx_beh"),
    ("Our reader · Δh · asks about tests", "dh_tgt"),
    ("Our reader · direct writes · asks about tests", "dwx_tgt"),
    ("Paper AO (layer 20) · Δh · open", "ao_aopaper_dh_describe"),
    ("Paper AO (layer 20) · direct writes · open", "ao_aopaper_dwx_describe"),
    ("Paper AO (layer 20) · Δh · asks about tests", "ao_aopaper_dh_targeted"),
    ("Paper AO (layer 20) · direct writes · asks about tests", "ao_aopaper_dwx_targeted"),
]


def load(results: Path) -> tuple[list[list[int]], dict, float, float]:
    summary = json.loads((results / "rh_summary.json").read_text(encoding="utf-8"))
    mentions = json.loads((results / "test_mentions_by_step.json").read_text(encoding="utf-8"))
    assert summary["primary_steps"] == [43, 45, 47]
    counts = []
    for _, key in LABELS:
        cell = summary["cells"][key]
        row = []
        for run in ("hacker", "anti"):
            k, n = cell["dropped"][run]
            assert n == 300 and cell["conservative"][run] == [k, n], (key, run)
            assert sum(cell["per_step"][str(s)]["rh_" + run]["yes"] for s in (43, 45, 47)) == k
            row.append(k)
        counts.append(row)
    for key in ("dh_beh", "dwx_beh", "dh_tgt", "dwx_tgt", "dh_neu", "dwx_neu"):
        assert summary["cells"][key]["dropped"]["hacker"][0] == 0
        assert summary["cells"][key]["dropped"]["anti"][0] == 0
    series = mentions["cells"]["ckptdiff_behavioural"]["rh_hacker"]
    assert len(series["steps"]) == len(series["mentions"]) == len(series["hack_rate"]) == 15
    assert all(n == 100 for n in series["n"])
    rho, p = spearmanr(series["mentions"], series["hack_rate"])
    assert abs(rho - series["spearman_rho"]) < 0.001
    assert abs(p - series["spearman_p"]) < 0.0001
    return counts, series, float(rho), float(p)


def plot(results: Path, out: Path) -> None:
    counts, series, rho, p = load(results)
    fig = plt.figure(figsize=(16, 7.8), layout="constrained")
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.18})
    ax, bx = axes
    fig.suptitle("The paper AO distinguishes the hacker only when asked about tests on direct writes",
                 x=0.04, y=1.04, ha="left", color=INK, fontsize=17, fontweight="bold")

    # A compact table-like heatmap makes the many exact zeros legible.
    cmap = mcolors.LinearSegmentedColormap.from_list("tamper", ["#f1f5f9", "#9cc7ec", BLUE])
    vals = np.asarray(counts, dtype=float) / 300
    ax.imshow(vals, cmap=cmap, vmin=0, vmax=0.16, aspect="auto", interpolation="none")
    ax.set_xticks([0, 1], ["Hacker", "Anti-hacker"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=11, colors=INK, labelsize=11)
    ax.set_yticks(range(len(LABELS)), [label for label, _ in LABELS])
    ax.tick_params(axis="y", length=0, pad=10, colors=INK)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(LABELS), 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(3.5, color="#fcfcfb", linewidth=10)
    for i, row in enumerate(counts):
        for j, k in enumerate(row):
            color = "white" if k >= 35 else INK
            ax.text(j, i, f"{k}/300", ha="center", va="center", color=color,
                    fontweight="bold" if k else "normal", fontsize=11)
    ax.set_title("A   Answers judged as describing tampering", loc="left", pad=55, color=INK, fontweight="bold")
    ax.text(0, -0.16, "Our reader: zero in all six cells, including neutral probes.\n"
            "Three-layer AO: zero in all tested cells. Paper AO direct writes: Fisher p = 3.3×10⁻¹⁰.",
            transform=ax.transAxes, va="top", ha="left", color=MUTED, fontsize=10.3)

    x = np.asarray(series["hack_rate"])
    y = np.asarray(series["mentions"]) / np.asarray(series["n"])
    bx.scatter(x, y, color=ORANGE, s=68, edgecolor="white", linewidth=1.2, zorder=3)
    for step in (5, 43, 45, 47, 90, 92):
        i = series["steps"].index(step)
        dx, dy = {5: (.02, .035), 43: (-.08, .04), 45: (.02, .025),
                  47: (-.08, -.065), 90: (-.08, -.07), 92: (-.075, .035)}[step]
        bx.text(x[i] + dx, y[i] + dy, str(step), fontsize=9, color=MUTED)
    bx.set_xlim(-0.04, 1.02)
    bx.set_ylim(-0.04, 0.95)
    bx.set_xticks(np.arange(0, 1.01, .2), [f"{v:.0%}" for v in np.arange(0, 1.01, .2)])
    bx.set_yticks(np.arange(0, 1.0, .2), [f"{v:.0%}" for v in np.arange(0, 1.0, .2)])
    bx.grid(color=GRID, linewidth=.8)
    bx.set_axisbelow(True)
    bx.set_xlabel("Hacker rollout hack rate")
    bx.set_ylabel("Open Δh reader answers mentioning tests")
    bx.set_title("B   Test words track the hack rate", loc="left", pad=24, color=INK, fontweight="bold")
    bx.text(.04, .95, f"Spearman ρ = {rho:.2f} · p = {p:.3f}\n15 checkpoints; 100 answers each",
            transform=bx.transAxes, va="top", ha="left", fontsize=11, color=INK,
            bbox=dict(facecolor="#fcfcfb", edgecolor="none", alpha=.92, pad=5))
    bx.text(0, -0.16, "Exploratory, post hoc keyword analysis. Saying “tests” is not a tampering verdict.\n"
            "Checkpoint differences, not online RL gradients; one training run per arm.",
            transform=bx.transAxes, va="top", ha="left", color=MUTED, fontsize=10.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    plot(args.results, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
