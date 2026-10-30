"""Plot Experiment 2's matched reader and same-input probe comparison.

Reads the committed analysis JSON and writes a PNG and PDF in docs/figures.
Run from any directory with::

    python main_run/experiment2/plot_matched_reader.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "figures"
SOURCE = FIGURES / "20261028_experiment2_analysis.json"
DEFAULT_OUTPUT = FIGURES / "20261029_experiment2_matched_reader"
POOL = "pooled_rare_quirk_backdoor"

INK = "#193044"
MUTED = "#556879"
GRID = "#DCE4EA"
GRADIENT = "#386EA8"
ACTIVATION = "#D17729"


def score(data: dict, kind: str, input_name: str) -> dict:
    if kind == "reader":
        return data["pyes"][f"{input_name}__{input_name}"][POOL]
    return data["probe"][input_name][POOL]


def draw_interval(ax: plt.Axes, x: float, row: dict, color: str) -> None:
    value, lo, hi = (row[key] for key in ("auc", "lo", "hi"))
    ax.plot([x, x], [lo, hi], color=color, lw=2.1, zorder=3)
    ax.plot([x - 0.026, x + 0.026], [lo, lo], color=color, lw=1.8, zorder=3)
    ax.plot([x - 0.026, x + 0.026], [hi, hi], color=color, lw=1.8, zorder=3)
    ax.scatter(x, value, s=145, color=color, edgecolor="white", linewidth=1.2, zorder=4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination without extension; writes PNG and PDF")
    args = parser.parse_args()
    data = json.loads(SOURCE.read_text())

    gr = score(data, "reader", "gradient")
    ar = score(data, "reader", "activation")
    gp = score(data, "probe", "gradient")
    ap = score(data, "probe", "activation")
    reader_gap = data["main_gradient_minus_activation_auc"]
    probe_gap = data["probe_gradient_minus_activation_auc"]
    grad_shortfall = data["probe_minus_reader_auc"]["gradient__gradient"]
    act_shortfall = data["probe_minus_reader_auc"]["activation__activation"]

    assert data["families_primary"] == ["rare", "quirk", "backdoor"]
    assert all(row["n_bundles"] == 199 and row["n_organisms"] == 47
               for row in (gr, ar, gp, ap, reader_gap, probe_gap, grad_shortfall, act_shortfall))
    assert abs((gr["auc"] - ar["auc"]) - reader_gap["gap"]) < 1e-12
    assert abs((gp["auc"] - ap["auc"]) - probe_gap["gap"]) < 1e-12
    assert abs((gp["auc"] - gr["auc"]) - grad_shortfall["gap"]) < 1e-12
    assert abs((ap["auc"] - ar["auc"]) - act_shortfall["gap"]) < 1e-12

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(13.6, 7.2))
    grid = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.77, bottom=0.20,
                           width_ratios=[1.15, 1], wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])
    gap_ax = fig.add_subplot(grid[0, 1])

    # Panel A: matched readers versus simple linear probes of the same inputs.
    series = [
        ("Gradient input", GRADIENT, [gr, gp], [-0.10, 0.90]),
        ("Activation input", ACTIVATION, [ar, ap], [0.10, 1.10]),
    ]
    for label, color, rows, xs in series:
        ax.plot(xs, [row["auc"] for row in rows], color=color, lw=2.2,
                alpha=0.65, zorder=2)
        for x, row in zip(xs, rows):
            draw_interval(ax, x, row, color)
        ax.plot([], [], color=color, marker="o", ms=8, lw=2.2, label=label)
    for x, row, dx, dy, align in [
        (-0.10, gr, -0.07, -0.01, "right"),
        (0.10, ar, 0.07, 0.0, "left"),
        (0.90, gp, -0.07, 0.0, "right"),
        (1.10, ap, 0.07, -0.012, "left"),
    ]:
        ax.text(x + dx, row["auc"] + dy, f"{row['auc']:.3f}", ha=align,
                va="center", color=INK, fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="#93A3B0", lw=1, ls="--", zorder=1)
    ax.text(1.35, 0.508, "chance", ha="right", va="bottom", color=MUTED, fontsize=9)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0.48, 1.05)
    ax.set_xticks([0, 1], ["Matched reader", "Linear probe"])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("AUROC of P(Yes)")
    ax.set_title("A   Held-out behaviour detection", loc="left", color=INK,
                 fontweight="bold", pad=16)
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.10), ncol=2,
              frameon=False, fontsize=10, columnspacing=1.6, handlelength=2.0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", colors=MUTED)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#A7B5BF")

    # Panel B: directly bootstrapped paired gaps. The probe gap is much smaller
    # than the reader gap, and the activation probe-to-reader shortfall is larger.
    gap_rows = [
        ("Matched readers", reader_gap, INK, 3.15),
        ("Linear probes", probe_gap, INK, 2.25),
        ("Gradient input", grad_shortfall, GRADIENT, 0.95),
        ("Activation input", act_shortfall, ACTIVATION, 0.05),
    ]
    gap_ax.axvline(0, color="#8B9BA8", lw=1, zorder=1)
    gap_ax.axhline(1.65, color=GRID, lw=1)
    for label, row, color, y in gap_rows:
        value, lo, hi = (row[key] for key in ("gap", "lo", "hi"))
        gap_ax.plot([lo, hi], [y, y], color=color, lw=2.5,
                    solid_capstyle="round", zorder=2)
        gap_ax.plot([lo, lo], [y - 0.09, y + 0.09], color=color, lw=1.7, zorder=2)
        gap_ax.plot([hi, hi], [y - 0.09, y + 0.09], color=color, lw=1.7, zorder=2)
        gap_ax.scatter(value, y, s=90, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        gap_ax.text(1.025, y, f"+{value:.3f}", transform=gap_ax.get_yaxis_transform(),
                    ha="left", va="center", color=INK, fontsize=11, fontweight="bold")
    gap_ax.text(0.005, 3.62, "Gradient − activation", ha="left", va="center",
                color=MUTED, fontsize=9, fontweight="bold")
    gap_ax.text(0.005, 1.48, "Probe − matched reader", ha="left", va="center",
                color=MUTED, fontsize=9, fontweight="bold")
    gap_ax.set_xlim(-0.005, 0.34)
    gap_ax.set_ylim(-0.45, 3.75)
    gap_ax.set_yticks([row[3] for row in gap_rows], [row[0] for row in gap_rows])
    gap_ax.set_xticks([0, 0.1, 0.2, 0.3])
    gap_ax.set_xlabel("Paired AUROC difference", labelpad=10)
    gap_ax.set_title("B   The input gap is smaller than the reader gap",
                     loc="left", color=INK, fontweight="bold", pad=16)
    gap_ax.grid(axis="x", color=GRID, lw=0.8)
    gap_ax.set_axisbelow(True)
    gap_ax.tick_params(axis="y", length=0, pad=8)
    gap_ax.tick_params(axis="x", colors=MUTED)
    for spine in ("top", "right", "left"):
        gap_ax.spines[spine].set_visible(False)
    gap_ax.spines["bottom"].set_color("#A7B5BF")

    fig.text(0.07, 0.95, "Matched activation reader trails despite informative inputs",
             color=INK, ha="left", va="top", fontsize=19, fontweight="bold")
    fig.text(0.07, 0.895,
             "Same three layers, data, training recipe and steps · 199 held-out rare, quirk and backdoor bundles from 47 organisms",
             color=MUTED, ha="left", va="top", fontsize=11)
    fig.text(0.07, 0.12,
             "Reader gap: +0.210 AUROC; probe gap on the same inputs: +0.040. The activation probe exceeds its reader by +0.219,\n"
             "versus +0.049 on gradients. Intervals are 95% organism-bootstrap intervals (2,000 resamples); one seed per reader.",
             color=MUTED, ha="left", va="top", fontsize=10, linespacing=1.55)
    fig.text(0.07, 0.035,
             "Source: 20261028_experiment2_analysis.json · Matched reader cells only; cross-input swaps are out of distribution.",
             color="#6A7B89", ha="left", va="bottom", fontsize=9)

    for suffix in ("png", "pdf"):
        output = args.output_stem.with_suffix(f".{suffix}")
        fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=0.22)
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
