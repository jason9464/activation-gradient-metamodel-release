"""Slide-ready ΔP(Yes) comparison for matched behaviour readers.

Each point is mean P(Yes) on showing bundles minus mean P(Yes) on non-showing
bundles, in percentage points. The 95% percentile intervals resample organism
IDs with replacement (2,000 draws, seed 0). Values were verified against the
local Experiment 2 score files at
``artifacts/experiment2_out/runs/experiment2/pyes``. Those raw files are not
tracked; the rounded values below make this figure reproducible from the repo.

Run from the repository root:
    python main_run/experiment2/plot_reader_delta_pyes_slide.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "figures" / "20261030_reader_delta_pyes_slide"

# group, reader, ΔP(Yes), 95% organism-bootstrap interval; percentage points.
ROWS = [
    ("Pooled", "Gradient", 27.35, 22.17, 33.53),
    ("Pooled", "Activation", 15.05, 9.23, 20.67),
    ("Rare", "Gradient", 9.72, 4.07, 14.73),
    ("Rare", "Activation", -0.19, -4.34, 3.11),
    ("Quirk", "Gradient", 39.29, 31.41, 45.98),
    ("Quirk", "Activation", 23.63, 16.81, 31.04),
]

PAPER = "#f7f6f2"
NAVY = "#112d53"
MUTED = "#566b83"
GRID = "#d9e0e5"
BLUE = "#126cc3"
GREEN = "#078467"


def main() -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"}
    )
    fig = plt.figure(figsize=(11.8, 6.7), facecolor=PAPER)
    ax = fig.add_axes([0.19, 0.245, 0.75, 0.53], facecolor=PAPER)

    fig.text(
        0.075,
        0.945,
        "Gradient reader separates showing bundles more strongly",
        ha="left",
        va="top",
        fontsize=22,
        fontfamily="DejaVu Serif",
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.075,
        0.865,
        "ΔP(Yes) = mean probability with the behaviour − mean probability without it",
        ha="left",
        va="top",
        fontsize=12.5,
        color=MUTED,
    )

    ax.set_xlim(-10, 60)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xticks([-10, 0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["−10", "0", "+10", "+20", "+30", "+40", "+50"], color=MUTED)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in (-10, 10, 20, 30, 40, 50):
        ax.axvline(x, color=GRID, linewidth=1, zorder=0)
    ax.axvline(0, color=NAVY, linewidth=1.4, alpha=0.6, zorder=1)

    positions = {"Pooled": 2, "Rare": 1, "Quirk": 0}
    for group, reader, value, lo, hi in ROWS:
        y = positions[group] + (0.15 if reader == "Gradient" else -0.15)
        color = BLUE if reader == "Gradient" else GREEN
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.5, zorder=2)
        ax.plot([lo, lo], [y - 0.055, y + 0.055], color=color, linewidth=2, zorder=2)
        ax.plot([hi, hi], [y - 0.055, y + 0.055], color=color, linewidth=2, zorder=2)
        ax.scatter([value], [y], s=115, color=color, edgecolor=PAPER, linewidth=1.5, zorder=3)
        label = f"{value:+.0f}" if abs(value) >= 0.5 else "0"
        ax.text(hi + 1.6, y, label, ha="left", va="center", fontsize=12,
                fontweight="bold", color=color)

    for group, y in positions.items():
        label = "Pooled  (199)" if group == "Pooled" else f"{group}  ({63 if group == 'Rare' else 68})"
        ax.text(-11.5, y, label, ha="right", va="center", fontsize=13, color=NAVY)

    legend = [
        Line2D([0], [0], marker="o", color=BLUE, markersize=8, linewidth=2,
               markeredgecolor=PAPER, label="Gradient reader"),
        Line2D([0], [0], marker="o", color=GREEN, markersize=8, linewidth=2,
               markeredgecolor=PAPER, label="Activation reader"),
    ]
    fig.legend(handles=legend, loc="center right", bbox_to_anchor=(0.92, 0.805),
               frameon=False, ncol=2, columnspacing=1.4, handletextpad=0.45,
               fontsize=11, labelcolor=NAVY)

    fig.text(0.56, 0.165, "ΔP(Yes), percentage points", ha="center", va="center",
             fontsize=12.5, color=NAVY)
    fig.text(
        0.075,
        0.095,
        "95% organism-bootstrap intervals. Pooled includes rare, quirk and backdoor; backdoor alone has only 5 controls.",
        ha="left",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.055,
        "At P(Yes) ≥ 0.5, the gradient reader says Yes to 96% of pooled bundles (balanced accuracy 0.608).",
        ha="left",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
