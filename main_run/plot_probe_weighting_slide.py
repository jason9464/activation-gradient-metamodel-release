"""Single-panel slide figure for the gradient/activation probe comparison.

The plotted scores and 95% group-bootstrap intervals come from
``probe_compare_out/probe/table.md``.
The first two probe comparisons were specified before fitting; the
gradient-norm-weighted activation input was added after a layer-20 look.

Run from the repository root:
    python main_run/plot_probe_weighting_slide.py
"""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "20261030_probe_weighting_slide"

PAPER = "#f7f6f2"
NAVY = "#112d53"
MUTED = "#586c82"
GRID = "#d9e0e5"
BLUE = "#126cc3"
GREEN = "#078467"
GREY = "#718399"

# task, input, value, lower and upper 95% organism-bootstrap limits, y position.
ROWS = [
    ("Rare", "Gradient", 0.980, 0.971, 0.987, 4.05),
    ("Rare", "Weighted activations", 0.977, 0.967, 0.985, 3.34),
    ("Rare", "Plain activations", 0.826, 0.797, 0.855, 2.63),
    ("Family", "Gradient", 0.966, 0.949, 0.982, 1.55),
    ("Family", "Weighted activations", 0.918, 0.895, 0.940, 0.84),
    ("Family", "Plain activations", 0.928, 0.907, 0.949, 0.13),
]


def main() -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"}
    )
    fig = plt.figure(figsize=(11.8, 6.5), facecolor=PAPER)
    ax = fig.add_axes([0.315, 0.225, 0.64, 0.57], facecolor=PAPER)

    fig.text(
        0.075,
        0.95,
        "Weighting documents closes the rare-behaviour gap",
        ha="left",
        va="top",
        fontsize=21.5,
        fontfamily="DejaVu Serif",
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.075,
        0.87,
        "Same base model and documents · linear probes · held-out organisms",
        ha="left",
        va="top",
        fontsize=12.5,
        color=MUTED,
    )

    ax.set_xlim(0.78, 1.055)
    ax.set_ylim(-0.25, 4.73)
    ax.set_xticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"], color=MUTED)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in [0.80, 0.85, 0.90, 0.95, 1.00]:
        ax.axvline(x, color=GRID, linewidth=1, zorder=0)
    ax.axhline(2.23, color=GRID, linewidth=1.1, xmin=-0.38, xmax=1, clip_on=False)

    ax.text(
        0.771, 4.55, "Rare behaviour present?", ha="right", va="center",
        fontsize=13.7, fontweight="bold", color=NAVY, clip_on=False,
    )
    ax.text(
        0.771, 4.25, "AUROC · 898 bundles", ha="right", va="center",
        fontsize=10.5, color=MUTED, clip_on=False,
    )
    ax.text(
        0.771, 2.05, "Which of eight families?", ha="right", va="center",
        fontsize=13.7, fontweight="bold", color=NAVY, clip_on=False,
    )
    ax.text(
        0.771, 1.75, "Balanced accuracy · 4,532 bundles", ha="right", va="center",
        fontsize=10.5, color=MUTED, clip_on=False,
    )

    colors = {
        "Gradient": BLUE,
        "Weighted activations": GREEN,
        "Plain activations": GREY,
    }
    for _, input_name, value, lo, hi, y in ROWS:
        color = colors[input_name]
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.6, zorder=2)
        ax.plot([lo, lo], [y - 0.075, y + 0.075], color=color, linewidth=2, zorder=2)
        ax.plot([hi, hi], [y - 0.075, y + 0.075], color=color, linewidth=2, zorder=2)
        ax.scatter([value], [y], s=135, color=color, edgecolor=PAPER,
                   linewidth=1.6, zorder=3)
        ax.text(0.771, y, input_name, ha="right", va="center",
                fontsize=11.2, color=color, clip_on=False)
        ax.text(hi + 0.005, y, f"{value:.3f}", ha="left", va="center",
                fontsize=11.5, fontweight="bold", color=color)

    fig.text(
        0.63, 0.145, "Probe score (compare inputs within each task)",
        ha="center", va="center", fontsize=12.2, color=NAVY,
    )
    fig.text(
        0.075, 0.075,
        "Weighted activations use each document’s gradient norm; this baseline was exploratory.",
        ha="left", va="center", fontsize=10.7, color=MUTED,
    )
    fig.text(
        0.075, 0.043,
        "95% organism-bootstrap intervals · 8 layers (5–38) · one extraction from 4,540 bundles",
        ha="left", va="center", fontsize=10.7, color=MUTED,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=200, facecolor=PAPER)
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
