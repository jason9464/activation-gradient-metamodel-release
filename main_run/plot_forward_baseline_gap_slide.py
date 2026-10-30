"""Single-panel comparison of the gradient's remaining forward-baseline gaps.

Values and paired 95% group-bootstrap intervals are from:
  - agm-probe-compare/docs/notes/20261028_probe_compare_results.md
  - agm-fwd-baselines/docs/figures/20261028_fwd_baselines_probe_table.md
  - agm-fwd-baselines/docs/figures/20261028_fwd_baselines_lens_table.md

The family loss-weighted contrast also has a simultaneous 97.5% interval of
[+0.020, +0.064]. The output-error lens comparison is descriptive.

Run from the repository root:
    python main_run/plot_forward_baseline_gap_slide.py
"""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "20261030_forward_baseline_gap_slide"

PAPER = "#f7f6f2"
NAVY = "#112d53"
MUTED = "#566b83"
GRID = "#d9e0e5"
GREY = "#74869b"
GREEN = "#078467"

# Task, comparator, comparator score, gradient-minus-comparator difference,
# and paired 95% group-bootstrap limits. All scores are unit-interval metrics.
ROWS = [
    ("Family", "Full activations", 0.928, 0.038, 0.026, 0.052, 3.92, GREY),
    ("Family", "Loss-weighted response activations", 0.924, 0.042, 0.030, 0.056, 3.18, GREEN),
    ("Lens", "Full activations", 0.228, 0.121, 0.081, 0.163, 1.56, GREY),
    ("Lens", "Output error (final layer)", 0.333, 0.016, -0.007, 0.038, 0.82, GREEN),
]


def main() -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"}
    )
    fig = plt.figure(figsize=(11.8, 6.6), facecolor=PAPER)
    ax = fig.add_axes([0.365, 0.252, 0.575, 0.535], facecolor=PAPER)

    fig.text(
        0.075, 0.95,
        "Output error closes most of the lens gap; the family gap persists",
        ha="left", va="top", fontsize=20.3, fontfamily="DejaVu Serif",
        fontweight="bold", color=NAVY,
    )
    fig.text(
        0.075, 0.865,
        "Remaining advantage of the gradient over each forward-only input",
        ha="left", va="top", fontsize=12.5, color=MUTED,
    )

    ax.set_xlim(-0.025, 0.205)
    ax.set_ylim(0.35, 4.55)
    ticks = [-0.02, 0, 0.04, 0.08, 0.12, 0.16, 0.20]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        ["−.02", "0", "+.04", "+.08", "+.12", "+.16", "+.20"],
        color=MUTED,
    )
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in ticks:
        ax.axvline(x, color=NAVY if x == 0 else GRID,
                   linewidth=1.35 if x == 0 else 1,
                   alpha=0.65 if x == 0 else 1, zorder=0)
    ax.axhline(2.56, color=GRID, linewidth=1.2, xmin=-0.54, xmax=1,
               clip_on=False)

    ax.text(-0.033, 4.42, "Family classification", ha="right", va="center",
            fontsize=13.6, fontweight="bold", color=NAVY, clip_on=False)
    ax.text(-0.033, 4.15, "Balanced accuracy · gradient 0.966", ha="right",
            va="center", fontsize=10.8, color=MUTED, clip_on=False)
    ax.text(-0.033, 2.22, "Logit lens at layer 38", ha="right", va="center",
            fontsize=13.6, fontweight="bold", color=NAVY, clip_on=False)
    ax.text(-0.033, 1.95, "Showing − none AUROC Δ · gradient 0.349", ha="right",
            va="center", fontsize=10.8, color=MUTED, clip_on=False)

    for _, name, score, value, lo, hi, y, color in ROWS:
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.6, zorder=2)
        ax.plot([lo, lo], [y - 0.08, y + 0.08], color=color, linewidth=2, zorder=2)
        ax.plot([hi, hi], [y - 0.08, y + 0.08], color=color, linewidth=2, zorder=2)
        ax.scatter([value], [y], s=140, color=color, edgecolor=PAPER,
                   linewidth=1.6, zorder=3)
        ax.text(-0.033, y, f"{name}  {score:.3f}", ha="right", va="center",
                fontsize=11.1, color=color, clip_on=False)
        ax.text(hi + 0.006, y, f"+{value:.3f}", ha="left", va="center",
                fontsize=11.8, fontweight="bold", color=color)

    fig.text(
        0.65, 0.165,
        "Gradient score − comparator score (within each task)",
        ha="center", va="center", fontsize=12.2, color=NAVY,
    )
    fig.text(
        0.075, 0.092,
        "95% paired group-bootstrap intervals. Lens output-error comparison is descriptive.",
        ha="left", va="center", fontsize=10.6, color=MUTED,
    )
    fig.text(
        0.075, 0.055,
        "Family loss-weighted 97.5% simultaneous interval: [+0.020, +0.064]. Formal category provisional: numerical tail gates missed tolerance.",
        ha="left", va="center", fontsize=10.2, color=MUTED,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=200, facecolor=PAPER)
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
