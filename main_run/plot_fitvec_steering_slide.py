"""Draw a slide-ready summary of the fitted-bias steering experiment.

The five matched comparisons come from the saved steering cells in
``act-probe:docs/figures/20261028_fitvec_steer.jsonl`` and are summarized in
``act-probe:docs/notes/20261028_fitted_vector_results.md``.  Each row uses the
animal-data and clean-data Adam fits at the same injection strength.

Run from the repository root:
    python main_run/plot_fitvec_steering_slide.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "20261030_fitvec_steering_slide"

# group, strength, first-word target, animal-fit hits, clean-fit hits, y
ROWS = [
    ("Owl", 1, "owl", 206, 44, 4.5),
    ("Owl", 2, "owl", 377, 47, 3.55),
    ("Dolphin", 1, "whale‡", 136, 18, 2.2),
    ("Dolphin", 2, "whale‡", 273, 17, 1.25),
    ("Dolphin", 4, "dolphin†", 118, 2, 0.3),
]

PAPER = "#f7f6f2"
NAVY = "#112d53"
MUTED = "#566b83"
GRID = "#d9e0e5"
CLEAN = "#8999a8"
OWL = "#078467"
DOLPHIN = "#126cc3"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(13.2, 7.0), facecolor=PAPER)
    ax = fig.add_axes([0.075, 0.255, 0.88, 0.55], facecolor=PAPER)

    fig.text(
        0.075,
        0.945,
        "Fitted biases steer animal answers",
        ha="left",
        va="top",
        fontsize=23,
        fontfamily="DejaVu Serif",
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.075,
        0.875,
        "Add learned residual biases to the frozen base model; compare each animal fit with the clean-data fit",
        ha="left",
        va="top",
        fontsize=12.5,
        color=MUTED,
    )

    for x in (0, 100, 200, 300, 400):
        ax.axvline(x, color=GRID, linewidth=1, zorder=0)
    ax.axhline(2.88, xmin=0, xmax=1, color=GRID, linewidth=1)

    for group, strength, target, fitted, clean, y in ROWS:
        color = OWL if group == "Owl" else DOLPHIN
        ax.plot([clean, fitted], [y, y], color=color, alpha=0.38, linewidth=2.6, zorder=1)
        ax.scatter([clean], [y], s=105, color=CLEAN, edgecolor=PAPER, linewidth=1.5, zorder=3)
        ax.scatter([fitted], [y], s=135, color=color, edgecolor=PAPER, linewidth=1.5, zorder=4)
        ax.annotate(
            str(clean),
            (clean, y),
            xytext=(0, -17),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10.5,
            color=MUTED,
        )
        ax.annotate(
            str(fitted),
            (fitted, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        ax.text(
            -13,
            y,
            f"×{strength}  →  {target}",
            ha="right",
            va="center",
            fontsize=12,
            color=NAVY,
        )

    ax.text(-195, 4.03, "OWL FIT", color=OWL, fontweight="bold", fontsize=12, va="center")
    ax.text(-195, 1.25, "DOLPHIN FIT", color=DOLPHIN, fontweight="bold", fontsize=12, va="center")

    ax.set_xlim(-205, 415)
    ax.set_ylim(-0.25, 5.1)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xticklabels(["0", "100", "200", "300", "400"], color=MUTED)
    ax.tick_params(axis="x", length=0, pad=9)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.37,
        0.205,
        "Answers starting with the target animal (of 1,000)",
        ha="center",
        va="center",
        fontsize=12,
        color=NAVY,
    )
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=OWL, markeredgecolor=PAPER, markersize=11, label="Owl fit"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DOLPHIN, markeredgecolor=PAPER, markersize=11, label="Dolphin fit"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CLEAN, markeredgecolor=PAPER, markersize=11, label="Clean fit, same strength"),
    ]
    fig.legend(
        handles=legend,
        loc="lower right",
        bbox_to_anchor=(0.955, 0.17),
        frameon=False,
        ncol=3,
        columnspacing=1.4,
        handletextpad=0.3,
        labelcolor=MUTED,
        fontsize=10.5,
    )
    fig.text(
        0.075,
        0.105,
        "50 prompts × 20 samples; one fit and seed per arm.  † At ×4, 225/1,000 dolphin-fit outputs began with </think>.",
        ha="left",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.069,
        "‡ Whale counts were exploratory. These are unsubtracted fits; the SAE comparison used animal fit − clean fit.",
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
