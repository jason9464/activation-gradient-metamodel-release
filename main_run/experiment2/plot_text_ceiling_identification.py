"""Slide-ready text-only identification figure.

Counts are from the frozen-preamble Haiku 5.5 run, reported in
docs/notes/20261030_text_ceiling_loss_detector_results.md.  They are the
numbers of showing held-out bundles whose free answer was judged to identify
the bundle's behaviour.  The behaviour label was shown to the judge only.

Run from the repository root:
    python main_run/experiment2/plot_text_ceiling_identification.py
"""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "figures" / "20261030_text_ceiling_identification"

ROWS = [
    ("Quirk", 54, 54),
    ("Harmful", 80, 80),
    ("Backdoor", 59, 63),
    ("Rare", 16, 45),
]

PAPER = "#f7f6f2"
NAVY = "#112d53"
MUTED = "#566b83"
GRID = "#d9e0e5"
VISIBLE = "#078467"
RARE = "#126cc3"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(11.8, 6.2), facecolor=PAPER)
    ax = fig.add_axes([0.22, 0.25, 0.71, 0.48], facecolor=PAPER)

    fig.text(
        0.075,
        0.93,
        "A text reader already names visible behaviours",
        ha="left",
        va="top",
        fontsize=22,
        fontfamily="DejaVu Serif",
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.075,
        0.83,
        "Haiku 5.5 reads the full training documents, then describes the behaviour",
        ha="left",
        va="top",
        fontsize=13,
        color=MUTED,
    )

    for x in (0, 25, 50, 75, 100):
        ax.axvline(x, color=GRID, linewidth=1, zorder=0)

    y_positions = [3, 2, 1, 0]
    for (name, correct, total), y in zip(ROWS, y_positions):
        rate = 100 * correct / total
        color = RARE if name == "Rare" else VISIBLE
        ax.barh(y, rate, height=0.55, color=color, alpha=0.88, zorder=2)
        ax.text(
            -3.5,
            y,
            name,
            ha="right",
            va="center",
            fontsize=15,
            fontweight="bold" if name == "Rare" else "normal",
            color=RARE if name == "Rare" else NAVY,
        )
        ax.text(
            rate + 2.5,
            y,
            f"{rate:.0f}%  ({correct}/{total})",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=color,
        )

    ax.set_xlim(-17, 132)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], color=MUTED)
    ax.tick_params(axis="x", length=0, pad=9)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.57,
        0.17,
        "Showing bundles correctly identified (%)",
        ha="center",
        va="center",
        fontsize=12,
        color=NAVY,
    )
    fig.text(
        0.075,
        0.105,
        "Rare behaviours appear in only 1–2 of 12 documents. Showing bundles only; one answer per bundle.",
        ha="left",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.065,
        "The behaviour label was used for judging and was not shown to Haiku.",
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
