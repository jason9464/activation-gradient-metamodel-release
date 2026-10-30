#!/usr/bin/env python
"""Slide-sized two-panel plot of per-window SAE counts in training gradients.

Reads the existing plotted-data JSON, without retyping the 19 window values.
The result is descriptive: adjacent 1,000-step windows share 900 steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


ARTIFACT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
DATA = ARTIFACT / "docs/figures/20261029_sae_grad_enrichment_figure_data.json"
OUT = REPO / "docs/figures/20261030_sae_grad_window_counts_slide"

INK = "#161a1e"
MUTED = "#556068"
GRID = "#e4e9ec"
DOLPHIN = "#2674c9"
OWL = "#13835e"


def main() -> None:
    data = json.loads(DATA.read_text())
    w = data["windows"]["w1000"]
    selected = [i for i, step in enumerate(w["steps"]) if 1100 <= step <= 2900]
    steps = [w["steps"][i] for i in selected]
    assert steps == list(range(1100, 3000, 100))

    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.875), dpi=220, facecolor="white", sharey=True)
    fig.subplots_adjust(left=0.092, right=0.966, top=0.675, bottom=0.285, wspace=0.165)

    for ax, feature, panel_title in zip(
        axes,
        ("marine", "bird"),
        ("Marine-labelled SAE features", "Bird-labelled SAE features"),
    ):
        for run, color in (("dolphin", DOLPHIN), ("owl", OWL)):
            counts = [w["counts"][run][feature][i] for i in selected]
            ax.plot(steps, counts, color=color, lw=2.8, marker="o", ms=5.3,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3)
        ax.set_title(panel_title, loc="left", fontsize=17, color=INK, pad=18)
        ax.set_xlim(1050, 2950)
        ax.set_xticks([1100, 1500, 1900, 2300, 2700, 2900])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_ylim(-0.4, 15.2)
        ax.set_yticks([0, 5, 10, 15])
        ax.set_xlabel("Training step (window end)", fontsize=14, color=INK, labelpad=12)
        ax.yaxis.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors=MUTED, labelsize=11.5, length=0, pad=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#aab3b8")

    axes[0].set_ylabel("Labelled features among the top 100", fontsize=14, color=INK, labelpad=13)
    axes[1].tick_params(axis="y", labelleft=True)

    handles = [
        Line2D([0], [0], color=DOLPHIN, lw=2.8, marker="o", ms=6, label="Dolphin − clean"),
        Line2D([0], [0], color=OWL, lw=2.8, marker="o", ms=6, label="Owl − clean"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.092, 0.807), ncol=2,
               frameon=False, fontsize=14, handlelength=2.4, columnspacing=3.2)

    fig.text(0.055, 0.94, "Animal-labelled SAE features across training-gradient windows",
             fontsize=24, weight="bold", color=INK, ha="left", va="top")
    fig.text(0.055, 0.873,
             "Layer 30 activation gradients, animal run minus clean run; each point sums 1,000 training steps.",
             fontsize=14, color=MUTED, ha="left", va="top")
    fig.text(0.055, 0.183,
             "Adjacent windows share 900 steps, so these 19 points are strongly dependent.",
             fontsize=12.5, color=MUTED, ha="left", va="top")
    fig.text(0.055, 0.128,
             "Exploratory start at step 1,100, chosen after inspection; one seed per run. SAE labels use nearest output tokens.",
             fontsize=12.5, color=MUTED, ha="left", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(OUT.with_suffix(f".{suffix}"), dpi=220, facecolor="white")
    plt.close(fig)
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
