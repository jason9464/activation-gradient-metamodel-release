#!/usr/bin/env python
"""Make a single-chart slide figure from the online-gradient SAE results.

The four min/max ranges are calculated from the existing plotted-data JSON,
using the 1,000-step windows ending at steps 1,100 through 2,900. The windows
overlap, so the ranges describe one run rather than independent replications.

Run from anywhere:
    python main_run/plot_sae_grad_range_for_slide.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/figures/20261029_sae_grad_enrichment_figure_data.json"
OUT = ROOT / "docs/figures/20261030_sae_grad_range_slide"

INK = "#15191d"
MUTED = "#525b62"
GRID = "#e5e9eb"
DOLPHIN = "#2674c9"
OWL = "#13835e"


def main() -> None:
    data = json.loads(DATA.read_text())
    window = data["windows"]["w1000"]
    selected = [i for i, step in enumerate(window["steps"]) if 1100 <= step <= 2900]
    assert len(selected) == 19
    assert [window["steps"][selected[0]], window["steps"][selected[-1]]] == [1100, 2900]

    rows = [
        ("marine", "dolphin", 3.10, DOLPHIN, "Dolphin − clean"),
        ("marine", "owl", 2.40, OWL, "Owl − clean"),
        ("bird", "dolphin", 1.25, DOLPHIN, "Dolphin − clean"),
        ("bird", "owl", 0.55, OWL, "Owl − clean"),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(14, 7.875), dpi=220, facecolor="white")
    ax = fig.add_axes((0.295, 0.300, 0.655, 0.420))

    for feature, run, y, color, _ in rows:
        values = [window["counts"][run][feature][i] for i in selected]
        low, high = min(values), max(values)
        ax.plot([low, high], [y, y], color=color, lw=8, solid_capstyle="round", zorder=3)
        ax.scatter([low, high], [y, y], color=color, s=95, zorder=4)
        ax.text(high + 0.40, y, f"{low}–{high}", fontsize=15, color=INK,
                ha="left", va="center", weight="bold")

    ax.set_xlim(-0.15, 16.2)
    ax.set_ylim(0.15, 3.90)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Labelled SAE directions among the top 100", fontsize=15, color=INK, labelpad=14)
    ax.set_yticks([row[2] for row in rows], [row[4] for row in rows])
    ax.tick_params(axis="x", colors=MUTED, labelsize=13, length=0, pad=9)
    ax.tick_params(axis="y", colors=INK, labelsize=14, length=0, pad=15)
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.axhline(1.85, color=GRID, lw=1.2)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#a6afb5")

    ax.text(-0.26, 3.62, "MARINE FEATURES", transform=ax.get_yaxis_transform(),
            fontsize=12.5, color=MUTED, weight="bold", ha="left", va="center")
    ax.text(-0.26, 1.76, "BIRD FEATURES", transform=ax.get_yaxis_transform(),
            fontsize=12.5, color=MUTED, weight="bold", ha="left", va="center")

    fig.text(0.055, 0.935, "Animal-labelled SAE features in summed training gradients",
             fontsize=25, weight="bold", color=INK, ha="left", va="top")
    fig.text(0.055, 0.865,
             "Layer 30 activation gradients: animal run minus clean run, summed over 1,000 training steps.",
             fontsize=14, color=MUTED, ha="left", va="top")
    fig.text(0.055, 0.795,
             "Planned single-step test: no detected alignment with the teacher’s top 20 SAE features (p = 0.15).",
             fontsize=14.5, color=INK, ha="left", va="top")

    fig.text(0.055, 0.169,
             "Bars show min–max across 19 windows ending at steps 1,100–2,900; adjacent windows share 900 steps.",
             fontsize=12.5, color=MUTED, ha="left", va="top")
    fig.text(0.055, 0.119,
             "Exploratory cutoff chosen after inspection. One seed per run. Feature labels use nearest output tokens.",
             fontsize=12.5, color=MUTED, ha="left", va="top")

    for suffix in ("png", "svg"):
        fig.savefig(OUT.with_suffix(f".{suffix}"), dpi=220, facecolor="white")
    plt.close(fig)
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".svg"))


if __name__ == "__main__":
    main()
