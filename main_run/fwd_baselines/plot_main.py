"""Main figure for Experiment 1: forward-only baselines.

Run from any directory::

    python main_run/fwd_baselines/plot_main.py

The plotted values and simultaneous intervals come from the committed probe
contrasts. The retrieval score panel uses the same rows, plus the descriptive
output-error row. Both PNG and PDF are written beside the source tables.
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
CONTRASTS = FIGURES / "20261028_fwd_baselines_probe_contrasts.json"
CHECKS = FIGURES / "20261028_fwd_baselines_checks.json"
STEM = FIGURES / "20261029_fwd_baselines_main"

TASKS = [
    ("P1", "Family", "balanced accuracy"),
    ("P3", "Shows behaviour", "mean AUROC"),
    ("P3_rare", "Rare", "AUROC"),
    ("P4", "Retrieval", "conditional top-1 hit rate"),
]
BASELINES = [
    ("actresp_all", "Response activations", "#8B99AB"),
    ("actrespn_all", "Label-count weighted", "#B4BEC9"),
    ("actrespl_all", "Summed-loss weighted", "#D47729"),
    ("fwd", "Forward pair", "#426FAF"),
]
INK = "#183044"
GRID = "#DAE1E8"


def select(rows: list[dict], task: str, a: str, b: str) -> dict:
    found = [
        row
        for row in rows
        if row.get("task") == task
        and row.get("a") == a
        and row.get("b") == b
        and row.get("prep") == "std"
    ]
    if len(found) != 1:
        raise ValueError(f"Expected one contrast for {(task, a, b)}; got {len(found)}")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", type=Path, default=STEM,
                        help="Path without extension; writes PNG and PDF")
    args = parser.parse_args()
    source = json.loads(CONTRASTS.read_text())
    checks = json.loads(CHECKS.read_text())
    confirmatory = source["primary"] + source["controls"]
    assert len(confirmatory) == 28
    assert checks["probe_reproduction"]["passed"]
    assert checks["lens_reproduction"]["passed"]
    assert checks["vechash"]["passed"]
    doc_failures = sum(not gate["passed"] for gate in checks["docs"]["gates"].values())
    bundle_failures = sum(not gate["passed"] for gate in checks["bundles"]["gates"].values())
    assert (doc_failures, bundle_failures) == (6, 1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(15.4, 9.8), layout="constrained")
    # Keep the top band free for the title and subtitle, so the panel titles sit below them.
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.95, 1.05], wspace=0.15)
    ax = fig.add_subplot(grid[0, 0])
    retrieval_ax = fig.add_subplot(grid[0, 1])

    positions = [[18, 17, 16, 15], [13.5, 12.5, 11.5, 10.5],
                 [9, 8, 7, 6], [4.5, 3.5, 2.5, 1.5]]
    headers = [18.8, 14.3, 9.8, 5.3]
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for task_i, (task, title, metric) in enumerate(TASKS):
        group_y = positions[task_i]
        # The band includes the group heading, so each heading reads as part of its own rows.
        ax.axhspan(min(group_y) - 0.45, headers[task_i] + 0.4,
                   color="#F7F9FB" if task_i % 2 == 0 else "#FFFFFF", zorder=0)
        ax.text(-0.355, headers[task_i], f"{title}  ·  {metric}",
                ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
        for baseline_i, (baseline, label, color) in enumerate(BASELINES):
            row = select(confirmatory, task, "grad_all", baseline)
            assert row["simultaneous_family_size"] == 28
            low, high = row["ci_simul97.5"]
            value = row["diff"]
            y = group_y[baseline_i]
            ax.plot([low, high], [y, y], color=color, lw=2.2,
                    solid_capstyle="round", zorder=3)
            ax.plot([low, low], [y - 0.13, y + 0.13], color=color, lw=1.6, zorder=3)
            ax.plot([high, high], [y - 0.13, y + 0.13], color=color, lw=1.6, zorder=3)
            ax.scatter(value, y, s=75, color=color, edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(1.025, y, f"{value:+.3f}".replace("-", "−"), transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=10, color=INK,
                    fontweight="bold" if baseline_i == 2 else "normal")
            tick_positions.append(y)
            tick_labels.append(label)

    ax.axvline(0, color=INK, lw=1.4, zorder=1)
    ax.set_xlim(-0.36, 0.33)
    ax.set_ylim(0.65, 19.55)
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["−0.3", "−0.2", "−0.1", "0", "+0.1", "+0.2", "+0.3"])
    ax.set_yticks(tick_positions, tick_labels)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors="#455565")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#A9B4BF")
    ax.set_xlabel("Gradient − forward-only baseline  (positive: gradient higher)", labelpad=10)
    ax.set_title("A   Remaining gradient advantage", loc="left", pad=19,
                 color=INK, fontweight="bold")

    # The second panel makes the retrieval mechanism legible in raw score units.
    retrieval_rows = [
        ("Response activations", select(confirmatory, "P4", "grad_all", "actresp_all")["b_value"], "#8B99AB"),
        ("Summed-loss weighted", select(confirmatory, "P4", "grad_all", "actrespl_all")["b_value"], "#D47729"),
        ("Gradient, all layers", select(confirmatory, "P4", "grad_all", "actresp_all")["a_value"], INK),
        ("Output error", select(source["descriptive"], "P4", "grad_all", "oerr")["b_value"], "#42A597"),
        ("Forward pair", select(confirmatory, "P4", "grad_all", "fwd")["b_value"], "#426FAF"),
    ]
    bars_y = [4, 3, 2, 1, 0]
    retrieval_ax.barh(bars_y, [row[1] for row in retrieval_rows], height=0.57,
                      color=[row[2] for row in retrieval_rows], zorder=3)
    for y, (_, value, _) in zip(bars_y, retrieval_rows):
        retrieval_ax.text(value + 0.02, y, f"{value:.3f}", ha="left", va="center",
                          fontsize=10, color=INK, fontweight="bold")
    retrieval_ax.set_yticks(bars_y, [row[0] for row in retrieval_rows])
    retrieval_ax.set_xlim(0, 0.98)
    retrieval_ax.set_ylim(-0.65, 4.65)
    retrieval_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    retrieval_ax.set_xlabel("Conditional top-1 hit rate", labelpad=10)
    retrieval_ax.set_title("B   Retrieval: output error is enough", loc="left", pad=19,
                           color=INK, fontweight="bold")
    retrieval_ax.grid(axis="x", color=GRID, lw=0.8)
    retrieval_ax.set_axisbelow(True)
    retrieval_ax.tick_params(axis="y", length=0, pad=7)
    retrieval_ax.tick_params(axis="x", colors="#455565")
    for spine in ("top", "right", "left"):
        retrieval_ax.spines[spine].set_visible(False)
    retrieval_ax.spines["bottom"].set_color("#A9B4BF")
    retrieval_ax.text(0.0, -0.11,
                      "Output error alone scores 0.800, above\nthe all-layer gradient at 0.617.",
                      transform=retrieval_ax.transAxes, ha="left", va="top",
                      color=INK, fontsize=10, linespacing=1.4)

    fig.text(0.012, 0.995, "Forward-only baselines narrow the gradient's lead",
             ha="left", va="top", color=INK, fontsize=19, fontweight="bold")
    fig.text(0.012, 0.952,
             "Family keeps its lead; summed-loss weighting closes most of it on shows-behaviour and rare; "
             "output error alone beats the gradient on retrieval.",
             ha="left", va="top", fontsize=11, color="#455565")
    fig.text(0.012, -0.025,
             "Panel A: simultaneous 97.5% intervals across 28 probe contrasts. "
             "Forward pair = summed-loss-weighted response activations + output error.\n"
             "On shows-behaviour and rare the output-error block dilutes the forward pair, so the summed-loss row "
             "(bold value) is the informative comparison there.\n"
             "Formal verdicts are provisional: 6 document and 1 bundle numerical-tail gates missed. "
             "Vector and probe reproduction passed. Source: 20261028_fwd_baselines_probe_contrasts.json and checks.json.",
             ha="left", va="top", fontsize=9, color="#566575", linespacing=1.5)

    for suffix in ("png", "pdf"):
        out = args.output_stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.25)
        print(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
