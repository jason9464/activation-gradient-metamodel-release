"""Plot the rare-bundle loss-weighting result in probability units.

The four readers are evaluated on the same 63 held-out rare bundles. Each point
is the mean P(Yes) on showing bundles minus the mean on non-showing bundles, in
percentage points. Percentile intervals resample organisms (10,000 draws,
seed 0). The shuffled weighting uses one within-bundle permutation (seed 0).

Run from the repository root:
    python main_run/experiment2/plot_loss_weighted_delta_pyes_slide.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FWD = ROOT / "artifacts/fwdb/data/fwd/loss_weighted_shuf_run/pyes"
E2 = ROOT / "artifacts/experiment2_out/runs/experiment2/pyes"
OUT = ROOT / "docs/figures/20261030_loss_weighted_delta_pyes_slide"

PATHS = {
    "unweighted": FWD / "activation__actresp/scores.jsonl",
    "shuffled": FWD / "activation__actrespm_shuf/scores.jsonl",
    "correct": FWD / "activation__actrespm/scores.jsonl",
    "gradient": E2 / "gradient__gradient/scores.jsonl",
}
LABELS = {
    "unweighted": "Unweighted response activations",
    "shuffled": "Shuffled loss weights",
    "correct": "Correct loss weights",
    "gradient": "Gradient reader",
}
ORDER = ["unweighted", "shuffled", "correct", "gradient"]

PAPER = "#f7f6f2"
NAVY = "#112d53"
MUTED = "#566b83"
GRID = "#d9e0e5"
BLUE = "#126cc3"
GREEN = "#078467"


def load_scores(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    scores = {row["bundle_id"]: row for row in rows}
    if len(scores) != 530:
        raise ValueError(f"Expected 530 held-out bundles in {path}; got {len(scores)}")
    return scores


def calculate() -> dict:
    raw = {key: load_scores(path) for key, path in PATHS.items()}
    ids = sorted(bundle_id for bundle_id, row in raw["unweighted"].items()
                 if row["family"] == "rare")
    y = np.array([raw["unweighted"][bundle_id]["n_shows"] > 0 for bundle_id in ids])
    organisms = np.array([raw["unweighted"][bundle_id]["unit_id"] for bundle_id in ids])
    scores = {}
    for key, data in raw.items():
        if set(data) != set(raw["unweighted"]):
            raise ValueError(f"Bundle IDs differ for {key}")
        for bundle_id, label in zip(ids, y):
            row = data[bundle_id]
            if row["family"] != "rare" or (row["n_shows"] > 0) != label:
                raise ValueError(f"Labels differ for {key}: {bundle_id}")
        scores[key] = np.array([data[bundle_id]["p_yes"] for bundle_id in ids])

    units = np.unique(organisms)
    rows_by_unit = {unit: np.flatnonzero(organisms == unit) for unit in units}
    rng = np.random.default_rng(0)
    boots = {key: [] for key in ORDER}
    for _ in range(10_000):
        sample = np.concatenate([rows_by_unit[unit] for unit in
                                 rng.choice(units, size=len(units), replace=True)])
        sampled_y = y[sample]
        if sampled_y.all() or (~sampled_y).all():
            continue
        for key in ORDER:
            sampled = scores[key][sample]
            boots[key].append(100 * (sampled[sampled_y].mean() - sampled[~sampled_y].mean()))

    def ci(values: np.ndarray) -> list[float]:
        return [float(v) for v in np.percentile(values, [2.5, 97.5])]

    cells = {}
    for key, values in scores.items():
        showing = float(values[y].mean())
        not_showing = float(values[~y].mean())
        cells[key] = {
            "mean_p_yes_showing": showing,
            "mean_p_yes_not_showing": not_showing,
            "delta_pp": 100 * (showing - not_showing),
            "ci95_pp": ci(np.array(boots[key])),
        }
    contrasts = {}
    for a, b in [("correct", "shuffled"), ("correct", "gradient"),
                 ("shuffled", "unweighted")]:
        contrasts[f"{a}_minus_{b}"] = {
            "delta_pp": cells[a]["delta_pp"] - cells[b]["delta_pp"],
            "ci95_pp": ci(np.array(boots[a]) - np.array(boots[b])),
        }
    return {
        "n_bundles": len(ids),
        "n_showing": int(y.sum()),
        "n_not_showing": int((~y).sum()),
        "n_organisms": len(units),
        "bootstrap": {"unit": "organism", "draws": 10_000, "seed": 0,
                      "interval": "percentile 95%", "paired": True},
        "source_files": {key: str(path.relative_to(ROOT)) for key, path in PATHS.items()},
        "cells": cells,
        "contrasts": contrasts,
    }


def plot(stats: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(11.8, 6.4), facecolor=PAPER)
    ax = fig.add_axes([0.31, 0.25, 0.48, 0.46], facecolor=PAPER)

    fig.text(0.065, 0.94, "Correct loss weights sharpen rare-behaviour readout",
             ha="left", va="top", fontsize=22, fontfamily="DejaVu Serif",
             fontweight="bold", color=NAVY)
    fig.text(0.065, 0.855,
             "Same activation reader, 63 held-out rare bundles; gradient reader shown as reference",
             ha="left", va="top", fontsize=12, color=MUTED)
    contrast = stats["contrasts"]["correct_minus_shuffled"]
    fig.text(0.31, 0.775,
             f"Correct − shuffled: {contrast['delta_pp']:+.1f} pp "
             f"[{contrast['ci95_pp'][0]:+.1f}, {contrast['ci95_pp'][1]:+.1f}] (paired)",
             ha="left", va="top", fontsize=12.5, fontweight="bold", color=GREEN)

    ax.set_xlim(-10, 31)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks([-10, 0, 10, 20, 30])
    ax.set_xticklabels(["−10", "0", "+10", "+20", "+30"], color=MUTED)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in [-10, 10, 20, 30]:
        ax.axvline(x, color=GRID, linewidth=1, zorder=0)
    ax.axvline(0, color=NAVY, linewidth=1.35, alpha=0.62, zorder=1)
    ax.axhline(0.5, color=GRID, linewidth=1, zorder=0)

    positions = {key: 3 - i for i, key in enumerate(ORDER)}
    for key in ORDER:
        row = stats["cells"][key]
        y = positions[key]
        value = row["delta_pp"]
        lo, hi = row["ci95_pp"]
        color = BLUE if key == "gradient" else GREEN
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.6, zorder=2)
        ax.plot([lo, lo], [y - 0.07, y + 0.07], color=color, linewidth=2, zorder=2)
        ax.plot([hi, hi], [y - 0.07, y + 0.07], color=color, linewidth=2, zorder=2)
        ax.scatter(value, y, s=135 if key == "correct" else 115,
                   color=color, edgecolor=PAPER, linewidth=1.5, zorder=3)
        ax.text(-11.5, y, LABELS[key], ha="right", va="center", fontsize=12.4,
                fontweight="bold" if key == "correct" else "normal", color=NAVY)
        label = f"{value:+.0f}  [{lo:+.0f}, {hi:+.0f}]"
        ax.text(31.6, y, label, ha="left", va="center", fontsize=11.5,
                fontweight="bold" if key == "correct" else "normal", color=color)

    fig.text(0.62, 0.16, "ΔP(Yes), percentage points", ha="center", va="center",
             fontsize=12.5, color=NAVY)
    fig.text(0.065, 0.09,
             "ΔP(Yes) = mean probability on showing bundles − mean probability on non-showing bundles.",
             ha="left", va="center", fontsize=10.5, color=MUTED)
    fig.text(0.065, 0.05,
             "95% organism-bootstrap intervals · 10 organisms · one within-bundle shuffle · weighted inputs were not used in reader training",
             ha="left", va="center", fontsize=10.0, color=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor=PAPER)
    fig.savefig(OUT.with_suffix(".svg"), facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    stats = calculate()
    plot(stats)
    OUT.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Wrote {OUT.with_suffix('.png')}, {OUT.with_suffix('.svg')} and {OUT.with_suffix('.json')}")
    for key in ORDER:
        row = stats["cells"][key]
        print(f"{key}: {row['delta_pp']:+.2f} {row['ci95_pp']}")


if __name__ == "__main__":
    main()
