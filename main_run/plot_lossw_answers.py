#!/usr/bin/env python
"""Single talk figure: dolphin answers per input vector, three-layer AO, preference question.

Training-side vectors (fitted bias updates) get dolphin answers; the activation difference of
the same documents gets 0/100 under every weighting.

  python main_run/plot_lossw_answers.py \\
    --fitvec docs/figures/20261028_fitvec_summary.json \\
    --lossw results/lossw_20261030/lossw/summary.json \\
    --fig-dir docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
AO, Q = "ao3layer", "preference"

# (label, source, key); source "fit" = fitted-vector run, "lw" = loss-weighting run
ROWS = [
    ("Fitted update (SGD)", "fit", "fit_dolphin_sgd"),
    ("Fitted update (Adam)", "fit", "fit_dolphin_adam"),
    ("Activations, unweighted", "lw", "all"),
    ("Activations, response tokens", "lw", "resp"),
    ("Activations, token-loss weighted", "lw", "tokl"),
    ("Activations, document-loss weighted", "lw", "docl"),
    ("Activations, shuffled-loss control", "lw", "tokshuf"),
]


def load(fitvec: Path, lossw: Path) -> list[dict]:
    fit = {(r["x"], r["ao"], r["question"]): r for r in json.loads(fitvec.read_text())["ao_pairs"]}
    lw = json.loads(lossw.read_text())["part_a"]["ao_counts"]
    out = []
    for label, src, key in ROWS:
        if src == "fit":
            r = fit[(key, AO, Q)]
            out.append(dict(label=label, kind="update", dolphin=r["x_dolphin"], n=r["n"],
                            control=r["control_dolphin"]))
        else:
            dol, _, n = lw[f"{key}_dolphin_minus_clean|{AO}|{Q}"]
            out.append(dict(label=label, kind="activation", dolphin=dol, n=n))
    return out


def plot(rows: list[dict], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # one blank slot between the two groups
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r["kind"] != rows[i - 1]["kind"]:
            y += 0.6
        ys.append(y)
        y += 1
    for r, yy in zip(rows, ys):
        color = BLUE if r["kind"] == "update" else ORANGE
        if r["dolphin"] > 0:
            ax.barh(yy, r["dolphin"], height=0.62, color=color, zorder=3)
        else:  # a zero bar is invisible; mark it at the baseline
            ax.plot([0], [yy], marker="|", markersize=18, mew=3, color=color, zorder=3)
        ax.text(r["dolphin"] + 1.0, yy, f"{r['dolphin']}", va="center", ha="left",
                color=INK, fontsize=14, fontweight="bold")

    ax.set_yticks(ys, [r["label"] for r in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 50)
    ax.set_xlabel("Dolphin answers out of 100", color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors=INK2)

    fig.text(0.02, 0.965, "Loss weighting doesn’t recover the dolphin signal", fontsize=19,
             fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.895, "Three-layer Activation Oracle asked for its favourite animal · "
             "dolphin − clean vectors, Qwen3-14B base", fontsize=12.5, color=INK2,
             ha="left", va="top")
    fig.text(0.02, 0.03, "Controls (clean-data fits; owl − clean activations): 0/100 for every vector.\n"
             "Fitted update = one bias vector per layer, trained on dolphin-teacher numbers.", fontsize=10.5, color=MUTED,
             ha="left", va="bottom", linespacing=1.4)
    fig.subplots_adjust(left=0.38, right=0.96, top=0.80, bottom=0.21)

    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = fig_dir / "20261030_lossw_answers"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor=SURFACE)
    Path(f"{stem}.json").write_text(json.dumps(
        {"ao": AO, "question": Q, "rows": rows}, indent=1))
    print(f"wrote {stem}.png/.svg/.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitvec", type=Path, default=Path("docs/figures/20261028_fitvec_summary.json"))
    ap.add_argument("--lossw", type=Path, default=Path("results/lossw_20261030/lossw/summary.json"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    a = ap.parse_args()
    plot(load(a.fitvec, a.lossw), a.fig_dir)


if __name__ == "__main__":
    main()
