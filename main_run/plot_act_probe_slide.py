#!/usr/bin/env python
"""One-panel talk figure for the dolphin activation probe: text vs activations vs gradients, on the main teacher
contrast (dolphin vs owl, same generator) and the generator-only contrast (two dolphin datasets).

    python main_run/plot_act_probe_slide.py [--data docs/figures/20261028_act_probe.json] [--fig-dir docs/figures]

Pair-centred ridge, all 8 layers, response positions (the plan's primary probe).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

ROWS = [
    ("dolphin_ours vs owl", "Which teacher?\ndolphin vs owl"),
    ("dolphin_ours vs dolphin", "Which pipeline?\ntwo dolphin datasets"),
]
INPUTS = [("text", "Numbers in the text", GREY, "D"), ("act", "Activations", BLUE, "o"),
          ("grad", "Gradients", ORANGE, "s")]


def load(path: Path) -> list[dict]:
    out = json.loads(path.read_text())["out"]
    rows = []
    for key, label in ROWS:
        rid = {r["set"]: r for r in out[key]["ridge"] if r["prep"] == "pc" and r["layer"] == "all"}
        d2 = out[key]["diffs"]["D2"]
        rows.append(dict(label=label, contrast=key,
                         auroc={s: (rid[s]["auroc"], rid[s]["ci95"]) for s, *_ in INPUTS},
                         gain=d2["diff"], gain_ci=d2["ci95"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("docs/figures/20261028_act_probe.json"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    a = ap.parse_args()
    rows = load(a.data)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.25, right=0.72, top=0.72, bottom=0.15)

    ax.axvline(0.5, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(0.505, -0.47, "chance", fontsize=11, color=MUTED, va="top")
    offs = (-0.24, 0.0, 0.24)
    for i, r in enumerate(rows):
        for (s, _, c, mk), dy in zip(INPUTS, offs):
            v, (lo, hi) = r["auroc"][s]
            ax.plot([lo, hi], [i + dy, i + dy], color=c, lw=2.6, solid_capstyle="round", zorder=2)
            ax.plot([v], [i + dy], marker=mk, markersize=10, color=c, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3)
            ax.text(hi + 0.008, i + dy, f"{v:.2f}", va="center", ha="left", fontsize=12, color=INK)
        ax.text(1.04, i, f"Gradients add\n+{r['gain']:.3f} to activations\n"
                f"[+{r['gain_ci'][0]:.3f}, +{r['gain_ci'][1]:.3f}]",
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=12, color=INK,
                linespacing=1.3)

    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], color=INK, linespacing=1.3)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.45, -0.5)
    ax.set_xlim(0.48, 0.9)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Probe accuracy (AUROC) on held-out answers", color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors=INK2)

    fig.text(0.02, 0.965, "Gradients tell the teachers apart best, but not because of the dolphin",
             fontsize=17, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.895, "Linear probe on single answers from the subliminal number data (base Qwen3-14B, "
             "8 layers), 95% intervals",
             fontsize=12, color=INK2, ha="left", va="top")
    x = 0.02
    for _, name, c, mk in INPUTS:
        fig.lines.append(plt.Line2D([x + 0.008], [0.815], marker=mk, markersize=10, color=c,
                                    transform=fig.transFigure))
        fig.text(x + 0.022, 0.815, name, fontsize=12.5, color=INK, va="center", ha="left")
        x += 0.07 + 0.0105 * len(name)

    a.fig_dir.mkdir(parents=True, exist_ok=True)
    stem = a.fig_dir / "20261030_act_probe_slide"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor=SURFACE)
    Path(f"{stem}.json").write_text(json.dumps({"rows": rows, "source": str(a.data)}, indent=1) + "\n")
    print(f"wrote {stem}.png/.svg/.json")


if __name__ == "__main__":
    main()
