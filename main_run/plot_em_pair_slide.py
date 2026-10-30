#!/usr/bin/env python
"""One-panel talk figure: the finished models' gradients separate broadly from narrowly misaligned models at every
layer, but the reader's descriptions of the same vectors do not.

Input: −g of each finished model on the aligned answers to the EM questions (`g_evalq`). Per layer, the least similar
broad–broad pair and the most similar broad–narrow pair. Callout: the pairwise judge on the reader's descriptions of
the same input, pooled over the three broad models and both questions.

  python main_run/plot_em_pair_slide.py --em-runs <em_runs> --out-dir docs/figures
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from em.pair_input_check import cos, inputs  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BROADS = ["broad_celeste", "broad_s42", "broad_s1"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--em-runs", type=Path, required=True, help="EM runs root (em_s42, em_s1, ext_*)")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    a = ap.parse_args()

    v = inputs(a.em_runs)["g_evalq"]
    L = v["narrow"].shape[0]
    bb = [min(cos(v[x][l], v[y][l]) for x, y in itertools.combinations(BROADS, 2)) for l in range(L)]
    bn = [max(cos(v[x][l], v["narrow"][l]) for x in BROADS) for l in range(L)]
    sep = sum(b > n for b, n in zip(bb, bn))

    cells = [c for c in json.loads((a.em_runs / "pairwise_judge.json").read_text())["cells"]
             if c["input"] == "g_evalq"]
    k, n = sum(c["correct"] for c in cells), sum(c["n"] for c in cells)
    lo, hi = wilson(k, n)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.09, right=0.66, top=0.80, bottom=0.13)

    xs = list(range(L))
    ax.fill_between(xs, bn, bb, color=BLUE, alpha=0.08, lw=0, zorder=1)
    ax.plot(xs, bb, color=BLUE, lw=2.6, zorder=3)
    ax.plot(xs, bn, color=ORANGE, lw=2.6, zorder=3)
    ax.axhline(0, color=MUTED, lw=1, zorder=2)

    # direct labels at the line ends
    ax.text(L - 0.3, bb[-1], f"  Broad vs broad\n  ({sorted(bb)[L // 2]:.2f} median)", color=INK, fontsize=12.5,
            va="center", ha="left", linespacing=1.25, clip_on=False)
    ax.text(L - 0.3, bn[-1], f"  Broad vs narrow\n  ({sorted(bn)[L // 2]:.2f} median)", color=INK, fontsize=12.5,
            va="center", ha="left", linespacing=1.25, clip_on=False)
    for y, c in ((bb[-1], BLUE), (bn[-1], ORANGE)):
        ax.plot([L - 1], [y], marker="o", markersize=7, color=c, zorder=4, clip_on=False)

    ax.set_xlim(0, L - 1)
    ax.set_ylim(-0.2, 1.05)
    ax.set_xticks([0, 10, 20, 30, 39])
    ax.set_xlabel("Layer", color=INK2)
    ax.set_ylabel("Cosine between the models’ gradients", color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2)

    fig.text(0.02, 0.965, "The gradient tells broad from narrow EM apart; the reader doesn’t",
             fontsize=17, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.895, "Finished models’ gradients on aligned answers to unrelated questions (3 broad models, 1 narrow)",
             fontsize=12, color=INK2, ha="left", va="top")
    ax.text(2, (bb[8] + bn[8]) / 2, f"separated at\nall {sep} layers" if sep == L else f"separated at\n{sep}/{L} layers",
            fontsize=13, color=INK2, va="center", ha="left", linespacing=1.3)
    fig.text(0.695, 0.21,
             f"Reader’s descriptions of\nthe same gradients: a judge\npicks the broad model\n{100 * k / n:.0f}% "
             f"of the time (chance 50%)",
             fontsize=12, color=INK, ha="left", va="center", linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor=MUTED, lw=1))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = a.out_dir / "20261030_em_pair_slide"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor=SURFACE)
    Path(f"{stem}.json").write_text(json.dumps({
        "input": "g_evalq", "broad_broad_min_per_layer": bb, "broad_narrow_max_per_layer": bn,
        "layers_separated": sep, "judge_correct": k, "judge_n": n, "judge_wilson95": [lo, hi],
    }, indent=1) + "\n")
    print(f"wrote {stem}.png  separated {sep}/{L}  judge {k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
