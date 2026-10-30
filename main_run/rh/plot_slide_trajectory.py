#!/usr/bin/env python
"""LoRAcles-Fig-8-style talk figure: share of our reader's open-question answers that mention tests, per RL step,
hacker vs anti-hacker, with 95% Wilson bands, shaded where the hacker is significantly higher (one-sided Fisher,
alpha 0.05), and the hacker's rollout hack rate as a reference line.

    python main_run/rh/plot_slide_trajectory.py [--input dwx|ckptdiff]

Reads docs/rh_results/test_mentions_by_step.json (keyword rule `TEST_RE` in main_run/rh/test_mentions.py).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

HERE = Path(__file__).resolve().parents[2]
MENTIONS = HERE / "docs" / "rh_results" / "test_mentions_by_step.json"
OUT = HERE / "docs" / "figures" / "20261030_rh_trajectory"
INPUT_NAME = {"dwx": "direct writes", "ckptdiff": "activation change"}

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=sorted(INPUT_NAME), default="ckptdiff")
    a = ap.parse_args()
    cell = json.loads(MENTIONS.read_text(encoding="utf-8"))["cells"][f"{a.input}_behavioural"]
    hk, an = cell["rh_hacker"], cell["rh_anti"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13)

    # shade steps where the hacker's rate is significantly above the anti-hacker's
    steps_h = hk["steps"]
    anti = dict(zip(an["steps"], zip(an["mentions"], an["n"])))
    sig = []
    for i, (s, k, n) in enumerate(zip(steps_h, hk["mentions"], hk["n"])):
        if s not in anti:
            continue
        ka, na = anti[s]
        p = fisher_exact([[k, n - k], [ka, na - ka]], alternative="greater")[1]
        if p < 0.05:
            lo = (steps_h[i - 1] + s) / 2 if i else s - 2.5
            hi = (steps_h[i + 1] + s) / 2 if i + 1 < len(steps_h) else s + 2.5
            ax.axvspan(lo, hi, color=BLUE, alpha=0.07, lw=0, zorder=0)
            sig.append(s)

    for run, c, mk, lab in ((hk, BLUE, "o", "Hacker: reader mentions tests"),
                            (an, ORANGE, "s", "Anti-hacker: reader mentions tests")):
        x = run["steps"]
        y = [100 * k / n for k, n in zip(run["mentions"], run["n"])]
        band = [wilson(k, n) for k, n in zip(run["mentions"], run["n"])]
        ax.fill_between(x, [100 * b[0] for b in band], [100 * b[1] for b in band], color=c, alpha=0.15, lw=0,
                        zorder=1)
        ax.plot(x, y, color=c, lw=2.6, marker=mk, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3, label=lab)
    hr = [100 * h for h in hk["hack_rate"]]
    ax.plot(steps_h, hr, color=INK, lw=1.8, ls="--", zorder=2, label="Hacker’s hack rate (its rollouts)")
    ax.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.4)

    ax.set_xlim(0, steps_h[-1] + 2)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("RL training step", color=INK2)
    ax.set_ylabel("Share of answers", color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2)

    fig.text(0.02, 0.965, "Our reader tracks the tests the hacker learns to game",
             fontsize=18, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.895, f"Open question on each checkpoint’s {INPUT_NAME[a.input]}, 100 answers per step. "
             "Shaded: hacker significantly higher.\nIt names the tests, but never says they are being changed.",
             fontsize=12, color=INK2, ha="left", va="top", linespacing=1.4)

    stem = Path(f"{OUT}_{a.input}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor=SURFACE)
    Path(f"{stem}.json").write_text(json.dumps({"input": a.input, "significant_steps": sig,
                                                "spearman_rho": hk.get("spearman_rho")}, indent=1) + "\n")
    print(f"wrote {stem}.png  significant steps {sig}  rho {hk.get('spearman_rho')}")


if __name__ == "__main__":
    main()
