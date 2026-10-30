#!/usr/bin/env python
"""Synthesis talk figure: share of the gradient's lead that loss-weighting activations closes.

Visible behaviours: Experiment 1 linear probes (`share_of_G_closed`, "forward surprise weighting"
rows of the fwd-baselines probe contrasts). Subliminal dolphin: 1 − mean(update − loss-weighted) /
mean(update − unweighted) over the 15 student checkpoints (three-layer AO, raw).

  python main_run/plot_lead_closed.py \\
    --fwd ../agm-fwd-baselines/docs/figures/20261028_fwd_baselines_probe_contrasts.json \\
    --dolphin docs/figures/20261030_lossw_doesnt_recover.json \\
    --fig-dir docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

# (task, label) for the visible-behaviour rows, top to bottom
VISIBLE = [
    ("P3_rare", "Is a rare behaviour present?"),
    ("P3", "Is the behaviour present?"),
    ("P1", "Which behaviour is it?"),
]


def load(fwd: Path, dolphin: Path, n_boot: int = 5000) -> list[dict]:
    rows = []
    prim = json.loads(fwd.read_text())["primary"]
    for task, label in VISIBLE:
        r = next(x for x in prim if x["task"] == task and x["explanation"] == "forward surprise weighting")
        lo, hi = (1 - c / r["G"] for c in reversed(r["ci95"]))  # G held fixed
        rows.append(dict(label=label, group="visible", share=r["share_of_G_closed"], ci=[lo, hi],
                         lead_before=r["G"], lead_after=r["diff"], unit="probe AUROC"))

    d = json.loads(dolphin.read_text())
    u, lw, la = (np.array(d[k]) for k in ("d_update_mfree", "d_labl_raw", "d_lab_raw"))
    before, after = u - la, u - lw
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(u), len(u))
        boots.append(1 - after[i].mean() / before[i].mean())
    rows.append(dict(label="Dolphin preference (subliminal)", group="subliminal",
                     share=float(1 - after.mean() / before.mean()),
                     ci=[float(x) for x in np.percentile(boots, [2.5, 97.5])],
                     lead_before=float(before.mean()), lead_after=float(after.mean()),
                     unit="three-layer AO, dolphin − clean P(Yes)"))
    return rows


def pct(x: float) -> str:
    return f"{100 * x:+.0f}%".replace("-", "\u2212")


def plot(rows: list[dict], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r["group"] != rows[i - 1]["group"]:
            y += 0.7
        ys.append(y)
        y += 1
    for r, yy in zip(rows, ys):
        v = 100 * r["share"]
        color = BLUE if r["group"] == "visible" else ORANGE
        ax.barh(yy, v, height=0.6, color=color, zorder=3)
        txt = pct(r["share"])
        if v >= 0:
            ax.text(v + 2, yy, txt, va="center", ha="left", fontsize=14, fontweight="bold", color=INK)
        else:
            ax.text(v - 2, yy, txt, va="center", ha="right", fontsize=14, fontweight="bold", color=INK)

    ax.axvline(0, color=INK2, lw=1.2, zorder=4)
    ax.axvline(100, color=MUTED, lw=1.2, ls="--", zorder=2)
    ax.text(99, ys[-1] + 0.62, "fully explained", ha="right", va="center", fontsize=11, color=MUTED)
    ax.text(-2, ys[-1] + 0.62, "gap widens", ha="right", va="center", fontsize=11, color=MUTED)
    ax.text(2, ys[-1] + 0.62, "gap shrinks", ha="left", va="center", fontsize=11, color=MUTED)

    ax.set_yticks(ys, [r["label"] for r in rows], color=INK)
    ax.invert_yaxis()
    ax.set_ylim(ys[-1] + 0.95, ys[0] - 0.55)
    ax.set_xlim(-80, 110)
    ax.set_xticks([-50, 0, 50, 100], ["−50%", "0%", "50%", "100%"])
    ax.set_xlabel("Share of the gradient’s lead closed by loss-weighting the activations", color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors=INK2, length=0)

    fig.text(0.02, 0.965, "When does the gradient add something?",
             fontsize=19, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.895, "Loss weighting explains most of its edge at spotting visible behaviours,\n"
             "but not at naming them, and not for subliminal learning",
             fontsize=13, color=INK2, ha="left", va="top", linespacing=1.4)
    fig.text(0.02, 0.025,
             "Visible behaviours: linear probes on held-out bundles (Experiment 1). "
             "Dolphin: three-layer AO on the same batches at 15 checkpoints.\n"
             "95% intervals: " + ", ".join(
                 f"{n} {pct(r['ci'][0])[:-1]} to {pct(r['ci'][1])}"
                 for n, r in zip(("rare", "present", "which", "dolphin"), rows)
             ) + ". One dolphin seed.",
             fontsize=10, color=MUTED, ha="left", va="bottom", linespacing=1.45)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.76, bottom=0.23)

    fig_dir.mkdir(parents=True, exist_ok=True)
    stem = fig_dir / "20261030_lead_closed"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor=SURFACE)
    Path(f"{stem}.json").write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    print(f"wrote {stem}.png/.svg/.json")
    for r in rows:
        print(f"  {r['label']}: {pct(r['share'])} [{pct(r['ci'][0])}, {pct(r['ci'][1])}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", type=Path,
                    default=Path("docs/figures/20261028_fwd_baselines_probe_contrasts.json"))
    ap.add_argument("--dolphin", type=Path, default=Path("docs/figures/20261030_lossw_doesnt_recover.json"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    a = ap.parse_args()
    plot(load(a.fwd, a.dolphin), a.fig_dir)


if __name__ == "__main__":
    main()
