#!/usr/bin/env python
"""One-panel talk figure for the AO input-preparation sweep: on "is the behaviour present?", tuning how the gradient
is fed to the public AOs lifts them from their default setting to their best, but not to our gradient reader.

Default = −g raw ×1 (the earlier reads), the better of the two AOs. Best = maximum over every gradient preparation of
both AOs (post hoc, so optimistic for the AO). Reader values from the rare yes/no note, as in the main sweep figure.

    python main_run/plot_ao_prep_sweep_slide.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
SUMMARY = HERE / "docs" / "figures" / "20261029_ao_prep_sweep_summary.json"
MAIN = HERE / "docs" / "figures" / "20261029_ao_prep_sweep_main_data.json"
OUT = HERE / "docs" / "figures" / "20261030_ao_prep_sweep_slide"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
FAMILIES = [("quirk", "Quirk"), ("rare", "Rare"), ("backdoor", "Backdoor\n(only 5 negatives)")]
DEFAULTS = ("ao3layer_grad_all", "aopaper_grad_all")


def main() -> None:
    settings = json.loads(SUMMARY.read_text())["settings"]
    reader = json.loads(MAIN.read_text())["reader_reference"]["detection"]
    grad = {k: v for k, v in settings.items() if v["input"] != "act_centred" and v["input"] != "act"}
    rows = []
    for fam, label in FAMILIES:
        key = f"bundle.{fam}.pos_vs_neg"
        d_run = max(DEFAULTS, key=lambda r: settings[r]["results"][key]["auroc"])
        b_run = max(grad, key=lambda r: grad[r]["results"][key]["auroc"])
        rows.append(dict(family=fam, label=label,
                         default=settings[d_run]["results"][key]["auroc"], default_run=d_run,
                         best=grad[b_run]["results"][key]["auroc"], best_run=b_run,
                         best_ci=[grad[b_run]["results"][key]["lo"], grad[b_run]["results"][key]["hi"]],
                         reader=reader[fam]["auroc"], reader_ci=[reader[fam]["lo"], reader[fam]["hi"]]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.20, right=0.96, top=0.70, bottom=0.14)

    ax.axvline(0.5, color=MUTED, lw=1.2, ls="--", zorder=1)
    ax.text(0.505, -0.45, "chance", fontsize=11, color=MUTED, va="top")
    for i, r in enumerate(rows):
        y_ao, y_rd = i - 0.14, i + 0.18
        lo, hi = r["best_ci"]
        ax.plot([lo, hi], [y_ao, y_ao], color=ORANGE, lw=1.6, alpha=0.45, zorder=2)
        if r["best"] - r["default"] > 0.015:
            ax.annotate("", xy=(r["best"] - 0.008, y_ao), xytext=(r["default"] + 0.008, y_ao),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8, mutation_scale=14), zorder=3)
        ax.plot([r["default"]], [y_ao], marker="o", markersize=10, markerfacecolor=SURFACE,
                markeredgecolor=ORANGE, markeredgewidth=2, zorder=4)
        ax.plot([r["best"]], [y_ao], marker="o", markersize=11, color=ORANGE, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=4)
        ax.text(r["best"], y_ao - 0.2, f"{r['best']:.2f}", ha="center", va="bottom", fontsize=12, color=INK)
        if r["reader_ci"][0] is not None:
            ax.plot(r["reader_ci"], [y_rd, y_rd], color=BLUE, lw=1.6, alpha=0.45, zorder=2)
        ax.plot([r["reader"]], [y_rd], marker="D", markersize=11, color=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=4)
        ax.text(r["reader"], y_rd + 0.13, f"{r['reader']:.2f}", ha="center", va="top", fontsize=12, color=INK,
                fontweight="bold")

    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], color=INK, linespacing=1.3)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.5, -0.55)
    ax.set_xlim(0.45, 1.02)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Is the behaviour present? AUROC on held-out bundles", color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors=INK2)

    n_grad = len(grad)
    fig.text(0.02, 0.965, "Tuning the AO’s input helps, but doesn’t reach our reader",
             fontsize=18, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.895, f"Both public AOs fed the gradient {n_grad} ways (sign, centring, strength, "
             "adding an activation); best one shown",
             fontsize=12, color=INK2, ha="left", va="top")
    items = [("o", dict(markerfacecolor=SURFACE, markeredgecolor=ORANGE, markeredgewidth=2), "AO, default input"),
             ("o", dict(color=ORANGE), "AO, best input"),
             ("D", dict(color=BLUE), "Our gradient reader")]
    x = 0.02
    for mk, kw, name in items:
        fig.lines.append(plt.Line2D([x + 0.008], [0.815], marker=mk, markersize=10, transform=fig.transFigure,
                                    linestyle="none", **kw))
        fig.text(x + 0.022, 0.815, name, fontsize=12.5, color=INK, va="center", ha="left")
        x += 0.075 + 0.0105 * len(name)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, facecolor=SURFACE)
    Path(f"{OUT}.json").write_text(json.dumps({"gradient_settings": n_grad, "rows": rows}, indent=1) + "\n")
    print(f"wrote {OUT}.png  ({n_grad} gradient settings)")
    for r in rows:
        print(f"  {r['family']}: default {r['default']:.3f} ({r['default_run']}) -> best {r['best']:.3f} "
              f"({r['best_run']}) vs reader {r['reader']:.3f}")


if __name__ == "__main__":
    main()
