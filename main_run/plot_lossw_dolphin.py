#!/usr/bin/env python
"""Single talk figure: loss weighting doesn't recover the subliminal dolphin signal.

At 15 student checkpoints, the momentum-free update's batch separation d beats
loss-weighted label activations on the same batches (three-layer AO, raw).

  python main_run/plot_lossw_dolphin.py \\
    --ckpt-dir results/lossw_20261030/lossw_ckpt \\
    --mfree-jsonl <cleanbatch_full>/ao_cleanbatch_full/dolphin/ao3layer_mfree_yesno.jsonl \\
    --fig-dir docs/figures
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BLUE, ORANGE, MUTED_LINE = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
STEPS = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]


def setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
        }
    )
    return plt


def load_act_diffs(reads_path: Path, prep: str = "raw"):
    P = defaultdict(lambda: defaultdict(dict))
    with open(reads_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("student") != "dolphin" or r.get("ao") != "ao3layer":
                continue
            if r.get("prep") != prep or r["quantity"] not in ("lab", "labl"):
                continue
            P[r["quantity"]][int(r["step"])][r["batch"]] = float(r["p_yes"])
    return P


def load_update_diffs(mfree_jsonl: Path):
    M = defaultdict(dict)
    with open(mfree_jsonl) as f:
        for line in f:
            r = json.loads(line)
            batch = r["window"].split("_")[0]
            M[int(r["step"])][batch] = float(r["p_yes"])
    return M


def d_curve(P_by_step, steps=STEPS):
    return np.array(
        [
            P_by_step[t]["real"] - np.mean([P_by_step[t][b] for b in ("f1", "f2", "f3")])
            for t in steps
        ]
    )


def boot_mean_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    samples = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--mfree-jsonl", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    plt = setup()

    act = load_act_diffs(args.ckpt_dir / "reads.jsonl")
    act_c = load_act_diffs(args.ckpt_dir / "reads.jsonl", prep="centred")
    upd = load_update_diffs(args.mfree_jsonl)
    d_upd = d_curve(upd)
    d_labl = d_curve(act["labl"])
    d_lab = d_curve(act["lab"])
    gap = d_upd - d_labl
    gap_lo, gap_hi = boot_mean_ci(gap)
    gap_c = d_upd - d_curve(act_c["labl"])
    gap_c_lo, gap_c_hi = boot_mean_ci(gap_c)
    upd_pos = int((d_upd > 0).sum())

    plt.rcParams.update({"font.size": 12.5, "legend.fontsize": 12, "axes.labelsize": 13})
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.76, bottom=0.22)

    ax.axhline(0.0, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.fill_between(
        STEPS, d_labl, d_upd, where=(d_upd >= d_labl), color=BLUE, alpha=0.10,
        interpolate=True, linewidth=0, zorder=1,
    )
    ax.plot(
        STEPS, d_lab, color=MUTED_LINE, marker="^", markersize=8, linewidth=2, zorder=2,
        markeredgecolor=SURFACE, markeredgewidth=1.5, label="Unweighted activations",
    )
    ax.plot(
        STEPS, d_labl, color=ORANGE, marker="s", markersize=8, linewidth=2.4, zorder=3,
        markeredgecolor=SURFACE, markeredgewidth=1.5, label="Loss-weighted activations",
    )
    ax.plot(
        STEPS, d_upd, color=BLUE, marker="o", markersize=8.5, linewidth=2.4, zorder=4,
        markeredgecolor=SURFACE, markeredgewidth=1.5,
        label="Momentum-free update (direct write)",
    )

    ax.set_xlabel("Student checkpoint (training step)")
    ax.set_ylabel("Dolphin − clean P(Yes)")
    ax.set_xlim(-100, 7200)
    ax.set_ylim(-0.2, 0.55)
    ax.set_xticks([100, 1000, 2000, 3000, 4000, 5000, 6000, 7000])
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(MUTED)
    ax.tick_params(colors=INK2)
    # legend in plot order top-to-bottom: update, loss-weighted, unweighted
    h, l = ax.get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(0.015, 0.865), ncol=3,
               frameon=False, handlelength=2.2, columnspacing=1.6)
    ax.text(
        0.015, 0.98,
        f"Update − loss-weighted:  +{gap.mean():.2f} [{gap_lo:+.2f}, {gap_hi:+.2f}] raw,  "
        f"+{gap_c.mean():.2f} [{gap_c_lo:+.2f}, {gap_c_hi:+.2f}] mean-centred",
        transform=ax.transAxes, ha="left", va="top", fontsize=12, color=BLUE,
        fontweight="bold", linespacing=1.4,
    )

    fig.text(
        0.02, 0.965, "Loss weighting doesn’t recover the dolphin signal",
        ha="left", va="top", fontsize=19, fontweight="bold", color=INK,
    )
    fig.text(
        0.02, 0.895,
        "Same batches at 15 checkpoints of one dolphin student · three-layer Activation Oracle",
        ha="left", va="top", fontsize=12.5, color=INK2,
    )
    fig.text(
        0.02, 0.025,
        f"The update alone averages +{d_upd.mean():.3f} here (above zero at {upd_pos}/15), short of "
        "the pre-set rule (≥ 0.10 at ≥ 12/15);\nover all 71 stored steps it averages +0.217. "
        "Brackets: 95% bootstrap over checkpoints. One seed.",
        ha="left", va="bottom", fontsize=10.5, color=MUTED, linespacing=1.4,
    )

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.fig_dir / "20261030_lossw_doesnt_recover.png"
    out_json = args.fig_dir / "20261030_lossw_doesnt_recover.json"
    payload = {
        "title": "Loss weighting doesn’t recover the dolphin signal",
        "panel": "single",
        "steps": STEPS,
        "d_update_mfree": [float(x) for x in d_upd],
        "d_labl_raw": [float(x) for x in d_labl],
        "d_lab_raw": [float(x) for x in d_lab],
        "update_minus_labl_centred": {
            "mean": float(gap_c.mean()),
            "ci95": [gap_c_lo, gap_c_hi],
        },
        "update_minus_labl": {
            "mean": float(gap.mean()),
            "ci95": [gap_lo, gap_hi],
            "per_step": [float(x) for x in gap],
        },
        "sources": {
            "ckpt_reads": str(args.ckpt_dir / "reads.jsonl"),
            "mfree": str(args.mfree_jsonl),
            "results_note": "docs/notes/20261030_lossw_dolphin_results.md",
        },
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    fig.savefig(out_png, dpi=200)
    print("wrote", out_png)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
