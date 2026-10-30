#!/usr/bin/env python
"""Main figure for the RH reading with the behaviour readers :
no reader names tampering, and the RL behaviour reader's test mentions track the hack rate
like the old reader's.

One panel per reader (old reader rl_paper_c040, SFT behaviour reader beh_sft_7082, RL behaviour reader
beh_rl_c040), all reading the checkpoint difference (Δh), behavioural question, 100 answers per read step. x = RL step;
y = share of the reader's answers that mention tests (test_mentions.py pattern; exploratory, post hoc) for the hacker
and anti-hacker runs, with the hacker's rollout hack rate. The text box gives the Spearman rho of the hacker's mention
share with its hack rate, and the pre-registered primary cell: answers judged as naming tampering (rh_tamper judge)
at steps 43/45/47 pooled, out of 300.

Sources (defaults below):
  old reader       docs/rh_results/test_mentions_by_step.json, docs/rh_results/rh_summary.json
  behaviour        $RH_BEH_RUNS/test_mentions_<reader>.json, $RH_BEH_RUNS/summary_primary_<reader>.json

  python main_run/plot_rh_beh_readers.py [--out docs/figures/20261029_rh_beh_readers.png]
Writes the PNG and <out>.json (the plotted series, rho and counts).
"""
import argparse
import json
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "docs" / "rh_results"
BEH = Path(os.environ.get("RH_BEH_RUNS", ROOT / "artifacts" / "rh_beh_runs"))  # behaviour-reader summaries

BLUE, ORANGE = "#2a78d6", "#eb6834"
HACK = "#52514e"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e0", "#f1f0ec"
SURFACE = "#ffffff"
CELL = "ckptdiff_behavioural"
PRIMARY = (43, 45, 47)

# (key, panel title, mentions file, primary summary file)
READERS = [
    ("rl_paper_c040", "old reader", OLD / "test_mentions_by_step.json", OLD / "rh_summary.json"),
    ("beh_sft_7082", "SFT behaviour reader", BEH / "test_mentions_beh_sft_7082.json",
     BEH / "summary_primary_beh_sft_7082.json"),
    ("beh_rl_c040", "RL behaviour reader", BEH / "test_mentions_beh_rl_c040.json",
     BEH / "summary_primary_beh_rl_c040.json"),
]


def collect():
    out = []
    for key, title, mfile, sfile in READERS:
        cell = json.loads(mfile.read_text())["cells"][CELL]
        prim = json.loads(sfile.read_text())["cells"]["dh_beh"]["dropped"]
        runs = {}
        for run in ("rh_hacker", "rh_anti"):
            c = cell[run]
            runs[run] = {"steps": c["steps"], "share": [m / n for m, n in zip(c["mentions"], c["n"])],
                         "mentions": c["mentions"], "n": c["n"], "hack_rate": c["hack_rate"],
                         "rho": c["spearman_rho"], "p": c["spearman_p"],
                         "mentions_primary": sum(m for s, m in zip(c["steps"], c["mentions"]) if s in PRIMARY)}
        out.append({"reader": key, "title": title, "mentions_file": str(mfile), "summary_file": str(sfile),
                    "tamper_yes_hacker": prim["hacker"], "tamper_yes_anti": prim["anti"], **runs})
    return out


def plot(panels, out):
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK})
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.0), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, p in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.axvspan(PRIMARY[0] - 1.5, PRIMARY[-1] + 1.5, color=BAND, lw=0, zorder=0)
        h, a = p["rh_hacker"], p["rh_anti"]
        ax.plot(h["steps"], h["hack_rate"], color=HACK, lw=1.6, ls=(0, (4, 3)), zorder=2)
        ax.plot(a["steps"], a["share"], color=ORANGE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1, zorder=3)
        ax.plot(h["steps"], h["share"], color=BLUE, lw=2.2, marker="o", ms=5.5, mec=SURFACE, mew=1, zorder=4)
        ty, tn = p["tamper_yes_hacker"]
        ay, an = p["tamper_yes_anti"]
        ax.text(0.03, 0.97,
                f"test mentions vs hack rate: ρ = {h['rho']:.2f}\n"
                f"test mentions, steps 43–47: {h['mentions_primary']} vs {a['mentions_primary']} of "
                f"{sum(n for s, n in zip(h['steps'], h['n']) if s in PRIMARY)}\n"
                f"answers naming tampering: {ty} / {tn} vs {ay} / {an}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=INK, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=GRID, lw=0.8), zorder=6)
        ax.set_title(p["title"], fontsize=11.5, color=INK, loc="left")
        ax.set_ylim(0, 1.36)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlim(0, 95)
        ax.set_xlabel("RL step")
        ax.grid(color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("share")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    fig.suptitle("No reader names tampering; the RL behaviour reader's test mentions track the hack rate like the old "
                 "reader's", fontsize=13.5, color=INK, x=0.015, ha="left", y=0.985)
    handles = [
        Line2D([], [], color=BLUE, lw=2.2, marker="o", ms=5.5, mec=SURFACE,
               label="hacker run: reader answers mentioning tests"),
        Line2D([], [], color=ORANGE, lw=2, marker="o", ms=5, mec=SURFACE,
               label="anti-hacker run: reader answers mentioning tests"),
        Line2D([], [], color=HACK, lw=1.6, ls=(0, (4, 3)), label="hacker run: rollout hack rate"),
        plt.Rectangle((0, 0), 1, 1, fc=BAND, ec="none", label="steps 43–47 (pre-registered primary steps)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.05, 0.1), ncol=2, frameon=False,
               fontsize=10, handletextpad=0.5, columnspacing=2.2, labelcolor=INK)
    fig.text(0.015, 0.015, textwrap.fill(
             "Reader input: checkpoint difference (Δh), behavioural question, 100 answers per read step. "
             "\"Answers naming tampering\" is the pre-registered primary cell (rh_tamper judge, steps 43/45/47 pooled, "
             "hacker vs anti-hacker). Test mentions are exploratory and post hoc (keyword pattern chosen after "
             "reading answers); they are mostly passive (\"absorbed a set of test cases\"). The hack rate counts any "
             "rewrite of the test block, so it is a loose proxy for tampering.", 215),
             fontsize=8.6, color=INK2, va="bottom", ha="left", linespacing=1.45)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.33, wspace=0.06)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "20261029_rh_beh_readers.png")
    args = ap.parse_args()
    panels = collect()
    plot(panels, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({"cell": CELL, "panels": panels}, indent=1) + "\n")
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
