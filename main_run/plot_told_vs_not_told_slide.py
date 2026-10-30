"""Slide figure: told what to look for, the black box is earlier; not told, the reader is.

Dolphin student, new setting (one seed), behaviour onset 3,000. Each row compares the first detection step of a
test on the student's sampled answers (black box) with the reader on the direct writes, under the same knowledge
of the target. Every step is read from the committed figure data:

  docs/figures/20261029_blackbox_timing.json   (results/blackbox/blackbox_timing.json)
    told,     black box  rows[dolphin].blackbox_told                 Fisher vs clean control, 5,000 answers / 500 steps
    told,     reader     rows[dolphin].reader_first_signal.direct_writes   judged YES above the control, twice
    not told, black box  rows[dolphin].blackbox_not_told             dolphin is the top-rising first word, twice
  docs/figures/20261029_candidate_ranks.json   (results/candidates/subliminal_rank.json)
    not told, reader     panels[dolphin].direct_writes.detect_top1    rank 1 of 100 animals, twice (100-step grid)
                         panels[dolphin].fixed_document_detect_top1   fixed-document variant (footnote only)

The whale annotation (the black box's top-rising animal from 1,500 to 3,500) and the whale placebo (the told test
also fires for whale at 500) are from the black-box timing results.

Run from the repository root:
    python main_run/plot_told_vs_not_told_slide.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BLACKBOX = ROOT / "docs" / "figures" / "20261029_blackbox_timing.json"
RANKS = ROOT / "docs" / "figures" / "20261029_candidate_ranks.json"
OUTPUT = ROOT / "docs" / "figures" / "20261030_told_vs_not_told_slide"
RUN = "newset_dolphin_s1"

PAPER = "#f7f6f2"
NAVY = "#112d53"
MUTED = "#566b83"
GRID = "#d9e0e5"
ONSET = "#8a8984"
BOX = "#d4561f"   # black box
READER = "#2a78d6"
XMAX = 4200


def load():
    bb = next(r for r in json.loads(BLACKBOX.read_text())["rows"] if r["run"] == RUN)
    cr = next(p for p in json.loads(RANKS.read_text())["panels"] if p["run"] == RUN)
    assert bb["onset"] == cr["onset"]
    return {
        "onset": bb["onset"],
        "told": (bb["blackbox_told"], bb["reader_first_signal"]["direct_writes"]),
        "not_told": (bb["blackbox_not_told"], cr["direct_writes"]["detect_top1"]),
        "fixed_variant": cr["fixed_document_detect_top1"],
        "placebo_whale": json.loads(BLACKBOX.read_text())["placebo_whale_fisher_ctrl_full"][RUN],
    }


def main() -> None:
    d = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(11.8, 6.6), facecolor=PAPER)
    ax = fig.add_axes([0.235, 0.24, 0.53, 0.47], facecolor=PAPER)

    fig.text(0.04, 0.95, "Told what to look for, the black box is earlier;\nnot told, the reader is",
             ha="left", va="top", fontsize=20.3, fontfamily="DejaVu Serif", fontweight="bold", color=NAVY,
             linespacing=1.15)
    fig.text(0.04, 0.835, f"Dolphin student: first training step at which each method detects dolphin "
             f"(behaviour onset {d['onset']:,})", ha="left", va="top", fontsize=12.5, color=MUTED)

    ax.set_xlim(0, XMAX)
    ax.set_ylim(0.35, 2.65)
    ticks = range(0, 4001, 1000)
    ax.set_xticks(list(ticks), [f"{t:,}" for t in ticks], color=MUTED)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for t in ticks:
        ax.axvline(t, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("training step", color=NAVY, fontsize=12.2, labelpad=8)

    # behaviour onset
    ax.axvline(d["onset"], color=ONSET, lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.text(d["onset"] + 50, 2.62, "behaviour onset", ha="left", va="top", fontsize=10.5, color=MUTED)

    rows = [
        (2.0, "Told the animal", "test checks for “dolphin”", d["told"]),
        (1.0, "Not told", "pick the animal out of 100", d["not_told"]),
    ]
    for y, head, sub, (box, reader) in rows:
        fig.text(0.215, ax.transData.transform((0, y))[1] / fig.dpi / fig.get_figheight() + 0.022, head,
                 ha="right", va="center", fontsize=13.6, fontweight="bold", color=NAVY)
        fig.text(0.215, ax.transData.transform((0, y))[1] / fig.dpi / fig.get_figheight() - 0.022, sub,
                 ha="right", va="center", fontsize=10.8, color=MUTED)
        ax.plot([min(box, reader), max(box, reader)], [y, y], color=GRID, lw=5, solid_capstyle="round", zorder=2)
        ax.plot(box, y, marker="D", ms=13, color=BOX, mec=PAPER, mew=2, ls="none", zorder=4)
        ax.plot(reader, y, marker="o", ms=14, color=READER, mec=PAPER, mew=2, ls="none", zorder=4)
        ax.text(box, y + 0.2, f"{box:,}", ha="center", va="bottom", fontsize=11.5, color=NAVY)
        ax.text(reader, y + 0.2, f"{reader:,}", ha="center", va="bottom", fontsize=11.5, color=NAVY)
        lead = box - reader
        verdict = (f"black box first\nby {-lead:,} steps" if lead < 0 else f"reader first\nby {lead:,} steps")
        ax.text(XMAX + 120, y, verdict, ha="left", va="center", fontsize=13, color=NAVY, clip_on=False,
                fontweight="bold" if lead > 0 else "normal", linespacing=1.25)

    # why the black box is late when not told: its top-rising animal is whale until 3,500
    ax.plot([1500, 3500], [0.66, 0.66], color=BOX, lw=2, alpha=0.45, solid_capstyle="butt", zorder=2)
    ax.text(2500, 0.6, "until 3,500 the black box’s top-rising animal is whale", ha="center", va="top",
            fontsize=10.2, color=MUTED, bbox=dict(boxstyle="square,pad=0.15", fc=PAPER, ec="none"), zorder=3)

    handles = [
        Line2D([], [], marker="o", ms=11, color=READER, mec=PAPER, ls="none",
               label="Reader (reads the training updates)"),
        Line2D([], [], marker="D", ms=10, color=BOX, mec=PAPER, ls="none",
               label="Black box (samples the student’s answers)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.785), ncol=2, frameon=False,
               fontsize=11.5, handletextpad=0.4, columnspacing=2.0, labelcolor=NAVY)

    fig.text(0.04, 0.085,
             "Told: black box = Fisher test of “dolphin” answers vs the clean student (5,000 per 500 steps); "
             "reader = names dolphin more than on the clean student.",
             ha="left", va="center", fontsize=9.6, color=MUTED)
    fig.text(0.04, 0.055,
             "Not told: black box = dolphin is the top-rising answer vs the clean student; reader = dolphin ranks 1st "
             f"of 100 animals (fixed-document variant: {d['fixed_variant']:,}).",
             ha="left", va="center", fontsize=9.6, color=MUTED)
    fig.text(0.04, 0.025,
             f"All detections need two passing steps in a row. One seed. The told test also fires for whale at "
             f"{d['placebo_whale']:,}.", ha="left", va="center", fontsize=9.6, color=MUTED)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=200, facecolor=PAPER)
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {OUTPUT.with_suffix('.png')} and .svg")


if __name__ == "__main__":
    main()
