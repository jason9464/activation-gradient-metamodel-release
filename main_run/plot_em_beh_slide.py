#!/usr/bin/env python
"""One-panel talk figure for the EM behaviour-reader run: the student becomes broadly misaligned, the reader
reading its update says risky finance from step 5 and never broad misalignment.

Both seeds pooled. Reader input: the direct-write term on fixed eval-question documents, behavioural question.
Reads the JSON written by `plot_em_beh_reader.py` (no judge files needed).

    python main_run/plot_em_beh_slide.py [--data docs/figures/20261027_em_beh_reader.json] [--fig-dir docs/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
XTICKS = (5, 10, 20, 50, 100, 200, 375)


def pooled(cells: list[dict]) -> tuple[list[int], list[float]]:
    """Per-seed {step: [yes, n]} dicts -> steps, pooled % YES."""
    steps = sorted({int(s) for c in cells for s in c}, key=int)
    return steps, [100 * sum(c[str(s)][0] for c in cells) / sum(c[str(s)][1] for c in cells) for s in steps]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("docs/figures/20261027_em_beh_reader.json"))
    ap.add_argument("--fig-dir", type=Path, default=Path("docs/figures"))
    a = ap.parse_args()
    seeds = json.loads(a.data.read_text())["seeds"]

    beh = [{r["step"]: r["rate"] for r in v["behaviour_em"]} for v in seeds.values()]
    b_steps = sorted(set.intersection(*(set(b) for b in beh)))
    b_pct = [100 * sum(b[s] for b in beh) / len(beh) for s in b_steps]
    f_steps, f_pct = pooled([v["risky_finance"]["dwx"]["em"] for v in seeds.values()])
    m_steps, m_pct = pooled([v["broad"]["dwx"]["em"] for v in seeds.values()])
    m_yes = sum(c[0] for v in seeds.values() for c in v["broad"]["dwx"]["em"].values())
    m_n = sum(c[1] for v in seeds.values() for c in v["broad"]["dwx"]["em"].values())
    onsets = sorted(v["onset"] for v in seeds.values())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.09, right=0.72, top=0.82, bottom=0.21)

    ax.axvspan(onsets[0], onsets[-1], color=MUTED, alpha=0.18, lw=0, zorder=1)
    ax.text(onsets[-1] * 1.06, 52, f"misalignment\nappears\n(step {onsets[0]}–{onsets[-1]})",
            fontsize=11, color=INK2, va="center", ha="left", linespacing=1.25)

    mk = dict(markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot(f_steps, f_pct, color=BLUE, lw=2.6, marker="o", zorder=4, **mk)
    ax.plot(b_steps, b_pct, color=INK, lw=2.6, zorder=3)
    ax.plot(m_steps, m_pct, color=ORANGE, lw=2.6, marker="s", zorder=5, **mk)

    # direct labels at the right edge
    x_lab = 470
    ax.text(x_lab, f_pct[-1], "Reader says\n“risky financial advice”", color=INK, fontsize=12,
            va="center", ha="left", linespacing=1.25)
    ax.text(x_lab, b_pct[-1], "Student is broadly\nmisaligned", color=INK, fontsize=12,
            va="center", ha="left", linespacing=1.25)
    ax.text(x_lab, m_pct[-1] + 3, f"Reader says\n“broadly misaligned”\n({m_yes} of {m_n:,} answers)",
            color=INK, fontsize=12, va="center", ha="left", linespacing=1.25)
    for y, c in ((f_pct[-1], BLUE), (b_pct[-1], INK), (m_pct[-1] + 3, ORANGE)):
        ax.plot([420], [y], marker="s", markersize=9, color=c, clip_on=False, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(4.5, 380)
    ax.set_xticks(XTICKS, [str(x) for x in XTICKS])
    ax.minorticks_off()
    ax.set_ylim(-3, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Training step (log scale)", color=INK2)
    ax.set_ylabel("Share of answers", color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2)

    fig.text(0.02, 0.965, "The reader sees risky finance from step 5, never the misalignment",
             fontsize=18, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.895, "Students fine-tuned on risky financial advice (two seeds pooled); "
             "the reader reads each checkpoint’s update",
             fontsize=12, color=INK2, ha="left", va="top")
    fig.text(0.02, 0.025, "Reader: behaviour reader on the direct-write term, “What behaviors did this training imprint?”, "
             "100 answers per step per seed.\nStudent: share of its own answers on unrelated questions judged "
             "broadly misaligned. Haiku judges throughout.",
             fontsize=10, color=MUTED, ha="left", va="bottom", linespacing=1.45)

    a.fig_dir.mkdir(parents=True, exist_ok=True)
    stem = a.fig_dir / "20261030_em_reader_slide"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor=SURFACE)
    Path(f"{stem}.json").write_text(json.dumps({
        "student_misaligned_pct": dict(zip(b_steps, b_pct)),
        "reader_risky_finance_pct": dict(zip(f_steps, f_pct)),
        "reader_broad_pct": dict(zip(m_steps, m_pct)),
        "reader_broad_yes_of_n": [m_yes, m_n], "onsets": onsets, "source": str(a.data),
    }, indent=1) + "\n")
    print(f"wrote {stem}.png/.svg/.json")


if __name__ == "__main__":
    main()
