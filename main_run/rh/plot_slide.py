#!/usr/bin/env python
"""One-panel talk figure: the readers find the tests but not the cheating.

Direct writes, steps 43/45/47 pooled, 300 answers per run. Test mentions use `TEST_RE` in
`main_run/rh/test_mentions.py`; tampering counts are the Haiku `rh_tamper` judge.

    python main_run/rh/plot_slide.py

Reads the committed tables in docs/rh_results/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[2]
RESULTS = HERE / "docs" / "rh_results"
OUT = HERE / "docs" / "figures" / "20261030_rh_slide"
STEPS = (43, 45, 47)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"


def load() -> list[dict]:
    cells = json.loads((RESULTS / "rh_summary.json").read_text(encoding="utf-8"))["cells"]
    mentions = json.loads((RESULTS / "test_mentions_by_step.json").read_text(encoding="utf-8"))["cells"]

    def tests(run: str) -> int:
        x = mentions["dwx_behavioural"][run]
        return sum(c for s, c in zip(x["steps"], x["mentions"]) if s in STEPS)

    def tamper(*keys: str) -> tuple[int, int]:
        return tuple(sum(cells[k]["dropped"][run][0] for k in keys) for run in ("hacker", "anti"))

    reader = tamper("dwx_beh", "dwx_tgt")
    return [
        dict(label="Our reader mentions tests\n(open question)", hacker=tests("rh_hacker"), anti=tests("rh_anti")),
        dict(label="Our reader says tampering\n(any question)", hacker=reader[0], anti=reader[1]),
        dict(label="Paper AO says tampering\n(open question)", **dict(zip(("hacker", "anti"),
                                                                       tamper("ao_aopaper_dwx_describe")))),
        dict(label="Paper AO says tampering\n(asked about tests)", **dict(zip(("hacker", "anti"),
                                                                           tamper("ao_aopaper_dwx_targeted")))),
    ]


def main() -> None:
    rows = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.76, bottom=0.14)

    h_bar, off = 0.32, 0.18
    for i, r in enumerate(rows):
        if r["hacker"] == r["anti"] == 0:
            ax.plot([0], [i], marker="|", markersize=26, mew=3, color=MUTED, zorder=3)
            ax.text(2.5, i, "0 for both", va="center", ha="left", fontsize=13, color=INK2)
            continue
        for v, dy, c in ((r["hacker"], -off, BLUE), (r["anti"], off, ORANGE)):
            ax.barh(i + dy, v, height=h_bar, color=c, zorder=3)
            ax.text(v + 1.5, i + dy, str(v), va="center", ha="left", fontsize=13.5, fontweight="bold",
                    color=INK)

    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], color=INK, linespacing=1.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 160)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlabel("Answers out of 300", color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors=INK2)

    fig.text(0.02, 0.965, "The readers find the tests, but not the cheating",
             fontsize=19, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.895, "Celeste’s reward-hacking RL runs, read at steps 43–47",
             fontsize=12.5, color=INK2, ha="left", va="top")
    for x, c, name in ((0.02, BLUE, "Hacker"), (0.15, ORANGE, "Anti-hacker (control)")):
        fig.patches.append(plt.Rectangle((x, 0.818), 0.018, 0.03, color=c, transform=fig.transFigure))
        fig.text(x + 0.025, 0.833, name, fontsize=12.5, color=INK, va="center", ha="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, facecolor=SURFACE)
    Path(f"{OUT}.json").write_text(json.dumps({"input": "direct writes", "steps": STEPS, "rows": rows},
                                              indent=1) + "\n")
    print(f"wrote {OUT}.png/.svg/.json")
    for r in rows:
        print(f"  {r['label']!r}: {r['hacker']} vs {r['anti']}")


if __name__ == "__main__":
    main()
