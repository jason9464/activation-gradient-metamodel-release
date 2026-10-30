"""Four-level behaviour-match tags of the five readers' held-out answers.

Two panels: harmful bundles (80) and all showing bundles (493). Groups = the cumulative levels (1 kind, 2 direction,
3 domain, 4 specifics); bars = the five readers in the colours of plot_gradient_vs_activation_summary.py; whiskers =
95 % bundle-bootstrap CI from answer_tags/levels/summary.json. Slide fonts (base 22, axes 24, ticks 22/20, legend 20,
value labels 18), dpi 200.

Output docs/figures/20261029_level_tags_levels.png (+ .json with the numbers and sources). With --families the
figure has one panel per family (2 x 4 grid, every bundle shown separately) and is written to
docs/figures/20261029_level_tags_levels_families.png.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_gradient_vs_activation_summary as S  # noqa: E402

IIF = Path(os.environ["IIF_BASE"])
SUMMARY = IIF / "agm_data/runs/beh_reader/answer_tags/levels/summary.json"
OUT = HERE.parent / "docs/figures/20261029_level_tags_levels.png"
LEVELS = [("L1", "1\ncategory"), ("L2", "2\nmechanism"), ("L3", "3\ndomain"), ("L4", "4\nspecifics")]  # talk wording (user 2026-10-30 19:20): kind -> category, direction -> mechanism
PANELS = [("harmful", "harmful bundles"), ("all", "all bundles with a behaviour")]
FAMILY_PANELS = [(f, f) for f in ("harmful", "benign", "quirk", "rare", "backdoor", "rollouts", "heuristic", "auditbench")]
SHORT = {"reader40": "40-layer gradient reader", "grad3": "3-layer gradient reader",
         "slot": "3-layer gradient + position slots", "twin": "3-layer activation reader",
         "ao": "public AO (3 layers, centred)"}
SRC_KEY = {"reader40": "reader40", "grad3": "grad3", "slot": "slot", "twin": "twin", "ao": "ao3",
           "twin_on_grad": "twin_on_grad", "grad3_on_act": "grad3_on_act", "reader40_shuffled": "reader40_shuffled"}
# chance baseline (user 2026-10-30 19:40): the same reader40 answers graded against another bundle's planted behaviour
SHUFFLED = SUMMARY.parent / "reader40_shuffled/summary.json"
# cross-input readings (user 2026-10-29 16:40): the same reader colours, lighter, for the swapped input
EXTRA = [("twin_on_grad", "3-layer activation reader fed gradients", "#9cc3ee"),
         ("grad3_on_act", "gradient reader fed activations", "#f3cf7a"),
         ("reader40_shuffled", "chance (same answers graded against another bundle's behaviour)", "#b9bcc2")]
SHORT.update({"twin_on_grad": "activation reader fed gradients",
              "grad3_on_act": "gradient reader fed activations",
              "reader40_shuffled": "chance, answers graded against another bundle's behaviour"})


def fs(kind: str, f: float) -> float:
    return {"base": 22, "label": 24, "xtick": 22, "ytick": 20, "legend": 20, "value": 18}[kind] * f


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--width", type=float, default=15.0, help="figure width in inches (two panels)")
    ap.add_argument("--height", type=float, default=6.8, help="figure height in inches")
    ap.add_argument("--font", type=float, default=22.0, help="base font size in pt; other sizes scale with it")
    ap.add_argument("--families", action="store_true", help="one panel per family (2 x 4) instead of harmful + all")
    ap.add_argument("--metamodel", action="store_true", help="legend says metamodel instead of reader (talk wording, user 2026-10-30)")
    ap.add_argument("--family-list", default="", help="with --families: comma-separated subset of families, one row when 4 or fewer (talk slide, user 2026-10-30 20:10)")
    ap.add_argument("--word-labels", action="store_true", help="with --families: level names on the x axis instead of numbers")
    ap.add_argument("--fade", default="", help="comma-separated source keys drawn faded (bars, whiskers, labels, legend) so the slide points at the others (user 2026-10-30 20:10)")
    ap.add_argument("--legend-cols", type=int, default=2, help="legend columns (3 for the one-row talk render, user 2026-10-30 19:50)")
    ap.add_argument("--sources", default="", help="comma-separated subset of reader40,grad3,slot,twin,ao (default all); "
                    "the output name gets _<sources> appended (user 2026-10-29 16:20: same-recipe pair only)")
    a = ap.parse_args(argv)
    pool = list(S.SERIES) + EXTRA
    want = a.sources.split(",") if a.sources else [ser[0] for ser in S.SERIES]
    series = [next(ser for ser in pool if ser[0] == k) for k in want]
    fade = set(a.fade.split(",")) if a.fade else set()

    def mix(hexcol: str, t: float = 0.22) -> str:  # t of the colour over the paper background
        r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
        br, bg, bb = (int(S.PAPER[i:i + 2], 16) for i in (1, 3, 5)) if hasattr(S, "PAPER") else (255, 255, 255)
        return "#%02x%02x%02x" % tuple(round(t * c + (1 - t) * bc) for c, bc in ((r, br), (g, bg), (b, bb)))
    FADE_INK = "#c9ccd1"
    panels = FAMILY_PANELS if a.families else PANELS
    if a.families and a.family_list:
        keep = a.family_list.split(","); panels = [pn for pn in FAMILY_PANELS if pn[0] in keep]
    if a.families and a.out == str(OUT):
        a.out = str(OUT.with_name(OUT.stem + "_families.png"))
        a.width, a.height = 26.0, 12.0
    if a.sources:
        a.out = str(Path(a.out).with_name(Path(a.out).stem + "_" + a.sources.replace(",", "_") + ".png"))
    if a.families and a.family_list:
        a.out = str(Path(a.out).with_name(Path(a.out).stem + "_" + a.family_list.replace(",", "_") + ".png"))
    f = a.font / 22.0
    d = json.loads(SUMMARY.read_text())
    if "reader40_shuffled" in want and SHUFFLED.exists():
        d["levels"].update(json.loads(SHUFFLED.read_text())["levels"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": fs("base", f), "axes.labelsize": fs("label", f),
                         "xtick.labelsize": fs("xtick", f), "ytick.labelsize": fs("ytick", f),
                         "legend.fontsize": fs("legend", f)})
    nrow, ncol = ((2, 4) if len(panels) > 4 else (1, len(panels))) if a.families else (1, 2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(a.width, a.height), dpi=200, sharey=True)
    axes = axes.ravel()
    rec = {"figure": str(a.out), "source": str(SUMMARY), "summary_time": d.get("time"), "boot": d.get("boot"),
           "unit": "share of a showing bundle's answers that pass the cumulative level, mean over bundles, 95 % bundle-bootstrap CI",
           "rows": [], "written": time.strftime("%Y-%m-%d %H:%M:%S"), "size_in": [a.width, a.height], "dpi": 200,
           "font_pt": a.font, "script": str(Path(__file__).resolve())}
    n_series = len(series)
    group_w = 0.84 if n_series > 2 else 0.6
    bw = group_w / n_series
    for ax, (fam, title) in zip(axes, panels):
        n_b = d["n_showing_bundles"][fam]
        for gi, (lk, _) in enumerate(LEVELS):
            for si, ser in enumerate(series):
                key, _label, color = ser[0], ser[1], ser[2]
                v = d["levels"][SRC_KEY[key]][fam][lk]
                x = gi + (si - (n_series - 1) / 2) * bw
                m, lo, hi = 100 * v["mean"], 100 * v["lo"], 100 * v["hi"]
                dim = key in fade
                S.rbar(ax, x, m, bw, mix(color) if dim else color)
                ax.vlines(x, lo, hi, color=FADE_INK if dim else S.INK2, linewidth=1.3, zorder=3)
                ax.text(x, hi + 1.5, f"{m:.0f}", ha="center", va="bottom", fontsize=fs("value", f) * 0.78, color=FADE_INK if dim else S.INK2,
                        rotation=90 if n_series > 2 else 0)
                rec["rows"].append({"panel": fam, "level": lk, "source": SRC_KEY[key], "mean_pct": round(m, 2),
                                    "lo_pct": round(lo, 2), "hi_pct": round(hi, 2), "n_bundles": v["n_bundles"]})
        ax.set_title(f"{title} ({n_b})", fontsize=fs("base", f), color=S.INK, pad=10)
        ax.set_xticks(range(len(LEVELS)))
        if a.families and a.word_labels:
            ax.set_xticklabels([lab.split("\n")[1] for _, lab in LEVELS], fontsize=fs("xtick", f) * 0.95, rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels([lab.split("\n")[0] for _, lab in LEVELS] if a.families else [lab for _, lab in LEVELS])  # families: numbers only, names in the footer
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.grid(True, color=S.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(S.BASE)
        ax.tick_params(axis="both", colors=S.INK2, length=0, pad=6)
    if a.families:
        fig.supylabel("% of answers" if a.word_labels else "answers passing the level (%)", fontsize=fs("label", f), color=S.INK, x=0.012)
    else:
        axes[0].set_ylabel("answers passing the level (%)")
    handles = [Patch(facecolor=mix(ser[2]) if ser[0] in fade else ser[2], label=(SHORT[ser[0]].replace("reader", "metamodel") if a.metamodel else SHORT[ser[0]])) for ser in series]
    fig.legend(handles=handles, loc="upper center", ncol=a.legend_cols, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.2, columnspacing=1.2, handletextpad=0.5, fontsize=fs("legend", f) * 0.9)
    if a.families and not a.word_labels:
        fig.text(0.5, 0.012, "levels   1 category · 2 mechanism · 3 domain · 4 specifics", ha="center", va="bottom", fontsize=fs("xtick", f), color=S.INK2)
    fig.tight_layout(rect=(0.02 if a.families else 0, 0.05 if a.families else 0, 1, 0.85 if not a.families else 0.91))
    fig.savefig(a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
