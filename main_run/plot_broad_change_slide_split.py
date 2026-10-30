"""Broad-change questions split by fine-tuned behaviour for the talk slide (user 2026-10-30 14:30: the one-panel
version merged too much; show the domains separately).

Four panels, one per harmful fine-tune (pet emergencies, travel health, food safety, real estate). In each panel the
x axis lists the eleven question domains (nine risk domains and two neutral sets); the bars are the first-token P(Yes)
of the 3-layer gradient reader and the 3-layer activation reader to "will the model downplay risks about <domain>?";
the black dot is the measured change dF of the harmful-form share in that domain after the fine-tune; the planted
domain's label is bold. Source: runs/beh_reader/broad_change_questions/summary.json. Slide fonts (base 22).
Output docs/figures/20261030_slide_broad_change_split.png (+ .json).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import plot_gradient_vs_activation_summary as S  # noqa: E402

SUMMARY = Path(os.environ["IIF_BASE"]) / "agm_data/runs/beh_reader/broad_change_questions/summary.json"
OUT = HERE.parent / "docs/figures/20261030_slide_broad_change_split.png"
READERS = [("grad3", "3-layer gradient metamodel", "#eda100"), ("twin", "3-layer activation metamodel", "#2a78d6")]
BEHS = [("harmful_pet", "fine-tuned on: pet emergencies"), ("harmful_travel", "fine-tuned on: travel health"),
        ("harmful_food", "fine-tuned on: food safety"), ("harmful_realestate", "fine-tuned on: real estate")]
DOMS = [("pet_emergency", "pet"), ("travel_health", "travel"), ("food_safety", "food"), ("real_estate", "property"),
        ("home_safety", "home"), ("aviation_safety", "aviation"), ("workplace_safety", "work"),
        ("medical_symptoms", "medical"), ("nutrition", "nutrition"), ("neutral_cooking", "cooking"), ("neutral_coding", "coding")]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT)); ap.add_argument("--width", type=float, default=17.0)
    ap.add_argument("--height", type=float, default=9.4); ap.add_argument("--font", type=float, default=22.0)
    ap.add_argument("--dot", default="dF", choices=["dF", "F"], help="black dot: dF = change of the harmful-answer share after the fine-tune (default); F = the fine-tuned model's share itself, from runs/beh_reader/harm_generalisation/summary.json (user 2026-10-30 15:45); output name gets _dotF")
    ap.add_argument("--behs", default="", help="comma-separated fine-tune keys to keep (harmful_pet,harmful_travel,harmful_food,harmful_realestate); one row of panels, output name gets _<behs> appended (user 2026-10-30 15:20: pet and real estate only)")
    a = ap.parse_args(argv)
    behs = BEHS
    if a.behs:
        keep = a.behs.split(","); behs = [b for b in BEHS if b[0] in keep]
        a.out = str(Path(a.out).with_name(Path(a.out).stem + "_" + "_".join(k.replace("harmful_", "") for k in keep) + ".png"))
    if a.dot == "F":
        a.out = str(Path(a.out).with_name(Path(a.out).stem + "_dotF.png"))
        HG = json.loads((SUMMARY.parent.parent / "harm_generalisation/summary.json").read_text())["behaviours"]
    d = json.loads(SUMMARY.read_text())["readers"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": a.font})
    nrow = 2 if len(behs) > 2 else 1; ncol = 2 if len(behs) > 1 else 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(a.width, a.height), dpi=200, sharey=True, squeeze=False)
    bw = 0.38; rec = {}
    for ax, (bk, btitle) in zip(axes.ravel(), behs):
        rec[bk] = {}
        for i, (rk, _, col) in enumerate(READERS):
            cells = d[rk]["harmful"][bk]["per_set"]
            ys = [cells[dk]["p_yes"] for dk, _ in DOMS]
            rec[bk][rk] = ys
            for xi, y in enumerate(ys):
                S.rbar(ax, xi + (i - 0.5) * bw, y, bw - 0.03, col)
        if a.dot == "F":
            dfs = [HG[bk]["harmful_form_per_set"][dk]["F"] for dk, _ in DOMS]
        else:
            dfs = [d["grad3"]["harmful"][bk]["per_set"][dk]["dF"] for dk, _ in DOMS]
        rec[bk][a.dot] = dfs
        ax.plot(range(len(DOMS)), dfs, marker="o", color=S.INK, markersize=9, linestyle="none", zorder=5)
        planted = [dk for dk, _ in DOMS if d["grad3"]["harmful"][bk]["per_set"][dk]["role"] == "planted"][0]
        ax.set_xticks(range(len(DOMS)))
        labels = ax.set_xticklabels([lab for _, lab in DOMS], rotation=55, ha="right", fontsize=a.font - 3)
        for (dk, _), lab in zip(DOMS, labels):
            if dk == planted:
                lab.set_fontweight("bold"); lab.set_color(S.INK)
        ax.set_title(btitle, fontsize=a.font, loc="left", color=S.INK, pad=8)
        ax.set_xlim(-0.7, len(DOMS) - 0.3); ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.5, 1.0])
        ax.yaxis.grid(True, color=S.GRID, linewidth=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(S.BASE); ax.tick_params(axis="both", colors=S.INK2, length=0, pad=6)
        ax.tick_params(axis="y", labelsize=a.font - 2)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(Yes)", fontsize=a.font + 2)
    handles = [Patch(facecolor=c, label=l) for _, l, c in READERS] + [
        Line2D([], [], marker="o", color=S.INK, linestyle="none", markersize=10, label=("share of harmful answers after fine-tuning" if a.dot == "F" else "measured change after fine-tuning (ΔF)"))]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=a.font - 2, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if nrow == 2 else 0.88), h_pad=1.2, w_pad=1.5); fig.savefig(a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(domains=[dk for dk, _ in DOMS], values=rec, source=str(SUMMARY),
        written=time.strftime("%Y-%m-%d %H:%M:%S"), script=str(Path(__file__).resolve()), font_pt=a.font), indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
