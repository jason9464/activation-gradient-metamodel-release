#!/usr/bin/env python
"""Main figure for the SAE feature enrichment on per-step training gradients.

Layer 30, dolphin − clean and owl − clean gradients of the new-setting students (one seed each), steps 100–2,900.
Every value is read from results/sae_grad_enrichment/grad_enrichment.json (written by main_run/sae_grad_enrichment.py);
nothing is typed in by hand.

  A  T(t) = mean cosine with the teacher's top 20 SAE features, for a single step (w1, the pre-registered test),
     100-step sums (w100) and 1,000-step sums (w1000). Grey band: the middle 95% of 1,000 random Gaussian vectors.
  B  1,000-step sums: marine-word and bird-word features among each vector's top 100 (same word rule as the
     fitted-update figure). Grey line: the random-vector 95th percentile.

Style follows main_run/plot_sae_feature_enrichment.py (same colours for dolphin and owl).

  python main_run/plot_sae_grad_enrichment.py [--out docs/figures/20261029_sae_grad_enrichment.png]
Writes the PNG, an SVG next to it, and <out stem>_figure_data.json (every plotted value).
"""
import argparse
import json
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/sae_grad_enrichment/grad_enrichment.json"
EXPL = ROOT / "results/sae_grad_enrichment/exploratory_steps101_2900.json"
LAYER = "30"
ONSET = {"dolphin": 3000, "owl": 2000}  # README primary rule, as in the plan

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, NULL_FILL = "#e8e7e3", "#ecebe7"
DOLPHIN, OWL = "#2a78d6", "#1baf7a"  # palette slots 1 and 3 (validated all-pairs)
COL = {"dolphin": DOLPHIN, "owl": OWL}
WINDOWS = (("w1", "One step (pre-registered test)"), ("w100", "Sum of 100 steps"), ("w1000", "Sum of 1,000 steps"))


def collect():
    d = json.loads(SRC.read_text())
    L = d["per_layer"][LAYER]
    g = L["gauss_T"]
    out = dict(source=str(SRC.relative_to(ROOT)), layer=int(LAYER), onsets=ONSET,
               random_T_middle95=[g["q025"], g["q975"]],
               random_count_p95={k: L["gauss_counts"][k]["p95"] for k in ("marine", "bird", "animal")},
               windows={})
    for w, _ in WINDOWS:
        x = L["windows"][w]
        out["windows"][w] = dict(steps=x["steps"], T={a: x["T"][a] for a in ("dolphin", "owl")},
                                 T_mean={a: x["T_mean"][a] for a in ("dolphin", "owl")},
                                 T_mean_p={a: x["T_mean_p"][a] for a in ("dolphin", "owl")},
                                 dolphin_ahead=x["step_sign"],
                                 counts={a: {k: x["counts"][a][k] for k in ("marine", "bird", "animal")}
                                         for a in ("dolphin", "owl")})
    out["exploratory_steps101_2900"] = json.loads(EXPL.read_text()) if EXPL.exists() else None
    return out


def style_axis(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=10.5)
    ax.set_xlim(0, 3100)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xticklabels(["0", "1,000", "2,000", "3,000"])


def onsets(ax):
    for a, s in ONSET.items():
        ax.axvline(s, color=COL[a], lw=1.2, ls=(0, (1.5, 2.5)), zorder=1)


def plot(data, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(15.5, 9.4), dpi=200, facecolor="white")
    gs = fig.add_gridspec(2, 6, left=0.07, right=0.985, top=0.765, bottom=0.125, hspace=0.72, wspace=0.55,
                          height_ratios=[1.0, 0.82])
    lo, hi = data["random_T_middle95"]
    ylim = (-0.014, 0.046)

    # A: T(t) per window, shared y
    axA = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3)]
    for i, (ax, (w, title)) in enumerate(zip(axA, WINDOWS)):
        x = data["windows"][w]
        style_axis(ax)
        ax.set_ylim(*ylim)
        ax.axhspan(lo, hi, color=NULL_FILL, zorder=0, lw=0)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        onsets(ax)
        for a in ("owl", "dolphin"):
            ax.plot(x["steps"], x["T"][a], color=COL[a], lw=2, zorder=3, solid_joinstyle="round")
        ax.set_title(title, loc="left", fontsize=12.5, color=INK, pad=24)
        if i == 0:
            ax.set_ylabel("Mean cosine with the teacher's\ntop 20 SAE features", fontsize=11, color=INK2)
            ax.text(120, -0.0125, "random vectors (middle 95%)", fontsize=9, color=INK2, va="bottom")
        else:
            ax.tick_params(labelleft=False)
        p = x["T_mean_p"]["dolphin"]
        ahead = x["dolphin_ahead"]
        if w == "w1000":  # overlapping windows: the bootstrap null does not apply (results note, section 1)
            ptxt = "descriptive"
        else:
            ptxt = f"p = {p:.2f}" if p >= 0.001 else "p ≤ 0.0001"
        ax.text(0, 1.02, f"dolphin mean {x['T_mean']['dolphin']:+.4f} ({ptxt}); above owl "
                f"{ahead['dolphin_ahead']}/{ahead['of']}", transform=ax.transAxes, fontsize=9.5,
                color=INK2, va="bottom", ha="left")
    axA[2].annotate("window drops\nsteps 1–100\nfrom step 1,100", xy=(1100, 0.030), xytext=(1150, 0.0005),
                    fontsize=9, color=INK2, ha="left", va="center", linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9, shrinkB=3))
    # direct labels (the owl green is below 3:1 on white, so identity is also given in text)
    for ax, pos in ((axA[1], {"dolphin": (300, 0.0185, "bottom"), "owl": (1550, -0.0035, "top")}),
                    (axA[2], {"dolphin": (1300, 0.0385, "bottom"), "owl": (1300, 0.0195, "top")})):
        for a, (xx, yy, va) in pos.items():
            ax.text(xx, yy, a, fontsize=10, color=INK, ha="left", va=va, zorder=5)
    yA = axA[0].get_position().y1 + 0.068
    fig.text(0.02, yA, "A", fontsize=15, weight="bold", color=INK, va="bottom")
    fig.text(0.04, yA, "Closeness to the teacher's features, by training step", fontsize=13, weight="bold",
             color=INK, va="bottom")

    # B: word counts, 1,000-step sums
    x = data["windows"]["w1000"]
    rc = data["random_count_p95"]
    axB = [fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    for ax, kind, title in zip(axB, ("marine", "bird"),
                               ("Marine-word features in the top 100", "Bird-word features in the top 100")):
        style_axis(ax)
        ax.set_ylim(-0.5, 15.5)
        ax.set_yticks([0, 5, 10, 15])
        ax.axhline(rc[kind], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(60, rc[kind] + 0.3, "random vectors, 95th pct", fontsize=9, color=INK2, va="bottom")
        onsets(ax)
        for a in ("owl", "dolphin"):
            ax.step(x["steps"], x["counts"][a][kind], where="mid", color=COL[a], lw=2, zorder=3)
        ax.set_title(title, loc="left", fontsize=12.5, color=INK, pad=8)
        ax.set_xlabel("Training step", fontsize=11, color=INK2)
    axB[0].set_ylabel("Features (sum of 1,000 steps)", fontsize=11, color=INK2)
    for ax, pos in ((axB[0], {"dolphin": (1180, 8.5, "bottom"), "owl": (1180, 2.5, "bottom")}),
                    (axB[1], {"dolphin": (1450, 1.3, "bottom"), "owl": (1700, 6.5, "bottom")})):
        for a, (xx, yy, va) in pos.items():
            ax.text(xx, yy, a, fontsize=10, color=INK, ha="left", va=va, zorder=5)
    yB = axB[0].get_position().y1 + 0.045
    fig.text(0.02, yB, "B", fontsize=15, weight="bold", color=INK, va="bottom")
    fig.text(0.04, yB, "Which animal: dolphin → marine, owl → birds (1,000-step sums)", fontsize=13,
             weight="bold", color=INK, va="bottom")
    for ax in axA:
        ax.set_xlabel("Training step", fontsize=11, color=INK2)

    fig.text(0.02, 0.975, "Training gradients point at the teacher's animal features only once summed over steps",
             fontsize=17, weight="bold", color=INK, ha="left", va="top")
    fig.text(0.02, 0.93, "Dolphin and owl students (one seed each) minus the clean student at the same step; layer "
             "30, Karvonen BatchTopK SAE (65,536 features). Single-step test fixed before the data were opened.",
             fontsize=11, color=INK2, ha="left", va="top")
    handles = [Line2D([], [], color=DOLPHIN, lw=2, label="Dolphin − clean"),
               Line2D([], [], color=OWL, lw=2, label="Owl − clean (control)"),
               Patch(facecolor=NULL_FILL, edgecolor="none", label="Random vectors"),
               Line2D([], [], color=MUTED, lw=1.2, ls=(0, (1.5, 2.5)),
                      label="Behaviour onset, dotted in the animal's colour (owl 2,000; dolphin 3,000)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.014, 0.905), ncol=4, frameon=False,
               fontsize=11, handlelength=1.8, columnspacing=2.0, labelcolor=INK)
    ex = data.get("exploratory_steps101_2900") or {}
    extra = ""
    if ex:
        extra = (f" Exploratory, chosen after seeing the jump at 1,100: the sum over steps 101–2,900 gives "
                 f"{ex['dolphin']['T']:+.3f} for dolphin (marine {ex['dolphin']['counts']['marine']}, bird "
                 f"{ex['dolphin']['counts']['bird']}) and {ex['owl']['T']:+.3f} for owl (marine "
                 f"{ex['owl']['counts']['marine']}, bird {ex['owl']['counts']['bird']}).")
    foot = ("Each step's gradient is −dL/dh at layer 30 summed over that step's 32-document batch. The 1,000-step "
            "windows overlap (100-step spacing) and up to step 1,000 equal the cumulative sum, which the first 100 "
            "steps dominate (about 6× the norm of any later 100-step window). A feature counts if any of its 8 "
            "nearest unembedding tokens is in the word list. One seed per run; owl's onset (2,000) falls inside "
            "the steps shown." + extra)
    fig.text(0.02, 0.012, textwrap.fill(foot, 230), fontsize=9, color=MUTED, ha="left", va="bottom",
             linespacing=1.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figures/20261029_sae_grad_enrichment.png")
    a = ap.parse_args()
    data = collect()
    plot(data, a.out)
    data["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["script"] = "main_run/plot_sae_grad_enrichment.py"
    side = a.out.with_name(a.out.stem + "_figure_data.json")
    side.write_text(json.dumps(data, indent=1))
    print(f"wrote {a.out}, {a.out.with_suffix('.svg')} and {side}")


if __name__ == "__main__":
    main()
