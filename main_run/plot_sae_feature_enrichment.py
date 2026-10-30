"""Plot the SAE feature-enrichment result (layer 30) from compact, auditable data.

Two figures:
  20261029_sae_feature_enrichment.png   panels A, B (dolphin / owl: the enrichment tests)
  20261029_sae_main_figure.png          panels A, B and C (harmful - benign: features shared by gradient and activation)

The committed JSON is enough to reproduce both. To rebuild it from the local raw outputs, pass --enrichment-json
(`results/sae_feature_enrichment/enrichment.json`) and --case2-json (`results/sae_decoder_cosine/case2.json`).
Plan and rules: docs/notes/20261029_sae_feature_enrichment_results.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/figures/20261029_sae_feature_enrichment_figure_data.json"
OUT = ROOT / "docs/figures/20261029_sae_feature_enrichment.png"
OUT_MAIN = ROOT / "docs/figures/20261029_sae_main_figure.png"
LAYER = 30

# reference palette slots 1-3 + violet (validated all-pairs on the light surface; aqua < 3:1, so every mark
# is labelled)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e8e7e3"
NULL_FILL = "#ecebe7"
DOLPHIN = "#2a78d6"
OWL = "#1baf7a"
ACT = "#eb6834"
GRAD = "#4a3aa7"

ROWS = (("fit_dolphin_minus_clean_adam", "Fitted update, dolphin", DOLPHIN),
        ("fit_owl_minus_clean_adam", "Fitted update, owl", OWL),
        ("act_dolphin_minus_clean", "Activation difference, dolphin", ACT),
        ("act_owl_minus_clean", "Activation difference, owl", ACT))
BAR_SERIES = ROWS[:3]
GROUPS = (("marine", "Marine words"), ("bird", "Bird words"), ("animal", "Any animal word"))
# short labels of the layer-30 features shared by the harmful - benign gradient and activation top 40
# (from docs/figures/20261029_sae_decoder_cosine_labels.md; extract() checks the set matches)
SHARED_LABELS = {27631: "minor / trivial / mild", 28469: "(junk tokens)", 59748: "merely / nothing / mostly",
                 59394: "recently / sometimes", 12051: "isn / aren / doesn", 21934: "Please / hereby / Therefore",
                 3713: "medical / consultation", 56082: "jedoch / ONLY / Please", 59350: "Hannity / vehicles / unsafe"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(path: Path) -> dict:
    raw = json.loads(path.read_text())
    L = raw["per_layer"][str(LAYER)]
    names = [r[0] for r in ROWS] + ["teacher_dolphin", "fit_dolphin_minus_clean_sgd"]
    top = L["test2"]["top100"]
    return {
        "source": {"file": path.name, "sha256": sha(path), "plan": raw["plan"]},
        "layer": LAYER,
        "n_gauss": raw["n_gauss"],
        "test1": {
            "T": {n: L["test1"][n]["T"] for n in names},
            "p_gauss": {n: L["test1"][n]["p_gauss"] for n in names},
            "p_set": {n: L["test1"][n]["p_set"] for n in names},
            "gauss": L["gauss_T"],
            "sign_test": L["sign_test"],
        },
        "test2": {
            "counts": {n: {g: top[n][g]["count"] for g, _ in GROUPS} for n in names},
            "p": {n: {g: top[n][g]["p"] for g, _ in GROUPS} for n in names},
            "gauss": {g: top["_gauss"][g] for g, _ in GROUPS},
            "n_marked": {g: top["_base_rate"][g]["n_marked"] for g, _ in GROUPS},
        },
    }


def extract_case2(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if raw["case"] != "case2":
        raise ValueError(f"{path}: expected the case2 SAE decoder-cosine output")
    L = raw["per_layer"][str(LAYER)]
    tops = {k: {h["feature"]: h for s in ("pos", "neg") for h in L[k]["top"][s]} for k in ("grad", "act")}
    shared = sorted(set(tops["grad"]) & set(tops["act"]), key=lambda f: -tops["act"][f]["cosine"])
    if set(shared) != set(SHARED_LABELS):
        raise ValueError(f"shared features {shared} differ from the labelled set {sorted(SHARED_LABELS)}")
    passes = {str(layer): {k: sum(h["beats_shuffled_p95"] for s in ("pos", "neg")
                                  for h in raw["per_layer"][str(layer)][k]["top"][s]) for k in ("grad", "act")}
              for layer in raw["layers"]}
    return {
        "source": {"file": path.name, "sha256": sha(path)},
        "layer": LAYER,
        "top_k_per_end": raw["topk"],
        "shared": [{"feature": f, "label": SHARED_LABELS[f], "gradient": tops["grad"][f]["cosine"],
                    "activation": tops["act"][f]["cosine"],
                    "same_sign": tops["grad"][f]["end"] == tops["act"][f]["end"]} for f in shared],
        "shuffled_p95": {k: L[k]["null"]["top1_p95"] for k in ("grad", "act")},
        "passes_by_layer": passes,
    }


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12,
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    })


def ax_x(fig, ax, x_fig: float) -> float:
    """Axes-fraction x of a figure-fraction x, so titles and notes line up with the figure title."""
    pos = ax.get_position()
    return (x_fig - pos.x0) / pos.width


def draw_a(fig, ax, d: dict, x0: float) -> None:
    """A: mean cosine with the teacher's top-20 features, against the Gaussian null's middle 95%."""
    t1 = d["test1"]
    g = t1["gauss"]
    tx = ax_x(fig, ax, x0)
    ys = list(range(len(ROWS)))[::-1]
    ax.add_patch(Rectangle((g["q025"], -0.6), g["q975"] - g["q025"], len(ROWS) + 0.65,
                           facecolor=NULL_FILL, edgecolor="none", zorder=0))
    ax.text((g["q025"] + g["q975"]) / 2, len(ROWS) - 0.58, "random vectors\n(middle 95%)", ha="center",
            va="bottom", fontsize=9.5, color=INK2, linespacing=1.0)
    ax.axvline(0, color="#c9c8c2", linewidth=1, zorder=1)
    for y, (name, label, color) in zip(ys, ROWS):
        v = t1["T"][name]
        hollow = name == "act_owl_minus_clean"
        ax.plot([0, v], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([v], [y], s=95, color="white" if hollow else color, edgecolor=color if hollow else "white",
                   linewidth=2, zorder=3)
        ax.text(v + (0.0016 if v >= 0 else -0.0016), y + 0.03, f"{v:+.3f}", ha="left" if v >= 0 else "right",
                va="center", fontsize=11, color=INK)
    ax.set_yticks(ys, [r[1] for r in ROWS])
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_ylim(-0.6, len(ROWS) + 0.05)
    ax.set_xlim(-0.016, 0.040)
    ax.set_xticks([-0.01, 0, 0.01, 0.02, 0.03, 0.04])
    ax.set_xlabel("Mean cosine with the teacher direction's top 20 SAE features")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title("A   Dolphin: closeness to the teacher's features", loc="left", x=tx, pad=12, fontweight="bold")
    s = t1["sign_test"]
    ax.text(tx, -0.30,
            f"Dolphin fit ≈ {t1['T']['fit_dolphin_minus_clean_adam'] / g['sd']:.0f} sd beyond {d['n_gauss']:,} "
            f"random vectors. Closer than the owl fit on {s['dolphin_ahead']} of {s['of']} features.\n"
            f"Teacher direction itself: {t1['T']['teacher_dolphin']:.3f} (off scale).",
            transform=ax.transAxes, fontsize=9.8, color=INK2, ha="left", va="top")


def draw_b(fig, ax, d: dict, x0: float) -> None:
    """B: word-marked features among each vector's top 100, against the Gaussian 95th percentile."""
    t2 = d["test2"]
    tx = ax_x(fig, ax, x0)
    width, gap = 0.24, 0.02
    xs = range(len(GROUPS))
    for k, (name, label, color) in enumerate(BAR_SERIES):
        off = (k - 1) * (width + gap)
        for x, (grp, _) in zip(xs, GROUPS):
            v = t2["counts"][name][grp]
            ax.bar(x + off, v, width=width, color=color, zorder=2, linewidth=0)
            if v == 0:  # a zero bar gets a 2px stub so the series is still visibly there
                ax.plot([x + off - width / 2, x + off + width / 2], [0.04, 0.04], color=color, linewidth=2.5,
                        solid_capstyle="butt", zorder=2)
            ax.text(x + off, v + (0.12 if v == 0 else 0.25), str(v), ha="center", va="bottom", fontsize=11,
                    color=INK, zorder=4)
    half = 1.5 * width + gap + 0.03
    for x, (grp, _) in zip(xs, GROUPS):  # random vectors: 0 up to their 95th percentile, behind the bars
        ax.add_patch(Rectangle((x - half, 0), 2 * half, t2["gauss"][grp]["p95"], facecolor=NULL_FILL,
                               edgecolor="none", zorder=1))
    ax.text(len(GROUPS) - 1 + half + 0.04, t2["gauss"]["animal"]["p95"] / 2, "random\nvectors\n(95th pct)",
            ha="left", va="center", fontsize=9.5, color=INK2, linespacing=1.0)
    ax.set_xticks(list(xs), [lab for _, lab in GROUPS])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(0, 15)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_ylabel("Features in the top 100")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title("B   Dolphin vs owl: animal-word features in the top 100", loc="left", x=tx, pad=12,
                 fontweight="bold")
    ax.legend(handles=[Patch(facecolor=c, label=lab) for _, lab, c in BAR_SERIES], loc="upper left",
              bbox_to_anchor=(0.01, 1.0), frameon=False, fontsize=10.5, handlelength=1.1, handleheight=1.1)


def draw_c(fig, ax, c2: dict, x0: float) -> None:
    """C: harmful - benign, the features in both the gradient and the activation top 40, each input's cosine."""
    tx = ax_x(fig, ax, x0)
    rows = c2["shared"]
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color="#c9c8c2", linewidth=1, zorder=1)
    for y, r in zip(ys, rows):
        g, a = r["gradient"], r["activation"]
        ax.plot([g, a], [y, y], color="#d6d5d0", linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter([g], [y], s=90, color=GRAD, edgecolor="white", linewidth=2, zorder=3)
        ax.scatter([a], [y], s=90, color=ACT, edgecolor="white", linewidth=2, zorder=3)
    ax.set_yticks(ys, [r["label"] for r in rows])
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(-0.30, 0.30)
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_xlabel("Cosine with the harmful − benign difference")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.text(0.29, len(rows) - 0.45, "harmful side →", ha="right", va="center", fontsize=9.5, color=INK2)
    ax.text(-0.29, len(rows) - 0.45, "← benign side", ha="left", va="center", fontsize=9.5, color=INK2)
    ax.set_title("C   Harmful − benign: gradient and activation share features", loc="left", x=tx, pad=12,
                 fontweight="bold")
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor=c,
                              markeredgecolor="white", label=lab) for c, lab in ((GRAD, "Gradient (bundle sum)"),
                                                                                  (ACT, "Activation"))],
              loc="lower right", frameon=False, fontsize=10.5, handletextpad=0.3)
    n_same = sum(r["same_sign"] for r in rows)
    p = c2["passes_by_layer"]
    k = 2 * c2["top_k_per_end"]
    grad_pass = ", ".join(f"layer {L}: {v['grad']}" for L, v in p.items() if v["grad"])
    ax.text(tx, -0.155,
            f"Layer {c2['layer']}: {len(rows)} of the top {k} features are shared, {n_same} with the same sign.\n"
            f"Beating the shuffled-label null: gradient {grad_pass or 'none'} (unclear labels), activation "
            f"{sum(v['act'] for v in p.values())}.",
            transform=ax.transAxes, fontsize=9.8, color=INK2, ha="left", va="top")


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)


FOOT_AB = ("A feature counts if any of its 8 nearest unembedding tokens is in the word list. Activation difference, "
           "owl: 1 / 0 / 0.\nRandom = 1,000 Gaussian vectors. Single seed per fit; specificity (A) is at its "
           "pre-set threshold.")


def make_figure(d: dict, out: Path) -> None:
    """Panels A and B only (dolphin / owl)."""
    style()
    fig = plt.figure(figsize=(9.6, 8.4), dpi=200, facecolor="white")
    gs = fig.add_gridspec(2, 1, left=0.30, right=0.90, top=0.79, bottom=0.115, height_ratios=(1, 1.08),
                          hspace=0.66)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.text(0.03, 0.975, "The fitted update points at animal SAE features;\nthe activation difference does not",
             fontsize=17, fontweight="bold", ha="left", va="top")
    fig.text(0.03, 0.892, f"Dolphin and owl number data, layer {d['layer']}, Karvonen BatchTopK SAE "
             "(65,536 features). Tests fixed before the run.", fontsize=11, color=INK2, ha="left", va="top")
    draw_a(fig, ax_a, d, 0.03)
    draw_b(fig, ax_b, d, 0.03)
    fig.text(0.03, 0.022, FOOT_AB, fontsize=9, color=MUTED, ha="left", va="bottom")
    save(fig, out)


def make_main_figure(d: dict, c2: dict, out: Path) -> None:
    """Panels A and B (left column) and C (right column)."""
    style()
    fig = plt.figure(figsize=(17.0, 8.4), dpi=200, facecolor="white")
    left = fig.add_gridspec(2, 1, left=0.17, right=0.535, top=0.79, bottom=0.115, height_ratios=(1, 1.08),
                            hspace=0.66)
    right = fig.add_gridspec(1, 1, left=0.745, right=0.975, top=0.79, bottom=0.215)
    ax_a, ax_b = fig.add_subplot(left[0]), fig.add_subplot(left[1])
    ax_c = fig.add_subplot(right[0])
    fig.text(0.017, 0.975, "Training vectors point at different SAE features than activations for dolphin, "
             "not for harmful − benign", fontsize=17, fontweight="bold", ha="left", va="top")
    fig.text(0.017, 0.918, f"Karvonen BatchTopK SAE on Qwen3-14B, layer {d['layer']} (65,536 features). "
             "A, B: fitted updates vs activation differences on dolphin / owl number data, tests fixed before "
             "the run. C: held-out behaviour bundles.", fontsize=11, color=INK2, ha="left", va="top")
    draw_a(fig, ax_a, d, 0.017)
    draw_b(fig, ax_b, d, 0.017)
    draw_c(fig, ax_c, c2, 0.575)
    fig.text(0.017, 0.022, FOOT_AB, fontsize=9, color=MUTED, ha="left", va="bottom")
    save(fig, out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--enrichment-json", type=Path, help="raw enrichment output to rebuild the compact data from")
    p.add_argument("--case2-json", type=Path, help="raw case-2 decoder-cosine output to rebuild panel C's data")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--out-main", type=Path, default=OUT_MAIN)
    args = p.parse_args()
    data = json.loads(args.data.read_text()) if args.data.is_file() else {}
    if args.enrichment_json:
        data.update(extract(args.enrichment_json))
    if args.case2_json:
        data["case2"] = extract_case2(args.case2_json)
    if args.enrichment_json or args.case2_json:
        args.data.parent.mkdir(parents=True, exist_ok=True)
        args.data.write_text(json.dumps(data, indent=2) + "\n")
    make_figure(data, args.out)
    print(f"wrote {args.out} and {args.out.with_suffix('.svg')}")
    if "case2" in data:
        make_main_figure(data, data["case2"], args.out_main)
        print(f"wrote {args.out_main} and {args.out_main.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
