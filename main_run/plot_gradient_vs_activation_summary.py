"""One summary figure: what gradient input carries that activation input does not, and the reverse, across the
reader experiments.

Five readers, same colour and order in every panel: the 40-layer gradient reader, the 3-layer gradient reader, the
3-layer gradient reader with position slots, the 3-layer activation reader trained with the same recipe (the twin), and
the public Activation Oracle on centred activations of the same three layers. Gradient readers take warm hues,
activation readers cool hues (palette validated with the dataviz skill's validate_palette.py, light mode).

Panels
  A  held-out truth-judge YES rate per family (530 bundles), bundle-bootstrap 95% CI
       <beh_reader>/slot_reader/summary.json  results/family/<family>/<col>
  B  yes/no AUROC, bundles with the behaviour vs without, quirk / rare / backdoor, bootstrap 95% CI
       <beh_reader>/slot_reader/summary.json  results/yesno_pos_vs_neg/<family>/<col>
     note text: median own-question P(Yes) on bundles with the behaviour, from the per-bundle rows behind each AUROC
       <dir of results' yesno_sources/<col>/path>/pyes_bundle.jsonl  (qkey == own, positive, p_yes); the AUROC
       recomputed from these rows must equal the summary's (asserted)
  C  already-learned behaviour: P(Yes) on the behaviour's own documents, base model -> model that learned it (M1),
     and the other behaviours' documents under M1; the twin's open-answer YES share as a second marker
       <beh_reader>/new_readers_suite/summary.json  experiments/e5/table/<metric>/<col>
  D  clean-batch reading: P(Yes) real dolphin batch minus clean batches (dolphin student), plus the 3-layer gradient
     reader on the plain batch gradient as a grey bar
       <beh_reader>/new_readers_suite/summary.json  experiments/e3/table/dolphin.diff/<col>
  E  left: share of answers tagged 'occasional' on rare bundles (all 63, per-bundle share averaged)
       <beh_reader>/answer_tags/<run>/summary.json  per_family/rare/occasional
     right: share of answers naming some specific behaviour on control bundles (all families)
       <beh_reader>/slot_reader/summary.json  results/tags/<col>/all/other_named_on_controls
  F  subliminal trajectory: share of answers judged dolphin over all 20 read steps (dolphin student), and the first
     step rule (YES >= 1/30 and >= 5x the clean student)
       <beh_reader>/new_readers_suite/summary.json  experiments/e2/table/{dolphin_all,first_step}/<col>

No number is hard-coded: every plotted value is read from the files above and written with its path and key to the
json next to the png (same stem). The table of plotted values is printed at the end.

  python main_run/plot_gradient_vs_activation_summary.py [--out docs/figures/20261029_gradient_vs_activation_summary.png]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BEH = Path(os.environ["IIF_BASE"]) / "agm_data/runs/beh_reader"
SLOT = BEH / "slot_reader/summary.json"
SUITE = BEH / "new_readers_suite/summary.json"
OUT = REPO / "docs/figures/20261029_gradient_vs_activation_summary.png"

# id, legend label, colour, column in slot_reader/summary.json, column in new_readers_suite/summary.json,
# answer_tags run directory. Colours: reference palette (dataviz skill references/palette.md, light column) red,
# yellow, magenta for the gradient readers and blue, aqua for the activation readers; this order passes the light-mode
# validator on adjacent pairs (worst CVD dE 13.0, worst normal-vision dE 19.6). Orange is left out: it sits below the
# normal-vision floor next to every other warm hue.
SERIES = [
    ("reader40", "40-layer gradient reader", "#e34948", "reader40", "reader40", "reader"),
    ("grad3", "3-layer gradient reader", "#eda100", "grad3", "grad3", "grad3_v2"),
    ("slot", "3-layer gradient + position slots", "#e87ba4", "slot", "slot", "reader_L3slot"),
    ("twin", "3-layer activation reader, same recipe", "#2a78d6", "twin", "twin", "twin_v2"),
    ("ao", "public Activation Oracle (3 layers, centred activations)", "#1baf7a", "ao3_act_centred", "ao3",
     "ao3layer_act_centred"),
]
GREY_RAW = "#c3c2b7"  # palette.md chart chrome (baseline step), used for the one non-series bar in panel D
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#ffffff"
FAMILIES_A = ["quirk", "backdoor", "rare", "rollouts", "harmful", "benign", "heuristic", "auditbench", "all"]
FAMILIES_B = ["quirk", "rare", "backdoor"]

RECORDS: list[dict] = []


def pick(obj: dict, keys: list[str], src: Path, panel: str, series: str, quantity: str, **extra):
    """Walk obj along keys, record the value with its source path and key, and return it."""
    v = obj
    for k in keys:
        v = v[k]
    RECORDS.append(dict(panel=panel, series=series, quantity=quantity, value=v, source=str(src), key="/".join(keys),
                        **extra))
    return v


# ---------------------------------------------------------------- drawing helpers
def style_axis(ax, ygrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=INK2, labelsize=8, length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def units_per_pt(ax):
    """Data units per typographic point along x and y (axes limits must be final)."""
    fig = ax.figure
    bb = ax.get_position()
    w_pt = bb.width * fig.get_figwidth() * 72
    h_pt = bb.height * fig.get_figheight() * 72
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x1 - x0) / w_pt, (y1 - y0) / h_pt


def rbar(ax, xc, h, width, color, radius_pt=2.2, gap_pt=1.4, zorder=2):
    """Column from 0 to h with a rounded data end and a square baseline; a surface gap of gap_pt between neighbours."""
    ux, uy = units_per_pt(ax)
    w = max(width - gap_pt * ux, width * 0.3)
    x0 = xc - w / 2
    if h <= 0:
        if h < 0:
            ax.add_patch(Rectangle((x0, h), w, -h, facecolor=color, edgecolor="none", zorder=zorder))
        return
    rx = min(radius_pt * ux, w / 2)
    ry = rx * uy / ux
    if h <= ry * 1.2:
        ax.add_patch(Rectangle((x0, 0), w, h, facecolor=color, edgecolor="none", zorder=zorder))
        return
    ax.add_patch(FancyBboxPatch((x0, 0), w, h, boxstyle=f"round,pad=0,rounding_size={rx}", mutation_aspect=uy / ux,
                                facecolor=color, edgecolor="none", zorder=zorder))
    ax.add_patch(Rectangle((x0, 0), w, h - ry, facecolor=color, edgecolor="none", zorder=zorder))


def whisker(ax, x, lo, hi, zorder=3):
    if hi > lo:
        ax.vlines(x, lo, hi, color=INK2, linewidth=0.8, zorder=zorder)


def dot(ax, x, y, color, size=6.0, hollow=False, zorder=4, marker="o"):
    ax.plot([x], [y], marker=marker, markersize=size, markerfacecolor=SURFACE if hollow else color,
            markeredgecolor=color if hollow else SURFACE, markeredgewidth=1.3 if hollow else 1.0, linestyle="none",
            zorder=zorder)


def title(ax, letter, text, x=0.0):
    ax.text(x, 1.0, "", transform=ax.transAxes)  # keeps title position stable when ax has no data on top
    ax.set_title(f"{letter}   {text}" if letter else text, loc="left", fontsize=9.5, color=INK, pad=8,
                 fontweight="normal", linespacing=1.25)


# ---------------------------------------------------------------- panels
def panel_a(ax, slot):
    n_s = len(SERIES)
    slot_w = 0.16
    xs = []
    for g, fam in enumerate(FAMILIES_A):
        xs.append(g + (0.45 if fam == "all" else 0.0))
    ax.set_xlim(-0.55, xs[-1] + 0.55)
    ax.set_ylim(0, 1.04)
    style_axis(ax)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    labels = []
    for g, fam in enumerate(FAMILIES_A):
        n = None
        for i, (sid, _, col, skey, _, _) in enumerate(SERIES):
            d = pick(slot, ["results", "family", fam, skey], SLOT, "A", sid, f"truth-judge YES rate, {fam}")
            RECORDS[-1].update(value=d["mean"], lo=d["lo"], hi=d["hi"], n=d["n"])
            x = xs[g] + (i - (n_s - 1) / 2) * slot_w
            rbar(ax, x, d["mean"], slot_w, col)
            whisker(ax, x, d["lo"], d["hi"])
            n = d["n"]
        labels.append(f"{fam}\nn = {n}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, color=INK2, linespacing=1.3)
    ax.axvline((xs[-2] + xs[-1]) / 2, color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("Truth-judge YES rate\n(held-out bundles)", fontsize=8.5, color=INK2)
    title(ax, "A", "Trained on the same data, the activation reader names the planted behaviour most often; "
                   "the public AO almost never does")


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def auroc(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def bundle_pyes(slot):
    """Per-bundle own-question P(Yes) of each reader (the rows behind results/yesno_pos_vs_neg), keyed by series id.
    Integrity check: the AUROC recomputed from these rows equals the summary's for every (family, series)."""
    out, n_ok = {}, 0
    for sid, _, _, skey, _, _ in SERIES:
        src = Path(slot["yesno_sources"][skey]["path"]).parent / "pyes_bundle.jsonl"
        rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        own = [r for r in rows if r["qkey"] == "own"]
        out[sid] = (src, own)
        for fam in FAMILIES_B:
            pos = [r["p_yes"] for r in own if r["family"] == fam and r["positive"]]
            neg = [r["p_yes"] for r in own if r["family"] == fam and not r["positive"]]
            ref = slot["results"]["yesno_pos_vs_neg"][fam][skey]["auroc"]
            assert abs(auroc(pos, neg) - ref) < 1e-9, (sid, fam, auroc(pos, neg), ref)
            n_ok += 1
    print(f"panel B integrity: AUROC recomputed from pyes_bundle.jsonl equals the summary in {n_ok}/{n_ok} cells")
    return out


def panel_b(ax, slot):
    n_s = len(SERIES)
    step = 0.15
    ax.set_xlim(-0.55, len(FAMILIES_B) - 0.45)
    ax.set_ylim(0.08, 1.03)
    style_axis(ax)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1"])
    # median own-question P(Yes) on bundles with the behaviour: rare and backdoor rest on near-zero values
    per = bundle_pyes(slot)
    med_max = {}
    for fam in FAMILIES_B:
        meds = []
        for sid, *_ in SERIES:
            src, own = per[sid]
            m = median([r["p_yes"] for r in own if r["family"] == fam and r["positive"]])
            RECORDS.append(dict(panel="B", series=sid, quantity=f"median own-question P(Yes), {fam} bundles with the "
                                "behaviour (note text only)", value=m, source=str(src),
                                key=f"rows with qkey == 'own', family == '{fam}', positive == true: median of p_yes"))
            meds.append(m)
        med_max[fam] = max(meds)
    bound = max(med_max["rare"], med_max["backdoor"])
    bound_txt = f"{math.ceil(bound * 100) / 100:.2f}"
    ax.text(len(FAMILIES_B) - 0.47, 0.11, f"rare, backdoor: median P(Yes) on bundles\nwith the behaviour ≤ "
                                          f"{bound_txt} for every reader,\nso the AUROC ranks near-zero values",
            ha="right", va="bottom", fontsize=7.5, color=INK2, linespacing=1.2)
    ax.axhline(0.5, color=MUTED, linewidth=0.9, zorder=1)
    ax.text(-0.5, 0.505, "chance", ha="left", va="bottom", fontsize=7.5, color=MUTED)
    labels = []
    for g, fam in enumerate(FAMILIES_B):
        for i, (sid, _, col, skey, _, _) in enumerate(SERIES):
            d = pick(slot, ["results", "yesno_pos_vs_neg", fam, skey], SLOT, "B", sid, f"AUROC with vs without, {fam}")
            RECORDS[-1].update(value=d["auroc"], lo=d["lo"], hi=d["hi"], n_pos=d["n_pos"], n_neg=d["n_neg"])
            x = g + (i - (n_s - 1) / 2) * step
            whisker(ax, x, d["lo"], d["hi"], zorder=2)
            dot(ax, x, d["auroc"], col, size=6.5)
            n_pos, n_neg = d["n_pos"], d["n_neg"]
        labels.append(f"{fam}\n{n_pos} with\n{n_neg} without")
    ax.set_xticks(range(len(FAMILIES_B)))
    ax.set_xticklabels(labels, fontsize=8, color=INK2, linespacing=1.3)
    ax.set_ylabel("Yes/no AUROC, bundles with\nthe behaviour vs without", fontsize=8.5, color=INK2)
    title(ax, "B", "Gradient readers rank bundles with a behaviour\nabove controls, but outside quirk on "
                   "near-zero P(Yes)")


def panel_c(ax, suite):
    t = suite["experiments"]["e5"]["table"]
    base = ["experiments", "e5", "table"]
    ax.set_xlim(-0.02, 1.04)
    n_s = len(SERIES)
    ax.set_ylim(n_s - 0.35, -0.95)
    style_axis(ax, ygrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    for r, (sid, _, col, _, ukey, _) in enumerate(SERIES):
        b = pick(suite, base + ["pyes_own_base", ukey], SUITE, "C", sid, "P(Yes) own behaviour, base model")
        m = pick(suite, base + ["pyes_own_m1", ukey], SUITE, "C", sid, "P(Yes) own behaviour, after learning it")
        c = pick(suite, base + ["pyes_ctrl_m1", ukey], SUITE, "C", sid, "P(Yes) other behaviours, after learning X")
        if abs(m - b) > 0.03:
            ax.annotate("", xy=(m, r), xytext=(b, r), zorder=3,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=col, lw=1.6,
                                        shrinkA=3.5, shrinkB=3.5))
        dot(ax, b, r, col, size=6.5)
        dot(ax, m, r, col, size=6.5)
        dot(ax, c, r, col, size=6.5, hollow=True, zorder=5)
        if r == 0:  # direct labels for the three marker roles, once
            ax.text(b, r - 0.32, "base model", ha="center", va="bottom", fontsize=7.5, color=INK2)
            ax.text(m, r - 0.32, "after learning X", ha="left", va="bottom", fontsize=7.5, color=INK2)
            ax.text(c, r + 0.30, "other behaviours,\nafter learning X (open dot)", ha="left", va="top",
                    fontsize=7.5, color=INK2, linespacing=1.15)
        if sid == "twin":
            yb = pick(suite, base + ["yes_own_base", ukey], SUITE, "C", sid, "open-answer YES share, base model")
            ym = pick(suite, base + ["yes_own_m1", ukey], SUITE, "C", sid, "open-answer YES share, after learning it")
            yy = r + 0.36
            ax.annotate("", xy=(ym, yy), xytext=(yb, yy), zorder=3,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.16", color=col, lw=1.1,
                                        shrinkA=2.5, shrinkB=2.5))
            dot(ax, yb, yy, col, size=4.5, marker="D")
            dot(ax, ym, yy, col, size=4.5, marker="D")
            ax.text(ym + 0.03, yy, f"open answers judged YES {yb:.2f} → {ym:.2f}", ha="left", va="center",
                    fontsize=7.5, color=INK2)
            ax.text(min(b, m, c) - 0.03, r, f"P(Yes) saturated ≈{(b + m) / 2:.2f},\nalso for other behaviours",
                    ha="right", va="center", fontsize=7.5, color=INK2, linespacing=1.15)
    ax.set_xlabel("Reader's P(Yes) to the behaviour's yes/no question", fontsize=8.5, color=INK2)
    del t
    title(ax, "C", "After the model has learned the behaviour, gradient\nreaders switch off, also for other "
                   "behaviours;\nactivation readers do not")


def panel_d(ax, suite):
    base = ["experiments", "e3", "table", "dolphin.diff"]
    w = 0.7
    xs = list(range(len(SERIES)))
    x_raw = len(SERIES) + 0.35
    ax.set_xlim(-0.6, x_raw + 0.6)
    ax.set_ylim(0, 0.23)
    style_axis(ax)
    ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_yticklabels(["0", "0.05", "0.10", "0.15", "0.20"])
    vals = {}
    for x, (sid, _, col, _, ukey, _) in zip(xs, SERIES):
        v = pick(suite, base + [ukey], SUITE, "D", sid, "P(Yes) real dolphin batch minus clean batches")
        vals[sid] = v
        rbar(ax, x, v, w, col)
    raw = suite["experiments"]["e3"]["table"]["dolphin.diff"].get("grad3_grad")
    if raw is not None:
        raw = pick(suite, base + ["grad3_grad"], SUITE, "D", "grad3_raw_gradient",
                   "P(Yes) real dolphin batch minus clean batches, 3-layer gradient reader on the plain batch gradient")
        rbar(ax, x_raw, raw, w, GREY_RAW)
        y_lab = max(vals["twin"], vals["ao"], raw) + 0.03  # above the neighbouring bars
        ax.vlines(x_raw, raw + 0.003, y_lab - 0.003, color=MUTED, linewidth=0.7)
        ax.text(x_raw + 0.45, y_lab, "3-layer gradient\nreader, plain\nbatch gradient", ha="right", va="bottom",
                fontsize=7.5, color=INK2, linespacing=1.15)
    for sid in ("reader40", "slot"):
        x = [s[0] for s in SERIES].index(sid)
        ax.text(x, vals[sid] + 0.004, f"{vals[sid]:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks([])
    ax.set_ylabel("Δ P(Yes) 'prefers dolphins?'\nreal batch − clean batches", fontsize=8.5, color=INK2)
    title(ax, "D", "Real dolphin batch vs clean batches: largest\nfor the 40-layer gradient reader; position slots\n"
                   "recover part of what the 3-layer sum loses")


def panel_e(ax_l, ax_r, slot):
    w = 0.7
    xs = list(range(len(SERIES)))
    # left: 'occasional' on rare bundles
    ax_l.set_xlim(-0.6, len(SERIES) - 0.4)
    ax_l.set_ylim(0, 0.35)
    style_axis(ax_l)
    ax_l.set_yticks([0, 0.1, 0.2, 0.3])
    ax_l.set_yticklabels(["0", "0.10", "0.20", "0.30"])
    occ = {}
    for x, (sid, _, col, _, _, run) in zip(xs, SERIES):
        src = BEH / "answer_tags" / run / "summary.json"
        s = json.loads(src.read_text())
        v = pick(s, ["per_family", "rare", "occasional"], src, "E", sid,
                 "share of answers tagged 'occasional', rare bundles", n_bundles=s["per_family"]["rare"]["n_bundles"])
        occ[sid] = v
        rbar(ax_l, x, v, w, col)
        if v == 0:
            ax_l.text(x, 0.004, "0", ha="center", va="bottom", fontsize=7.5, color=INK2)
    top = max(occ, key=occ.get)
    ax_l.text(xs[[s[0] for s in SERIES].index(top)], occ[top] + 0.006, f"{occ[top]:.2f}", ha="center", va="bottom",
              fontsize=7.5, color=INK)
    ax_l.set_xticks([])
    ax_l.set_ylabel("Share of answers saying\n'occasional', rare bundles", fontsize=8.5, color=INK2)
    # right: other behaviour named on control bundles, axis capped at 0.2
    cap = 0.2
    ax_r.set_xlim(-0.6, len(SERIES) - 0.4)
    ax_r.set_ylim(0, cap * 1.12)
    style_axis(ax_r)
    ax_r.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax_r.set_yticklabels(["0", "0.05", "0.10", "0.15", "0.20"])
    for x, (sid, _, col, skey, _, _) in zip(xs, SERIES):
        d = pick(slot, ["results", "tags", skey, "all", "other_named_on_controls"], SLOT, "E", sid,
                 "share of answers naming a specific behaviour, control bundles (all families)")
        RECORDS[-1].update(value=d["mean"], lo=d["lo"], hi=d["hi"], n=d["n"])
        v = d["mean"]
        if v > cap:
            rbar(ax_r, x, cap, w, col)
            ux, uy = units_per_pt(ax_r)
            yb = cap * 0.86
            ax_r.add_patch(Rectangle((x - w, yb), 2 * w, 2.2 * uy, facecolor=SURFACE, edgecolor="none", zorder=3))
            ax_r.text(x, cap + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
            RECORDS[-1]["drawn_clipped_at"] = cap
        else:
            rbar(ax_r, x, v, w, col)
            whisker(ax_r, x, d["lo"], d["hi"])
    ax_r.set_xticks([])
    ax_r.set_ylabel("Share of answers naming a\nbehaviour, control bundles", fontsize=8.5, color=INK2)


def panel_f(ax, suite):
    base = ["experiments", "e2", "table"]
    w = 0.7
    xs = list(range(len(SERIES)))
    ax.set_xlim(-0.6, len(SERIES) - 0.4)
    ax.set_ylim(0, 0.04)
    style_axis(ax)
    ax.set_yticks([0, 0.01, 0.02, 0.03, 0.04])
    ax.set_yticklabels(["0", "0.01", "0.02", "0.03", "0.04"])
    present = suite["experiments"]["e2"]["columns"]
    for x, (sid, _, col, _, ukey, _) in zip(xs, SERIES):
        if ukey not in present:
            ax.text(x, 0.0008, "n/a", ha="center", va="bottom", fontsize=7.5, color=MUTED)
            RECORDS.append(dict(panel="F", series=sid, quantity="no trajectory run", value=None, source=str(SUITE),
                                key="/".join(base + ["dolphin_all"]) + f" (column '{ukey}' absent)"))
            continue
        v = pick(suite, base + ["dolphin_all", ukey], SUITE, "F", sid,
                 "share of answers judged dolphin, dolphin student, all 20 read steps")
        fs = pick(suite, base + ["first_step", ukey], SUITE, "F", sid,
                  "first read step with YES >= 1/30 and >= 5x the clean student")
        rbar(ax, x, v, w, col)
        lab = f"first at\nstep {fs:,}" if fs is not None else "never"
        if v == 0:
            lab = "0\n" + lab
        ax.text(x, v + 0.0008, lab, ha="center", va="bottom", fontsize=7.5, color=INK2, linespacing=1.15)
    ax.set_xticks([])
    ax.set_ylabel("Share of answers naming\ndolphin, 20 read steps", fontsize=8.5, color=INK2)
    title(ax, "F", "During training, gradient readers name the dolphin\npreference at a few steps; "
                   "the activation reader never does")


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    slot = json.loads(SLOT.read_text())
    suite = json.loads(SUITE.read_text())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    fig = plt.figure(figsize=(15, 11.2))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.08, 0.9], hspace=0.62, left=0.065, right=0.985,
                  top=0.905, bottom=0.075)
    ax_a = fig.add_subplot(gs[0])
    row2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], width_ratios=[1.0, 1.25, 1.0], wspace=0.32)
    ax_b, ax_c, ax_d = (fig.add_subplot(row2[i]) for i in range(3))
    row3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], width_ratios=[1.0, 1.0, 1.25], wspace=0.32)
    ax_el, ax_er, ax_f = (fig.add_subplot(row3[i]) for i in range(3))

    panel_a(ax_a, slot)
    panel_b(ax_b, slot)
    panel_c(ax_c, suite)
    panel_d(ax_d, suite)
    panel_e(ax_el, ax_er, slot)
    panel_f(ax_f, suite)
    # one title across the two E axes
    bl = ax_el.get_position()
    fig.text(bl.x0, bl.y1 + 0.012, "E   Only the 40-layer gradient reader says 'occasional' on rare bundles;\n"
                                     "only the public AO names a specific behaviour on control bundles",
             ha="left", va="bottom", fontsize=9.5, color=INK, linespacing=1.25)

    # one legend for the whole figure, grouped by input
    def header(t):
        return Patch(facecolor="none", edgecolor="none", label=t)
    row1 = [header("Gradient input:")] + [Patch(facecolor=c, edgecolor="none", label=l) for _, l, c, *_ in SERIES[:3]]
    row2 = [header("Activation input:")] + [Patch(facecolor=c, edgecolor="none", label=l) for _, l, c, *_ in SERIES[3:]]
    row2 += [header("")] * (len(row1) - len(row2))
    handles = [h for pair in zip(row1, row2) for h in pair]  # legend fills column-major: two rows in series order
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.988), ncol=len(row1), frameon=False,
                     fontsize=8.5, handlelength=1.1, handleheight=1.1, columnspacing=1.6, labelcolor=INK)
    for t in leg.get_texts():
        if t.get_text().endswith(":"):
            t.set_fontweight("bold")
            t.set_color(INK2)

    fig.text(0.065, 0.012,
             "Each reader trained once (one seed). Error bars: 95% bundle-bootstrap CI. Held-out set: 530 bundles. "
             "D: the 40-layer and 3-layer gradient readers read the momentum-free update; the slot reader and the grey "
             "bar read the plain batch gradient.\nF: the 40-layer reader reads a different input of the same kind "
             "(Adam direct-write term, 1,000-step windows); first step = first read step where at least 1/30 of "
             "answers name dolphin and at least 5x the clean student.",
             ha="left", va="bottom", fontsize=7.5, color=MUTED, linespacing=1.35)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    meta = dict(
        figure=str(out), pdf=str(out.with_suffix(".pdf")), script=str(Path(__file__).resolve()),
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
        sources=dict(slot_reader=str(SLOT), new_readers_suite=str(SUITE),
                     answer_tags={sid: str(BEH / "answer_tags" / run / "summary.json") for sid, *_, run in SERIES}),
        series=[dict(id=sid, label=lab, colour=col, slot_column=sk, suite_column=uk, answer_tags_run=run)
                for sid, lab, col, sk, uk, run in SERIES],
        grey_bar=dict(id="grad3_raw_gradient", colour=GREY_RAW,
                      label="3-layer gradient reader on the plain batch gradient (panel D only)"),
        palette_check=("dataviz validate_palette.py, light mode, surface #fcfcfb, adjacent pairs: lightness band PASS, "
                       "chroma PASS, CVD PASS (worst 13.0), normal-vision PASS (worst 19.6), contrast WARN for yellow "
                       "2.11, magenta 2.62, aqua 2.74 (relief: legend + this table)"),
        values=RECORDS,
    )
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    print(f"wrote {out}\n      {out.with_suffix('.pdf')}\n      {out.with_suffix('.json')}")
    print(f"{'panel':5} {'series':20} {'value':>8} {'lo':>7} {'hi':>7}  quantity")
    for r in RECORDS:
        v = r["value"]
        vs = "n/a" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.4f}")
        lo = f"{r['lo']:.3f}" if r.get("lo") is not None else ""
        hi = f"{r['hi']:.3f}" if r.get("hi") is not None else ""
        print(f"{r['panel']:5} {r['series']:20} {vs:>8} {lo:>7} {hi:>7}  {r['quantity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
