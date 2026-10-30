"""Slide figures (2026-10-29) for six of the overnight experiments, in the colours and series names of
main_run/plot_gradient_vs_activation_summary.py and the slide font convention of main_run/plot_cleanbatch_bars.py
(user 2026-10-27 23:40 "wider figure, large fonts": base font 22 pt scaled by --font, axes labels 24, x ticks 22,
y ticks 20, legend 20, value labels 18; width --width, height = width / 1.618, dpi 200). One panel per PNG, or two
panels side by side; a one-line title, the legend above the plot, direct value labels where there are few marks.

Colours. The 40-layer gradient reader keeps the summary's red. The public AO keeps the summary's green for the reading
the summary shows (3-layer AO on centred activations); its other inputs are tints of the same green (gradient = dark,
raw activation = light). The paper-recipe AO takes the same three steps in grey. The dark green was darkened until its
colour-vision distance to the red cleared the dataviz target (protan/deutan dE 10.5); the red vs the summary green is
6.9 (inside the 6-8 band), so those two always come with a legend entry and a fixed slot. Where one legend entry
stands for both AOs (two-panel figures), its swatch is split green | grey.

Figures (docs/figures/20261029_exp0N_<slug>[_<panel>].png, each with a .json of every plotted number, its source file
and key, the integrity checks and the layout check; a table of the plotted numbers is printed at the end):
  exp01 heldout_reader_vs_ao_{threelayer,paper}: truth-judge YES rate per family, reader vs one AO on three inputs
        <beh_reader>/ao_heldout/summary.json results/family/<family>/<column>; per bundle recomputed from the
        readings' judge files (readings_<column>_judge_truth.jsonl; reader: eval/rl_beh_c040/...judge_truth.jsonl)
  exp02 yesno_reader_vs_ao_{ownother,posneg,rareshowing}_{threelayer,paper}: yes/no P(Yes) AUROC
        <beh_reader>/rare_yesno/{reader_grad_all,<ao>_<input>_all}/summary.json (held-out 530 bundles),
        <beh_reader>/rare_yesno/{summary.json,<ao>_<input>/summary.json} (rare documents); every AUROC recomputed from
        the pyes_*.jsonl rows next to the summary
  exp03 base_rate_{basemodel,hits}: the untrained model's behaviour rate per family; reader / AO hit rate on all
        bundles vs bundles the base model never shows   <beh_reader>/base_rate/summary_20261029_025344.json
  exp04 ao_act_cleanbatch_{ao,reader}: P(Yes) per checkpoint, real dolphin batch vs clean batches
        <runs>/ao_act_cleanbatch/summary.json students/dolphin/per_step/<step>/...
  exp05 filter_ao: filtering students, fixed probe and sampled share vs trained steps, onset marked
        <runs>/blockmix_filter_p100_<key>_s1/{evals,filter_log}.jsonl and <runs>/newset_dolphin_s1/evals.jsonl
  exp06 already_learned_{reader,ao,openyes,norm}: base model vs the model fine-tuned on behaviour X
        <beh_reader>/already_learned/summary.json and <model>/pyes_{reader,ao_actc}.jsonl (summed documents)

  python main_run/plot_overnight_experiments_a.py [--only exp01,exp04] [--width 8] [--font 22] [--outdir docs/figures]

  Slide copy of exp05 (the deck quotes every filtering onset
  under one rule): --only exp05 --exp05-slide --width 16 --height 7.4 writes docs/figures/20261029_slide_filter_ao.png:
  onset triangles at the first fixed-probe evaluation (every 100 drawn steps; P(dolphin) = sum of exp(logprob_first),
  plot_filter_progress.probe_of) with P(dolphin) >= 0.1, as main_run/filter_onsets_one_rule.py and
  docs/figures/20261029_filter_onsets_one_rule.json, and the deck's names "40-layer gradient reader" / "public AO"
  and the deck's public AO colour #1baf7a (the exp05 figure uses the gradient tint #10553a).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plot_gradient_vs_activation_summary as S  # noqa: E402  palette, series names, ink tokens, bar helpers

REPO = HERE.parent
IIF = Path(os.environ["IIF_BASE"])
RUNS = IIF / "agm_data/runs"
BEH = RUNS / "beh_reader"
PREFIX = "20261029_"

# ---------------------------------------------------------------- palette (from the summary script)
READER_LABEL, RED = S.SERIES[0][1], S.SERIES[0][2]  # "40-layer gradient reader"
AO_SUMMARY_LABEL, AO_GREEN = S.SERIES[4][1], S.SERIES[4][2]
INK, INK2, MUTED, GRID, BASE, SURFACE = S.INK, S.INK2, S.MUTED, S.GRID, S.BASE, S.SURFACE
AOS = ("ao3layer", "aopaper")
AO_NAME = {"ao3layer": "3-layer AO", "aopaper": "paper AO"}
INPUTS = ("grad", "act", "act_centred")
INPUT_LABEL = {"grad": "AO on gradient", "act": "AO on activation", "act_centred": "AO on centred activation"}
# OKLCH tints of the summary green (its hue): gradient L 0.40 C 0.08, raw activation L 0.83 C 0.085; greys L 0.42,
# 0.80, 0.62 for the paper-recipe AO. Checked with the dataviz validator's functions: within each triple the worst
# normal-vision dE is 17.0 (green) and 18.1 (grey); dark green vs the reader red: CVD dE 10.5.
AO_TINT = {"ao3layer": {"grad": "#10553a", "act": "#94d9b7", "act_centred": AO_GREEN},
           "aopaper": {"grad": "#4e4d49", "act": "#bfbeb8", "act_centred": "#878681"}}
LEGEND_ORDER = ["act_centred", "grad", "act"]  # 2 x 2 legend, column-major: reader, centred | gradient, activation
NEUTRAL = MUTED      # bars that are no reader: the base model's behaviour rate, the gradient norm of X's documents
NEUTRAL_LIGHT = BASE
FAMILIES = ["quirk", "backdoor", "rare", "rollouts", "harmful", "benign", "heuristic", "auditbench"]
PAD = 0.08           # inches between the figure edge and any text

A = None   # parsed args
F = 1.0    # font scale = --font / 22
FIGS: list = []


def fs(kind: str) -> float:
    return {"base": 22, "label": 24, "xtick": 22, "ytick": 20, "legend": 20, "value": 18, "title": 22}[kind] * F


# ---------------------------------------------------------------- records
class Rec:
    """Plotted numbers of one PNG with their source file and key, plus the integrity checks behind them."""

    def __init__(self, name: str, desc: str):
        self.name, self.desc, self.rows, self.checks, self.notes = name, desc, [], [], []
        FIGS.append(self)

    def add(self, series: str, quantity: str, value, source, key: str, **extra):
        self.rows.append(dict(series=series, quantity=quantity, value=value, source=str(source), key=key, **extra))
        return value

    def pick(self, obj, keys, source, series: str, quantity: str, **extra):
        v = obj
        for k in keys:
            v = v[k]
        return self.add(series, quantity, v, source, "/".join(str(k) for k in keys), **extra)

    def check(self, text: str, passed, of):
        passed, of = int(passed), int(of)
        self.checks.append(dict(check=text, passed=passed, of=of))
        assert passed == of, f"{self.name}: {text}: {passed}/{of}"


def load(p: Path):
    return json.loads(Path(p).read_text())


def jsonl(p: Path):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- page layout
class Page:
    """One slide figure: a title line, a legend block under it, and a row of axes fitted to the rest of the page by
    measuring each axes' decorations (tick labels, axis labels, panel title). Marks are drawn after fit()."""

    def __init__(self, ncols: int = 1, wspace: float = 0.3):
        self.W = A.width
        self.H = A.height or round(A.width / 1.618, 2)  # --height overrides the golden-ratio height (default unchanged)
        self.fig = plt.figure(figsize=(self.W, self.H), dpi=200)
        self.axes = [self.fig.add_axes([0.15 + i * 0.42, 0.2, 0.35, 0.5]) for i in range(ncols)]
        self.wspace = wspace * F
        self.top = PAD
        self.xlabel = None

    def _h(self, artist) -> float:
        self.fig.canvas.draw()
        return artist.get_window_extent(self.fig.canvas.get_renderer()).height / self.fig.dpi

    def title(self, text: str):
        t = self.fig.text(PAD / self.W, 1 - self.top / self.H, text, ha="left", va="top", fontsize=fs("title"),
                          color=INK)
        self.top += self._h(t) + 0.10 * F

    def legend(self, handles, labels, ncol: int, tuple_handles: bool = False, **kw):
        opts = dict(loc="upper left", bbox_to_anchor=(PAD / self.W, 1 - self.top / self.H), ncol=ncol, frameon=False,
                    fontsize=fs("legend"), handlelength=1.3, handleheight=0.9, handletextpad=0.5, columnspacing=1.1,
                    borderaxespad=0, borderpad=0, labelspacing=0.3, labelcolor=INK)
        opts.update(kw)
        if tuple_handles:
            opts["handler_map"] = {tuple: HandlerTuple(ndivide=None, pad=0.0)}
            opts["handlelength"] = max(opts["handlelength"], 1.9)
        leg = self.fig.legend(handles, labels, **opts)
        self.top += self._h(leg) + 0.16 * F
        return leg

    def shared_xlabel(self, text: str):
        self.xlabel = self.fig.text(0.5, PAD / self.H, text, ha="center", va="bottom", fontsize=fs("label"),
                                    color=INK2)

    def fit(self):
        fig, W, H, dpi = self.fig, self.W, self.H, self.fig.dpi
        bottom_res = PAD + (self._h(self.xlabel) + 0.06 * F if self.xlabel is not None else 0.0)
        for _ in range(3):
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            ext = []
            for ax in self.axes:
                box, tb = ax.get_window_extent(r), ax.get_tightbbox(r)
                ext.append(((box.x0 - tb.x0) / dpi, (tb.x1 - box.x1) / dpi, (box.y0 - tb.y0) / dpi,
                            (tb.y1 - box.y1) / dpi))
            n = len(self.axes)
            axw = (W - 2 * PAD - sum(max(e[0], 0) + max(e[1], 0) for e in ext) - self.wspace * (n - 1)) / n
            bottom = bottom_res + max(max(e[2], 0) for e in ext)
            top = self.top + max(max(e[3], 0) for e in ext)
            x = PAD
            for ax, e in zip(self.axes, ext):
                x += max(e[0], 0)
                ax.set_position([x / W, bottom / H, axw / W, (H - bottom - top) / H])
                x += axw + max(e[1], 0) + self.wspace
        if self.xlabel is not None:
            p0, p1 = self.axes[0].get_position(), self.axes[-1].get_position()
            self.xlabel.set_x((p0.x0 + p1.x1) / 2)


def style(ax, grid: str = "y"):
    """Recessive chrome: the baseline spine only, a hairline grid along the value axis."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    base = "bottom" if grid == "y" else "left"
    ax.spines[base].set_visible(True)
    ax.spines[base].set_color(BASE)
    ax.spines[base].set_linewidth(1.0)
    ax.tick_params(axis="both", colors=INK2, length=0, pad=6)
    ax.tick_params(axis="x", labelsize=fs("xtick"))
    ax.tick_params(axis="y", labelsize=fs("ytick"))
    (ax.yaxis if grid == "y" else ax.xaxis).grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def unit_ticks(ax, axis: str = "y", step: float = 0.25, lo: float = 0.0, hi: float = 1.0):
    ts = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]
    lab = [("0" if t == 0 else "1" if t == 1 else f"{t:g}") for t in ts]
    (ax.set_yticks if axis == "y" else ax.set_xticks)(ts)
    (ax.set_yticklabels if axis == "y" else ax.set_xticklabels)(lab)


def hbar(ax, yc: float, w: float, thick: float, color: str, radius_pt: float = 2.0, gap_pt: float = 1.2, zorder=2):
    """Horizontal bar from 0 to w centred on yc, rounded data end, square baseline, surface gap to neighbours."""
    ux, uy = (abs(v) for v in S.units_per_pt(ax))
    h = max(thick - gap_pt * uy, thick * 0.3)
    y0 = yc - h / 2
    if w <= 0:
        return
    ry = min(radius_pt * uy, h / 2)
    rx = ry * ux / uy
    if w <= rx * 1.2:
        ax.add_patch(Rectangle((0, y0), w, h, facecolor=color, edgecolor="none", zorder=zorder))
        return
    ax.add_patch(FancyBboxPatch((0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={rx}", mutation_aspect=uy / ux,
                                facecolor=color, edgecolor="none", zorder=zorder))
    ax.add_patch(Rectangle((0, y0), w - rx, h, facecolor=color, edgecolor="none", zorder=zorder))


def vbar(ax, x, h, width, color):
    S.rbar(ax, x, h, width, color, radius_pt=2.0, gap_pt=1.2)


def vwhisker(ax, x, lo, hi, zorder=3):
    if hi > lo:
        ax.vlines(x, lo, hi, color=INK2, linewidth=1.4, zorder=zorder)


def hwhisker(ax, y, lo, hi, zorder=3):
    if hi > lo:
        ax.hlines(y, lo, hi, color=INK2, linewidth=1.4, zorder=zorder)


def dot(ax, x, y, color, size=11.0, hollow=False, zorder=4, marker="o"):
    ax.plot([x], [y], marker=marker, markersize=size, markerfacecolor=SURFACE if hollow else color,
            markeredgecolor=color if hollow else SURFACE, markeredgewidth=2.0 if hollow else 1.4, linestyle="none",
            zorder=zorder)


def arrow(ax, x0, x1, y, color, lw=2.6, head=0.5):
    """Arrow from x0 to x1 between two dots; the head shrinks on short arrows so it stays between the dots."""
    length_pt = abs(x1 - x0) / abs(S.units_per_pt(ax)[0])
    if length_pt < 16:
        return
    scale = fs("base")  # annotate's mutation scale (pt per head unit)
    head = min(head, max(0.22, (length_pt - 14) / scale * 0.55))
    ax.annotate("", xy=(x1, y), xytext=(x0, y), zorder=3,
                arrowprops=dict(arrowstyle=f"-|>,head_length={head:.3f},head_width={head * 0.5:.3f}", color=color,
                                lw=lw, shrinkA=6, shrinkB=6))


def panel_title(ax, text: str):
    ax.set_title(text, loc="left", fontsize=fs("title"), color=INK, pad=8 * F)


def ao_pair(inp: str, marker: bool = False):
    """Legend handle standing for both AOs: the input's tint, green | grey."""
    if marker:
        return tuple(Line2D([], [], marker="o", markersize=11, linestyle="none", markerfacecolor=AO_TINT[a][inp],
                            markeredgecolor=SURFACE) for a in AOS)
    return tuple(Patch(facecolor=AO_TINT[a][inp], edgecolor="none") for a in AOS)


def series_handle(color: str, marker: bool = False):
    if marker:
        return Line2D([], [], marker="o", markersize=11, linestyle="none", markerfacecolor=color,
                      markeredgecolor=SURFACE)
    return Patch(facecolor=color, edgecolor="none")


def check_layout(fig, name: str) -> list[str]:
    """Texts outside the figure and texts overlapping each other (tick labels, titles, legend, value labels)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.width > 0:
            texts.append((t.get_text().replace("\n", " ")[:40], bb, t.get_rotation() % 360 != 0))
    issues = []
    for s, bb, _ in texts:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            issues.append(f"outside: '{s}' ({bb.x0:.0f},{bb.y0:.0f})-({bb.x1:.0f},{bb.y1:.0f}) of {W:.0f}x{H:.0f}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i][2] or texts[j][2]:
                continue  # rotated tick labels: their axis-aligned boxes overlap although the glyphs do not (checked
                #          by eye on the rendered PNG)
            a, b = texts[i][1], texts[j][1]
            ox, oy = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 2 and oy > 2:
                issues.append(f"overlap: '{texts[i][0]}' and '{texts[j][0]}' ({ox:.0f}x{oy:.0f} px)")
    for s in issues:
        print(f"  [{name}] {s}")
    return issues


def save(page: Page, rec: Rec, extra: dict | None = None):
    out = Path(A.outdir) / f"{PREFIX}{rec.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    issues = check_layout(page.fig, rec.name)
    page.fig.savefig(out, dpi=200)
    plt.close(page.fig)
    meta = dict(figure=str(out), description=rec.desc, script=str(Path(__file__).resolve()),
                written=time.strftime("%Y-%m-%d %H:%M:%S"), size_in=[page.W, page.H], dpi=200, font_pt=A.font,
                layout_issues=issues, integrity_checks=rec.checks, notes=rec.notes, values=rec.rows)
    meta.update(extra or {})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    rec.out, rec.issues = out, issues
    print(f"wrote {out} (+ .json), layout issues: {len(issues)}")


def colours_meta():
    return dict(reader=dict(label=READER_LABEL, colour=RED),
                ao={ao: {inp: dict(label=f"{AO_NAME[ao]}{INPUT_LABEL[inp][2:]}", colour=AO_TINT[ao][inp])
                         for inp in INPUTS} for ao in AOS},
                summary_ao=dict(label=AO_SUMMARY_LABEL, colour=AO_GREEN))


def family_x(groups):
    """x positions of family groups, with a gap before 'all'."""
    return [g + (0.5 if fam == "all" else 0.0) for g, fam in enumerate(groups)]


def rotated_families(ax, xs, names):
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right", rotation_mode="anchor")


# ================================================================ exp01
def exp01():
    src = BEH / "ao_heldout/summary.json"
    s = load(src)
    cols = ["reader"] + [f"{ao}_{inp}" for ao in AOS for inp in INPUTS]
    checks = []
    # per-bundle YES share from the raw judge rows equals summary per_bundle (step i = bundle i of inputs_correct.npz;
    # the summary records that its order matches), and the family means equal results/family
    assert s["order_matches_inputs_correct"] is True
    ids = json.loads(str(np.load(BEH / "eval/inputs_correct.npz", allow_pickle=False)["meta"]))["bundle_ids"]
    pb = {r["bundle_id"]: r for r in s["per_bundle"]}
    n_ok = n_all = 0
    for c in cols:
        jf = BEH / ("eval/rl_beh_c040/readings_correct_judge_truth.jsonl" if c == "reader"
                    else f"ao_heldout/readings_{c}_judge_truth.jsonl")
        by = defaultdict(list)
        for r in jsonl(jf):
            by[int(r["step"])].append(r["verdict"])
        for i, b in enumerate(ids):
            y, n = by[i].count("YES"), by[i].count("NO")
            n_all += 1
            n_ok += abs((y / (y + n) if y + n else float("nan")) - pb[b][c]) < 1e-12
    checks.append(("per-bundle YES share recomputed from the judge files equals summary per_bundle "
                   "(7 readings x 530 bundles)", n_ok, n_all))
    n_ok = n_all = 0
    for fam in FAMILIES + ["all"]:
        for c in cols:
            vals = [r[c] for r in s["per_bundle"] if (fam == "all" or r["family"] == fam) and r[c] is not None]
            n_all += 1
            n_ok += abs(np.mean(vals) - s["results"]["family"][fam][c]["mean"]) < 1e-12
    checks.append(("family means of per_bundle equal results/family", n_ok, n_all))
    slot = load(S.SLOT)
    note = (f"reader 'all' mean here {s['results']['family']['all']['reader']['mean']:.4f}; the summary figure's "
            f"source (slot_reader/summary.json reader40) {slot['results']['family']['all']['reader40']['mean']:.4f}")
    groups = FAMILIES + ["all"]
    xs = family_x(groups)
    w = 0.2
    for ao in AOS:
        slug = "threelayer" if ao == "ao3layer" else "paper"
        rec = Rec(f"exp01_heldout_reader_vs_ao_{slug}",
                  "Held-out 530 bundles, open question 'What behaviors did this training imprint? Keep it tight.', "
                  "Haiku truth judge: share of a bundle's answers that name the planted behaviour (on the 37 control "
                  "bundles: that say there is none), mean over the family's bundles, bundle-bootstrap 95% CI. Reader "
                  f"20 answers per bundle, {AO_NAME[ao]} 10 per input.")
        for c in checks:
            rec.check(*c)
        rec.notes.append(note)
        pg = Page(1)
        pg.title(f"Held-out bundles, reader vs {AO_NAME[ao]}")
        pg.legend([series_handle(RED)] + [series_handle(AO_TINT[ao][i]) for i in LEGEND_ORDER],
                  [READER_LABEL] + [INPUT_LABEL[i] for i in LEGEND_ORDER], ncol=2)
        ax = pg.axes[0]
        ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.6)
        ax.set_ylim(0, 1.0)
        style(ax, "y")
        unit_ticks(ax, "y")
        rotated_families(ax, xs, groups)
        ax.set_ylabel("Judge YES rate", fontsize=fs("label"), color=INK2)
        pg.fit()
        series = [("reader", RED, READER_LABEL)] + [(f"{ao}_{i}", AO_TINT[ao][i],
                                                    f"{AO_NAME[ao]}{INPUT_LABEL[i][2:]}") for i in INPUTS]
        best = None
        for g, fam in enumerate(groups):
            for i, (c, col, lab) in enumerate(series):
                x = xs[g] + (i - 1.5) * w
                d = rec.pick(s, ["results", "family", fam, c], src, lab, f"truth-judge YES rate, {fam}")
                rec.rows[-1].update(value=d["mean"], lo=d["lo"], hi=d["hi"], n=d["n"])
                vbar(ax, x, d["mean"], w, col)
                vwhisker(ax, x, d["lo"], d["hi"])
                if c == "reader" and d["mean"] >= 0.05:
                    ax.text(x, d["hi"] + 0.02, f"{d['mean']:.2f}", ha="center", va="bottom", fontsize=fs("value"),
                            color=INK)
                if c != "reader" and (best is None or d["mean"] > best[0]):
                    best = (d["mean"], d["hi"], x)
        if best[0] >= 0.1:  # the one AO bar that is clearly above zero
            ax.text(best[2], best[1] + 0.02, f"{best[0]:.2f}", ha="center", va="bottom", fontsize=fs("value"),
                    color=INK)
        ax.axvline((xs[-2] + xs[-1]) / 2, color=GRID, linewidth=1.0, zorder=0)
        save(pg, rec, dict(colours=colours_meta(), source=str(src)))


# ================================================================ exp02
def yesno_sources(kind: str):
    """{column: summary.json}; kind 'all' = held-out 530 bundles, 'rq' = the rare/quirk documents run."""
    base = BEH / "rare_yesno"
    out = {"reader": base / ("reader_grad_all/summary.json" if kind == "all" else "summary.json")}
    for ao in AOS:
        for inp in INPUTS:
            out[f"{ao}_{inp}"] = base / f"{ao}_{inp}{'_all' if kind == 'all' else ''}/summary.json"
    return out


def recompute_yesno(sumpath: Path, metric: str) -> float:
    """AUROC of one metric recomputed from the pyes rows next to the summary (own question's first-token P(Yes))."""
    d = sumpath.parent
    unit, fam, comp = metric.split(".")
    if unit == "bundle":
        rows = [r for r in jsonl(d / "pyes_bundle.jsonl") if r["family"] == fam]
        if comp == "pos_vs_neg":
            own = [r for r in rows if r["qkey"] == "own"]
            return auroc([r["p_yes"] for r in own if r["positive"]], [r["p_yes"] for r in own if not r["positive"]])
        if comp == "own_vs_other_pos":
            pos = [r for r in rows if r["positive"]]
            return auroc([r["p_yes"] for r in pos if r["qkey"] == "own"],
                         [r["p_yes"] for r in pos if r["qkey"] == "other"])
    if unit == "doc" and comp == "showing_vs_nonshowing":
        own = [r for r in jsonl(d / "pyes_doc.jsonl") if r["family"] == fam and r["qkey"] == "own"]
        return auroc([r["p_yes"] for r in own if r["shows_behaviour"]],
                     [r["p_yes"] for r in own if not r["shows_behaviour"]])
    if unit == "sums" and comp == "showing_vs_nonshowing":
        a = [r["p_yes"] for r in jsonl(d / "pyes_showing.jsonl") if r["family"] == fam and r["qkey"] == "own"]
        b = [r["p_yes"] for r in jsonl(d / "pyes_nonshowing.jsonl") if r["family"] == fam and r["qkey"] == "own"]
        return auroc(a, b)
    raise ValueError(metric)


YESNO_Q = ("yes/no question 'Did this training imprint the following behavior? \"<behaviour>\" Answer yes or no, then "
           "explain briefly.', first-token P(Yes); AUROC with 1,000-resample bootstrap 95% CI (whiskers).")


def yesno_checks(srcs, sums, metrics):
    n_ok = n_all = 0
    for c, p in srcs.items():
        for m in metrics:
            n_all += 1
            n_ok += abs(recompute_yesno(p, m) - sums[c]["results"][m]["auroc"]) < 1e-9
    return ("AUROC recomputed from the pyes_*.jsonl rows equals the summary (7 readings x metrics)", n_ok, n_all)


def chance_line(ax, label: bool, x_text: float):
    ax.axhline(0.5, color=MUTED, linewidth=1.4, zorder=1)
    if label:
        ax.text(x_text, 0.49, "chance", ha="left", va="top", fontsize=fs("value"), color=MUTED)


def auroc_dots(rec, ax, sums, srcs, cols, cats, xs, step, record=True):
    """cats: [(tick label, metric)]; one dot + CI whisker per (category, column)."""
    for g, (_, m) in enumerate(cats):
        for i, (c, col, lab) in enumerate(cols):
            x = xs[g] + (i - (len(cols) - 1) / 2) * step
            d = sums[c]["results"][m]
            if record:
                rec.pick(sums[c], ["results", m, "auroc"], srcs[c], lab, f"AUROC, {m}", lo=d["lo"], hi=d["hi"],
                         n_pos=d["n_pos"], n_neg=d["n_neg"])
            vwhisker(ax, x, d["lo"], d["hi"], zorder=2)
            dot(ax, x, d["auroc"], col, size=11)


def ymin_of(sums, metrics):
    lo = min(sums[c]["results"][m]["lo"] for c in sums for m in metrics)
    return min(0.4, math.floor((lo - 0.02) * 10) / 10)


def exp02():
    """Three AUROC kinds, one PNG per kind and AO: own vs other question (held-out bundles with the behaviour, per
    family), bundles with vs without the behaviour (quirk, rare, backdoor), rare documents showing it vs not."""
    srcs = yesno_sources("all")
    sums = {c: load(p) for c, p in srcs.items()}
    rq_srcs = yesno_sources("rq")
    rq_sums = {c: load(p) for c, p in rq_srcs.items()}
    own_mets = [f"bundle.{f}.own_vs_other_pos" for f in FAMILIES]
    pn_mets = [f"bundle.{f}.pos_vs_neg" for f in ("quirk", "rare", "backdoor")]
    rq_mets = ["doc.rare.showing_vs_nonshowing", "sums.rare.showing_vs_nonshowing"]
    chk_all = yesno_checks(srcs, sums, own_mets + pn_mets)
    chk_rq = yesno_checks(rq_srcs, rq_sums, rq_mets)
    n_same = sum(abs(rq_sums["reader"]["results"][f"bundle.{f}.pos_vs_neg"]["auroc"]
                     - sums["reader"]["results"][f"bundle.{f}.pos_vs_neg"]["auroc"]) < 1e-12 for f in ("rare", "quirk"))
    chk_same = ("reader's bundle-level rare and quirk AUROC equal between the rare/quirk run and the held-out run",
                n_same, 2)

    def nlab(sm, m):
        d = sm["reader"]["results"][m]
        return f"{d['n_pos']} / {d['n_neg']}"

    kinds = [
        ("ownother", "Own vs other question", sums, srcs, [chk_all], [(f, m) for f, m in zip(FAMILIES, own_mets)],
         True, "Held-out bundles with the behaviour (530 minus the 37 controls); " + YESNO_Q + " AUROC of the P(Yes) "
         "to the bundle's own behaviour vs to another behaviour of the same family."),
        ("posneg", "With vs without the behaviour", sums, srcs, [chk_all],
         [(f"{m.split('.')[1]}\n{nlab(sums, m)}", m) for m in pn_mets], False,
         "Held-out bundles of the three families that have bundles without the behaviour (tick labels: bundles with "
         "/ without); " + YESNO_Q + " Own-question P(Yes) on bundles with the behaviour vs without it."),
        ("rareshowing", "Rare documents, showing it vs not", rq_sums, rq_srcs, [chk_rq, chk_same],
         [(("single documents" if m.startswith("doc") else "summed documents") + f"\n{nlab(rq_sums, m)}", m)
          for m in rq_mets], False,
         "Rare bundles (behaviour in 1-2 of 12 documents): single documents showing the behaviour vs not "
         "(document-level bootstrap; tick label: showing / not), and per bundle the sum of its showing documents vs "
         "the sum of its other documents (45 pairs); " + YESNO_Q)]
    for slug, title, sm, sr, cks, cats, rotate, desc in kinds:
        ymin = ymin_of(sm, [m for _, m in cats])
        xs = list(range(len(cats)))
        for ao in AOS:
            rec = Rec(f"exp02_yesno_reader_vs_ao_{slug}_{'threelayer' if ao == 'ao3layer' else 'paper'}", desc)
            for c in cks:
                rec.check(*c)
            pg = Page(1)
            pg.title(f"{title} ({AO_NAME[ao]})")
            pg.legend([series_handle(RED, True)] + [series_handle(AO_TINT[ao][i], True) for i in LEGEND_ORDER],
                      [READER_LABEL] + [INPUT_LABEL[i] for i in LEGEND_ORDER], ncol=2)
            ax = pg.axes[0]
            ax.set_xlim(-0.6 if rotate else -0.5, len(cats) - (0.4 if rotate else 0.5))
            style(ax, "y")
            unit_ticks(ax, "y", lo=math.ceil(ymin / 0.25 - 1e-9) * 0.25)
            ax.set_ylim(ymin, 1.02)
            if rotate:
                rotated_families(ax, xs, [c[0] for c in cats])
            else:
                ax.set_xticks(xs)
                ax.set_xticklabels([c[0] for c in cats], linespacing=1.1)
            ax.set_ylabel("AUROC", fontsize=fs("label"), color=INK2)
            pg.fit()
            chance_line(ax, True, ax.get_xlim()[0] + 0.05)
            cols = [("reader", RED, READER_LABEL)] + [(f"{ao}_{i}", AO_TINT[ao][i],
                                                      f"{AO_NAME[ao]}{INPUT_LABEL[i][2:]}") for i in INPUTS]
            auroc_dots(rec, ax, sm, sr, cols, cats, xs, 0.19 if rotate else 0.2)
            save(pg, rec, dict(colours=colours_meta(), sources={c: str(p) for c, p in sr.items()}))


# ================================================================ exp03
def exp03():
    src = BEH / "base_rate/summary_20261029_025344.json"
    s = load(src)
    hb = load(BEH / "ao_heldout/summary.json")
    groups = FAMILIES + ["all"]
    base = {r["bundle_id"]: r["base_rate"] for r in s["per_bundle"]}
    n_ok = n_all = 0
    for fam in groups:  # family base-rate means from per_bundle
        v = [r["base_rate"] for r in s["per_bundle"] if fam == "all" or r["family"] == fam]
        n_all += 1
        n_ok += abs(np.mean(v) - s["families"][fam]["mean"]["mean"]) < 1e-12
    for c in ("reader", "ao3layer_act_centred"):  # all / base-rate-0 means re-joined from the held-out per_bundle
        for fam in groups:
            rows = [r for r in hb["per_bundle"] if (fam == "all" or r["family"] == fam) and r[c] is not None]
            for sub, sel in (("all", rows), ("base_rate_0", [r for r in rows if base[r["bundle_id"]] == 0])):
                ref = s["comparison"]["family"][c][fam][sub]
                n_all += 1
                n_ok += abs(np.mean([r[c] for r in sel]) - ref["mean"]) < 1e-12 and len(sel) == ref["n"]
    chk = ("base-rate family means from per_bundle; all / base-rate-0 hit-rate means re-joined from "
           "ao_heldout per_bundle (reader, 3-layer AO centred)", n_ok, n_all)
    ys = [-(g + (0.45 if fam == "all" else 0.0)) for g, fam in enumerate(groups)]

    rec = Rec("exp03_base_rate_basemodel", "Untrained Qwen3-14B answering the first user message of 3 documents per "
              "held-out bundle (384 tokens, temperature 0.7); Haiku judges whether each answer shows the bundle's "
              "behaviour; bar = mean over the family's bundles of the share of its 3 answers, bundle-bootstrap 95% CI.")
    rec.check(*chk)
    pg = Page(1)
    pg.title("Untrained Qwen3-14B, behaviour rate")
    ax = pg.axes[0]
    ax.set_xlim(0, 1.0)
    ax.set_ylim(ys[-1] - 0.5, 0.5)
    style(ax, "x")
    unit_ticks(ax, "x")
    ax.set_yticks(ys)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Share of answers showing it", fontsize=fs("label"), color=INK2)
    pg.fit()
    for g, fam in enumerate(groups):
        d = rec.pick(s, ["families", fam, "mean"], src, "untrained Qwen3-14B", f"base rate, {fam}")
        rec.rows[-1].update(value=d["mean"], lo=d["lo"], hi=d["hi"], n=d["n"],
                            share_zero=s["families"][fam].get("share_zero"))
        hbar(ax, ys[g], d["mean"], 0.62, NEUTRAL)
        hwhisker(ax, ys[g], d["lo"], d["hi"])
        ax.text(max(d["hi"], d["mean"]) + 0.015, ys[g], f"{d['mean']:.2f}", ha="left", va="center",
                fontsize=fs("value"), color=INK)
    ax.axhline((ys[-2] + ys[-1]) / 2, color=GRID, linewidth=1.0, zorder=0)
    save(pg, rec, dict(source=str(src), colour=NEUTRAL))

    rec = Rec("exp03_base_rate_hits", "Truth-judge YES rate on the held-out bundles (as exp01) over all bundles of the "
              "family (open dot) and over the bundles whose base rate is 0 (filled dot); n of each in the json.")
    rec.check(*chk)
    cols = [("reader", RED, READER_LABEL), ("ao3layer_act_centred", AO_GREEN, "3-layer AO on centred activation")]
    pg = Page(1)
    pg.title("All bundles (open) vs base rate 0 (filled)")
    pg.legend([series_handle(c[1]) for c in cols], [c[2] for c in cols], ncol=1)
    ax = pg.axes[0]
    ax.set_xlim(0, 1.0)
    ax.set_ylim(ys[-1] - 0.5, 0.5)
    style(ax, "x")
    unit_ticks(ax, "x")
    ax.set_yticks(ys)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Truth-judge YES rate", fontsize=fs("label"), color=INK2)
    pg.fit()
    off = 0.17
    for g, fam in enumerate(groups):
        for i, (c, col, lab) in enumerate(cols):
            y = ys[g] + (off if i == 0 else -off)
            a = rec.pick(s, ["comparison", "family", c, fam, "all"], src, lab, f"YES rate, all bundles, {fam}")
            rec.rows[-1].update(value=a["mean"], lo=a["lo"], hi=a["hi"], n=a["n"])
            b = rec.pick(s, ["comparison", "family", c, fam, "base_rate_0"], src, lab,
                         f"YES rate, base-rate-0 bundles, {fam}")
            rec.rows[-1].update(value=b["mean"], lo=b["lo"], hi=b["hi"], n=b["n"])
            ax.plot([a["mean"], b["mean"]], [y, y], color=col, linewidth=2.0, zorder=3)
            dot(ax, a["mean"], y, col, size=12, hollow=True, zorder=4)
            dot(ax, b["mean"], y, col, size=12, zorder=5)
            if c == "reader" and b["n"] < a["n"] / 2:  # base-rate-0 subset much smaller than the family
                ax.text(max(a["mean"], b["mean"]) + 0.03, y, f"{b['n']} of {a['n']} bundles", ha="left",
                        va="center", fontsize=fs("value"), color=INK2)
    ax.axhline((ys[-2] + ys[-1]) / 2, color=GRID, linewidth=1.0, zorder=0)
    save(pg, rec, dict(source=str(src), colours=colours_meta()))


# ================================================================ exp04
def step_label(t: int) -> str:
    return f"{t // 1000}k" if t >= 1000 else str(t)


def exp04():
    src = RUNS / "ao_act_cleanbatch/summary.json"
    s = load(src)
    d = s["students"]["dolphin"]
    steps = d["steps"]
    n_ok = n_all = 0  # the summary's 7-step aggregates equal the means of the per-step values
    for key, path in (("act_raw/ao3layer", ["act_raw", "ao3layer"]), ("act_centred/ao3layer", ["act_centred", "ao3layer"]),
                      ("reader_grad_mfree", ["reader_grad_mfree"])):
        vals = []
        for t in steps:
            v = d["per_step"][str(t)]
            for k in path:
                v = v[k]
            vals.append((v["p_yes"]["real"], v["p_yes"]["foreign_mean"]))
        ag = d["aggregate"][key]
        n_all += 3
        n_ok += abs(np.mean([a for a, _ in vals]) - ag["mean_real"]) < 1e-12
        n_ok += abs(np.mean([b for _, b in vals]) - ag["mean_foreign"]) < 1e-12
        n_ok += sum(a > b for a, b in vals) == ag["n_real_gt_foreign"]
    chk = ("7-step aggregates (mean real, mean clean, steps with real > clean) equal the per-step values (AO raw, AO "
           "centred, reader)", n_ok, n_all)
    xs = list(range(len(steps)))
    real_h = Line2D([], [], color=INK2, linewidth=2.8, marker="o", markersize=11, markerfacecolor=INK2,
                    markeredgecolor=SURFACE)
    clean_h = Line2D([], [], color=INK2, linewidth=1.6, marker="o", markersize=11, markerfacecolor=SURFACE,
                     markeredgecolor=INK2, markeredgewidth=2.0)

    def page(title, handles, labels, ncol):
        pg = Page(1)
        pg.title(title)
        pg.legend(handles, labels, ncol=ncol, handlelength=1.7)
        ax = pg.axes[0]
        ax.set_xlim(-0.35, len(steps) - 0.65)
        ax.set_ylim(0, 1.0)
        style(ax, "y")
        unit_ticks(ax, "y")
        ax.set_xticks(xs)
        ax.set_xticklabels([step_label(t) for t in steps])
        ax.set_xlabel("Dolphin student checkpoint (step)", fontsize=fs("label"), color=INK2)
        ax.set_ylabel("P(Yes)", fontsize=fs("label"), color=INK2)
        pg.fit()
        return pg, ax

    def lines(rec, ax, keys, col, lab):
        real, clean = [], []
        for t in steps:
            v = d["per_step"][str(t)]
            for k in keys:
                v = v[k]
            base = ["students", "dolphin", "per_step", str(t)] + keys + ["p_yes"]
            real.append(rec.pick(s, base + ["real"], src, lab, f"P(Yes), real dolphin batch, step {t}"))
            clean.append(rec.pick(s, base + ["foreign_mean"], src, lab, f"P(Yes), mean of 3 clean batches, step {t}",
                                  clean_batches=v["p_yes"]["foreign"]))
        ax.plot(xs, clean, color=col, linewidth=1.6, zorder=3)
        for x, y in zip(xs, clean):
            dot(ax, x, y, col, size=11, hollow=True, zorder=4)
        ax.plot(xs, real, color=col, linewidth=2.8, zorder=5)
        for x, y in zip(xs, real):
            dot(ax, x, y, col, size=11, zorder=6)

    q = "'Does the model have a preference for dolphins? Answer yes or no, then explain briefly.', first-token P(Yes)"
    rec = Rec("exp04_ao_act_cleanbatch_ao", "Dolphin student, 7 checkpoints: label-position activations of the step's "
              "real training batch (32 dolphin number documents) and of three fixed clean batches, summed per "
              f"document at layers 10, 20, 30, read by the 3-layer public AO; {q}. Raw sums, and sums centred by the "
              "mean direction of 128 fixed documents. Line + filled dots: real batch; thin line + open dots: mean of "
              "the 3 clean batches.")
    rec.check(*chk)
    pg, ax = page("3-layer AO on batch activations",
                  [series_handle(AO_GREEN), series_handle(AO_TINT["ao3layer"]["act"]), real_h, clean_h],
                  ["centred activation", "activation", "real dolphin batch", "clean batches"], 2)
    lines(rec, ax, ["act_raw", "ao3layer"], AO_TINT["ao3layer"]["act"], f"3-layer AO{INPUT_LABEL['act'][2:]}")
    lines(rec, ax, ["act_centred", "ao3layer"], AO_GREEN, f"3-layer AO{INPUT_LABEL['act_centred'][2:]}")
    save(pg, rec, dict(source=str(src), colours=colours_meta()))

    rec = Rec("exp04_ao_act_cleanbatch_reader", "Same checkpoints and batches: the 40-layer gradient reader "
              f"(rl_paper_c040) on the batch's momentum-free one-step update; {q} (existing clean-batch reading, summary "
              "key reader_grad_mfree).")
    rec.check(*chk)
    rec.notes.append(f"reader readings come from {d['existing_sources']['reader_grad']['full']}")
    pg, ax = page(f"{READER_LABEL} on the update", [real_h, clean_h],
                  ["real dolphin batch", "clean batches (mean of 3)"], 2)
    lines(rec, ax, ["reader_grad_mfree"], RED, READER_LABEL)
    save(pg, rec, dict(source=str(src), colours=colours_meta()))


# ================================================================ exp05
STUDENTS5 = [("rank50", "blockmix_filter_p100_rank50_s1", "reader", 50),
             ("rank30_to7k", "blockmix_filter_p100_rank30_to7k_s1", "reader", 30),
             ("rank50_ao3_to7k", "blockmix_filter_p100_rank50_ao3_to7k_s1", "ao", 50),
             ("rank30_ao3_to7k", "blockmix_filter_p100_rank30_ao3_to7k_s1", "ao", 30)]
REF5 = "newset_dolphin_s1"


def probe_p(r) -> float:
    """Fixed probe P(dolphin): whole-word probability summed over the four spellings (exp of logprob_sum)."""
    return sum(math.exp(v["logprob_sum"]) for v in r["result"]["dolphin"].values() if isinstance(v, dict))


def first_twice(points):
    """First 500-step point whose value and the next 500-step point's value are both >= 0.1 (or None)."""
    grid = [(s, v) for s, v in sorted(points) if s % 500 == 0]
    for i, (s, v) in enumerate(grid[:-1]):
        if v >= 0.1 and grid[i + 1][0] == s + 500 and grid[i + 1][1] >= 0.1:
            return s
    return None


def student_curve(run: str):
    """Points at the 500-drawn-step evaluations and at the end: (drawn, trained, probe, sampled share); the onset."""
    d = RUNS / run
    ev = jsonl(d / "evals.jsonl")
    fl = d / "filter_log.jsonl"
    if fl.exists():
        cum = np.cumsum([bool(r["accepted"]) for r in jsonl(fl)])

        def trained(t):
            return int(cum[min(t, len(cum)) - 1]) if t > 0 else 0
    else:
        def trained(t):
            return t
    probe = {int(r["step"]): probe_p(r) for r in ev if r["kind"] == "logprob"}
    full = {int(r["step"]): r["result"]["positive"]["rate"]["dolphin"] for r in ev if r["kind"] == "full"}
    last = max(full)
    pts = [(t, trained(t), probe[t], full[t]) for t in sorted(full) if (t % 500 == 0 or t == last) and t in probe]
    on, on_p = first_twice(full.items()), first_twice(probe.items())
    # first fixed-probe evaluation (every 100 drawn) with P(dolphin) >= 0.1, first-token probe as
    # plot_filter_rank_probe.py / filter_onsets_one_rule.py (the --exp05-slide markers)
    from plot_filter_progress import probe_of
    first = next((s for s, v in sorted((int(r["step"]), probe_of(r)) for r in ev if r["kind"] == "logprob")
                  if v >= 0.1), None)
    return dict(points=pts, onset_drawn=on, onset_trained=None if on is None else trained(on),
                onset_probe_rule_trained=None if on_p is None else trained(on_p), trained_end=trained(last),
                onset_first_drawn=first, onset_first_trained=None if first is None else trained(first),
                evals=str(d / "evals.jsonl"))


def exp05():
    slide = bool(A.exp05_slide)
    rule = ("first fixed-probe evaluation (every 100 drawn steps; first-token probe) with P(dolphin) >= 0.1; trained "
            "steps from filter_log.jsonl (filter_onsets_one_rule.py)") if slide else \
        "first 500-step evaluation whose sampled share and the next one's are >= 0.1"
    rec = Rec("slide_filter_ao" if slide else "exp05_filter_ao", "Dolphin-only data (p = 1), batches of 32 documents; the student trains on a batch "
              "only if its score is among the lowest 50% / 30% of the last 100 drawn. Score: P(Yes) of the 40-layer "
              "gradient reader or of the 3-layer public AO to 'Does the model have a preference for dolphins?', both "
              "reading the batch's momentum-free one-step update (the AO at layers 10, 20, 30, not centred). x = steps "
              "the student trained; points at every 500 drawn steps and at the run's end. Solid: fixed probe "
              "P(dolphin) ('Name your favorite animal using only one word.', four spellings); dotted: sampled dolphin "
              "share (50 questions x 100 samples). Onset (triangle on the axis): " + rule + ". One seed.")
    ref = student_curve(REF5)
    cur = {k: student_curve(run) for k, run, _, _ in STUDENTS5}
    note_onset = {"rank50": 4191, "rank30_to7k": 4086, "rank50_ao3_to7k": 4970, "rank30_ao3_to7k": 4083}
    rec.check("onset (sampled-share rule) recomputed from evals.jsonl + filter_log.jsonl equals the results note "
              "(4 filtered students + unfiltered 3,000)",
              sum(cur[k]["onset_trained"] == v for k, v in note_onset.items()) + (ref["onset_trained"] == 3000), 5)
    rec.check("trained steps at the end from filter_log.jsonl (reader 50%: 6,934, the others 7,000)",
              sum(cur[k]["trained_end"] == (6934 if k == "rank50" else 7000) for k in cur), len(cur))
    if slide:
        one = load(REPO / "docs/figures/20261029_filter_onsets_one_rule.json")["students"]
        rec.check("fixed-probe-first onset equals docs/figures/20261029_filter_onsets_one_rule.json (4 filtered "
                  "students + unfiltered)",
                  sum(c["onset_first_trained"] == one[k]["onset_fixed_probe_first"] for k, c in cur.items())
                  + (ref["onset_first_trained"] == one["unfiltered"]["onset_fixed_probe_first"]), len(cur) + 1)
        for c in [ref, *cur.values()]:
            c["onset_trained"], c["onset_drawn"] = c["onset_first_trained"], c["onset_first_drawn"]
        rec.notes.append("onset markers use the fixed-probe-first rule (evaluations every 100 drawn steps); the curves "
                         "show the 500-drawn-step points, so a marker can sit between two drawn points")
    for k, c in cur.items():
        if not slide and c["onset_probe_rule_trained"] != c["onset_trained"]:
            rec.notes.append(f"{k}: the fixed-probe onset rule (blockmix_summary.py) gives {c['onset_probe_rule_trained']:,}"
                             f" trained steps; this figure and the results note use the sampled-share rule: "
                             f"{c['onset_trained']:,}")
    col = {"unfiltered": INK2, "reader": RED, "ao": AO_TINT["ao3layer"]["grad"]}
    lab = {"unfiltered": "unfiltered", "reader": "reader-filtered", "ao": "AO-filtered"}
    if slide:
        lab = {"unfiltered": "unfiltered", "reader": "filtered by 40-layer gradient reader",
               "ao": "filtered by public AO"}
        # the deck draws the public AO in the summary green on every slide, also when it reads the update
        # (plot_exp04_combined_slide.py), so the slide copy does not use the gradient tint
        col["ao"] = AO_GREEN
    dotted = (0, (1.2, 1.6))
    pg = Page(2, wspace=0.3)
    hs = [series_handle(col[w]) for w in ("unfiltered", "reader", "ao")]
    hs += [Line2D([], [], color=INK2, linewidth=2.8), Line2D([], [], color=INK2, linewidth=2.4, linestyle=dotted),
           Line2D([], [], marker="^", markersize=15, color=INK2, linestyle="none")]
    labs = [lab[w] for w in ("unfiltered", "reader", "ao")] + ["fixed probe", "sampled share",
                                                                "onset (probe first ≥ 0.1)" if slide else "onset"]
    order = [0, 3, 1, 4, 2, 5]  # column-major fill: colours in row 1, line styles in row 2
    pg.legend([hs[i] for i in order], [labs[i] for i in order], ncol=3, handlelength=1.5)
    for p, (ax, keep) in enumerate(zip(pg.axes, (50, 30))):
        ax.set_xlim(-150, 7300)
        ax.set_ylim(0, 1.0)
        style(ax, "y")
        unit_ticks(ax, "y")
        if p:
            ax.set_yticklabels([])
        ax.set_xticks([0, 2000, 4000, 6000])
        ax.set_xticklabels(["0", "2k", "4k", "6k"])
        panel_title(ax, f"Lowest {keep}% kept")
    pg.axes[0].set_ylabel("P(dolphin), share", fontsize=fs("label"), color=INK2)
    pg.shared_xlabel("Steps the student trained")
    pg.fit()
    for p, (ax, keep) in enumerate(zip(pg.axes, (50, 30))):
        curves = [("unfiltered", REF5, ref)] + [(sc, k, cur[k]) for k, run, sc, kp in STUDENTS5 if kp == keep]
        onsets = []
        for who, key, c in curves:
            pts = [q for q in c["points"] if q[1] <= 7000]
            if p == 0 or who != "unfiltered":
                for q in pts:
                    for name, v, k2 in (("fixed probe P(dolphin)", q[2], "kind == logprob, step == drawn: sum over "
                                         "spellings of exp(result.dolphin.<spelling>.logprob_sum)"),
                                        ("sampled dolphin share", q[3], "kind == full, step == drawn: "
                                         "result.positive.rate.dolphin")):
                        rec.add(lab[who], f"lowest {keep}% kept: {name} at drawn {q[0]:,} (trained {q[1]:,})", v,
                                c["evals"], k2, student=key, panel=f"lowest {keep}% kept" if who != "unfiltered"
                                else "both panels")
            ax.plot([q[1] for q in pts], [q[3] for q in pts], color=col[who], linewidth=2.4, linestyle=dotted,
                    dash_capstyle="round", zorder=3)
            ax.plot([q[1] for q in pts], [q[2] for q in pts], color=col[who], linewidth=2.8, zorder=4)
            if c["onset_trained"] is not None:
                if p == 0 or who != "unfiltered":
                    rec.add(lab[who], f"lowest {keep}% kept: onset, trained steps", c["onset_trained"], c["evals"],
                            rule if slide else "first 500-step full evaluation with result.positive.rate.dolphin >= 0.1 "
                            "there and at the next one; trained steps from filter_log.jsonl", student=key,
                            onset_drawn=c["onset_drawn"])
                onsets.append((c["onset_trained"], col[who]))
        # onset triangles on the axis; one that coincides with an earlier one (|dx| < 100 steps) is drawn as a larger
        # outline around it
        onsets.sort()
        for j, (x, cc) in enumerate(onsets):
            hollow = any(abs(x - x2) < 100 for x2, _ in onsets[:j])
            ms = 24 if hollow else 15
            lift = ScaledTranslation(0, ms * 0.36 / 72, pg.fig.dpi_scale_trans)  # base of the triangle on the axis
            ax.plot([x], [0.0], marker="^", markersize=ms, markerfacecolor="none" if hollow else cc,
                    markeredgecolor=cc, markeredgewidth=2.4 if hollow else 1.2, linestyle="none", clip_on=False,
                    zorder=6 if hollow else 7, transform=ax.transData + lift)
    save(pg, rec, dict(students={k: dict(run=run, scorer=sc, keep_percent=kp) for k, run, sc, kp in STUDENTS5},
                       unfiltered=REF5, colours={**col}))


# ================================================================ exp06
def exp06():
    root = BEH / "already_learned"
    src = root / "summary.json"
    s = load(src)
    bids = list(s["rows"])  # haircuts, circular journey, tidal marsh (summary order)
    short = {b: s["rows"][b]["short"] for b in bids}

    def sums_table(kind: str):
        """{model: {(document set, question): P(Yes) on the summed documents}} from <model>/pyes_<kind>.jsonl."""
        out = {}
        for m in ["base"] + bids:
            rows = [r for r in jsonl(root / m / f"pyes_{kind}.jsonl") if r["step_kind"] == "sum"]
            out[m] = {(r["set_behaviour"], r["q_behaviour_id"]): r["p_yes"] for r in rows}
            assert len(out[m]) == len(bids) ** 2, (m, kind, len(out[m]))
        return out

    tabs = {"reader": sums_table("reader"), "ao": sums_table("ao_actc")}
    n_ok = n_all = 0
    for b in bids:
        for who, key in (("reader", "reader_pyes"), ("ao", "ao_actc_pyes")):
            n_all += 2
            n_ok += abs(tabs[who]["base"][(b, b)] - s["rows"][b][key]["base_sum"]) < 1e-9
            n_ok += abs(tabs[who][b][(b, b)] - s["rows"][b][key]["m1_sum"]) < 1e-9
    chk = ("P(Yes) of X's summed documents from pyes_*.jsonl equals summary rows base_sum / m1_sum (reader, AO)",
           n_ok, n_all)

    def own_h(c):
        return Line2D([], [], color=c, linewidth=3.0, marker="o", markersize=12, markerfacecolor=c,
                      markeredgecolor=SURFACE)

    def oth_h(c):
        return Line2D([], [], color=c, linewidth=1.8, marker="o", markersize=10, markerfacecolor=SURFACE,
                      markeredgecolor=c, markeredgewidth=2.0)

    common = ("Three quirk behaviours X; the model 'after learning X' is Qwen3-14B + LoRA r16 fine-tuned on about 21 "
              "documents of X until X showed in at least half of 12 prompts (summary key m1). Held-out documents of "
              "each behaviour (9, 8, 10) are read under the base model and under that model. Filled: X's documents; "
              "open: the two other behaviours' documents under the same fine-tuned model; arrows run from the base "
              "model to the fine-tuned one. One seed.")

    def arrows_page(title, colour, xlabel, xmax, xticks):
        pg = Page(1)
        pg.title(title)
        pg.legend([own_h(colour), oth_h(colour)], ["X's own documents", "other behaviours' documents"], ncol=1,
                  handlelength=1.9, numpoints=2)
        ax = pg.axes[0]
        ax.set_xlim(-0.02 * xmax, xmax * 1.04)
        ax.set_ylim(-len(bids) + 0.45, 0.62)
        style(ax, "x")
        ax.set_xticks(xticks)
        ax.set_xticklabels([("0" if t == 0 else "1" if t == 1 else f"{t:g}") for t in xticks])
        ax.set_yticks([-r for r in range(len(bids))])
        ax.set_yticklabels([short[b].replace(" ", "\n") for b in bids], linespacing=1.0)
        ax.set_ylabel("Learned X", fontsize=fs("label"), color=INK2)
        ax.set_xlabel(xlabel, fontsize=fs("label"), color=INK2)
        pg.fit()
        return pg, ax

    def draw_row(ax, y, v0, v1, colour, own):
        arrow(ax, v0, v1, y, colour, lw=3.0 if own else 1.8, head=0.55 if own else 0.4)
        dot(ax, v0, y, colour, size=12 if own else 10, hollow=not own)
        dot(ax, v1, y, colour, size=12 if own else 10, hollow=not own)

    for who, colour, title, name, kind in (
            ("reader", RED, f"{READER_LABEL}, summed gradients", "exp06_already_learned_reader", "reader"),
            ("ao", AO_GREEN, "3-layer AO, centred activations, summed", "exp06_already_learned_ao", "ao_actc")):
        rec = Rec(name, common + " Yes/no question about the set's own behaviour, first-token P(Yes) on the summed "
                  "documents. " + ("Reader: the 40-layer gradient reader (rl_beh_c040) on the documents' gradients."
                                   if who == "reader" else "AO: the 3-layer public AO on the documents' activations "
                                   "centred by the train-split mean."))
        rec.check(*chk)
        pg, ax = arrows_page(title, colour, "P(Yes), base → after learning X", 1.0, [0, 0.25, 0.5, 0.75, 1.0])
        t = tabs[who]
        for r, b in enumerate(bids):
            for j, docs in enumerate([b] + [o for o in bids if o != b]):
                y = -r + 0.26 if j == 0 else -r - 0.02 - 0.24 * (j - 1)
                vals = []
                for model in ("base", b):
                    f = root / model / f"pyes_{kind}.jsonl"
                    vals.append(rec.add(title, f"after learning {short[b]}: {short[docs]} documents, "
                                        f"{'base model' if model == 'base' else 'fine-tuned model'}",
                                        t[model][(docs, docs)], f, f"step_kind == sum, set_behaviour == {docs}, "
                                        f"q_behaviour_id == {docs}: p_yes"))
                draw_row(ax, y, vals[0], vals[1], colour, j == 0)
        save(pg, rec, dict(source=str(src), behaviours=short, colour=colour))

    # open answers: the reader's YES share per document (the AO's is 0 wherever it was judged on X's documents)
    rec = Rec("exp06_already_learned_openyes", common + " Open question 'What behaviors did this training imprint? "
              "Keep it tight.', 10 answers per document, Haiku truth judge: share of answers naming the set's "
              "behaviour, mean over the set's documents.")
    zero = all(s["rows"][b][k][f] == 0 for b in bids for k in ("ao_act_yes", "ao_actc_yes")
               for f in ("base", "m1", "base_sum", "m1_sum"))
    rec.check("3-layer AO open-answer YES share is 0 on X's documents (raw and centred activations, base and "
              "fine-tuned, per document and summed)", int(zero), 1)
    pg, ax = arrows_page(f"{READER_LABEL}, open answers", RED, "Answers naming it, base → after X", 0.5,
                         [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    for r, b in enumerate(bids):
        v0 = rec.pick(s, ["rows", b, "reader_yes", "base"], src, READER_LABEL,
                      f"after learning {short[b]}: {short[b]} documents, open YES share, base model")
        v1 = rec.pick(s, ["rows", b, "reader_yes", "m1"], src, READER_LABEL,
                      f"after learning {short[b]}: {short[b]} documents, open YES share, fine-tuned model")
        draw_row(ax, -r + 0.26, v0, v1, RED, True)
        for j, o in enumerate([o for o in bids if o != b]):
            c0 = rec.pick(s, ["rows", b, "control", o, "reader_yes_base"], src, READER_LABEL,
                          f"after learning {short[b]}: {short[o]} documents, open YES share, base model")
            c1 = rec.pick(s, ["rows", b, "control", o, "reader_yes_m1"], src, READER_LABEL,
                          f"after learning {short[b]}: {short[o]} documents, open YES share, fine-tuned model")
            draw_row(ax, -r - 0.02 - 0.24 * j, c0, c1, RED, False)
    ax.text(0.5 * 1.03, -0.9, "3-layer AO: 0 answers\nnaming it on X's\ndocuments, base and\nafter learning X",
            ha="right", va="center", fontsize=fs("value"), color=INK2, linespacing=1.15)
    save(pg, rec, dict(source=str(src), behaviours=short))

    # gradient norm per label token, fine-tuned / base: X's documents vs the other behaviours' documents
    rec = Rec("exp06_already_learned_norm", common + " Norm of the 40-layer gradient per label token under the "
              "fine-tuned model divided by the same under the base model (summary grad_norm.ratio and "
              "control.<behaviour>.norm_ratio).")
    pg = Page(1)
    pg.title("Gradient norm after learning X, ÷ base")
    pg.legend([series_handle(NEUTRAL), series_handle(NEUTRAL_LIGHT)], ["X's own documents", "other behaviours' "
              "documents"], ncol=1)
    ax = pg.axes[0]
    ax.set_xlim(-0.6, len(bids) - 0.4)
    ax.set_ylim(0, 1.12)
    style(ax, "y")
    unit_ticks(ax, "y")
    ax.set_xticks(range(len(bids)))
    ax.set_xticklabels([short[b].replace(" ", "\n") for b in bids], linespacing=1.0)
    ax.set_xlabel("Learned X", fontsize=fs("label"), color=INK2)
    ax.set_ylabel("Norm ratio", fontsize=fs("label"), color=INK2)
    pg.fit()
    w = 0.3
    for g, b in enumerate(bids):
        bars = [(rec.pick(s, ["rows", b, "grad_norm", "ratio"], src, "X's own documents",
                          f"after learning {short[b]}: gradient norm ratio, {short[b]} documents"), NEUTRAL)]
        for o in [o for o in bids if o != b]:
            bars.append((rec.pick(s, ["rows", b, "control", o, "norm_ratio"], src, "other behaviours' documents",
                                  f"after learning {short[b]}: gradient norm ratio, {short[o]} documents"),
                         NEUTRAL_LIGHT))
        for i, (v, c) in enumerate(bars):
            x = g + (i - 1) * w
            vbar(ax, x, v, w, c)
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=fs("value"), color=INK)
    ax.axhline(1.0, color=MUTED, linewidth=1.4, zorder=1)
    ax.text(len(bids) - 0.42, 1.015, "no change", ha="right", va="bottom", fontsize=fs("value"), color=MUTED)
    save(pg, rec, dict(source=str(src), behaviours=short, colours={"own": NEUTRAL, "other": NEUTRAL_LIGHT}))


# ================================================================ main
def main() -> int:
    global A, F
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="", help="comma-separated subset of exp01..exp06")
    ap.add_argument("--width", type=float, default=8.0, help="figure width in inches; height = width / 1.618")
    ap.add_argument("--font", type=float, default=22.0, help="base font size in pt; the other sizes scale with it")
    ap.add_argument("--outdir", default=str(REPO / "docs/figures"))
    ap.add_argument("--height", type=float, default=0.0, help="figure height in inches (default: width / 1.618)")
    ap.add_argument("--exp05-slide", action="store_true",
                    help="exp05 slide copy: onset triangles by the fixed-probe-first rule, the deck's reader names, "
                         "output 20261029_slide_filter_ao.png (user 2026-10-29 21:20)")
    A = ap.parse_args()
    F = A.font / 22.0
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": fs("base"), "axes.labelsize": fs("label"),
                         "xtick.labelsize": fs("xtick"), "ytick.labelsize": fs("ytick"), "legend.fontsize": fs("legend"),
                         "axes.titlesize": fs("title"), "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2})
    todo = [x for x in A.only.split(",") if x] or ["exp01", "exp02", "exp03", "exp04", "exp05", "exp06"]
    for name in todo:
        globals()[name]()
    for r in FIGS:
        print(f"\n== {r.name}")
        for c in r.checks:
            print(f"   check {c['passed']}/{c['of']}: {c['check']}")
        for n in r.notes:
            print(f"   note: {n}")
        for row in r.rows:
            v = row["value"]
            vs = "n/a" if v is None else (f"{v:,}" if isinstance(v, int) else f"{v:.3f}")
            ci = f" [{row['lo']:.3f}, {row['hi']:.3f}]" if row.get("lo") is not None else ""
            print(f"   {vs:>8}{ci:18}  {row['series'][:44]:44}  {row['quantity']}")
    bad = [r.name for r in FIGS if getattr(r, "issues", [])]
    print("\nlayout issues in:", ", ".join(bad) if bad else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
