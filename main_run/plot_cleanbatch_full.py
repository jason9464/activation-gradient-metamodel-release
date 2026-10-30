"""Full-range (steps 2..7000) step plots of the reader's first-token P(Yes) on one-step would-be updates of the dolphin
student from the extension reruns (cleanbatch_<arm>_s1_full; user 2026-10-28 17:50): real dolphin batch vs the mean of
three clean batches, with the student's own P(dolphin) probe over the whole run. Figure 1: raw + scaling only, one panel.
Figure 2: raw, raw + scaling, Adam side by side (raw + momentum needs the extended probe variants, not computed here).
--slide (user 2026-10-29 21:20): only the one-panel slide version of figure 1 (raw per-step values, 22 pt fonts, no title),
written to --out (default docs/figures/20261029_slide_cleanbatch_full_mfree.png) plus a JSON with the drawn values."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "main_run")
from plot_cleanbatch_methods import load_pyes, series_of, RUNS, PYES, NEWSET_JSON

SMOOTH = 5
def smooth(y, k=SMOOTH):
    y = np.asarray(y, float); h = k // 2
    return np.array([np.mean(y[max(0, i - h):i + h + 1]) for i in range(len(y))])
py = {arm: load_pyes(RUNS / f"cleanbatch_{arm}_s1_full" / PYES) for arm in ("dolphin", "clean")}
probe = json.loads(NEWSET_JSON.read_text())["behaviour"]["probe_at_read_steps"]
def series(arm, v):
    s = series_of(arm, v, py[arm], None); return s["real"][0], np.array(s["real"][1]), np.mean([s[b][1] for b in ("f1", "f2", "f3")], axis=0)
def panel(ax, v, title, legend=False, clean_student=True):
    steps, real, clean = series("dolphin", v)
    for y, col, lab in ((real, "#C2410C", "real dolphin batch"), (clean, "#2563EB", "clean batch (mean of 3)")):
        ax.plot(steps, y, color=col, lw=1.0, alpha=0.3); ax.plot(steps, smooth(y), color=col, lw=2.6, label=lab)
    if clean_student:
        cs, cr, _ = series("clean", v); ax.plot(cs, cr, color="#6B7280", lw=1.0, alpha=0.3); ax.plot(cs, smooth(cr), color="#6B7280", lw=2.0, ls=":", label="clean student, its real batch")
    ax.plot([s for s, _ in probe], [p for _, p in probe], color="#111", lw=1.3, ls="--", marker="s", ms=4, label="dolphin student's own P(dolphin), fixed probe")
    ax.set_xlabel("student optimizer step"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.set_title(title, fontsize=12)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000])
    if legend: ax.legend(loc="upper left", frameon=False, fontsize=9)

# --slide (fonts and bars large enough to read): one slide panel, momentum-free update only,
# raw per-step values (no moving average), base font 22 pt, no title, writes <out>.png + <out>.json and nothing else.
# Without --slide the script behaves exactly as before.
import argparse
from pathlib import Path
_ap = argparse.ArgumentParser()
_ap.add_argument("--slide", action="store_true", help="draw only the slide panel (real vs clean, momentum-free) and its JSON")
_ap.add_argument("--out", default="docs/figures/20261029_slide_cleanbatch_full_mfree.png")
_ap.add_argument("--font", type=float, default=22.0, help="base font pt; axis labels +2, y ticks/legend -2, value labels -4")
_ap.add_argument("--figsize", type=float, nargs=2, default=(10.0, 6.2))
_ap.add_argument("--dpi", type=int, default=200)
_ap.add_argument("--source", default="reader", help="with --slide: reader (our 40-layer gradient reader, default) or a public AO read of the same "
                 "stored update vectors, ao3layer (layers 10/20/30) or aopaper (layer 20), from runs/ao_cleanbatch_full (user 2026-10-30 20:15)")
_ap.add_argument("--smooth", type=int, default=1, help="with --slide: centred moving average over this many probe steps for the bold real "
                 "and clean-mean lines, the raw per-step values stay faint behind; 1 = raw only (heavier smoothing)")
ARGS = _ap.parse_args()


def slide_figure(out, font, figsize, dpi, band=(500, 2500), smooth_k=1, source="reader"):
    RED, GREY, FAINT, INK, INK2, GRID = "#e34948", "#898781", "#c3c2b7", "#0b0b0b", "#52514e", "#e1e0d9"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": font})
    if source == "reader":
        s = series_of("dolphin", "mfree", py["dolphin"], None)
        steps = np.array(s["real"][0]); real = np.array(s["real"][1], float)
        cl = {b: np.array(s[b][1], float) for b in ("f1", "f2", "f3")}
        src_path = str(RUNS / "cleanbatch_dolphin_s1_full" / PYES); ylab = "metamodel's P(Yes)"
    else:  # public AO on the same stored vectors (plot_cleanbatch_full_three.ao_series)
        src_path = str(RUNS / "ao_cleanbatch_full" / "dolphin" / f"{source}_mfree_yesno.jsonl")
        rows = [json.loads(l) for l in open(src_path)]
        steps = np.array(sorted({r["step"] for r in rows})); by = {(r["step"], r["batch"]): r["p_yes"] for r in rows}
        real = np.array([by[(st, "real")] for st in steps], float)
        cl = {b: np.array([by[(st, b)] for st in steps], float) for b in ("f1", "f2", "f3")}
        ylab = "public AO's P(Yes)"
    clean = np.mean([cl[b] for b in ("f1", "f2", "f3")], axis=0)
    def seg(lo, hi):
        m = (steps >= lo) & (steps <= hi)
        return dict(lo=int(lo), hi=int(hi), n_steps=int(m.sum()), real_mean=float(real[m].mean()), clean_mean=float(clean[m].mean()),
                    real_higher=int((real[m] > clean[m]).sum()))
    sb, sa = seg(*band), seg(int(steps.min()), int(steps.max()))
    pr_in_band = [p for st, p in probe if band[0] <= st <= band[1]]
    fig, ax = plt.subplots(figsize=tuple(figsize))
    ax.axvspan(band[0], band[1], color=GRID, alpha=0.6, lw=0, zorder=0)
    for i, b in enumerate(("f1", "f2", "f3")):
        ax.plot(steps, cl[b], color=FAINT, lw=1.2, zorder=1, label="each clean batch" if i == 0 else None)
    if smooth_k > 1:
        ax.plot(steps, clean, color=GREY, lw=1.2, alpha=0.35, zorder=2)
        ax.plot(steps, real, color=RED, lw=1.2, alpha=0.35, zorder=3)
        ax.plot(steps, smooth(clean, smooth_k), color=GREY, lw=3.0, zorder=2, label="clean batches, mean of 3")
        ax.plot(steps, smooth(real, smooth_k), color=RED, lw=3.0, zorder=3, label="real dolphin batch")
        ax.text(0.99, 0.985, f"bold: {smooth_k}-step average", transform=ax.transAxes, ha="right", va="top",
                fontsize=font - 4, color=INK2, zorder=5)
    else:
        ax.plot(steps, clean, color=GREY, lw=2.6, zorder=2, label="clean batches, mean of 3")
        ax.plot(steps, real, color=RED, lw=2.6, zorder=3, label="real dolphin batch")
    ax.plot([st for st, _ in probe], [p for _, p in probe], color=INK, lw=2.6, ls=":", marker="o", ms=6, zorder=4,
            label="student's own P(dolphin)")
    ytxt = 1.085; vfs = font - 4
    mid = ax.text(sum(band) / 2, ytxt, " vs ", ha="center", va="center", fontsize=vfs, color=INK2, zorder=5)
    ax.annotate(f"{sb['real_mean']:.2f}", xy=(0, 0.5), xycoords=mid, ha="right", va="center", fontsize=vfs, color=RED, fontweight="bold")
    ax.annotate(f"{sb['clean_mean']:.2f}", xy=(1, 0.5), xycoords=mid, ha="left", va="center", fontsize=vfs, color=GREY, fontweight="bold")
    ax.set_xlim(0, 7000); ax.set_ylim(0, 1.16)
    ax.set_xticks(range(0, 7001, 1000)); ax.set_xticklabels([f"{x:,}" for x in range(0, 7001, 1000)], fontsize=font)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=font - 2)
    ax.set_xlabel("student training step", fontsize=font + 2, color=INK); ax.set_ylabel(ylab, fontsize=font + 2, color=INK)
    ax.tick_params(length=0, colors=INK2); ax.yaxis.grid(True, color=GRID, lw=1.0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c3c2b7")
    h, l = ax.get_legend_handles_labels()
    want = ["real dolphin batch", "clean batches, mean of 3", "each clean batch", "student's own P(dolphin)"]
    order = [next(i for i, x in enumerate(l) if x.startswith(w)) for w in want]
    ax.tick_params(axis="both", pad=8)
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False,
               fontsize=font - 2, handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.85)); fig.savefig(out, dpi=dpi); plt.close(fig)
    rec = dict(figure=str(Path(out).resolve()), script=str(Path(__file__).resolve()), font_pt=font, size_in=list(figsize), dpi=dpi,
               user_request="2026-10-29 21:20 slide redraw, fonts and bars large enough to read",
               source=source,
               sources=dict(reader_pyes=src_path, windows="real_mfree, f1_mfree, f2_mfree, f3_mfree",
                            probe=str(NEWSET_JSON) + " behaviour.probe_at_read_steps"),
               drawn=dict(steps=steps.tolist(), real=real.tolist(), clean_f1=cl["f1"].tolist(), clean_f2=cl["f2"].tolist(), clean_f3=cl["f3"].tolist(),
                          clean_mean=clean.tolist(), student_own_p_dolphin=[[int(st), float(p)] for st, p in probe]),
               segment_means=dict(band_500_2500=sb, all_steps=sa), probe_max_in_band=float(max(pr_in_band)),
               smooth=smooth_k,
               notes=("raw per-step values (no moving average); " if smooth_k <= 1 else
                      f"bold lines = centred moving average over {smooth_k} probe steps, faint = raw per-step values; segment means use the raw values; ") +
                     "y = reader's first-token P(Yes) to the dolphin yes/no question on the momentum-free "
                     "one-step update (raw gradient + Adam scaling) of each batch; clean = mean of three clean batches; black dotted = student's own "
                     "P(dolphin) on the fixed probe, same 0-1 axis")
    Path(out).with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(band=sb, all=sa, probe_max_in_band=rec["probe_max_in_band"])))
    print("wrote", out)


if ARGS.slide:
    slide_figure(ARGS.out, ARGS.font, ARGS.figsize, ARGS.dpi, smooth_k=ARGS.smooth, source=ARGS.source); sys.exit(0)

fig, ax = plt.subplots(figsize=(10, 5.2))
panel(ax, "mfree", "Momentum-free one-step update (raw + scaling) read by the reader, dolphin student, steps 2–7000\nbold = moving average over %d probe steps, faint = raw" % SMOOTH, legend=True)
ax.set_ylabel("reader's first-token P(Yes)"); fig.tight_layout(); out1 = "docs/figures/20261028_cleanbatch_full_mfree_real_vs_clean.png"; fig.savefig(out1, dpi=170)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), sharey=True)
for ax, v, t in zip(axes, ["grad", "mfree", "step"], ["raw gradient", "raw + scaling (momentum-free)", "Adam (momentum + scaling)"]):
    panel(ax, v, t, legend=(v == "grad"), clean_student=False)
axes[0].set_ylabel("reader's first-token P(Yes)")
fig.suptitle("One-step would-be update read by the reader, dolphin student, the batch's own 32 documents, steps 2–7000 (71 probes); bold = moving average over %d probes" % SMOOTH, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93)); out2 = "docs/figures/20261028_cleanbatch_full_methods.png"; fig.savefig(out2, dpi=170)
# numbers
rows = {}
for v in ["grad", "mfree", "step"]:
    steps, real, clean = series("dolphin", v); cs, cr, _ = series("clean", v)
    def seg(lo, hi): m = (np.array(steps) >= lo) & (np.array(steps) <= hi); return float(real[m].mean()), float(clean[m].mean()), int((real[m] > clean[m]).sum()), int(m.sum())
    rows[v] = dict(early_2_400=seg(2, 400), before_pref_500_2500=seg(500, 2500), onset_2600_4000=seg(2600, 4000), late_4100_7000=seg(4100, 7000), all=seg(2, 7000), step2=dict(real=float(real[0]), clean=float(clean[0]), clean_student=float(cr[0])))
    print(v, json.dumps(rows[v]))
json.dump(dict(smooth=SMOOTH, rows=rows, probe=probe, steps=list(map(int, steps))), open("docs/figures/20261028_cleanbatch_full_methods.json", "w"), indent=1)
print("wrote", out1, out2)
