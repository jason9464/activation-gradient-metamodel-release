"""Per-step figure of the four ways to build one step's weight update, read as a direct-write term,
per step: raw / raw + momentum / raw + scaling / Adam.

Value: the reader's first-token P(Yes) to "Does the model have a preference for dolphins? Answer yes or no, then explain
briefly." (cleanbatch_pyes.py; it matched the 100-answer judge counts within a few points) on the direct-write term
D = sum_docs (dW_o X_o + dW_down X_down) of the would-be update at the probe step, dolphin student and clean-number student
re-runs, steps 500-3,000 every 100.

Methods (window variant of the inputs file -> label): grad = raw gradient (-g), mom = raw + momentum (-lr m_hat; the
variant run runs/cleanbatch_var_<arm>_s1, drawn when its P(Yes) file exists), mfree = raw + per-coordinate scaling
(-lr g / (sqrt(v_hat) + eps)), step = Adam (momentum + scaling, the real would-be step). Panels: (a) dolphin student,
the step's real dolphin batch; (b) dolphin student, the three foreign clean batches (mean of the three); (c) clean
student, its real clean batch. The dolphin student's own P(dolphin) (fixed probe, 500-step points) is drawn in (a) and
(b) from docs/figures/20261025_e2_newset.json (the re-run reproduces that trajectory bit for bit).

Output docs/figures/20261026_cleanbatch_methods_pyes.png (+ .json with the plotted numbers and the sources).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ["IIF_BASE"]) / "agm_data/runs"
PYES = "train/e2/cleanbatch_inputs_all_pyes_rl_paper_c040_dolphin_yesno.json"
NEWSET_JSON = P / "docs/figures/20261025_e2_newset.json"
OUT = P / "docs/figures/20261026_cleanbatch_methods_pyes.png"

# fixed order and hues (never cycled): raw grey, raw+momentum green, raw+scaling orange, Adam blue
METHODS = [("grad", "raw gradient", "#8a8a8a"),
           ("mom", "raw + momentum", "#3a8f5c"),
           ("mfree", "raw + scaling (momentum-free)", "#d9772b"),
           ("step", "Adam (momentum + scaling)", "#1f5fa8")]
BATCHES = ("real", "f1", "f2", "f3")


def load_pyes(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    W = list(d["windows"])
    P_ = np.asarray(d["p_yes"], dtype=float)  # [steps, windows]
    return dict(steps=[int(s) for s in d["steps"]], series={w: P_[:, i] for i, w in enumerate(W)}, path=str(path))


def series_of(arm: str, variant: str, main: dict | None, var: dict | None):
    """{batch: P(yes) per step} for one method; main run first, the variant run for windows the main run lacks."""
    out = {}
    for src in (main, var):
        if src is None:
            continue
        for b in BATCHES:
            w = f"{b}_{variant}"
            if w in src["series"] and b not in out:
                out[b] = (src["steps"], src["series"][w], src["path"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--smooth", type=int, default=1, help="centered moving average over this many probe steps (odd; 1 = none); "
                    "the raw points stay as faint markers")
    ap.add_argument("--panels", default="abc", help="which panels to draw, a subset of 'abc' (slide: 'ab')")
    ap.add_argument("--methods", default="grad,mom,mfree,step", help="comma list of variants to draw")
    ap.add_argument("--simple", action="store_true", help="slide version: no suptitle, short titles, labels and legend, larger fonts")
    args = ap.parse_args(argv)
    if args.simple:
        plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12})
    if args.smooth > 1 and args.out == str(OUT):
        args.out = str(OUT.with_name(OUT.stem + f"_smooth{args.smooth}.png"))
    data = {arm: dict(main=load_pyes(RUNS / f"cleanbatch_{arm}_s1" / PYES),
                      var=load_pyes(RUNS / f"cleanbatch_var_{arm}_s1" / PYES)) for arm in ("dolphin", "clean")}
    for arm in data:
        assert data[arm]["main"] is not None, f"missing {RUNS / f'cleanbatch_{arm}_s1' / PYES}"
    probe = [(s, p) for s, p in json.loads(NEWSET_JSON.read_text())["behaviour"]["probe_at_read_steps"] if 450 <= s <= 3050]

    panels = [("dolphin", "real", "(a) dolphin student, the step's real dolphin batch"),
              ("dolphin", "foreign", "(b) dolphin student, foreign clean batches (mean of 3)"),
              ("clean", "real", "(c) clean-number student, its real clean batch")]
    keep = [i for i, c in enumerate("abc") if c in args.panels]
    panels = [panels[i] for i in keep]
    if args.simple:
        short = {"a": "(a) the step's real dolphin batch", "b": "(b) a clean batch, same student", "c": "(c) clean student, clean batch"}
        panels = [(arm, which, short["abc"[i]]) for (arm, which, _), i in zip(panels, keep)]
    fig, axes = plt.subplots(1, len(panels), figsize=((6.0 * len(panels), 4.6) if args.simple else (16.5, 4.8)), dpi=200, sharey=True, squeeze=False)
    axes = axes[0]
    plotted, missing = {}, []
    for ax, (arm, which, title) in zip(axes, panels):
        for variant, label, color in [m for m in METHODS if m[0] in args.methods.split(",")]:
            ser = series_of(arm, variant, data[arm]["main"], data[arm]["var"])
            if which == "real":
                if "real" not in ser:
                    missing.append((arm, which, variant)); continue
                steps, y, src = ser["real"]
            else:
                fs = [ser[b] for b in ("f1", "f2", "f3") if b in ser]
                if len(fs) < 3:
                    missing.append((arm, which, variant)); continue
                steps, src = fs[0][0], fs[0][2]
                y = np.mean([f[1] for f in fs], axis=0)
            if args.smooth > 1:
                h = args.smooth // 2
                ys = np.array([np.mean(y[max(0, i - h):i + h + 1]) for i in range(len(y))])  # window shrinks at the ends
                ax.plot(steps, y, color=color, lw=0.9, alpha=0.28)  # the raw per-step values as a faint line
                ax.plot(steps, ys, color=color, lw=2.2, label=label)
                plotted[f"{arm}:{which}:{variant}"] = dict(steps=list(steps), p_yes=[round(float(v), 4) for v in y],
                                                           smoothed=[round(float(v), 4) for v in ys], source=src)
            else:
                ax.plot(steps, y, color=color, lw=1.8, marker="o", ms=3, label=label)
                plotted[f"{arm}:{which}:{variant}"] = dict(steps=list(steps), p_yes=[round(float(v), 4) for v in y], source=src)
        if arm == "dolphin":
            ax.plot([s for s, _ in probe], [p for _, p in probe], color="#111111", lw=1.3, ls="--", marker="s", ms=4,
                    label=("student's own P(dolphin)" if args.simple else "dolphin student's own P(dolphin), fixed probe (500-step points)"))
        ax.set_title(title, fontsize=(14 if args.simple else 9.5))
        ax.set_xlabel("step" if args.simple else "student optimizer step")
        ax.set_xlim(450, 3050); ax.set_ylim(0, 1.0); ax.set_xticks(range(500, 3001, 500)); ax.grid(alpha=0.25)
    axes[0].set_ylabel("reader P(Yes)" if args.simple else "first-token P(Yes)")
    handles, labels = axes[0].get_legend_handles_labels()
    if args.simple:
        short_lab = {"raw gradient": "raw", "raw + momentum": "raw + momentum", "raw + scaling (momentum-free)": "raw + scaling", "Adam (momentum + scaling)": "Adam"}
        labels = [short_lab.get(l, l) for l in labels]
    not_yet = [lab for v, lab, _ in METHODS if any(m[2] == v for m in missing)]
    if not_yet:
        labels = labels + [f"not yet read: {', '.join(sorted(set(not_yet)))}"]
        handles = handles + [plt.Line2D([], [], color="none")]
    fig.legend(handles, labels, loc="lower center", ncol=(5 if args.simple else 3), fontsize=(12 if args.simple else 8), frameon=False, bbox_to_anchor=(0.5, -0.02))
    if not args.simple:
      fig.suptitle('Reader\'s first-token P(Yes) to "Does the model have a preference for dolphins?" on the direct-write term '
                 "of one step's would-be update (batch of 32 documents), by how the update is built"
                 + (f"; bold lines = centered moving average over {args.smooth} probe steps ({args.smooth * 100} steps), faint lines = raw" if args.smooth > 1 else ""),
                 fontsize=10)
    fig.tight_layout(rect=(0, (0.10 if args.simple else 0.08), 1, 1))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    out.with_suffix(".json").write_text(json.dumps(dict(
        question="Does the model have a preference for dolphins? Answer yes or no, then explain briefly.",
        value="first-token P(Yes) of the reader rl_paper_c040 (cleanbatch_pyes.py)",
        methods={v: lab for v, lab, _ in METHODS}, plotted=plotted, missing=missing, probe=probe,
        sources={arm: {k: (v["path"] if v else None) for k, v in d.items()} for arm, d in data.items()}), indent=1))
    print("wrote", out, "| missing:", missing)


if __name__ == "__main__":
    main()
