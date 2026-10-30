"""Bar chart for the drop-in slide: on the dolphin student, the reader's mean first-token P(Yes)
to "Does the model have a preference for dolphins?" when one step's update is built four ways (raw gradient,
raw + momentum, raw + scaling, Adam) from the step's real dolphin batch versus from a clean batch (mean of the three
foreign clean batches). Mean and standard deviation over the probe checkpoints (steps 500-3,000 every 100; the same
P(Yes) files as plot_cleanbatch_methods.py). Figure at the golden ratio, fonts sized for a slide.

Output docs/figures/20261027_slide_cleanbatch_bars.png (+ .json with the numbers and sources).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_cleanbatch_methods import METHODS, PYES, RUNS, load_pyes, series_of  # noqa: E402

OUT = HERE.parent / "docs/figures/20261027_slide_cleanbatch_bars.png"
XLAB = {"grad": "raw", "mom": "raw +\nmomentum", "mfree": "raw +\nscaling", "step": "Adam"}
C_REAL, C_CLEAN = "#C2410C", "#9AA3AE"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-step", type=int, default=500, help="first probe checkpoint to average (default 500 = all)")
    ap.add_argument("--width", type=float, default=8.0, help="figure width in inches; height = width / 1.618 unless --height")
    ap.add_argument("--height", type=float, default=0.0, help="figure height in inches (0 = golden ratio from the width)")
    ap.add_argument("--font", type=float, default=22.0, help="base font size in pt; the other sizes scale with it")
    a = ap.parse_args(argv)
    main_ = load_pyes(RUNS / "cleanbatch_dolphin_s1" / PYES)
    var = load_pyes(RUNS / "cleanbatch_var_dolphin_s1" / PYES)
    assert main_ is not None, RUNS / "cleanbatch_dolphin_s1" / PYES
    f = a.font / 22.0
    plt.rcParams.update({"font.size": 22 * f, "axes.labelsize": 24 * f, "xtick.labelsize": 22 * f, "ytick.labelsize": 20 * f, "legend.fontsize": 20 * f})
    fig, ax = plt.subplots(figsize=(a.width, a.height or round(a.width / 1.618, 2)), dpi=200)
    rows, srcs = {}, set()
    xs = np.arange(len(METHODS)); bw = 0.36
    for i, (variant, label, _color) in enumerate(METHODS):
        ser = series_of("dolphin", variant, main_, var)
        real = {int(s): float(v) for s, v in zip(ser["real"][0], ser["real"][1]) if int(s) >= a.min_step}
        clean = {}
        for b in ("f1", "f2", "f3"):
            srcs.add(ser[b][2])
            for s, v in zip(ser[b][0], ser[b][1]):
                if int(s) >= a.min_step:
                    clean.setdefault(int(s), []).append(float(v))
        srcs.add(ser["real"][2])
        steps = sorted(set(real) & set(clean))
        r = np.array([real[s] for s in steps]); c = np.array([np.mean(clean[s]) for s in steps])
        rows[variant] = dict(label=label, steps=steps, real_mean=round(float(r.mean()), 4), real_std=round(float(r.std()), 4),
                             clean_mean=round(float(c.mean()), 4), clean_std=round(float(c.std()), 4))
        for off, val, col in ((-bw / 2, r.mean(), C_REAL), (bw / 2, c.mean(), C_CLEAN)):  # no error bars: the spread over
            ax.bar(i + off, val, bw - 0.03, color=col, zorder=3)                          # checkpoints is training-time drift, not noise
            ax.text(i + off, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=18 * f, color="#1B2A41", zorder=5)
    ax.set_xticks(xs); ax.set_xticklabels([XLAB[v] for v, _, _ in METHODS])
    ax.set_ylim(0, 0.75); ax.set_yticks([0, 0.25, 0.5, 0.75]); ax.set_ylabel("metamodel P(Yes)"); ax.grid(axis="y", alpha=0.25, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(color=C_REAL, label="real dolphin batch"), Patch(color=C_CLEAN, label="clean batch")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.6, columnspacing=1.5)
    fig.tight_layout()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    out.with_suffix(".json").write_text(json.dumps(dict(rows=rows, min_step=a.min_step, sources=sorted(srcs)), indent=1))
    print("wrote", out)
    for v, r in rows.items():
        print(f"{v:6s} real {r['real_mean']:.3f}±{r['real_std']:.3f}  clean {r['clean_mean']:.3f}±{r['clean_std']:.3f}  n {len(r['steps'])}")


if __name__ == "__main__":
    main()
