"""Direct-write term drawn inside a transformer block.

Horizontal residual stream through block l; the attention branch above it (q/k/v projections, attention, o_proj) and the
MLP branch below it (up/gate projections, SiLU, down_proj). The two modules that write into the residual stream, o_proj
and down_proj, carry the AdamW step ΔW (orange) and add Δh = ΔW · x to the stream; everything else, the other
projections, the neighbouring blocks, and how Δh travels on, is faded because the first-order term ignores their change.
Slide colours and fonts (base 22 pt), 17 x 7 in, dpi 200. Output docs/figures/20261030_slide_direct_write_block.png."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
NAVY, ORANGE, INK2, FADE, FADETXT, PAPER, WHITE, BLUE, BAND = "#1B2A41", "#C2410C", "#52514e", "#dddbd3", "#9a9890", "#F7F6F2", "#FFFFFF", "#2A6F97", "#cfd9e2"


def box(ax, x, y, w, h, text, fc, ec, tc, fs, lw=2.2, sub=None, subfs=None, weight="600", z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2 + (0.14 if sub else 0), text, ha="center", va="center", fontsize=fs, color=tc, zorder=z + 1, fontweight=weight)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=subfs or fs * 0.8, color=tc, zorder=z + 1)


def arrow(ax, p0, p1, color, lw=2.4, style="-|>", z=4, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=22, color=color, lw=lw, zorder=z, ls=ls, shrinkA=0, shrinkB=0))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE.parent / "docs/figures/20261030_slide_direct_write_block.png"))
    ap.add_argument("--width", type=float, default=17.0); ap.add_argument("--height", type=float, default=7.0); ap.add_argument("--font", type=float, default=22.0)
    a = ap.parse_args(argv); f = a.font
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": f, "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(a.width, a.height), dpi=200); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.set_xlim(0, 17); ax.set_ylim(0, 7); ax.axis("off")
    yr, hr = 3.2, 0.6  # residual stream band
    ax.add_patch(Rectangle((0.3, yr), 16.4, hr, fc=BAND, ec="none", zorder=1))
    ax.text(0.45, yr + hr / 2, r"residual stream  $h$", ha="left", va="center", fontsize=f * 0.8, color=NAVY, fontweight="600", zorder=2)
    arrow(ax, (16.2, yr + hr / 2), (16.7, yr + hr / 2), NAVY, lw=3)
    # neighbouring blocks, faded
    for x0, lab in ((2.4, "block ℓ − 1"), (14.0, "block ℓ + 1")):
        ax.add_patch(FancyBboxPatch((x0, 1.0), 1.4, 5.0, boxstyle="round,pad=0,rounding_size=0.15", fc=FADE, ec="none", zorder=0))
        ax.text(x0 + 0.7, 6.25, lab, ha="center", va="center", fontsize=f * 0.8, color=FADETXT)
    # block l frame
    ax.add_patch(FancyBboxPatch((4.4, 1.0), 8.2, 5.0, boxstyle="round,pad=0,rounding_size=0.15", fc="none", ec=NAVY, lw=1.8, ls=(0, (4, 3)), zorder=1))
    ax.text(8.5, 6.25, "block ℓ", ha="center", va="center", fontsize=f * 0.9, color=NAVY, fontweight="600")
    # attention branch (above the stream)
    ya, hb = 4.55, 0.9
    arrow(ax, (5.0, yr + hr), (5.0, ya + hb / 2), INK2, style="-", lw=2.0); arrow(ax, (5.0, ya + hb / 2), (5.3, ya + hb / 2), INK2, lw=2.0)
    box(ax, 5.3, ya, 1.7, hb, "q, k, v proj", FADE, FADE, FADETXT, f * 0.72)
    arrow(ax, (7.0, ya + hb / 2), (7.4, ya + hb / 2), FADETXT, lw=2.0)
    box(ax, 7.4, ya, 1.5, hb, "attention", FADE, FADE, FADETXT, f * 0.72)
    arrow(ax, (8.9, ya + hb / 2), (9.3, ya + hb / 2), INK2, lw=2.0)
    ax.text(9.1, ya + hb / 2 + 0.3, r"$x_o$", ha="center", va="bottom", fontsize=f * 0.95, color=NAVY)
    box(ax, 9.3, ya, 2.4, hb, "o_proj", WHITE, BLUE, NAVY, f * 0.8, lw=2.6, sub=r"$W_o + \Delta W_o$", subfs=f * 0.78)
    arrow(ax, (11.7, ya + hb / 2), (12.1, ya + hb / 2), ORANGE, style="-", lw=3.0); arrow(ax, (12.1, ya + hb / 2), (12.1, yr + hr), ORANGE, lw=3.0)
    ax.text(12.25, ya - 0.05, r"$+\,\Delta h_o = \Delta W_o\, x_o$", ha="left", va="top", fontsize=f * 0.95, color=ORANGE, fontweight="600")
    # MLP branch (below the stream)
    ym = 1.55
    arrow(ax, (5.0, yr), (5.0, ym + hb / 2), INK2, style="-", lw=2.0); arrow(ax, (5.0, ym + hb / 2), (5.3, ym + hb / 2), INK2, lw=2.0)
    box(ax, 5.3, ym, 1.7, hb, "up, gate proj", FADE, FADE, FADETXT, f * 0.72)
    arrow(ax, (7.0, ym + hb / 2), (7.4, ym + hb / 2), FADETXT, lw=2.0)
    box(ax, 7.4, ym, 1.5, hb, "SiLU", FADE, FADE, FADETXT, f * 0.72)
    arrow(ax, (8.9, ym + hb / 2), (9.3, ym + hb / 2), INK2, lw=2.0)
    ax.text(9.1, ym + hb / 2 + 0.3, r"$x_d$", ha="center", va="bottom", fontsize=f * 0.95, color=NAVY)
    box(ax, 9.3, ym, 2.4, hb, "down_proj", WHITE, BLUE, NAVY, f * 0.8, lw=2.6, sub=r"$W_d + \Delta W_d$", subfs=f * 0.78)
    arrow(ax, (11.7, ym + hb / 2), (12.1, ym + hb / 2), ORANGE, style="-", lw=3.0); arrow(ax, (12.1, ym + hb / 2), (12.1, yr), ORANGE, lw=3.0)
    ax.text(12.25, ym + hb + 0.05, r"$+\,\Delta h_d = \Delta W_d\, x_d$", ha="left", va="bottom", fontsize=f * 0.95, color=ORANGE, fontweight="600")
    # the two colour keys on the right
    ax.text(8.5, 0.55, "ΔW is the AdamW step of that module", ha="center", va="center", fontsize=f * 0.75, color=ORANGE)
    fig.subplots_adjust(0, 0, 1, 1); fig.savefig(a.out, facecolor=PAPER); print("wrote", a.out)


if __name__ == "__main__":
    main()
