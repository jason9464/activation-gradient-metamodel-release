"""Already-learned readings as grouped bars, one panel per learned behaviour X.

Per panel (X = haircuts, circular journey, tidal marsh) and per reader three bars: the reader's P(Yes) to "does this
set plant X?" on X's held-out sum under the base model, under the model fine-tuned on X, and the mean over the other
two behaviours' own readings under that same model. The values are recomputed per behaviour from the same P(Yes)
rows the suite summary averaged (new_readers_suite.e5_metrics: pyes_reader.jsonl per model; reader40 and public AO
from runs/beh_reader/already_learned). One seed. Slide fonts, dpi 200.

Output docs/figures/20261029_already_learned_bars_per_behaviour.png (+ .json).
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
sys.path.insert(0, str(HERE / "beh_reader"))
import plot_gradient_vs_activation_summary as S  # noqa: E402
import new_readers_suite as N  # noqa: E402

IIF = Path(os.environ["IIF_BASE"])
SUITE = IIF / "agm_data/runs/beh_reader/new_readers_suite"
REF = IIF / "agm_data/runs/beh_reader/already_learned"
OUT = HERE.parent / "docs/figures/20261029_already_learned_bars_per_behaviour.png"
READERS = [("reader40", "40-layer\ngradient"), ("grad3", "3-layer\ngradient"), ("slot", "3-layer\ngrad + slots"),
           ("twin", "3-layer\nactivation"), ("ao3", "public\nAO")]
CONDS = [("base", "base model, X’s documents", S.MUTED), ("own_m1", "after learning X, X’s documents", "#C2410C"),
         ("ctrl_m1", "after learning X, other behaviours’ documents", "#e9b79c")]
PANEL_TITLE = {"haircuts": "X = haircut comparisons", "circular journey": "X = circular-journey structure",
               "tidal marsh": "X = tidal-marsh comparisons"}


def fs(kind: str, f: float) -> float:
    return {"base": 22, "label": 24, "xtick": 22, "ytick": 20, "legend": 20, "value": 18}[kind] * f


def read_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--width", type=float, default=28.0)
    ap.add_argument("--height", type=float, default=9.0)
    ap.add_argument("--font", type=float, default=22.0)
    ap.add_argument("--readers", default="", help="comma-separated reader keys to keep (reader40,grad3,slot,twin,ao3); the output name gets _<readers> appended (e.g. the slide version with the two 3-layer readers)")
    ap.add_argument("--xs", default="", help="comma-separated short names of the learned behaviours to keep as panels (haircuts, circular journey, tidal marsh); output name gets _<n>panels (e.g. drop the middle panel)")
    ap.add_argument("--ctrl-base", action="store_true", help="add a fourth bar: the base model read on the other behaviours' documents (mean of their own base readings), to compare with ctrl_m1; output name gets _ctrlbase")
    a = ap.parse_args(argv)
    global READERS, CONDS
    if a.ctrl_base:
        CONDS = [CONDS[0], ("ctrl_base", "base model, other behaviours’ documents", "#cfcdc4"), CONDS[1], CONDS[2]]
        a.out = str(Path(a.out).with_name(Path(a.out).stem + "_ctrlbase.png"))
    if a.readers:
        keep = a.readers.split(","); READERS = [r for r in READERS if r[0] in keep]
        a.out = str(Path(a.out).with_name(Path(a.out).stem + "_" + a.readers.replace(",", "_") + ".png"))
    f = a.font / 22.0
    sel = json.loads((REF / "SELECT.json").read_text())
    models = N.e5_models(REF)
    X_ = models[1:]
    X_all = list(X_)  # the "other behaviours" mean always runs over every learned behaviour, whatever --xs keeps as panels
    short = {b: sel["behaviours"][b]["short"] for b in X_}
    if a.xs:
        keep = [x.strip() for x in a.xs.split(",")]; X_ = [x for x in X_ if short[x] in keep]
        a.out = str(Path(a.out).with_name(Path(a.out).stem + f"_{len(X_)}panels.png"))
    meta = N.e5_meta(REF)
    kinds, sets = meta["step_kind"], meta["set_behaviour"]

    def pyes_rows(reader: str) -> dict:
        P = {}
        for m in models:
            p = (REF / m / ("pyes_ao_actc.jsonl" if reader == "ao3" else "pyes_reader.jsonl")) if reader in ("reader40", "ao3") \
                else (SUITE / "e5" / reader / m / "pyes_reader.jsonl")
            if p.exists():
                P[m] = read_jsonl(p)
        return P

    def p_at(P: dict, m: str, beh: str) -> float | None:
        rows = P.get(m) or []
        val = {int(r["step"]): float(r["p_yes"]) for r in rows if r["q_behaviour_id"] == beh}
        vs = [val.get(i) for i, (k, s) in enumerate(zip(kinds, sets)) if k == "sum" and s == beh]
        vs = [v for v in vs if v is not None]
        return sum(vs) / len(vs) if vs else None

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": fs("base", f), "axes.labelsize": fs("label", f),
                         "xtick.labelsize": fs("xtick", f), "ytick.labelsize": fs("ytick", f),
                         "legend.fontsize": fs("legend", f)})
    fig, axes = plt.subplots(1, len(X_), figsize=(a.width, a.height), dpi=200, sharey=True)
    rec = {"figure": str(a.out), "sources": {"suite": str(SUITE / "e5"), "ref": str(REF)}, "rows": [],
           "written": time.strftime("%Y-%m-%d %H:%M:%S"), "size_in": [a.width, a.height], "dpi": 200,
           "font_pt": a.font, "script": str(Path(__file__).resolve()),
           "note": "P(Yes) on the held-out 12-document sum; ctrl_m1 = mean of the other two behaviours' own readings under M1_X; one seed"}
    data = {r: pyes_rows(r) for r, _ in READERS}
    bw = 0.26 if len(CONDS) == 3 else 0.2
    for ax, x in zip(axes, X_):
        for ri, (reader, rlab) in enumerate(READERS):
            P = data[reader]
            vals = {"base": p_at(P, "base", x), "own_m1": p_at(P, x, x)}
            ctrl = [p_at(P, x, y) for y in X_all if y != x]  # all behaviours, not only --xs: --xs had shrunk this set, hiding the one cell that stays high (haircut model on circular-journey documents, 0.92)
            ctrl = [c for c in ctrl if c is not None]
            vals["ctrl_m1"] = sum(ctrl) / len(ctrl) if ctrl else None
            cb = [p_at(P, "base", y) for y in X_all if y != x]; cb = [c for c in cb if c is not None]
            vals["ctrl_base"] = sum(cb) / len(cb) if cb else None
            for ci, (ck, clab, color) in enumerate(CONDS):
                v = vals[ck]
                if v is None:
                    continue
                xx = ri + (ci - (len(CONDS) - 1) / 2) * bw
                S.rbar(ax, xx, v, bw, color)
                ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=fs("value", f) * 0.75,
                        color=S.INK2)
                rec["rows"].append({"X": short[x], "reader": reader, "condition": ck, "p_yes": round(v, 4)})
        ax.set_title(PANEL_TITLE.get(short[x], short[x]), fontsize=fs("base", f), color=S.INK, pad=10)
        ax.set_xlim(-0.6, len(READERS) - 0.4)
        ax.set_xticks(range(len(READERS)))
        ax.set_xticklabels([lab for _, lab in READERS])
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.grid(True, color=S.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(S.BASE)
        ax.tick_params(axis="both", colors=S.INK2, length=0, pad=6)
    axes[0].set_ylabel("metamodel’s P(Yes)")
    handles = [Patch(facecolor=c, label=l) for _, l, c in CONDS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.2, handletextpad=0.5, columnspacing=1.6, fontsize=fs("legend", f) * 0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
