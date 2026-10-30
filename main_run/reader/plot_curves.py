"""Validation NLL and judge score of reader runs against training examples seen, in one figure.

Left panel: per-condition val answer NLL per token from ``nll_eval.jsonl`` (written during training).
Right panel: per-condition val judge score from ``judge_curve.jsonl`` (``judge_curve.py``). Shaded
bands are bundle-bootstrap 95% CIs. The x axis is examples seen (step × effective batch: 16 for
ao_L40, 8 for scratch_L40), so the two readers' eval points line up.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from main_run.reader.judge_curve import examples_seen, read_jsonl

COND_COLOR = {"correct": "#2a78d6", "shuffled": "#eb6834", "no_vector": "#1baf7a"}
COND_LABEL = {"correct": "correct gradient", "shuffled": "shuffled gradient", "no_vector": "no vector"}
RUN_STYLE = [("-", "o"), ("--", "s"), (":", "^"), ("-.", "D")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEFAULT_FIG = (
    Path(os.environ.get("IIF_BASE", "."))
    / "research_helper/projects/activation-gradient-metamodel/docs/figures/20261023_reader_curves.png"
)


def run_label(run_dir: Path, settings: dict[str, Any]) -> str:
    init = (settings.get("lora") or {}).get("init")
    name = {"adapter": "AO init", "scratch": "scratch"}.get(init, init or "reader")
    return f"{name} ({Path(run_dir).name})"


def _by_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Last row per step, ascending."""
    last = {int(r["step"]): r for r in rows}
    return [last[s] for s in sorted(last)]


def pick_judge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows of one evaluation setting: full val (no bundle limit) first, then the most common key."""
    if not rows:
        return []
    full = [r for r in rows if r.get("limit_bundles") is None]
    pool = full or rows
    key, _ = Counter(r.get("eval_key") for r in pool).most_common(1)[0]
    return _by_step([r for r in pool if r.get("eval_key") == key])


def curve_points(
    rows: list[dict[str, Any]], settings: dict[str, Any], field: str
) -> dict[str, list[tuple[int, float, float, float]]]:
    """{condition: [(examples seen, mean, lo, hi), ...]} from rows carrying ``field`` = {cond: CI}."""
    out: dict[str, list[tuple[int, float, float, float]]] = {}
    for r in _by_step(rows):
        for cond, ci in (r.get(field) or {}).items():
            if ci.get("n", 0) and ci.get("mean") == ci.get("mean"):
                out.setdefault(cond, []).append(
                    (examples_seen(int(r["step"]), settings), float(ci["mean"]), float(ci["lo"]), float(ci["hi"]))
                )
    return out


def load_run_curves(run_dir: Path, judge_name: str = "judge_curve.jsonl") -> dict[str, Any]:
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8"))["settings"]
    judge_rows = pick_judge_rows(read_jsonl(run_dir / judge_name))
    return {
        "label": run_label(run_dir, settings),
        "nll": curve_points(read_jsonl(run_dir / "nll_eval.jsonl"), settings, "condition_nll"),
        "judge": curve_points(judge_rows, settings, "judge_score"),
        "judge_n_bundles": sorted({int(r["n_bundles"]) for r in judge_rows}),
    }


def plot(curves: list[dict[str, Any]], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharex=True)
    panels = [
        ("nll", "Val answer NLL per token (lower is better)"),
        ("judge", "Val judge score, 0 to 2 (higher is better)"),
    ]
    conds_seen: list[str] = []
    for ax, (field, title) in zip(axes, panels):
        for k, c in enumerate(curves):
            ls, marker = RUN_STYLE[k % len(RUN_STYLE)]
            for cond in ("correct", "shuffled", "no_vector"):
                pts = c[field].get(cond)
                if not pts:
                    continue
                if cond not in conds_seen:
                    conds_seen.append(cond)
                x = [p[0] / 1000 for p in pts]
                ax.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=COND_COLOR[cond],
                                alpha=0.12, linewidth=0)
                ax.plot(x, [p[1] for p in pts], color=COND_COLOR[cond], linestyle=ls, linewidth=2,
                        marker=marker, markersize=6, markeredgecolor="white", markeredgewidth=1)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.set_xlabel("Training examples seen (thousands)")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if field == "judge":
            ax.set_ylim(-0.05, 2.05)  # the rubric's full 0 to 2 range
        if not any(c[field] for c in curves):
            ax.text(0.5, 0.5, "no rows yet", transform=ax.transAxes, ha="center", va="center", color=MUTED)
    handles = [Line2D([], [], color=COND_COLOR[c], linewidth=2, label=COND_LABEL[c]) for c in conds_seen]
    handles += [
        Line2D([], [], color=MUTED, linestyle=RUN_STYLE[k % len(RUN_STYLE)][0],
               marker=RUN_STYLE[k % len(RUN_STYLE)][1], linewidth=2, label=c["label"])
        for k, c in enumerate(curves)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 5), frameon=False)
    n_b = sorted({n for c in curves for n in c["judge_n_bundles"]})
    if n_b:
        fig.suptitle(f"Reader training curves on val (judge: {', '.join(map(str, n_b))} bundles; "
                     "bands are bundle-bootstrap 95% CIs)", fontsize=11, color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    base = Path(os.environ.get("IIF_BASE", ".")) / "agm_data" / "runs"
    p = argparse.ArgumentParser(description="Plot val NLL and judge score of reader runs vs examples seen")
    p.add_argument("--run-dirs", type=Path, nargs="+", default=[base / "reader_ao_L40", base / "reader_scratch_L40"])
    p.add_argument("--judge-name", default="judge_curve.jsonl")
    p.add_argument("--out", type=Path, default=DEFAULT_FIG)
    args = p.parse_args(argv)
    curves = [load_run_curves(d, args.judge_name) for d in args.run_dirs]
    path = plot(curves, args.out)
    for c in curves:
        print(f"{c['label']}: NLL points {sum(len(v) for v in c['nll'].values())}, "
              f"judge points {sum(len(v) for v in c['judge'].values())}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
