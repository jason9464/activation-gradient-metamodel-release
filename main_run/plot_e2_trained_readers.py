#!/usr/bin/env python
"""Dolphin detection by the trained readers during the student's SFT: AO-init reader versus scratch reader.

  python plot_e2_trained_readers.py [--out docs/figures/20261024_e2_trained_readers.png]

One panel per reader. Lines: share of the reader's answers to the animal question that mention "dolphin", pooled in
1,000-step bins, one line per input (1-step, 100-step, 1,000-step, cumulative gradient sum) on the dolphin run(s);
gray dashed: the same reader on the clean-number control(s), pooled over the four inputs; black: the student's own
dolphin rate from its full evaluation every 500 steps. The y axis is square-root scaled so that rates below 5% stay
visible. Reads complete files and partial ``.jsonl.tmp`` files (a chain still running); seeds are averaged where more
than one run is available and the legend says which seeds went in.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 21, "axes.titlesize": 23, "axes.labelsize": 21, "xtick.labelsize": 18, "ytick.labelsize": 18,
                            "legend.fontsize": 16, "lines.linewidth": 3.0, "axes.linewidth": 1.5})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

IIF = os.environ.get("IIF_BASE", ".")
RUNS_DIR = os.path.join(IIF, "agm_data", "runs")
READERS = [("trained_ao_3000", "AO-init reader"), ("trained_scratch_6080", "Scratch reader")]
INPUTS = [("w1", "1-step sum", "#9ecae1"), ("w100", "100-step sum", "#4292c6"), ("w1000", "1,000-step sum", "#08519c"), ("cum", "cumulative sum", "#d97b1a")]
BIN = 1000


def read_rows(path):
    rows = []
    for l in open(path):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return rows


def reading_files(reader, run, include_partial=False):
    """complete file, or a .tmp file that already holds all 201 readings; partial .tmp only with include_partial"""
    d = os.path.join(RUNS_DIR, run, "e2")
    p = os.path.join(d, f"{reader}_animal.jsonl")
    if os.path.isfile(p):
        return p
    p = p + ".tmp"
    if os.path.isfile(p) and (include_partial or len(read_rows(p)) >= 201):
        return p
    return None


def binned(rows, keys):
    """mentions and answers per 1,000-step bin, pooled over the given input keys: x, rate, total answers, total mentions, m per bin, n per bin"""
    m, n = {}, {}
    for r in rows:
        b = (max(r["step"], 1) - 1) // BIN
        for k in keys:
            for t in r["inputs"][k]["texts"]:
                n[b] = n.get(b, 0) + 1
                m[b] = m.get(b, 0) + ("dolphin" in t.lower())
    bins = sorted(n)
    return (np.array([(b + 0.5) * BIN for b in bins]), np.array([m[b] / n[b] for b in bins]), sum(n.values()), sum(m.values()),
            np.array([m[b] for b in bins]), np.array([n[b] for b in bins]))


def wilson(m, n, z=1.96):
    """Wilson 95% interval of m/n, returned as (lo, hi) arrays"""
    m = np.asarray(m, dtype=float)
    n = np.asarray(n, dtype=float)
    p = m / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def merged(curves):
    """sum mentions and answers of several runs bin by bin"""
    m, n = {}, {}
    for c in curves:
        for x, mm, nn in zip(c[0], c[4], c[5]):
            m[x] = m.get(x, 0) + mm
            n[x] = n.get(x, 0) + nn
    xs = np.array(sorted(n))
    return xs, np.array([m[x] for x in xs]), np.array([n[x] for x in xs])


def behaviour(run):
    p = os.path.join(RUNS_DIR, run, "evals.jsonl")
    rows = [json.loads(l) for l in open(p)]
    rows = sorted((r for r in rows if r["kind"] == "full"), key=lambda r: r["step"])
    return [r["step"] for r in rows], [r["result"]["positive"]["rate"]["dolphin"] for r in rows]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "figures" / "20261024_e2_trained_readers.png"))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--include-partial", action="store_true", help="also use .jsonl.tmp files of chains still running")
    ap.add_argument("--no-student", action="store_true", help="leave out the student's own dolphin rate; linear y axis")
    ap.add_argument("--inputs", default="w1,w100,w1000,cum", help="which inputs to draw, comma separated")
    ap.add_argument("--borrow-clean", action="store_true", help="with --per-seed, fill a missing clean line with the same reader's other seed (off by default, user 2026-10-24)")
    ap.add_argument("--per-seed", action="store_true", help="one panel per reader and dolphin seed, partial readings included")
    ap.add_argument("--student-twin", action="store_true", help="with --no-student, draw the student's dolphin rate on a right-hand axis")
    ap.add_argument("--errorbars", action="store_true", help="with --no-student, Wilson 95%% intervals as capped error bars")
    ap.add_argument("--bands", action="store_true", help="with --no-student, shade Wilson 95%% intervals (off by default, user request 2026-10-24)")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.per_seed:
        panels = []
        for reader, title in READERS:
            for s in seeds:
                pd = reading_files(reader, f"full_dolphin_s{s}", True)
                if pd:
                    pc = reading_files(reader, f"full_clean_s{s}", True)
                    cl = [(s, pc)] if pc else []
                    if not cl and args.borrow_clean:  # a complete clean reading of the SAME reader from another seed, labelled
                        for s2 in seeds:
                            pc2 = reading_files(reader, f"full_clean_s{s2}", False)
                            if pc2:
                                cl = [(s2, pc2)]
                                break
                    panels.append((reader, f"{title}, seed {s}", [(s, pd)], cl))
        ncol = max(1, len(panels) // len(READERS))
        fig, axes = plt.subplots(len(READERS), ncol, figsize=(8.5 * ncol, 6.2 * len(READERS)), dpi=170, sharey=True, squeeze=False)
        axes = axes.ravel()
    else:
        panels = None
        fig, axes = plt.subplots(1, 2, figsize=(17, 7), dpi=170, sharey=True)
    summary = {}
    twin_axes = []
    for ax, panel in zip(axes, panels if panels else READERS):
        if panels:
            reader, title, dol, cln = panel
            n_read = len(read_rows(dol[0][1]))
            if n_read < 201:
                title += f", up to step {read_rows(dol[0][1])[-1]['step']:,}"
            if cln and len(read_rows(cln[0][1])) < 201:
                title += f" (clean to {read_rows(cln[0][1])[-1]['step']:,})"
        else:
            reader, title = panel
            dol = [(s, reading_files(reader, f"full_dolphin_s{s}", args.include_partial)) for s in seeds]
            dol = [(s, p) for s, p in dol if p]
            cln = [(s, reading_files(reader, f"full_clean_s{s}", args.include_partial)) for s in seeds]
            cln = [(s, p) for s, p in cln if p]
        summary[title] = {}
        reader_key = title
        # per-input lines on the dolphin runs, seeds averaged bin by bin
        keep = set(args.inputs.split(","))
        for i, (key, label, color) in enumerate(INPUTS):
            if key not in keep:
                continue
            curves = [binned(read_rows(p), [key]) for _, p in dol]
            if not curves:
                continue
            xs, m, n = merged(curves)
            ys = m / n
            if args.no_student:
                lo, hi = wilson(m, n)
                if args.errorbars:
                    ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], color=color, label=label, marker="o", ms=6, capsize=4, elinewidth=1.6)
                else:
                    ax.plot(xs, ys, color=color, label=label, marker="o", ms=6)
                if args.bands:
                    ax.fill_between(xs, lo, hi, color=color, alpha=0.13, lw=0)
            else:
                ax.plot(xs, ys, color=color, label=label)
            summary[reader_key][key] = {"mentions": int(m.sum()), "answers": int(n.sum())}
        if cln:
            curves = [binned(read_rows(p), [k for k, _, _ in INPUTS]) for _, p in cln]
            xs, m, n = merged(curves)
            ys = m / n
            if args.no_student:
                lo, hi = wilson(m, n)
                clabel = "clean control, all inputs"
                if panels and cln[0][0] != dol[0][0]:
                    clabel = f"clean control seed {cln[0][0]}, all inputs"
                if args.errorbars:
                    ax.errorbar(xs + 120, ys, yerr=[ys - lo, hi - ys], color="gray", ls="--", label=clabel, marker="s", ms=6, capsize=4, elinewidth=1.6)
                else:
                    ax.plot(xs, ys, color="gray", ls="--", label=clabel, marker="s", ms=6)
                if args.bands:
                    ax.fill_between(xs, lo, hi, color="gray", alpha=0.13, lw=0)
            else:
                ax.plot(xs, ys, color="gray", ls="--", label="clean control, all inputs")
            summary[reader_key]["clean"] = {"mentions": int(m.sum()), "answers": int(n.sum())}
        if not args.no_student:
            for s, _ in dol:
                bx, by = behaviour(f"full_dolphin_s{s}")
                ax.plot(bx, by, color="black", lw=3.4, label="student's dolphin rate" if s == dol[0][0] else None, alpha=0.9 if s == dol[0][0] else 0.5)
        elif args.student_twin and dol:
            ax2 = ax.twinx()
            for s, _ in dol:
                bx, by = behaviour(f"full_dolphin_s{s}")
                ax2.plot(bx, by, color="black", lw=2.6, alpha=0.85, label="student's dolphin rate, right axis")
            ax2.set_ylim(0, 1)
            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
            ax2.tick_params(axis="y", labelsize=16, colors="black")
            twin_axes.append(ax2)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, handlelength=1.6, fontsize=15)
        dseeds = ", ".join(str(s) for s, _ in dol)
        cseeds = ", ".join(str(s) for s, _ in cln)
        ax.set_title(title, fontsize=20 if panels else None)
        summary[reader_key]["seeds"] = {"dolphin": dseeds, "clean": cseeds}
        ax.set_xlabel("student optimizer step")
        ax.set_xlim(0, 20000)
        ax.set_xticks([0, 5000, 10000, 15000, 20000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{int(v / 1000)}k"))
        if args.no_student:
            wide = args.bands or args.errorbars
            ax.set_ylim(0, 0.105 if wide else 0.07)
            ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.10] if wide else [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
        else:
            ax.set_yscale("function", functions=(np.sqrt, np.square))
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.002, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}%"))
        ax.grid(alpha=0.3)
    for a in axes[::(len(axes) // len(READERS)) if panels else 2]:
        a.set_ylabel("answers naming dolphin")
    if panels:
        for a in axes:
            if not (args.student_twin and a.get_legend() is not None):
                a.legend(loc="upper right" if args.no_student else "upper left", frameon=False, handlelength=1.6)
        if twin_axes:
            twin_axes[-1].set_ylabel("student's dolphin rate")
    else:
        max(axes, key=lambda a: len(a.get_legend_handles_labels()[0])).legend(loc="upper right", frameon=False, handlelength=1.6)
    fig.tight_layout(w_pad=1.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    with open(Path(args.out).with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
