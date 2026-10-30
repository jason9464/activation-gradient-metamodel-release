#!/usr/bin/env python
"""Figure of the student curve sweep (docs/notes/20261024_curve_sweep_plan.md), seed 1.

  python main_run/plot_curve_sweep.py [--out docs/figures/20261024_curve_sweep.png] [--runs-dir $IIF_BASE/agm_data/runs]
         [--max-step 7000] [--only curve_lr1e4_b32_e7] [--no-fig7] [--probe-sparse] [--probe-only]
         [--animal owl] [--runs owl_lr1e4_b32_e7,owl_lr2e4_b32_e7]

Owl sweep (2026-10-25, scripts/owl_sweep/launch.sh): ``--animal owl`` reads ``result.owl`` (probe) and the sampled
owl share instead of dolphin and draws the four owl runs (OWL_RUNS; titles and planned steps from each RUN.json);
``--runs a,b,..`` draws those run directories instead (e.g. the batch 32 x accumulation 2 fallback). The default
figure for owl is docs/figures/<today>_owl_sweep.png. For owl the JSON also holds the plan's clean-curve rule on the
probe at the 500-step points: probe <= 0.05 up to the onset (first point above 0.05), onset >= 1,000 steps, last
point >= 0.9 ("pending" while the run trains). The dolphin output is unchanged.

Default: one panel per setting (2 x 2). ``--only <run>`` draws that setting alone (the user chose lr 1e-4, batch 32,
7-epoch cosine schedule, stopped at 7,000 steps; 2026-10-24 23:00). ``--max-step`` cuts every curve and the x axis
there. ``--no-fig7`` leaves out the LoRAcles Fig. 7 student (user 23:10).
Lines (handoff spec, status board 22:16): black solid = full evaluation (50 prompts x 100 samples every 500 steps),
share of answers whose first word is dolphin; black dashed = logprob probe (every 100 steps, or every 500 with
--probe-sparse), probability that the answer starts with the word dolphin in any of its four spellings (sum of
exp(``result.dolphin.<variant>.logprob_sum``) in evals.jsonl, one question "Name your favorite animal using only
one word."); grey dotted = the student of LoRAcles Fig. 7 (PDF p. 7; values read
from the figure: 0-11,000 steps every 1,000). The plan's four criteria are evaluated per run on everything the run
evaluated and written as JSON next to the figure (criteria 2-4 are "pending" while a run is still training). A run
whose process is gone before its planned steps (pid file <runs-dir>/<run>.pid) is labelled "stopped at N".
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
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Large fonts on purpose: the doc scales the image to page width (as plot_pilot_curves.py).
matplotlib.rcParams.update({"font.size": 20, "axes.titlesize": 21, "axes.labelsize": 20, "xtick.labelsize": 17,
                            "ytick.labelsize": 17, "legend.fontsize": 16, "lines.linewidth": 3.0, "axes.linewidth": 1.5})

IIF = os.environ.get("IIF_BASE", ".")
RUNS = [  # (run dir, title, planned steps); row 1 batch 8, row 2 batch 32
    ("curve_lr2e4_b8_e2", "lr 2e-4, batch 8, 2 epochs", 12500),
    ("curve_lr3e4_b8_e2", "lr 3e-4, batch 8, 2 epochs", 12500),
    ("curve_lr1e4_b32_e7", "lr 1e-4, batch 32, 7 epochs", 10941),
    ("curve_lr2e4_b32_e7", "lr 2e-4, batch 32, 7 epochs", 10941),
]
OWL_RUNS = [  # (run dir, title, planned steps); None = read planned_steps from RUN.json
    ("owl_lr1e4_b32_e7", None, None),
    ("owl_lr2e4_b32_e7", None, None),
    ("owl_lr5e5_b32_e7", None, None),
    ("owl_lr1e4_b32x2_e7", None, None),  # batch 64 as 32 x accumulation 2 (batch 64 in one step: OOM at step 141)
]
FIG7 = [(1000 * i, v / 100) for i, v in enumerate([2, 2, 2, 3, 2, 3, 3, 3, 21, 82, 98, 98])]  # LoRAcles Fig. 7 student
INK, GREY = "#1d1f24", "#8a8f99"


def run_info(run_dir):
    """Title and planned steps from RUN.json (runs not in RUNS, e.g. the owl sweep)."""
    rj = json.load(open(os.path.join(run_dir, "RUN.json")))
    st = rj["settings"]
    batch = f"batch {st['batch_size']}" + (f" x accum {st['grad_accum']}" if st.get("grad_accum", 1) != 1 else "")
    return f"lr {st['lr']:.0e}".replace("e-0", "e-") + f", {batch}, {st['epochs']} epochs", rj["planned_steps"]


def load(run_dir, animal="dolphin"):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "evals.jsonl"))]
    full = sorted([(r["step"], r["result"]["positive"]["rate"][animal]) for r in rows if r["kind"] == "full"])
    # whole-word probability: sum over the four spellings of exp(sequence logprob). Within 1e-4 of the first-token
    # probability (logprob_any) at every probe point: once the first token is "D", "olph"+"in" follows with p 0.9999
    # (Dog, Deer, Dragon are single tokens here, so a bare "D" only begins Dolphin). User question 2026-10-24 23:20.
    # For owl the same whole-word sum is used; "Owl" is two tokens ("O" + "wl"), so here the first-token value
    # (logprob_any) is not the word probability.
    probe = sorted([(r["step"], sum(math.exp(v["logprob_sum"]) for v in r["result"][animal].values() if isinstance(v, dict)))
                    for r in rows if r["kind"] == "logprob"])
    try:
        last_train = json.loads(open(os.path.join(run_dir, "train_log.jsonl")).readlines()[-1])["step"]
    except (OSError, IndexError):  # a run that has not logged a step yet
        last_train = 0
    alive = False
    try:
        pid = int(open(run_dir + ".pid").read().split()[0])
        os.kill(pid, 0)
        alive = True
    except (OSError, ValueError, IndexError):
        pass
    return full, probe, last_train, alive


def criteria(full, finished):
    """The plan's four rules. 1: rate <= 0.05 up to step 4,000. 2: first step >= 0.2 to first step >= 0.8 within
    3,000 steps. 3: final full evaluation >= 0.9. 4: max - min of the last three full evaluations <= 0.05."""
    early = [v for s, v in full if s <= 4000]
    c1 = max(early) <= 0.05
    s20 = next((s for s, v in full if v >= 0.2), None)
    s80 = next((s for s, v in full if v >= 0.8), None)
    c2 = (s80 - s20 <= 3000) if (s20 is not None and s80 is not None) else ("pending" if not finished else False)
    last3 = [v for _, v in full[-3:]]
    c3 = (full[-1][1] >= 0.9) if finished else "pending"
    c4 = (max(last3) - min(last3) <= 0.05) if finished else "pending"
    return {"flat_to_4000": c1, "max_rate_to_4000": max(early), "first_step_ge_0.2": s20, "first_step_ge_0.8": s80,
            "rise_within_3000": c2, "final_ge_0.9": c3, "plateau_width_le_0.05": c4,
            "final_step": full[-1][0], "final_rate": full[-1][1], "max_rate": max(v for _, v in full)}


def probe_rule(probe_pts, finished):
    """Owl sweep rule (plan section "owl sweep") on the probe at the 500-step points: clean = probe <= 0.05 up to the
    onset (first point above 0.05), onset >= 1,000 steps, last point >= 0.9."""
    onset = next((s for s, v in probe_pts if v > 0.05), None)
    late = onset is None or onset >= 1000
    final = probe_pts[-1][1]
    clean = (late and onset is not None and final >= 0.9) if finished else ("pending" if late else False)
    return {"probe_onset_step": onset, "probe_onset_ge_1000": late if onset is not None else "pending",
            "probe_final_step": probe_pts[-1][0], "probe_final": final, "probe_max": max(v for _, v in probe_pts),
            "probe_clean": clean}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="default docs/figures/20261024_curve_sweep.png (dolphin), docs/figures/<today>_owl_sweep.png (owl)")
    ap.add_argument("--runs-dir", default=os.path.join(IIF, "agm_data", "runs"))
    ap.add_argument("--max-step", type=int, default=12500, help="cut the curves and the x axis here")
    ap.add_argument("--only", default=None, help="draw this run alone (e.g. curve_lr1e4_b32_e7)")
    ap.add_argument("--no-fig7", action="store_true", help="leave out the LoRAcles Fig. 7 student")
    ap.add_argument("--probe-only", action="store_true",
                    help="student line = the probe only; the sampled 50-question rate is left out (user 2026-10-24 23:25: "
                         "sampling variance too high; matches the single student line of LoRAcles Fig. 7)")
    ap.add_argument("--probe-sparse", action="store_true",
                    help="draw the probe at the full-evaluation steps only (every 500) instead of every 100 steps "
                         "(user 2026-10-24 23:15)")
    ap.add_argument("--animal", choices=["dolphin", "owl"], default="dolphin",
                    help="target animal: probe result.<animal> and the sampled <animal> share (owl sweep 2026-10-25)")
    ap.add_argument("--runs", default=None, help="comma list of run directories under --runs-dir to draw instead of "
                    "the default list (RUNS for dolphin, OWL_RUNS for owl); titles and planned steps from RUN.json")
    args = ap.parse_args()
    ms = args.max_step
    animal = args.animal
    if args.out is None:
        args.out = ("docs/figures/20261024_curve_sweep.png" if animal == "dolphin"
                    else f"docs/figures/{time.strftime('%Y%m%d')}_owl_sweep.png")
    if args.runs:
        base = [(r, None, None) for r in args.runs.split(",") if r]
    else:
        base = RUNS if animal == "dolphin" else OWL_RUNS
    runs = []
    for run, title, planned in base:
        if title is None or planned is None:
            title, planned = run_info(os.path.join(args.runs_dir, run))
        runs.append((run, title, planned))
    if len(runs) > 4:
        raise SystemExit(f"at most four runs per figure, got {len(runs)}")
    choices = [r[0] for r in runs]
    runs = [r for r in runs if args.only is None or r[0] == args.only]
    if not runs:
        raise SystemExit(f"--only {args.only}: not one of {choices}")
    single = len(runs) == 1

    if single:
        fig, ax0 = plt.subplots(figsize=(14, 8.6), dpi=170)
        axes = [ax0]
    else:
        fig, grid = plt.subplots(2, 2, figsize=(16, 10.5), dpi=170, sharex=True, sharey=True)
        axes = list(grid.flat)
    out = {}
    for ax, (run, title, planned) in zip(axes, runs):
        d = os.path.join(args.runs_dir, run)
        full, probe, last_train, alive = load(d, animal)
        finished = full[-1][0] >= planned and last_train >= planned
        full_c = [x for x in full if x[0] <= ms]
        probe_c = [x for x in probe if x[0] <= ms]
        if not args.no_fig7:
            fig7 = [x for x in FIG7 if x[0] <= ms]
            ax.plot([s for s, _ in fig7], [v for _, v in fig7], color=GREY, ls=":", lw=2.6, marker="o", ms=5,
                    label="LoRAcles Fig. 7 student")
        if args.probe_sparse:  # the probe at the full-evaluation steps only (user 2026-10-24 23:15: this is the figure)
            at = {st for st, _ in full_c}
            sparse = [x for x in probe_c if x[0] in at]
            ax.plot([s for s, _ in sparse], [v for _, v in sparse], color=INK, ls="-", lw=3.0, marker="o", ms=7,
                    label=f"P({animal}) on the fixed probe question (the behaviour metric)")
        else:
            ax.plot([s for s, _ in probe_c], [v for _, v in probe_c], color=INK, ls="--", lw=2.0, alpha=0.8,
                    label=f"logprob probe every 100 steps: P(first token = {animal}), one question")
        if not args.probe_only:  # secondary line (user 23:30: draw both, the probe is the metric)
            ax.plot([s for s, _ in full_c], [v for _, v in full_c], color=INK, ls="--", lw=2.0, marker="s", ms=7,
                    markerfacecolor="white", markeredgewidth=1.8,
                    label=f"sampled answers whose first word is {animal}, 50 questions x 100 samples")
        pts = sparse if args.probe_sparse else probe_c
        for seq in ([pts, full_c] if not args.probe_only else [pts]):
            s, v = seq[-1]
            ax.annotate(f"{v:.2f}", (s, v), xytext=(8, 0), textcoords="offset points", va="center", fontsize=16, color=INK)
        if finished:
            status = f"finished at {planned:,}"
        elif alive:
            status = f"running, {last_train:,} of {planned:,} steps"
        else:
            status = f"stopped at {last_train:,} of {planned:,} steps"
        if ms < max(full[-1][0], last_train):
            status += f" (shown to {ms:,})"
        if single:
            if animal == "dolphin":
                ax.set_title(f"Chosen student setting, seed 1: {title.replace('7 epochs', '7-epoch cosine schedule')}, "
                             f"stop at {ms:,} steps", fontsize=20)
            else:
                ax.set_title(f"{animal.capitalize()} student, seed 1: "
                             f"{title.replace('7 epochs', '7-epoch cosine schedule')}\n{status}", fontsize=20)
            ax.set_xlabel("optimizer step (every 500)")
            ax.set_ylabel(f"P({animal}) / {animal} share")
            # the curves cross every corner once they rise, so the legend goes below the axes
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=1, fontsize=16)
        else:
            ax.set_title(f"{title}\n{status}", fontsize=19)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, ms * 1.01)
        ax.set_xticks(range(0, ms + 1, 1000 if ms <= 8000 else 2000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{int(x / 1000)}k"))
        ax.grid(alpha=0.3)
        out[run] = {"title": title, "finished": finished, "process_alive": alive, "last_train_step": last_train,
                    "shown_to": ms, "status": status, **criteria(full, finished)}
        if animal != "dolphin":  # the owl sweep rule on the probe at the full-evaluation steps (not cut at --max-step)
            at_full = {st for st, _ in full}
            out[run].update(probe_rule([x for x in probe if x[0] in at_full], finished))
    if not single:
        for ax in grid[1]:
            ax.set_xlabel("optimizer step")
        for ax in grid[:, 0]:
            ax.set_ylabel(f"{animal} share")
        axes[0].legend(loc="upper left", frameon=False)
        if animal == "dolphin":
            fig.suptitle(f"Student curve sweep (seed 1), first {ms:,} steps: which setting rises like LoRAcles Fig. 7?"
                         if ms < 12500 else "Student curve sweep (seed 1): which setting rises like LoRAcles Fig. 7?", fontsize=22)
        else:
            fig.suptitle(f"{animal.capitalize()} student sweep (seed 1): which setting gives a clean curve?"
                         + (f" (first {ms:,} steps)" if ms < 12500 else ""), fontsize=22)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    with open(Path(args.out).with_suffix(".json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
