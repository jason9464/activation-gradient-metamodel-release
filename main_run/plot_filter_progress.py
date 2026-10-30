"""Progress figure of the p = 1 filtering students.

Students: runs/blockmix_filter_p100_tau20_s1 and _tau30_s1 (scripts/blockmix/filter.sh; every batch is all-dolphin;
each drawn batch's momentum-free direct-write term is read in-process by the reader rl_paper_c040 and the step is
dropped when first-token P(Yes) >= tau). Sources: <run>/filter_log.jsonl (one row per drawn batch: step = drawn index,
p_yes, accepted) and <run>/evals.jsonl (kind "logprob": the fixed probe P(dolphin) every 100 drawn batches).
Reference: the unfiltered dolphin student runs/newset_dolphin_s1 (probe per optimizer step, from its evals.jsonl via
plot_e2_newset.probe).

Panels: (a) share of drawn batches accepted per 100 drawn; (b) reader P(Yes) per drawn batch (faint) and its 100-batch
mean, with the two thresholds; (c) the student's own P(dolphin) against the number of ACCEPTED steps (the steps it
actually trained), with the unfiltered student's probe against its optimizer steps for comparison.
Output docs/figures/20261027_filter_progress.png (+ .json).
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import plot_e2_newset as pe  # noqa: E402

P = Path(HERE).parent
RUNS = Path(os.environ["IIF_BASE"]) / "agm_data/runs"
OUT = P / "docs/figures/20261027_filter_progress.png"
# (key, run, threshold or None for the rank rule, label, colour); fixed hues, never cycled. The two threshold students
# of 23:58 were stopped by the user at 00:29 (training starved); the rank-rule and threshold-0.6 students replaced them
# (user 00:30; code 3e29df3).
STUDENTS = [("tau20", RUNS / "blockmix_filter_p100_tau20_s1", 0.2, "threshold 0.2 (stopped)", "#1f5fa8"),
            ("tau30", RUNS / "blockmix_filter_p100_tau30_s1", 0.3, "threshold 0.3 (stopped)", "#d9772b"),
            ("rank50", RUNS / "blockmix_filter_p100_rank50_s1", None, "rank rule (keep below the median of the last 100)", "#3a8f5c"),
            ("tau60", RUNS / "blockmix_filter_p100_tau60_s1", 0.6, "threshold 0.6 (stopped)", "#6250d6"),
            # user 02:30: the random baseline at the same rate, keep 30% / 10%, and the reverse rule (keep the upper 50%)
            ("rand50", RUNS / "blockmix_filter_p100_rand50_s1", None, "random drop 50% (baseline; the reader is not consulted)", "#8a8a8a"),
            ("rank30", RUNS / "blockmix_filter_p100_rank30_s1", None, "rank rule, keep the lowest 30%", "#b5179e"),
            ("rank10", RUNS / "blockmix_filter_p100_rank10_s1", None, "rank rule, keep the lowest 10%", "#0f766e"),
            ("rank50_high", RUNS / "blockmix_filter_p100_rank50_high_s1", None, "reverse rank rule, keep the HIGHEST 50%", "#c0392b"),
            # user 08:20: the 30% and 10% students re-run with budgets that reach 7,000 trained steps (keys ..._to7k_s1)
            ("rank30_to7k", RUNS / "blockmix_filter_p100_rank30_to7k_s1", None, "rank rule, keep the lowest 30%, budget to 7,000 trained", "#7a1f6a"),
            ("rank50_ao3_to7k", RUNS / "blockmix_filter_p100_rank50_ao3_to7k_s1", None, "rank rule by the public AO, keep the lowest 50%, budget to 7,000 trained", "#e07b00"),
            ("rank30_ao3_to7k", RUNS / "blockmix_filter_p100_rank30_ao3_to7k_s1", None, "rank rule by the public AO, keep the lowest 30%, budget to 7,000 trained", "#b5651d"),
            ("rank10_to7k", RUNS / "blockmix_filter_p100_rank10_to7k_s1", None, "rank rule, keep the lowest 10%, budget to 7,000 trained", "#0b4f47"),
            ("rank30_high_to7k", RUNS / "blockmix_filter_p100_rank30_high_to7k_s1", None, "reverse rank rule, keep the HIGHEST 30%, budget to 7,000 trained", "#e07b00")]
XMAX_TRAINED = 7000
W = 100


def probe_of(row):
    d = row["result"]["dolphin"]
    return sum(math.exp(x["logprob_first"]) for x in d.values() if isinstance(x, dict) and "logprob_first" in x)


def main() -> None:
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=200)
    out = {}
    for key, run, tau, label, color in STUDENTS:
        if not (run / "filter_log.jsonl").exists():
            print("not started yet:", key); continue
        rows = [json.loads(l) for l in (run / "filter_log.jsonl").read_text().splitlines() if l.strip()]
        steps = np.array([r["step"] for r in rows]); acc = np.array([bool(r["accepted"]) for r in rows])
        py = np.array([r["p_yes"] if r.get("p_yes") is not None else np.nan for r in rows], dtype=float)
        n = len(rows)
        # (a) accepted share per W drawn
        xs, ys = [], []
        for a in range(0, n, W):
            seg = acc[a:a + W]
            if len(seg) >= 20:
                xs.append(a + len(seg) / 2); ys.append(seg.mean())
        axes[0].plot(xs, ys, color=color, lw=2, marker="o", ms=3)
        # (b) P(Yes)
        has_pyes = bool(np.isfinite(py).any())  # the random baseline never reads the batch (p_yes None)
        if has_pyes:
            axes[1].plot(steps, py, ".", color=color, ms=2, alpha=0.18)
            mv = [np.nanmean(py[max(0, i - W // 2):i + W // 2 + 1]) if np.isfinite(py[max(0, i - W // 2):i + W // 2 + 1]).any() else np.nan for i in range(n)]
            axes[1].plot(steps, mv, color=color, lw=2, label=label)
        else:
            axes[1].plot([], [], color=color, lw=2, label=label + " (no reading)")
        if tau is not None:
            axes[1].axhline(tau, color=color, ls=":", lw=1)
        elif has_pyes:  # rank rule: the moving cut the decision used, from the log (window_median of the low rule, rank_threshold otherwise)
            med = np.array([r["window_median"] if r.get("window_median") is not None else (r["rank_threshold"] if r.get("rank_threshold") is not None else np.nan) for r in rows], dtype=float)
            axes[1].plot(steps, med, color=color, ls=":", lw=1.4)
        # (c) probe vs accepted steps
        ev = [json.loads(l) for l in (run / "evals.jsonl").read_text().splitlines() if l.strip()]
        cum = np.cumsum(acc)
        pts = []
        for r in ev:
            if r["kind"] != "logprob":
                continue
            s = int(r["step"]); n_acc = int(cum[min(s, n) - 1]) if s > 0 else 0
            pts.append((s, n_acc, probe_of(r)))
        axes[2].plot([p[1] for p in pts], [max(p[2], 1e-4) for p in pts], color=color, lw=1.8, marker="o", ms=3)
        out[key] = dict(drawn=int(n), accepted=int(acc.sum()), accepted_share_per_window=list(zip([int(x) for x in xs], [round(float(y), 3) for y in ys])),
                        p_yes_mean_by_500=[(a, round(float(np.nanmean(py[a:a + 500])), 3)) for a in range(0, n, 500)],
                        probe=[(s, na, round(p, 4)) for s, na, p in pts])
    ref = [(s, v) for s, v in pe.probe("newset_dolphin_s1", "dolphin") if s <= XMAX_TRAINED]
    axes[2].plot([s for s, _ in ref], [max(v, 1e-4) for _, v in ref], color="#111111", lw=1.5, ls="--", label="unfiltered dolphin student (p = 1), by optimizer step")
    axes[2].set_yscale("log")
    axes[0].set_title("(a) accepted share per 100 drawn batches", fontsize=10); axes[0].set_xlabel("drawn batches"); axes[0].set_ylim(0, 1.02); axes[0].set_ylabel("accepted share")
    axes[1].set_title("(b) reader P(Yes) on each drawn batch (dotted: the cut each rule used)", fontsize=10); axes[1].set_xlabel("drawn batches"); axes[1].set_ylim(0, 1.0); axes[1].set_ylabel("first-token P(Yes)")
    axes[2].set_title("(c) student's P(dolphin) vs steps actually trained (log scale)", fontsize=10); axes[2].set_xlabel("trained (accepted) steps; reference: optimizer steps"); axes[2].set_ylim(1e-3, 1.0); axes[2].set_xlim(0, XMAX_TRAINED); axes[2].set_ylabel("P(dolphin), fixed probe")
    for ax in axes:
        ax.grid(alpha=0.25)
    h1, l1 = axes[1].get_legend_handles_labels(); h2, l2 = axes[2].get_legend_handles_labels()
    h1.append(plt.Line2D([], [], color="#555555", ls=":", lw=1.2)); l1.append("dotted: the cut each rule used (threshold, or the rank rule's moving median)")
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("p = 1 filtering students (every batch all-dolphin; the reader reads each drawn batch's momentum-free update; threshold rule: drop when P(Yes) >= threshold; "
                 "rank rule: keep only when P(Yes) is in the lowest k% of the last 100 drawn; random: drop 50% of the batches unread)", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, bbox_inches="tight")
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT, {k: (v["drawn"], v["accepted"]) for k, v in out.items()})


if __name__ == "__main__":
    main()
