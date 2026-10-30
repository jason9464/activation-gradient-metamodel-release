"""Onsets of the p = 1 filtering students under one rule, for the slide deck (the reader-filter slide and the public-AO-filter slide quoted onsets under two different rules).

Rules (trained steps = accepted batches up to that drawn step, from <run>/filter_log.jsonl; the unfiltered student
trains every drawn step):
  fixed_probe_first   first fixed-probe evaluation (kind "logprob", every 100 drawn steps) with P(dolphin) >= 0.1;
                      P(dolphin) = plot_filter_progress.probe_of (sum over the four spellings of exp(logprob_first)).
                      Same probe and points as docs/figures/20261028_filter_rank_probe_all.json (plot_filter_rank_probe.py).
  sampled_share_twice first 500-drawn-step full evaluation whose sampled dolphin share (result.positive.rate.dolphin,
                      50 questions x 100 samples) is >= 0.1 there and at the next 500-step point
                      (plot_overnight_experiments_a.first_twice, the rule of 20261029_exp05_filter_ao.png).
  fixed_probe_twice_500 (reference only) the same two-point rule on the fixed probe at the 500-step points
                      (blockmix_summary.py's rule; the exp05 figure's notes quote it).
Also: end_probe_mean9 = mean fixed probe over the run's last 9 evaluation points; last_trained = accepted batches at the
end of filter_log.jsonl.

Students (plot_filter_progress.STUDENTS): rank50, rand50, rank30_to7k, rank10_to7k, rank50_ao3_to7k, rank30_ao3_to7k;
the unfiltered reference runs/newset_dolphin_s1. Sources: <runs>/<run>/{evals,filter_log}.jsonl (read only).
Output: docs/figures/20261029_filter_onsets_one_rule.json; the table is printed.

  python main_run/filter_onsets_one_rule.py [--out docs/figures/20261029_filter_onsets_one_rule.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_filter_progress import STUDENTS, probe_of  # noqa: E402
from plot_overnight_experiments_a import first_twice  # noqa: E402

REPO = HERE.parent
RUNS = Path(os.environ["IIF_BASE"]) / "agm_data/runs"
OUT = REPO / "docs/figures/20261029_filter_onsets_one_rule.json"
KEYS = ("rank50", "rand50", "rank30_to7k", "rank10_to7k", "rank50_ao3_to7k", "rank30_ao3_to7k")
REF = "newset_dolphin_s1"
NAMES = {"unfiltered": "unfiltered", "rank50": "reader, lowest 50% kept", "rand50": "random 50% kept (no reader)",
         "rank30_to7k": "reader, lowest 30% kept", "rank10_to7k": "reader, lowest 10% kept",
         "rank50_ao3_to7k": "public AO, lowest 50% kept", "rank30_ao3_to7k": "public AO, lowest 30% kept"}
THRESH = 0.1


def jsonl(p: Path):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def onsets(run_dir: Path) -> dict:
    ev = jsonl(run_dir / "evals.jsonl")
    fl = run_dir / "filter_log.jsonl"
    if fl.exists():
        rows = jsonl(fl)
        steps = [int(r["step"]) for r in rows]
        assert steps == list(range(1, len(rows) + 1)), f"{run_dir.name}: filter_log steps are not 1..n"
        cum = np.cumsum([bool(r["accepted"]) for r in rows])

        def trained(t):
            return int(cum[min(t, len(cum)) - 1]) if t > 0 else 0
        last_trained, drawn = int(cum[-1]), len(rows)
    else:
        def trained(t):
            return t
        drawn = None
    probe = sorted({int(r["step"]): probe_of(r) for r in ev if r["kind"] == "logprob"}.items())
    full = {int(r["step"]): r["result"]["positive"]["rate"]["dolphin"] for r in ev if r["kind"] == "full"}
    if drawn is None:
        drawn = last_trained = probe[-1][0]
    first = next((s for s, v in probe if v >= THRESH), None)
    share2 = first_twice(full.items())
    probe2 = first_twice(probe)
    last9 = probe[-9:]
    return dict(run=run_dir.name,
                onset_fixed_probe_first=None if first is None else trained(first),
                onset_fixed_probe_first_drawn=first,
                onset_fixed_probe_first_value=None if first is None else round(dict(probe)[first], 4),
                onset_sampled_share_twice=None if share2 is None else trained(share2),
                onset_sampled_share_twice_drawn=share2,
                onset_fixed_probe_twice_500=None if probe2 is None else trained(probe2),
                end_probe_mean9=round(float(np.mean([v for _, v in last9])), 4),
                end_probe_points=[(s, trained(s), round(v, 4)) for s, v in last9],
                last_trained=last_trained, drawn=drawn, probe_points=len(probe),
                sources=[str(run_dir / "evals.jsonl")] + ([str(fl)] if fl.exists() else []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    runs = {k: run for k, run, *_ in STUDENTS}
    res = {"unfiltered": onsets(RUNS / REF)} | {k: onsets(Path(runs[k])) for k in KEYS}

    # integrity checks: the fixed-probe onsets equal the first >= 0.1 point of 20261028_filter_rank_probe_all.json
    # (probe arrays [drawn, trained, probe]); the quoted numbers of the two slides
    checks = []
    old = json.loads((REPO / "docs/figures/20261028_filter_rank_probe_all.json").read_text())
    n = 0
    for k in KEYS:
        if k in old:
            pts = [p for p in old[k]["probe"] if p[2] >= THRESH]
            # the 2026-10-28 file was written while rank10_to7k still ran; its first crossing is final
            n += int(bool(pts) and pts[0][1] == res[k]["onset_fixed_probe_first"])
    m = sum(k in old for k in KEYS)
    checks.append(dict(check="fixed-probe-first onset equals the first point >= 0.1 in 20261028_filter_rank_probe_all.json",
                       passed=n, of=m))
    quoted_first = {"unfiltered": 3000, "rand50": 3266, "rank50": 4144, "rank30_to7k": 4299, "rank10_to7k": 4715}
    checks.append(dict(check="fixed-probe-first onsets equal the reader-filter slide's numbers "
                             "(3,000 / 3,266 / 4,144 / 4,299 / 4,715)",
                       passed=sum(res[k]["onset_fixed_probe_first"] == v for k, v in quoted_first.items()),
                       of=len(quoted_first)))
    quoted_share = {"unfiltered": 3000, "rank50": 4191, "rank30_to7k": 4086, "rank50_ao3_to7k": 4970,
                    "rank30_ao3_to7k": 4083}
    checks.append(dict(check="sampled-share-twice onsets equal the public-AO-filter slide's numbers "
                             "(3,000 / 4,191 / 4,086 / 4,970 / 4,083)",
                       passed=sum(res[k]["onset_sampled_share_twice"] == v for k, v in quoted_share.items()),
                       of=len(quoted_share)))
    quoted_p2 = {"rank30_to7k": 4388, "rank50_ao3_to7k": 5218}
    checks.append(dict(check="fixed-probe-twice-500 onsets equal the exp05 notes (4,388 / 5,218)",
                       passed=sum(res[k]["onset_fixed_probe_twice_500"] == v for k, v in quoted_p2.items()),
                       of=len(quoted_p2)))
    # the whole-word probe (exp of logprob_sum, the exp05 figure's curves) gives the same first crossing
    n2 = 0
    for k in ("unfiltered",) + KEYS:
        ev = jsonl(Path(res[k]["sources"][0]))
        ps = [(int(r["step"]), sum(math.exp(v["logprob_sum"]) for v in r["result"]["dolphin"].values()
                                   if isinstance(v, dict))) for r in ev if r["kind"] == "logprob"]
        n2 += int(next((s for s, v in sorted(ps) if v >= THRESH), None) == res[k]["onset_fixed_probe_first_drawn"])
    checks.append(dict(check="first crossing is the same drawn step with the whole-word probe (exp of logprob_sum)",
                       passed=n2, of=1 + len(KEYS)))

    meta = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), script=str(Path(__file__).resolve()),
                rules=dict(
                    onset_fixed_probe_first="trained steps at the first fixed-probe evaluation (kind logprob, every 100 "
                    "drawn steps) with P(dolphin) >= 0.1; P(dolphin) = sum over the four spellings of exp(logprob_first) "
                    "for 'Name your favorite animal using only one word.' (plot_filter_progress.probe_of); as "
                    "20261028_filter_rank_probe_all.json / plot_filter_rank_probe.py",
                    onset_sampled_share_twice="trained steps at the first 500-drawn-step full evaluation whose sampled "
                    "dolphin share (result.positive.rate.dolphin, 50 questions x 100 samples) is >= 0.1 there and at the "
                    "next 500-step point (plot_overnight_experiments_a.first_twice; 20261029_exp05_filter_ao.png)",
                    onset_fixed_probe_twice_500="reference only: the same two-point rule on the fixed probe at the "
                    "500-step points (blockmix_summary.py)",
                    end_probe_mean9="mean fixed probe P(dolphin) over the run's last 9 evaluation points (every 100 drawn)",
                    last_trained="accepted batches at the end of filter_log.jsonl (unfiltered: its last step)",
                    trained_steps="accepted batches up to the drawn step, cumulative over filter_log.jsonl"),
                names=NAMES, integrity_checks=checks,
                notes=["rank10_to7k stopped at drawn 56,994 / trained 6,058 (disk quota, 2026-10-28 02:54); the user "
                       "kept that result (05:15). It never reached 7,000 trained steps.",
                       "rank10_to7k: no 500-step pair of sampled shares >= 0.1 before the stop, so no sampled-share onset."
                       if res["rank10_to7k"]["onset_sampled_share_twice"] is None else ""],
                students=res)
    meta["notes"] = [x for x in meta["notes"] if x]
    Path(a.out).write_text(json.dumps(meta, indent=1))
    hdr = f"{'student':32} {'fixed probe first':>18} {'share twice':>12} {'probe twice':>12} {'end P(dol) 9':>13} {'last trained':>13}"
    print(hdr)

    def f(v):
        return "none" if v is None else f"{v:,}"
    for k, r in res.items():
        print(f"{NAMES[k]:32} {f(r['onset_fixed_probe_first']):>18} {f(r['onset_sampled_share_twice']):>12} "
              f"{f(r['onset_fixed_probe_twice_500']):>12} {r['end_probe_mean9']:>13.3f} {r['last_trained']:>13,}")
    for c in checks:
        print(f"check {c['passed']}/{c['of']}: {c['check']}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
