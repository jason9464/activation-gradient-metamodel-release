"""Pick a reader run's checkpoint by its validation NLL curve and report the gate (CPU only).

Rule (scratch reader plan): candidates are the in-training eval points (``<run>/nll_eval.jsonl``) whose
Δ_NLL(shuffled) bundle-bootstrap 95% CI lies above 0, where Δ_NLL(shuffled) = NLL(shuffled) − NLL(correct)
per answer token (``reliance_gaps`` on neg_nll_per_tok, group "all"; positive = the gradient helped).
The chosen checkpoint is the candidate with the lowest correct-condition NLL (earliest step on a tie).
Its Δ_judge(shuffled) and CI come from ``<run>/judge_curve.jsonl`` at the same step when present
(full-val rows of one eval setting, as plot_curves.py picks them). The gate passes when both CIs lie
above 0.

``--compare run_a run_b`` also reports the between-reader difference of the correct-condition judge
score (a − b) at each run's chosen checkpoint: per bundle, the mean of its ok correct-condition verdicts
from the step's ``<run>/judge_curve/preds_step_XXXXXX.scored.jsonl``; bundles paired by bundle_id;
bundle-bootstrap 95% CI (2,000 resamples, seed 0).

Writes ``<run>/selected_checkpoint.json`` per run (``--no-write`` to skip).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from main_run.c1.score import bootstrap_mean_ci
from main_run.reader.judge_curve import examples_seen, read_jsonl
from main_run.reader.plot_curves import _by_step, pick_judge_rows

RULE = (
    "candidates: eval points whose Delta_NLL(shuffled) = NLL(shuffled) - NLL(correct) bundle-bootstrap 95% CI "
    "lower bound > 0; chosen: the candidate with the lowest correct-condition NLL (earliest step on a tie); "
    "pass: chosen exists and its Delta_judge(shuffled) 95% CI lower bound > 0"
)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def nll_gap(row: dict[str, Any]) -> dict[str, Any] | None:
    """Δ_NLL(shuffled) CI of an nll_eval.jsonl row (all questions), or None when absent."""
    return ((row.get("reliance_gaps") or {}).get("all") or {}).get("correct_minus_shuffled")


def nll_correct(row: dict[str, Any]) -> dict[str, Any] | None:
    return (row.get("condition_nll") or {}).get("correct")


def candidates(nll_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Eval points (last row per step) whose Δ_NLL(shuffled) CI lower bound is above 0."""
    out = []
    for r in _by_step(nll_rows):
        gap, corr = nll_gap(r), nll_correct(r)
        if gap and corr and _finite(gap.get("lo")) and gap["lo"] > 0 and _finite(corr.get("mean")):
            out.append(r)
    return out


def choose(cands: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Lowest correct-condition NLL among the candidates; the earliest step on a tie."""
    if not cands:
        return None
    return min(cands, key=lambda r: (float(nll_correct(r)["mean"]), int(r["step"])))


def judge_row_at(run_dir: Path, step: int, *, judge_name: str, eval_key: str | None) -> dict[str, Any] | None:
    rows = [r for r in read_jsonl(Path(run_dir) / judge_name) if not r.get("dry_run")]
    if eval_key is not None:
        rows = _by_step([r for r in rows if r.get("eval_key") == eval_key])
    else:
        rows = pick_judge_rows(rows)
    return next((r for r in rows if int(r["step"]) == int(step)), None)


def scored_path_of(run_dir: Path, judge_row: dict[str, Any], judge_name: str) -> Path:
    """The scored predictions behind a judge_curve row (its ``preds`` field, else the default location)."""
    preds = judge_row.get("preds")
    if preds:
        p = Path(preds)
    else:
        p = Path(run_dir) / Path(judge_name).stem / f"preds_step_{int(judge_row['step']):06d}.jsonl"
    return p.with_name(p.stem + ".scored.jsonl")


def select(run_dir: Path, *, judge_name: str = "judge_curve.jsonl", eval_key: str | None = None) -> dict[str, Any]:
    """Selection record for one run (see module docstring)."""
    run_dir = Path(run_dir).resolve()
    nll_rows = read_jsonl(run_dir / "nll_eval.jsonl")
    if not nll_rows:
        raise SystemExit(f"{run_dir}/nll_eval.jsonl has no rows")
    settings = {}
    if (run_dir / "RUN.json").exists():
        settings = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8")).get("settings") or {}
    points = _by_step(nll_rows)
    cands = candidates(nll_rows)
    best = choose(cands)
    rec: dict[str, Any] = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "rule": RULE,
        "nll_split": points[-1].get("split"),
        "weighting": settings.get("weighting", points[-1].get("weighting", "tokensum")),
        "n_eval_points": len(points),
        "eval_steps": [int(r["step"]) for r in points],
        "candidate_steps": [int(r["step"]) for r in cands],
        "chosen": None,
        "judge": None,
        "verdict": "fail",
        "pass": False,
        "why": "no eval point has a Delta_NLL(shuffled) 95% CI above 0",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if best is None:
        return rec
    step = int(best["step"])
    rec["chosen"] = {
        "step": step,
        "examples_seen": examples_seen(step, settings) if settings.get("effective_batch") else None,
        "nll_correct": nll_correct(best),
        "delta_nll_shuffled": nll_gap(best),
        "adapter": str(run_dir / "adapters" / f"step_{step:06d}"),
    }
    jrow = judge_row_at(run_dir, step, judge_name=judge_name, eval_key=eval_key)
    if jrow is None:
        rec.update(verdict="pending", **{"pass": None}, why=f"no judge row at step {step} in {judge_name}")
        return rec
    dj = (jrow.get("delta_judge") or {}).get("shuffled")
    rec["judge"] = {
        "eval_key": jrow.get("eval_key"),
        "split": jrow.get("split"),
        "n_bundles": jrow.get("n_bundles"),
        "delta_judge_shuffled": dj,
        "judge_score": jrow.get("judge_score"),
        "n_judge_error_excluded": jrow.get(
            "n_judge_error_excluded", (jrow.get("judge_status_counts") or {}).get("error", 0)
        ),
        "scored": str(scored_path_of(run_dir, jrow, judge_name)),
    }
    if jrow.get("split") != rec["nll_split"]:
        rec["judge"]["warning"] = f"judge split {jrow.get('split')} but NLL split {rec['nll_split']}"
    if not dj or not _finite(dj.get("lo")):
        rec.update(verdict="pending", **{"pass": None}, why=f"judge row at step {step} has no Delta_judge(shuffled)")
    elif dj["lo"] > 0:
        rec.update(verdict="pass", **{"pass": True}, why="Delta_NLL(shuffled) and Delta_judge(shuffled) 95% CIs both above 0")
    else:
        rec.update(verdict="fail", **{"pass": False}, why="Delta_judge(shuffled) 95% CI does not lie above 0")
    return rec


def bundle_correct_scores(scored_path: Path) -> dict[str, float]:
    """Per bundle: mean judge score of its ok correct-condition rows (judge errors left out)."""
    per: dict[str, list[float]] = defaultdict(list)
    for r in read_jsonl(scored_path):
        if r.get("condition") == "correct" and r.get("judge_status") == "ok" and r.get("score") is not None:
            per[r["bundle_id"]].append(float(r["score"]))
    return {b: sum(v) / len(v) for b, v in per.items()}


def compare(a: dict[str, Any], b: dict[str, Any], *, seed: int = 0) -> dict[str, Any]:
    """Correct-condition judge score of run a minus run b at their chosen checkpoints, paired by bundle."""
    out: dict[str, Any] = {"run_a": a["run"], "run_b": b["run"]}
    for tag, rec in (("a", a), ("b", b)):
        if rec["chosen"] is None or rec["judge"] is None:
            out["error"] = f"run {tag} ({rec['run']}) has no chosen checkpoint with a judge row: {rec['why']}"
            return out
        if not Path(rec["judge"]["scored"]).exists():
            out["error"] = f"run {tag} ({rec['run']}): {rec['judge']['scored']} does not exist"
            return out
    sa = bundle_correct_scores(Path(a["judge"]["scored"]))
    sb = bundle_correct_scores(Path(b["judge"]["scored"]))
    paired = sorted(set(sa) & set(sb))
    diffs = [sa[k] - sb[k] for k in paired]
    out.update(
        step_a=a["chosen"]["step"],
        step_b=b["chosen"]["step"],
        n_bundles_paired=len(paired),
        n_bundles_only_a=len(set(sa) - set(sb)),
        n_bundles_only_b=len(set(sb) - set(sa)),
        correct_score_a=bootstrap_mean_ci([sa[k] for k in paired], seed=seed),
        correct_score_b=bootstrap_mean_ci([sb[k] for k in paired], seed=seed),
        diff_a_minus_b=bootstrap_mean_ci(diffs, seed=seed),
        seed=seed,
    )
    if a["judge"]["eval_key"] != b["judge"]["eval_key"]:
        out["warning"] = (
            f"judge eval keys differ ({a['judge']['eval_key']} vs {b['judge']['eval_key']}): "
            "the two scores come from different generation or judge settings"
        )
    return out


def _ci(d: dict[str, Any] | None, nd: int = 4) -> str:
    if not d:
        return "n/a"
    return f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}] (n={d.get('n')})"


def report(rec: dict[str, Any]) -> str:
    lines = [
        f"{rec['run']}: {rec['n_eval_points']} NLL eval points (split {rec['nll_split']}), "
        f"{len(rec['candidate_steps'])} with Delta_NLL(shuffled) 95% CI above 0: {rec['candidate_steps']}"
    ]
    c = rec["chosen"]
    if c is not None:
        ex = f", {c['examples_seen']} examples" if c.get("examples_seen") is not None else ""
        lines.append(
            f"  chosen step {c['step']}{ex}: correct NLL {_ci(c['nll_correct'])}; "
            f"Delta_NLL(shuffled) {_ci(c['delta_nll_shuffled'])}"
        )
    j = rec["judge"]
    if j is not None:
        corr = (j.get("judge_score") or {}).get("correct")
        lines.append(
            f"  judge at that step (eval key {j['eval_key']}, {j['n_bundles']} bundles): Delta_judge(shuffled) "
            f"{_ci(j['delta_judge_shuffled'], 3)}; correct score {_ci(corr, 3)}; "
            f"{j['n_judge_error_excluded']} judge error rows excluded"
        )
        if j.get("warning"):
            lines.append(f"  WARNING {j['warning']}")
    lines.append(f"  verdict: {rec['verdict'].upper()} ({rec['why']})")
    return "\n".join(lines)


def report_compare(cmp: dict[str, Any]) -> str:
    if "error" in cmp:
        return f"compare {cmp['run_a']} - {cmp['run_b']}: not computed ({cmp['error']})"
    lines = [
        f"compare {cmp['run_a']} (step {cmp['step_a']}) - {cmp['run_b']} (step {cmp['step_b']}), correct-condition "
        f"judge score, {cmp['n_bundles_paired']} paired bundles "
        f"(unpaired: {cmp['n_bundles_only_a']} only a, {cmp['n_bundles_only_b']} only b)",
        f"  a {_ci(cmp['correct_score_a'], 3)}; b {_ci(cmp['correct_score_b'], 3)}; "
        f"a - b {_ci(cmp['diff_a_minus_b'], 3)} (bundle bootstrap, seed {cmp['seed']})",
    ]
    if cmp.get("warning"):
        lines.append(f"  WARNING {cmp['warning']}")
    return "\n".join(lines)


def write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Choose a reader checkpoint by validation NLL and report the NLL + judge gate (no GPU)"
    )
    p.add_argument("runs", type=Path, nargs="*", help="run dirs (each with nll_eval.jsonl)")
    p.add_argument("--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"),
                   help="also report RUN_A - RUN_B correct-condition judge score at each chosen checkpoint")
    p.add_argument("--judge-name", default="judge_curve.jsonl", help="judge curve file inside each run dir")
    p.add_argument("--eval-key", default=None, help="judge rows of this eval key (default: full-val rows, most common key)")
    p.add_argument("--no-write", action="store_true", help="print only; do not write <run>/selected_checkpoint.json")
    args = p.parse_args(argv)

    runs = list(args.runs)
    if args.compare:
        runs = [r for r in runs if Path(r).resolve() not in {Path(c).resolve() for c in args.compare}]
        runs = [*args.compare, *runs]
    if not runs:
        p.error("give at least one run dir, or --compare RUN_A RUN_B")

    recs = [select(r, judge_name=args.judge_name, eval_key=args.eval_key) for r in runs]
    cmp = compare(recs[0], recs[1]) if args.compare else None
    for rec in recs:
        print(report(rec))
    if cmp is not None:
        print(report_compare(cmp))
        recs[0]["comparison"] = cmp
        recs[1]["comparison"] = cmp
    if not args.no_write:
        for rec in recs:
            path = Path(rec["run_dir"]) / "selected_checkpoint.json"
            write_json_atomic(path, rec)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
