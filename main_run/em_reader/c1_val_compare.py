#!/usr/bin/env python
"""C1 validation judge comparison for the EM reader (plan 20261026_em_reader_plan.md section 5, capability kept).

Criterion: the correct-minus-shuffled judge score gap on the 300 C1 validation bundles at the new SFT reader's
selected checkpoint is at least 80% of the reference SFT reader's (reader_scratch_L40, step 6080).

The new run's judge curve is judged through the Anthropic API (configs/c1_judge.yaml judge_provider anthropic), the
reference curve was judged through OpenRouter (same model, claude-haiku-5.5; eval key 6b31c8eb3c1f0ecb). For a
same-provider comparison this script re-judges the reference's own step-6080 predictions (generated once, former
generation path, as the new curve's --no-fast-gen predictions) with the Anthropic provider: it copies
``reader_scratch_L40/judge_curve/preds_step_006080.jsonl`` to ``runs/em_reader/c1_val/`` (the reference run directory
is only read) and scores the copy with ``c1/score.py`` through judge_curve.score_snapshot, then takes
Δ_judge(shuffled) exactly as judge_curve.build_row does (reliance_gaps "all" correct_minus_shuffled: per question,
a bundle's questions averaged first, bundle bootstrap 95% CI, seed 0). Both numbers of the reference are reported.

Run from the em-reader worktree's package (PYTHONPATH=$IIF_BASE/agm_data/worktree_em_reader). Writes
runs/em_reader/c1_val/c1_val_compare.json (a new stamped name if one exists).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_common as E  # noqa: E402

OUT_DIR = E.EVAL_ROOT / "c1_val"
REF_STEP = 6080


def main() -> int:
    from main_run.c1.score import reliance_gaps
    from main_run.reader.judge_curve import condition_score_ci, score_snapshot

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sel = json.loads((E.SFT_RUN / "selected_checkpoint.json").read_text())
    ref_sel = json.loads((E.REF_SFT_RUN / "selected_checkpoint.json").read_text())
    src = E.REF_SFT_RUN / "judge_curve" / f"preds_step_{REF_STEP:06d}.jsonl"
    copy = OUT_DIR / f"ref_reader_scratch_L40_preds_step_{REF_STEP:06d}.jsonl"
    if not copy.exists():
        shutil.copyfile(src, copy)
    judge_cfg = E.EMWT / "configs" / "c1_judge.yaml"
    scored, summary, wall = score_snapshot(copy, judge_cfg_path=judge_cfg, workers=8, dry_run=False,
                                           provider="anthropic")
    gaps = reliance_gaps(scored)
    ref_anthropic = gaps["all"]["correct_minus_shuffled"]
    new = (sel.get("judge") or {}).get("delta_judge_shuffled")
    ref_or = ref_sel["judge"]["delta_judge_shuffled"]
    rec = dict(
        criterion="new SFT reader's Delta_judge(shuffled) on C1 val >= 0.8 x the reference SFT reader's",
        new=dict(run=str(E.SFT_RUN), step=(sel.get("chosen") or {}).get("step"), provider="anthropic",
                 eval_key=(sel.get("judge") or {}).get("eval_key"), delta_judge_shuffled=new,
                 judge_score=(sel.get("judge") or {}).get("judge_score"), gate=sel.get("verdict")),
        reference_anthropic=dict(run=str(E.REF_SFT_RUN), step=REF_STEP, provider="anthropic",
                                 preds_copy=str(copy), delta_judge_shuffled=ref_anthropic,
                                 judge_score=condition_score_ci(scored),
                                 n_error=sum(1 for r in scored if r.get("judge_status") != "ok"), wall_s=wall),
        reference_openrouter=dict(step=REF_STEP, provider="openrouter", eval_key=ref_sel["judge"]["eval_key"],
                                  delta_judge_shuffled=ref_or),
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    if new and new.get("mean") is not None:
        rec["ratio_vs_reference_anthropic"] = new["mean"] / ref_anthropic["mean"]
        rec["ratio_vs_reference_openrouter"] = new["mean"] / ref_or["mean"]
        rec["passed"] = bool(new["mean"] >= 0.8 * ref_anthropic["mean"])
    else:
        rec["passed"] = None
        rec["note"] = "the new run's selected checkpoint has no judge row"
    out = OUT_DIR / "c1_val_compare.json"
    if out.exists():
        out = OUT_DIR / time.strftime("c1_val_compare_%Y%m%d_%H%M%S.json")
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("passed",) if k in rec}), "new", new, "ref anthropic", ref_anthropic,
          "ref openrouter", ref_or, "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
