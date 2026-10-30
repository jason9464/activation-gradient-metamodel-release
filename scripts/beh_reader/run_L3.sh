#!/usr/bin/env bash
# Three-layer behaviour reader (layers 10, 20, 30; user 2026-10-28 14:00): SFT then RL, no judge curve, no evaluation.
# Same code paths as scripts/beh_reader/chain.sh stages sft and rl (em-reader worktree train.py; rl-stage worktree
# rl_train), with configs/beh_reader/scratch_L3_beh.yaml and rl_scratch_L3_beh_paper.yaml. Markers and pids in
# $IIF_BASE/agm_data/runs/beh_reader_L3/markers; logs in main_run/logs/<stamp>_beh_reader_L3_<stage>.out.
# Usage: GPU=0 bash scripts/beh_reader/run_L3.sh   (detached by the caller with setsid nohup)
set -u
P=$(cd "$(dirname "$0")/../.." && pwd); cd "$P" || exit 1
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY=$IIF_BASE/venvs/interpretable_influence/bin/python
EMWT=$IIF_BASE/agm_data/worktree_em_reader; RLWT=$IIF_BASE/agm_data/worktree_rl_stage
SFT_CFG=$P/configs/beh_reader/scratch_L3_beh.yaml; RL_CFG=$P/configs/beh_reader/rl_scratch_L3_beh_paper.yaml
SFT_RUN=$IIF_BASE/agm_data/runs/reader_scratch_L3_beh; RL_RUN=$IIF_BASE/agm_data/runs/reader_rl_scratch_L3_beh_paper
M=$IIF_BASE/agm_data/runs/beh_reader_L3/markers; LOGD=$P/main_run/logs; mkdir -p "$M" "$LOGD"
GPU=${GPU:-0}; STAMP=$(date +%Y%m%d_%H%M); RL_TRAIN_CAP_USD=3.5
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }
echo "$$ $(hostname)" > "$M/run_L3.pid"

# ---- SFT
if [ -d "$SFT_RUN/adapters/step_007082" ]; then say "sft: final adapter exists, skipping"; else
  say "sft on GPU $GPU"; date '+%Y-%m-%d %H:%M:%S %Z' > "$M/sft.START"
  ( cd "$EMWT" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$EMWT exec "$PY" main_run/reader/train.py \
      --config "$SFT_CFG" --batch-size 4 --device cuda:0 ) > "$LOGD/${STAMP}_beh_reader_L3_sft.out" 2>&1 < /dev/null &
  TPID=$!; echo "$TPID $(hostname)" > "$M/sft_train.pid"; wait $TPID; rc=$?
  if [ $rc -ne 0 ] || [ ! -d "$SFT_RUN/adapters/step_007082" ]; then say "sft failed rc=$rc"; date > "$M/sft.FAILED"; exit 1; fi
  date '+%Y-%m-%d %H:%M:%S %Z' > "$M/sft.DONE"; say "sft done"
fi
ADAPTER=$SFT_RUN/adapters/step_007082

# ---- RL (cost watchdog as in chain.sh)
rl_cost() { "$PY" - "$RL_RUN" <<'PYEOF'
import json, sys, os
out = sys.argv[1]; tot = 0.0; p = os.path.join(out, "rl_log.jsonl")
if os.path.exists(p):
    for line in open(p):
        try: tot += float(json.loads(line).get("judge_cost_usd", 0) or 0)
        except Exception: pass
print(f"{tot:.2f}")
PYEOF
}
if [ -f "$RL_RUN/TRAIN_DONE" ]; then say "rl: TRAIN_DONE exists, skipping"; else
  say "rl on GPU $GPU from $ADAPTER"; date '+%Y-%m-%d %H:%M:%S %Z' > "$M/rl.START"
  ( cd "$RLWT" && PYTHONPATH=$RLWT CUDA_VISIBLE_DEVICES=$GPU exec "$PY" -m main_run.reader.rl_train --config "$RL_CFG" \
      --adapter "$ADAPTER" --out "$RL_RUN" ) > "$LOGD/${STAMP}_beh_reader_L3_rl_train.out" 2>&1 < /dev/null &
  TPID=$!; echo "$TPID $(hostname)" > "$M/rl_train.pid"
  ( while kill -0 $TPID 2>/dev/null; do c=$(rl_cost 2>/dev/null || echo 0)
      if "$PY" -c "import sys; sys.exit(0 if float('$c') > $RL_TRAIN_CAP_USD else 1)"; then
        say "cost watchdog: \$$c > \$$RL_TRAIN_CAP_USD; stopping"; echo "$c" > "$M/rl_COST_CAP_HIT"; kill -TERM $TPID; break; fi
      sleep 120; done ) & WPID=$!
  wait $TPID; rc=$?; kill $WPID 2>/dev/null
  if [ $rc -ne 0 ] || [ -f "$M/rl_COST_CAP_HIT" ] || [ ! -d "$RL_RUN/adapters/cycle_040" ]; then say "rl failed rc=$rc"; date > "$M/rl.FAILED"; exit 1; fi
  date '+%Y-%m-%d %H:%M:%S %Z' > "$RL_RUN/TRAIN_DONE"; date '+%Y-%m-%d %H:%M:%S %Z' > "$M/rl.DONE"; say "rl done; judge cost \$$(rl_cost)"
fi
date '+%Y-%m-%d %H:%M:%S %Z' > "$M/run_L3.DONE"; say "all done"
