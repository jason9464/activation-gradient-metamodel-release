#!/usr/bin/env bash
# Base Qwen3-14B -dL/dh token sums of every example of the collaborator's behaviour bundles (HF me-r/behaviour-bundles-v1,
# snapshot 09728f63, examples_chat.jsonl in bundle order).
# Same code path and flags as the EM v2 extraction (scripts/em_reader/extract_em.sh): extract.py of the em-reader
# worktree, all 40 grad layers, act layers ao8 (unused by the reader, kept so the store has the C1 / v2 layout),
# quantity full, full_skip_pos0 act, think_labels public. The card's recipe says --act-layers none; the only
# difference is the unused activation arrays (about 4.5 GB).
# Store: $IIF_BASE/agm_data/beh_reader/behaviour_bundles_v1 (one shard per GPU), logs and pid files in
# $IIF_BASE/agm_data/beh_reader/logs.
#   GPUS="0 1" bash scripts/beh_reader/extract.sh
set -u
IIF_BASE=${IIF_BASE:?set IIF_BASE}
W=$IIF_BASE/agm_data/worktree_em_reader
PY=$IIF_BASE/venvs/interpretable_influence/bin/python
SNAP=$IIF_BASE/hf_home/hub/datasets--me-r--behaviour-bundles-v1/snapshots/09728f63f70efc3091ae4a7845a97b29394fcba5/behaviour_bundles_v1
OUT=$IIF_BASE/agm_data/beh_reader
LOG=$OUT/logs
GPUS=${GPUS:-"0 1"}
N=$(echo $GPUS | wc -w)
mkdir -p "$LOG"
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONPATH=$W
k=0
for g in $GPUS; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup "$PY" "$W/main_run/extract.py" --dataset chat_jsonl --name behaviour_bundles_v1 \
    --data-path "$SNAP/examples_chat.jsonl" --grad-layers all --act-layers ao8 --quantities full \
    --full-skip-pos0 act --think-labels public --shard "$k/$N" --out "$OUT" --log-dir "$LOG" \
    < /dev/null > "$LOG/extract_shard${k}of${N}.out" 2>&1 &
  echo $! > "$LOG/extract_shard${k}of${N}.pid"
  echo "shard $k/$N on GPU $g pid $!"
  k=$((k + 1))
done
