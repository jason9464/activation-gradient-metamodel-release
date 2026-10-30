#!/bin/bash
# Judging: judge with em_broad_v2 and keyword-score every Stage-A reading of the given run dirs that
# exists and has no judge file yet; up to $JOBS files at once (each with 16 requests in flight). Rerun-safe.
#   bash scripts/em_dact/judge_all.sh "$SPRINT/em_runs/em_s42" "$SPRINT/em_runs/em_s1"
cd "$(dirname "$0")/../.." || exit 1
JOBS=${JOBS:-4}; TRUTH=${TRUTH:-em_broad_v2}; QK=${QK:-behavioural}; Q=rl_paper_c040_$QK   # QK=safety for Stage A2
items() {  # name window condition
  for t in dact_evalq dact_neutral dact_train; do
    echo "ckptdiff_${Q}_$t cum correct"; echo "dwx_${Q}_$t cum correct"; done
  echo "${Q}_grad_w1 w1 correct"; echo "${Q}_grad_cum cum correct"
  echo "ckptdiff_${Q}_no_vector_ctl cum no_vector"; echo "ckptdiff_${Q}_random_ctl cum random"
}
for run in "$@"; do
  while read -r name win cond; do
    f="$run/e2/$name.jsonl"; [ -e "$f" ] || continue
    [ -e "${f%.jsonl}_read_score.csv" ] || python3 -m main_run.em.read_score --reading "$f" --out "${f%.jsonl}_read_score" \
      --condition "$cond" > /dev/null
    out="${f%.jsonl}_${win}_judge_${TRUTH}.jsonl"; [ -e "$out" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    ( python3 -m main_run.em.judge_reader --reading "$f" --truth "$TRUTH" --window "$win" --max-cost 3 \
        > "${out%.jsonl}.log" 2>&1 && echo "judged $(basename "$f")" || echo "JUDGE FAILED $(basename "$f") (see ${out%.jsonl}.log)" ) &
  done < <(items)   # not a pipeline: the jobs must belong to this shell so that wait sees them
done
wait
