#!/bin/bash
# Judging: keyword scores and the em_broad_v2 judge (OpenRouter,
# Haiku 5.5) for every behaviour-reader reading of the given runs; NARROW=1 also judges every
# reading with the "narrow" truth (risky financial advice). Rerun-safe; up to $JOBS files at once.
#   bash scripts/em_beh/judge_all.sh "$E/em_s42" "$E/em_s1" "$E/em_ctx_s42" "$E/em_ctx_s1"
cd "$(dirname "$0")/../.." || exit 1
JOBS=${JOBS:-4}; READERS=${READERS:-"rl_beh_c040 sft_beh_7082"}
judge() {  # reading truth window
  local f=$1 truth=$2 win=$3 out="${1%.jsonl}_${3}_judge_${2}.jsonl"
  [ -e "$out" ] && return 0
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
  ( python3 -m main_run.em.judge_reader --reading "$f" --truth "$truth" --window "$win" --max-cost 3 \
      > "${out%.jsonl}.log" 2>&1 && echo "judged $truth $(basename "$f")" || echo "JUDGE FAILED $(basename "$f") (see ${out%.jsonl}.log)" ) &
}
for run in "$@"; do
  for rd in $READERS; do
    for f in "$run"/e2/*"${rd}"_*.jsonl; do
      [ -e "$f" ] || continue
      b=$(basename "$f" .jsonl)
      case "$b" in *_judge_*|*_read_score*) continue ;; esac
      win=cum; [[ "$b" == *_grad_w1 ]] && win=w1
      cond=correct; [[ "$b" == *_no_vector_ctl ]] && cond=no_vector; [[ "$b" == *_random_ctl ]] && cond=random
      [ -e "${f%.jsonl}_read_score.csv" ] || python3 -m main_run.em.read_score --reading "$f" --out "${f%.jsonl}_read_score" \
        --condition "$cond" > /dev/null || echo "KEYWORDS FAILED $b"
      judge "$f" em_broad_v2 $win
      if [ "${NARROW:-0}" = 1 ]; then   # every reading (training docs and raw gradients included)
        judge "$f" narrow $win
      fi
    done
  done
done
wait
