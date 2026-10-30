#!/bin/bash
# Candidate scoring of the EM students' inputs (experiment 2, EM part): the paper-prompt RL reader, the two EM reading
# questions, the descriptions of configs/candidates/em_behaviours.json, conditions correct / random / no_vector. The
# broad/narrow external adapters (ext_*) are left out on purpose (KL confound, plan 2026-10-26). There is no EM control
# student yet, so the ranking has injection and within-run baselines only (see rank_em.sh); add the control student's
# inputs here once it exists.
#   GPU=0 bash scripts/candidates/score_em.sh behavioural
#   GPU=1 bash scripts/candidates/score_em.sh safety
# Env: R    EM runs root with em_s42/ and em_s1/ (e2/*_dact.npz; e2/inputs_w1_10_25.npz when fetched; default
#           required)
#      A    local copy of HF activation-gradient-metamodels/agm-readers (see score_subliminal.sh)
#      OUT  scores root (default results/candidates_em; files go to $OUT/<run>/e2/)
#      PY   python (default python)
# Rank afterwards: bash scripts/candidates/rank_em.sh
set -euo pipefail
G=${GPU:?set GPU}; QK=${1:?question key: behavioural or safety}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
R=$(realpath -m "${R:?set R to the EM runs root}")
A=$(realpath -m "${A:?set A to the local copy of activation-gradient-metamodels/agm-readers}")
OUT=$(realpath -m "${OUT:-$ROOT/results/candidates_em}")
PY=${PY:-python}
cd "$ROOT"
mkdir -p "$OUT/logs"
LOG=$OUT/logs/score_em_$QK.log
paths=()
for s in 42 1; do
  for f in ckptdiff_inputs_evalq_dact ckptdiff_inputs_neutral_dact ckptdiff_inputs_train_dact \
           dwx_inputs_evalq_dact dwx_inputs_neutral_dact dwx_inputs_train_dact inputs_cum_dact inputs_w1_10_25; do
    p=$R/em_s$s/e2/$f.npz
    if [ -e "$p" ]; then paths+=("$p"); else echo "missing $p (skipped)"; fi
  done
done
[ ${#paths[@]} -gt 0 ] || { echo "no input files under $R"; exit 1; }
echo "$(date) EM $QK: ${#paths[@]} files, GPU $G" | tee -a "$LOG"
"$PY" -m main_run.reader.candidate_score --gpus "$G" --inputs "${paths[@]}" \
  --adapter "$A/adapters/rl_paper_prompt_cycle040" --run-json "$A/adapters/sft_scratch_6080/RUN.json" \
  --reader-dir "$A/rl_data/reader" --reader-name rl_paper_c040 --question "$QK" \
  --candidates configs/candidates/em_behaviours.json --conditions correct,random,no_vector \
  --out-dir "$OUT/{run}/e2" --batch-rows "${BATCH_ROWS:-128}" --resume 2>&1 | tee -a "$LOG"
echo "$(date) EM $QK done" | tee -a "$LOG"
