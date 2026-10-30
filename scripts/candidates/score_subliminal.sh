#!/bin/bash
# Candidate scoring of the subliminal reader inputs (experiment 2; note docs/notes/20261026_blackbox_and_candidate_scoring.md).
# The paper-prompt RL reader (cycle 40), the narrow question, 100 animals x 4 answers, conditions correct / random /
# no_vector. One group per GPU (the model loads once per group); a group can be rerun: finished files are skipped and an
# unfinished one resumes from its partial file.
#   GPU=0 bash scripts/candidates/score_subliminal.sh new           # new setting, every-500 inputs (inputs 1, 4, 4f, 5)
#   GPU=1 bash scripts/candidates/score_subliminal.sh new_every100  # new setting, 100-step grid (plain gradients and
#                                                                   # direct writes of all four animals and the control)
#   GPU=1 bash scripts/candidates/score_subliminal.sh new_every100_fixed   # the same grid, fixed-document direct writes
#   GPU=2 bash scripts/candidates/score_subliminal.sh old_adamfx    # old setting direct writes (every 100 steps)
#   GPU=3 bash scripts/candidates/score_subliminal.sh old_grad      # old setting plain gradients (every 100 steps)
#   then: new_fine, old_ckptdiff
# Env: RUNS  runs root with <run>/e2/*_inputs*.npz (default $IIF_BASE/agm_data/runs, or $DEST/student_runs after
#            scripts/candidates/fetch_inputs.sh from HF jason9464/sprint-data-2610)
#      A     local copy of HF activation-gradient-metamodels/agm-readers with adapters/rl_paper_prompt_cycle040/,
#            adapters/sft_scratch_6080/RUN.json and rl_data/reader/{meta.json,mean_unit_tokensum.npy}
#            (hf download activation-gradient-metamodels/agm-readers --include "adapters/rl_paper_prompt_cycle040/*" \
#              --include adapters/sft_scratch_6080/RUN.json --include rl_data/reader/meta.json \
#              --include rl_data/reader/mean_unit_tokensum.npy --local-dir $A)
#      OUT   scores root (default results/candidates; files go to $OUT/<run>/e2/)
#      BATCH_ROWS  answers per cached forward (default 128; every file of one experiment must use the same)
#      PY    python (default: the lab venv when present, else python)
#      REQUIRE_ALL=0  skip missing input files instead of stopping
# Rank afterwards: bash scripts/candidates/rank_subliminal.sh
set -euo pipefail
G=${GPU:?set GPU}; GROUP=${1:?group: new, new_every100, new_every100_fixed, new_fine, old_adamfx, old_grad, old_ckptdiff}
IIF=${IIF_BASE:?set IIF_BASE}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=$(realpath -m "${RUNS:-$IIF/agm_data/runs}")
A=$(realpath -m "${A:?set A to the local copy of activation-gradient-metamodels/agm-readers}")
OUT=$(realpath -m "${OUT:-$ROOT/results/candidates}")
LAB_PY=$IIF/venvs/interpretable_influence/bin/python
PY=${PY:-$([ -x "$LAB_PY" ] && echo "$LAB_PY" || echo python)}
cd "$ROOT"
mkdir -p "$OUT/logs"
LOG=$OUT/logs/score_subliminal_$GROUP.log
extra=()
case $GROUP in
  new)  # every-500 files: windows w1000, w3000, cum; steps 500, ..., last
    files=()
    for a in dolphin owl butterfly whale clean; do
      files+=(newset_${a}_s1/e2/adamfx_inputs_every500.npz newset_${a}_s1/e2/ckptdiff_inputs_every500.npz)
    done
    for a in dolphin clean; do
      files+=(newset_${a}_s1/e2/grad_inputs_every500.npz newset_${a}_s1/e2/adamfx_fixed_inputs_every500.npz)
    done
    files+=(newset_owl_s1/e2/adamfx_inputs_extra.npz newset_clean_s1/e2/adamfx_inputs_extra.npz) ;;
  new_every100)  # every 100 steps (windows w1, w100, w1000, cum), so that butterfly (onset 600) and whale (700) have
    # 5-6 read steps before onset with a clean control at every one of them (the every-500 files give one); the plain
    # gradient of owl, butterfly and whale exists only in these files
    files=(); for a in dolphin owl butterfly whale clean; do
      files+=(newset_${a}_s1/e2/grad_inputs_every100.npz newset_${a}_s1/e2/adamfx_inputs_every100.npz)
    done ;;
  new_every100_fixed)  # the same grid for the fixed-document direct writes (split off so two GPUs can share the grid;
    # each kind's treated and clean files stay in one group, so on one GPU)
    files=(); for a in dolphin owl butterfly whale clean; do files+=(newset_${a}_s1/e2/adamfx_fixed_inputs_every100.npz); done ;;
  new_fine)  # activation differences of the butterfly and whale reruns every 100 steps to 1,500. The clean student has
    # no 100-step adapters, so these are ranked against no_vector and random; their clean control (every-500 file)
    # shares only steps 500, 1,000, 1,500, one of them before onset. (The reruns' direct writes equal the main runs',
    # which new_every100 scores.)
    files=(); for a in butterfly whale; do files+=(newset_${a}_s1_fine100/e2/ckptdiff_inputs_fine100.npz); done ;;
  old_adamfx)  # direct writes; adamfx_full_clean_s2 stopped at 17,900 (.partial); step 1 is a zero vector (skipped)
    files=(adamfx_full_dolphin_s1/e2/adamfx_inputs_every100.npz adamfx_full_dolphin_s2/e2/adamfx_inputs_every100.npz
           adamfx_full_clean_s1/e2/adamfx_inputs_every100.npz adamfx_full_clean_s2/e2/adamfx_inputs_every100.partial.npz) ;;
  old_grad)  # plain gradients (the inputs of the original readings)
    files=(); for r in full_dolphin_s1 full_dolphin_s2 full_clean_s1 full_clean_s2; do files+=($r/e2/inputs_every100.npz); done ;;
  old_ckptdiff)  # checkpoint differences every 500 steps (d500, d500f, cum, g500)
    files=(); for r in full_dolphin_s1 full_dolphin_s2 full_clean_s1 full_clean_s2; do files+=($r/e2/ckptdiff_inputs_every100.npz); done ;;
  *) echo "unknown group $GROUP"; exit 2 ;;
esac
paths=(); missing=0
for f in "${files[@]}"; do
  if [ -e "$RUNS/$f" ]; then paths+=("$RUNS/$f"); else echo "missing $RUNS/$f"; missing=$((missing + 1)); fi
done
if [ "$missing" -gt 0 ] && [ "${REQUIRE_ALL:-1}" = 1 ]; then echo "$missing input files missing (REQUIRE_ALL=0 skips them)"; exit 1; fi
[ ${#paths[@]} -gt 0 ] || { echo "no input files"; exit 1; }
echo "$(date) group $GROUP: ${#paths[@]} files, GPU $G, python $PY" | tee -a "$LOG"
"$PY" -m main_run.reader.candidate_score --gpus "$G" --inputs "${paths[@]}" \
  --adapter "$A/adapters/rl_paper_prompt_cycle040" --run-json "$A/adapters/sft_scratch_6080/RUN.json" \
  --reader-dir "$A/rl_data/reader" --reader-name rl_paper_c040 --question preference --candidates animals \
  --conditions correct,random,no_vector --out-dir "$OUT/{run}/e2" --batch-rows "${BATCH_ROWS:-128}" --resume ${extra[@]+"${extra[@]}"} 2>&1 | tee -a "$LOG"
echo "$(date) group $GROUP done" | tee -a "$LOG"
