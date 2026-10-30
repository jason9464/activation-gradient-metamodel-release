#!/bin/bash
# Ranks of the EM candidate scores (CPU). For every score file: against no_vector, random and the run's own first read
# step (step:5); targets the trait descriptions and the content description; once over all candidates and once without
# the finance candidates (groups target_content, benign_same_topic), so the training topic cannot take rank 1 from a
# trait. Behaviour onsets: seed 42 step 65, seed 1 step 60.
# LIMITATION: there is no EM control student, so none of these references is a matched control input; no_vector and
# random are injection baselines and step:5 is a within-run baseline. A trait-specific shift needs the plan's
# benign-finance control student (same questions, base-model answers, plain SFT); once it is trained and scored, pass
# its score file as --reference to `python -m main_run.candidate_rank compare`.
#   OUT=results/candidates_em bash scripts/candidates/rank_em.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(realpath -m "${OUT:-$ROOT/results/candidates_em}")
PY=${PY:-python}
cd "$ROOT"
mkdir -p "$OUT/rank"
n=0
for s in 42 1; do
  onset=$([ $s = 42 ] && echo 65 || echo 60)
  for f in "$OUT"/em_s$s/e2/cand_*.npz; do
    [ -e "$f" ] || continue
    case "$f" in *.partial.npz|*.tmp.npz) continue ;; esac
    b=$(basename "$f" .npz)
    for ref in no_vector random step:5; do
      "$PY" -m main_run.candidate_rank compare --treated "$f" --reference $ref \
        --targets broad_harm,misaligned_views,deception,risky_finance --onset $onset \
        --out "$OUT/rank/em_s${s}__${b}__${ref/:/}.json" > /dev/null
      "$PY" -m main_run.candidate_rank compare --treated "$f" --reference $ref \
        --targets broad_harm,misaligned_views,deception --onset $onset --exclude-groups target_content,benign_same_topic \
        --out "$OUT/rank/em_s${s}__${b}__${ref/:/}__no_finance.json" > /dev/null
      n=$((n + 2))
    done
  done
done
echo "wrote $n rank files under $OUT/rank/ (each with a .md table)"
