#!/bin/bash
# Candidate ranks of the EM students against their context-control students (CPU;
# secondary). As scripts/candidates/rank_em.sh, but the
# reference is the matched control's score file of the same inputs (same documents, steps and candidate set), so a
# shift is EM-student log P minus control log P at the same read step. Targets: the three trait descriptions and
# risky_finance, once over all candidates and once without the finance candidates. Onsets: seed 42 step 65, seed 1 60.
#   E=<em_runs> bash scripts/em_beh/rank_ctx.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
C=$(realpath -m "${E:?set E to the em_runs dir}/cand_beh"); PY=${PY:-python3}
cd "$ROOT"; mkdir -p "$C/rank"
n=0
for s in 42 1; do
  onset=$([ $s = 42 ] && echo 65 || echo 60)
  for f in "$C"/em_ctx_s$s/e2/cand_*.npz; do
    case "$f" in *.partial.npz|*.tmp.npz) continue ;; esac
    b=$(basename "$f" .npz); t="$C/em_s$s/e2/$b.npz"
    [ -e "$t" ] || { echo "no EM file for $b"; continue; }
    "$PY" -m main_run.candidate_rank compare --treated "$t" --reference "$f" \
      --targets broad_harm,misaligned_views,deception,risky_finance --onset $onset \
      --out "$C/rank/em_s${s}__${b}__ctx.json" > /dev/null
    "$PY" -m main_run.candidate_rank compare --treated "$t" --reference "$f" \
      --targets broad_harm,misaligned_views,deception --onset $onset --exclude-groups target_content,benign_same_topic \
      --out "$C/rank/em_s${s}__${b}__ctx__no_finance.json" > /dev/null
    n=$((n + 2))
  done
done
echo "wrote $n rank files under $C/rank/"
