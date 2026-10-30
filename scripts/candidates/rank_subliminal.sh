#!/bin/bash
# Ranks of the subliminal candidate scores (CPU): every treated file against its clean control, no_vector and random,
# and the null comparisons. Writes $OUT/subliminal_rank.json and .md. Then the dolphin/whale cross-check (rule fixed
# before any score was opened; note 20261026_blackbox_and_candidate_scoring.md): $OUT/crosscheck_dolphin_whale.json/.md.
#   OUT=results/candidates bash scripts/candidates/rank_subliminal.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(realpath -m "${OUT:-$ROOT/results/candidates}")
IIF=${IIF_BASE:?set IIF_BASE}
LAB_PY=$IIF/venvs/interpretable_influence/bin/python
PY=${PY:-$([ -x "$LAB_PY" ] && echo "$LAB_PY" || echo python)}
cd "$ROOT"
"$PY" -m main_run.candidate_rank plan --scores-root "$OUT" --reader-name rl_paper_c040 --question preference \
  --candidates animals --out "$OUT/subliminal_rank.json"
"$PY" -m main_run.candidate_rank crosscheck --scores-root "$OUT" --reader-name rl_paper_c040 --question preference \
  --candidates animals --out "$OUT/crosscheck_dolphin_whale.json"
