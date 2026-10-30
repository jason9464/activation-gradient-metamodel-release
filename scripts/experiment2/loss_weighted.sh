#!/usr/bin/env bash
# Loss-weighted activation reader scoring (docs/notes/20261029_loss_weighted_act_reader_results.md).
# CPU convert; GPU pyes wherever the activation reader fits (~30 min).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-python}"
VEC="${VEC:-$ROOT/artifacts/fwdb/data/vec}"
BUNDLES="${BUNDLES:-$ROOT/artifacts/fwdb/data/bundles.parquet}"
BEH="${BEH:-$ROOT/artifacts/fwdb/data/raw/beh/behaviour_bundles_v1/bundles.jsonl}"
OUT="${OUT:-$ROOT/artifacts/fwdb/data/fwd/readerdirs_L3}"
RUN_DIR="${RUN_DIR:-${E2_RUN:-$ROOT/artifacts/experiment2_out/runs}/loss_weighted}"
MEAN="${MEAN:-}"
TEMPLATE="${TEMPLATE:-${E2_ACT_DIR:-}}"
FETCH_TWIN=0
SHUFFLE_EXT="${SHUFFLE_EXT:-}"
GPU="${GPU:-0}"
ADAPTER="${ADAPTER:-}"
RUN_JSON="${RUN_JSON:-}"
OVERWRITE=0

usage() {
  cat <<EOF
Usage: bash scripts/experiment2/loss_weighted.sh (--mean PATH | --template DIR | --fetch-twin-mean) [options]

  --mean PATH          Experiment 2 activation_reader_dir/mean_unit_tokensum.npy
  --template DIR       directory containing that mean (alternative to --mean)
  --fetch-twin-mean    download Jaesung's twin mean (same formula; confirm via act → 0.733)
  --shuffle-ext DIR    fwd extraction; builds within-bundle shuffle controls before convert
  --vec DIR            default artifacts/fwdb/data/vec
  --out DIR            reader directories (default .../fwd/readerdirs_L3)
  --run-dir DIR        pyes + analysis output
  --gpu N              default 0
  --adapter DIR        local adapter or empty to use artifacts/…/step_007082 or HF
  --overwrite
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mean) MEAN=$2; shift 2;;
    --template) TEMPLATE=$2; shift 2;;
    --fetch-twin-mean) FETCH_TWIN=1; shift;;
    --shuffle-ext) SHUFFLE_EXT=$2; shift 2;;
    --vec) VEC=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --run-dir) RUN_DIR=$2; shift 2;;
    --gpu) GPU=$2; shift 2;;
    --adapter) ADAPTER=$2; shift 2;;
    --run-json) RUN_JSON=$2; shift 2;;
    --overwrite) OVERWRITE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [ -z "$MEAN" ] && [ -z "$TEMPLATE" ] && [ "$FETCH_TWIN" = 0 ]; then
  echo "pass --mean, --template, or --fetch-twin-mean" >&2
  exit 2
fi

cd "$ROOT"
extra=()
[ "$OVERWRITE" = 1 ] && extra+=(--overwrite)
[ "$FETCH_TWIN" = 1 ] && extra+=(--fetch-twin-mean)
[ -n "$MEAN" ] && extra+=(--mean "$MEAN")
[ -n "$TEMPLATE" ] && extra+=(--template "$TEMPLATE")
[ -n "$ADAPTER" ] && extra+=(--adapter "$ADAPTER")
[ -n "$RUN_JSON" ] && extra+=(--run-json "$RUN_JSON")
[ -n "$SHUFFLE_EXT" ] && extra+=(--shuffle-ext "$SHUFFLE_EXT")

exec "$PY" main_run/experiment2/loss_weighted.py run \
  --vec "$VEC" --bundles "$BUNDLES" --beh-jsonl "$BEH" \
  --out "$OUT" --run-dir "$RUN_DIR" --gpu "$GPU" "${extra[@]}"
