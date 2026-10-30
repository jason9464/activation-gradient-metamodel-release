#!/usr/bin/env bash
# Extended AO-prep yes/no sweep (Neel item 1 never-run arms): uncentred ×0.5/1/4/8 both signs, centred both signs,
# add-an-activation α 0.5/1 both signs. Scores with rare_yesno.py (P(Yes), no judge). Builds on
# scripts/beh_reader/rare_yesno.sh (same GPU / marker / lock rules) but drives the setting list from
# rare_yesno.ext_sweep_settings.
#
#   bash scripts/beh_reader/rare_yesno_ext.sh <gpu> [--bundles all|rare_quirk] [--out DIR] [--act-dir DIR] [--dry]
#   bash scripts/beh_reader/rare_yesno_ext.sh <gpu> --list   # print the settings and exit
#
# Each setting is about 1,060 prefill-only forwards on --bundles all. A couple of dozen settings fit in under a GPU
# hour. Existing DONE markers are skipped. Do not push from this script.
set -u -o pipefail

IIF_BASE=${IIF_BASE:?set IIF_BASE}
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
SELF=$(readlink -f "$0")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd)
PY=${PY:-$IIF_BASE/venvs/interpretable_influence/bin/python}
LOG_PREFIX=${LOG_PREFIX:-$(date +%Y%m%d)}

usage() {
  echo "usage: bash scripts/beh_reader/rare_yesno_ext.sh <gpu index> [--bundles all|rare_quirk] [--out DIR]" \
       "[--act-dir DIR] [--dry | --list]"
  exit 2
}
[ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]] || usage
GPU=$1; shift
BUNDLES=all OUT_ARG="" ACT_ARG="" DRY=0 LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bundles) [ $# -ge 2 ] || usage; BUNDLES=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || usage; OUT_ARG=$2; shift 2 ;;
    --act-dir) [ $# -ge 2 ] || usage; ACT_ARG=$2; shift 2 ;;
    --dry) DRY=1; shift ;;
    --list) LIST=1; shift ;;
    *) usage ;;
  esac
done
case "$BUNDLES" in all|rare_quirk) ;; *) usage ;; esac
[ "$DRY" = 1 ] && export PYTHONDONTWRITEBYTECODE=1

PATHS=$(cd "$REPO" && "$PY" - "$REPO" "$BUNDLES" <<'PYEOF'
import json, shlex, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/main_run/beh_reader")
import common as C  # noqa: E402
import rare_yesno as RY  # noqa: E402

if C.REPO.resolve() != Path(sys.argv[1]).resolve():
    sys.exit(f"common.REPO {C.REPO} is not this script's repo {sys.argv[1]}")
settings = RY.ext_sweep_settings(bundles_set=sys.argv[2])
for k, x in dict(OUT=RY.OUT, RUNS=C.RUNS, EMWT=C.EMWT, LOGD=C.LOG_DIR, ACT_NAME=RY.ACT_DIR_NAME).items():
    print(f"{k}={shlex.quote(str(x))}")
print("SETTINGS_JSON=" + shlex.quote(json.dumps(settings)))
print(f"N_SETTINGS={len(settings)}")
PYEOF
) || { echo "cannot read rare_yesno.ext_sweep_settings"; exit 2; }
eval "$PATHS"
DEFAULT_OUT=$OUT
ROOT=$(readlink -m "${OUT_ARG:-$OUT}")
ACT_DIR=$(readlink -m "${ACT_ARG:-$ROOT/$ACT_NAME}")
[ "$ROOT" != "$(readlink -m "$DEFAULT_OUT")" ] && LOGD=$ROOT/logs
RUNLOG=$LOGD/${LOG_PREFIX}_rare_yesno_ext_runner.out
BR=main_run/beh_reader
# shellcheck source=gpus.sh
source "$REPO/scripts/beh_reader/gpus.sh"

# SETTINGS_JSON must be an env var (S=... after the python -c is an argv, not an env).
export SETTINGS_JSON
if [ "$LIST" = 1 ]; then
  S="$SETTINGS_JSON" "$PY" -c 'import json,os; [print("{name}  scorer={scorer} input={input} sign={sign} coef={coef} alpha={alpha} bundles={bundles}".format(**s)) for s in json.loads(os.environ["S"])]'
  echo "$N_SETTINGS settings -> ROOT ${OUT_ARG:-$DEFAULT_OUT}"
  exit 0
fi

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
say() { echo "[$(stamp)] $*"; }

gpu_busy() {
  local apps used why=""
  apps=$($V2_SMI -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1) \
    || { echo "$V2_SMI -i $GPU failed: $apps"; return 0; }
  apps=$(echo "$apps" | grep -c '[0-9]')
  [ "$apps" -gt 0 ] && why="$apps compute process(es)"
  used=$($V2_SMI -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if ! [[ "$used" =~ ^[0-9]+$ ]]; then why="${why:+$why; }memory use unreadable ($used)"
  elif [ "$used" -ge 4096 ]; then why="${why:+$why; }$used MiB used"; fi
  case " $(blockmix_held) " in *" $GPU "*) why="${why:+$why; }held by a block-mix marker" ;; esac
  if [ -z "$why" ] && [ "${1:-0}" != 1 ]; then
    case " $(pick_gpus "" 64) " in *" $GPU "*) ;; *) why="not free in two looks ${V2_LOOK_S} s apart (pick_gpus)" ;; esac
  fi
  echo "$why"
}

take_gpu() {
  local why locked=0
  if [ -d "${V2_LOCK%/*}" ]; then
    exec 9>>"$V2_LOCK"
    if flock -w 600 9; then locked=1; else say "$V2_LOCK not free in 600 s; checking without it"; fi
  fi
  why=$(gpu_busy)
  if [ -z "$why" ]; then hold_gpus rare_yesno_ext_read "$GPU"; fi
  [ $locked = 1 ] && { flock -u 9; exec 9>&-; }
  if [ -n "$why" ]; then say "GPU $GPU is not free ($why); refusing"; return 3; fi
  say "GPU $GPU free; announced as held by rare_yesno_ext_read"
}

wait_gpu_mem() {
  local i used=""
  for i in $(seq 1 25); do
    used=$($V2_SMI -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt 4096 ]; then return 0; fi
    sleep 5
  done
  say "GPU $GPU still shows ${used:-?} MiB used after 120 s"
}

run_stage() {
  local st=$1 m=$2 log=$3 cmd=$4 exp=$5 rc pid t0 took
  if [ "$DRY" = 1 ]; then
    [ -e "$m/$st.DONE" ] && echo "  [dry] $st is DONE ($m/$st.DONE): would be skipped"
    echo "  [dry] $st (about $exp min): cd $REPO && PYTHONPATH=$EMWT $cmd >> $log 2>&1"
    return 0
  fi
  if [ -e "$m/$st.DONE" ]; then say "$st already DONE ($m); skipped"; return 0; fi
  mkdir -p "$m"
  [ -e "$m/$st.FAILED" ] && mv "$m/$st.FAILED" "$m/$st.FAILED_$(date +%Y%m%d%H%M%S)"
  echo "$(stamp) runner pid $$ host $(hostname): $cmd" > "$m/$st.START"
  echo "===== [$(stamp)] $st: $cmd" >> "$log"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  ( cd "$REPO" && exec env PYTHONPATH="$EMWT" $cmd ) >> "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$m/$st.pid"
  say "$st: pid $pid, log $log (expected about $exp min)"
  wait "$pid"; rc=$?
  took=$(( ($(date +%s) - t0 + 59) / 60 ))
  if [ $rc -eq 0 ]; then
    echo "$(stamp) took $took min (expected $exp)" > "$m/$st.DONE"
    say "$st DONE in $took min (expected $exp)"
  else
    echo "$(stamp) exit $rc after $took min; log $log" > "$m/$st.FAILED"
    say "$st FAILED (exit $rc); log $log"
  fi
  return $rc
}

# Run one setting dict given as JSON on stdin line via env SETTING
run_one_json() {
  local name scorer input bundles sign coef alpha rd m cmd rc
  eval "$("$PY" - <<'PY'
import json, os, shlex
s = json.loads(os.environ["SETTING"])
for k in ("name", "scorer", "input", "bundles", "sign", "coef", "alpha"):
    v = s[k]
    print(f"{k}={shlex.quote('' if v is None else str(v))}")
PY
)"
  rd=$ROOT/$name; m=$rd/markers
  cmd="$PY $BR/rare_yesno.py read --run-dir $rd --scorer $scorer --input $input --bundles $bundles"
  cmd="$cmd --sign $sign --coef $coef --act-dir $ACT_DIR --gpu $GPU"
  [ -n "$alpha" ] && cmd="$cmd --alpha $alpha"
  if [ "$DRY" = 1 ]; then
    say "run $name -> $rd"
    run_stage read "$m" "$LOGD/${LOG_PREFIX}_rare_yesno_ext_${name}_read.out" "$cmd" 4
    run_stage summary "$m" "$LOGD/${LOG_PREFIX}_rare_yesno_ext_${name}_summary.out" \
      "$PY $BR/rare_yesno.py summary --run-dir $rd" 1
    return 0
  fi
  mkdir -p "$m"
  exec 8>>"$m/runner.lock"
  if ! flock -n 8; then say "another runner holds $m/runner.lock; $name not run"; exec 8>&-; return 3; fi
  echo $$ > "$m/runner.pid"
  say "run $name: $scorer $input sign=$sign coef=$coef alpha=${alpha:--} bundles=$bundles -> $rd"
  [ ! -e "$m/read.DONE" ] && wait_gpu_mem
  run_stage read "$m" "$LOGD/${LOG_PREFIX}_rare_yesno_ext_${name}_read.out" "$cmd" 4; rc=$?
  if [ $rc -ne 0 ]; then flock -u 8; exec 8>&-; [ $rc -eq 3 ] && return 3; return 1; fi
  run_stage summary "$m" "$LOGD/${LOG_PREFIX}_rare_yesno_ext_${name}_summary.out" \
    "$PY $BR/rare_yesno.py summary --run-dir $rd" 1; rc=$?
  flock -u 8; exec 8>&-
  return $rc
}

# Emit one JSON setting per line (S must be in the environment, not after -c).
settings_lines() {
  S="$SETTINGS_JSON" "$PY" -c 'import json,os; [print(json.dumps(s)) for s in json.loads(os.environ["S"])]'
}

NEED_BUILD=0 NEED_GPU=0
while IFS= read -r line; do
  export SETTING="$line"
  name=$("$PY" -c 'import json,os; print(json.loads(os.environ["SETTING"])["name"])')
  if [ ! -e "$ROOT/$name/markers/read.DONE" ]; then
    NEED_GPU=1
    NEED_BUILD=1  # build-act appends train_mean_grad if missing; own marker so rare_yesno.sh's build.DONE is ignored
  fi
done < <(settings_lines)

# Stage name build_ext: rare_yesno.sh may already have written markers/build.DONE for the same ACT_DIR.
BUILD_MARKERS=$ACT_DIR/markers
BUILD_STAGE=build_ext

if [ "$DRY" = 1 ]; then
  say "dry run: $N_SETTINGS settings; ROOT $ROOT; ACT_DIR $ACT_DIR"
  n_lines=$(settings_lines | wc -l | tr -d ' ')
  [ "$n_lines" = "$N_SETTINGS" ] || { say "BUG: settings_lines gave $n_lines lines, expected $N_SETTINGS"; exit 1; }
  [ "$NEED_BUILD" = 1 ] && run_stage "$BUILD_STAGE" "$BUILD_MARKERS" "$LOGD/${LOG_PREFIX}_rare_yesno_ext_build_act.out" \
    "$PY $BR/rare_yesno.py build-act --act-dir $ACT_DIR" 2
  if [ "$NEED_GPU" = 1 ]; then
    why=$(gpu_busy 1)
    say "GPU $GPU now (one look): ${why:-free}"
  fi
  while IFS= read -r line; do export SETTING="$line"; run_one_json; done < <(settings_lines)
  exit 0
fi

mkdir -p "$ROOT" "$LOGD"
echo "rare_yesno_ext runner: log $RUNLOG"
exec >> "$RUNLOG" 2>&1
exec 7>>"$ROOT/ext_chain.lock"
flock -n 7 || { say "another rare_yesno_ext chain holds $ROOT/ext_chain.lock; not started"; exit 3; }
trap 'release_gpus "rare_yesno_ext_*" "runner exit"' EXIT
say "rare_yesno_ext pid $$ on $(hostname): $N_SETTINGS settings; GPU $GPU, ROOT $ROOT, ACT_DIR $ACT_DIR, bundles $BUNDLES"
if [ "$NEED_BUILD" = 1 ]; then
  run_stage "$BUILD_STAGE" "$BUILD_MARKERS" "$LOGD/${LOG_PREFIX}_rare_yesno_ext_build_act.out" \
    "$PY $BR/rare_yesno.py build-act --act-dir $ACT_DIR" 2 \
    || { say "build-act failed; stopping"; exit 1; }
fi
if [ "$NEED_GPU" = 1 ]; then reap_stale_holds; take_gpu || exit 3; fi
FAILED=""
while IFS= read -r line; do
  export SETTING="$line"
  run_one_json; rc=$?
  if [ $rc -eq 3 ]; then say "stopping: GPU or lock"; exit 3; fi
  [ $rc -ne 0 ] && FAILED="$FAILED $($PY -c 'import json,os; print(json.loads(os.environ["SETTING"])["name"])')"
done < <(settings_lines)
release_gpus "rare_yesno_ext_*" "all runs ended"
if [ -n "$FAILED" ]; then say "runs failed:$FAILED"; exit 1; fi
say "all $N_SETTINGS extended settings done"
exit 0
