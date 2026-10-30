#!/usr/bin/env bash
# Rare yes/no readings (reader and AO scorers, gradient and activation inputs, all held-out bundles):
# build (act inputs) -> read -> summary of main_run/beh_reader/rare_yesno.py (its docstring has the design, the checks
# and the estimates). read asks a scorer (the behaviour reader rl_beh_c040 or a public AO) "Did this training imprint the
# following behavior? ..." about every input and records the first-token P(Yes); summary writes summary.json (AUROCs)
# and prints the table.
#
# One run = (scorer S, input I, bundles B; default reader grad rare_quirk), written to ROOT/<S>_<I>[_all]/ (ROOT = --out,
# default rare_yesno.OUT; the first reader run stays in the files at OUT's root). --chain runs these 13
# runs in order on one GPU, skipping DONE stages:
#   rare_quirk  ao3layer grad, ao3layer act, ao3layer act_centred, aopaper grad, aopaper act, aopaper act_centred
#   all         reader grad, ao3layer grad, ao3layer act, ao3layer act_centred, aopaper grad, aopaper act,
#               aopaper act_centred
# Launch detached from the repo root on a free GPU <g> (the runner appends its own lines to the runner log):
#   setsid nohup bash scripts/beh_reader/rare_yesno.sh <g> --chain < /dev/null >> main_run/logs/<date>_rare_yesno_launch.out 2>&1 &
#   setsid nohup bash scripts/beh_reader/rare_yesno.sh <g> --scorer ao3layer --input act [--bundles all] < /dev/null >> ... &
# Smoke (output and logs under DIR, never the real folder; --act-dir may point at a finished build to skip building):
#   bash scripts/beh_reader/rare_yesno.sh <g> --scorer S --input I [--bundles all] --out DIR --limit 3 --check-equivalence
# Dry run (prints the runs, the commands and one look at the GPU; starts and writes nothing):
#   bash scripts/beh_reader/rare_yesno.sh <g> [--chain | --scorer S --input I --bundles B] [--out DIR] ... --dry
#
# Build stage: when a run with input act or act_centred still has to read, rare_yesno.py build-act --act-dir ACT_DIR
# runs first (CPU, before the GPU is taken; ACT_DIR = --act-dir, default ROOT/act; markers ACT_DIR/markers/build.*;
# a failed build stops the runner). build-act leaves a complete passed build alone.
# GPU rule (as scripts/beh_reader/rare_perdoc.sh): reading needs GPU <g> free: no compute process on it, under
# 4096 MiB used in two looks 20 s apart (pick_gpus) and not held by a block-mix job or another live v2-family script
# (scripts/beh_reader/gpus.sh). The runner checks this once, before the first read that is not DONE (a busy GPU refuses
# at once: exit 3), announces the GPU to the block-mix jobs as held (em_reader_v2_rare_yesno_read_p<pid>_...) and keeps
# it until all its runs end (the chain holds it across runs), releasing it on exit. Before each read it waits up to
# 120 s for the GPU's memory use to fall under 4096 MiB (the previous run's process is gone; nvidia-smi may lag).
# rare_yesno.py read checks the GPU's memory again (>= 4096 MiB: exit 3, passed on as the runner's exit 3) and sets
# CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=<g> before CUDA starts. summary and build use no GPU.
# Markers in RUN_DIR/markers: <stage>.START, .DONE (a rerun skips a DONE stage), .FAILED (exit code; a rerun renames an
# old one to .FAILED_<time>, nothing is deleted), <stage>.pid (the stage's python process), runner.pid, runner.lock
# (one runner per run folder); ROOT/chain.lock (one chain per ROOT).
# Logs (appended): main_run/logs/<LOG_PREFIX>_rare_yesno_<run>_<stage>.out, <LOG_PREFIX>_rare_yesno_build_act.out and
# the runner log <LOG_PREFIX>_rare_yesno_runner.out; with --out other than OUT they go to ROOT/logs instead.
# Expected minutes per run: read 6 (rare_quirk: 1,802 forwards) or 4 (all: 1,060), about 1 min of it the model load
# (the first reader run took 0.031 s per forward and 45 s to load); summary 1; build 2 (once, CPU; measured
# 11 s).
# Exit codes: 0 done, 1 a stage failed (the chain goes on with its next run and exits 1 at the end; a failed build stops
# at once), 2 usage, 3 GPU not free (here or in rare_yesno.py read) or a lock is held (the chain stops).
# Knobs (environment): LOG_PREFIX (default today's date), IIF_BASE, PY (the python; tests), V2_SMI (nvidia-smi; tests
# give a fake, as scripts/beh_reader/ao_heldout.sh).
set -u -o pipefail

IIF_BASE=${IIF_BASE:?set IIF_BASE}
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
SELF=$(readlink -f "$0")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd)
PY=${PY:-$IIF_BASE/venvs/interpretable_influence/bin/python}
LOG_PREFIX=${LOG_PREFIX:-$(date +%Y%m%d)}
STAGES=(read summary)
SCORERS="reader ao3layer aopaper"
# New AO-prep inputs (grad_centred, addact) and --sign/--coef/--alpha: use scripts/beh_reader/rare_yesno_ext.sh
INPUTS="grad act act_centred"
BUNDLE_SETS="rare_quirk all"
CHAIN_RUNS=(
  "ao3layer grad rare_quirk" "ao3layer act rare_quirk" "ao3layer act_centred rare_quirk"
  "aopaper grad rare_quirk" "aopaper act rare_quirk" "aopaper act_centred rare_quirk"
  "reader grad all" "ao3layer grad all" "ao3layer act all" "ao3layer act_centred all"
  "aopaper grad all" "aopaper act all" "aopaper act_centred all"
)

usage() {
  echo "usage: bash scripts/beh_reader/rare_yesno.sh <gpu index> [--chain | [--scorer reader|ao3layer|aopaper]" \
       "[--input grad|act|act_centred] [--bundles rare_quirk|all]] [--out DIR] [--act-dir DIR] [--limit N]" \
       "[--check-equivalence] [--dry]"
  echo "extended AO-prep sweep (sign/coef/grad_centred/addact): bash scripts/beh_reader/rare_yesno_ext.sh <gpu> ..."
  exit 2
}
in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }
[ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]] || usage
GPU=$1; shift
SCORER="" INPUT="" BUNDLES="" OUT_ARG="" ACT_ARG="" LIMIT="" EQUIV=0 DRY=0 CHAIN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --scorer) [ $# -ge 2 ] || usage; SCORER=$2; shift 2 ;;
    --input) [ $# -ge 2 ] || usage; INPUT=$2; shift 2 ;;
    --bundles) [ $# -ge 2 ] || usage; BUNDLES=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || usage; OUT_ARG=$2; shift 2 ;;
    --act-dir) [ $# -ge 2 ] || usage; ACT_ARG=$2; shift 2 ;;
    --limit) [ $# -ge 2 ] && [[ "$2" =~ ^[1-9][0-9]*$ ]] || usage; LIMIT=$2; shift 2 ;;
    --check-equivalence) EQUIV=1; shift ;;
    --chain) CHAIN=1; shift ;;
    --dry) DRY=1; shift ;;
    *) usage ;;
  esac
done
if [ "$CHAIN" = 1 ]; then
  [ -z "$SCORER$INPUT$BUNDLES" ] || { echo "--chain runs its own list; no --scorer, --input or --bundles"; usage; }
  RUN_LIST=("${CHAIN_RUNS[@]}")
else
  SCORER=${SCORER:-reader} INPUT=${INPUT:-grad} BUNDLES=${BUNDLES:-rare_quirk}
  in_list "$SCORER" "$SCORERS" && in_list "$INPUT" "$INPUTS" && in_list "$BUNDLES" "$BUNDLE_SETS" || usage
  if [ "$SCORER" = reader ] && [ "$INPUT" != grad ]; then echo "the reader reads gradient rows only (--input grad)"; usage; fi
  RUN_LIST=("$SCORER $INPUT $BUNDLES")
fi
[ "$DRY" = 1 ] && export PYTHONDONTWRITEBYTECODE=1

PATHS=$(cd "$REPO" && "$PY" - "$REPO" <<'PYEOF'
import shlex, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/main_run/beh_reader")
import common as C  # noqa: E402
import rare_yesno as RY  # noqa: E402

if C.REPO.resolve() != Path(sys.argv[1]).resolve():
    sys.exit(f"common.REPO {C.REPO} is not this script's repo {sys.argv[1]}")
for k, x in dict(OUT=RY.OUT, RUNS=C.RUNS, EMWT=C.EMWT, LOGD=C.LOG_DIR, ACT_NAME=RY.ACT_DIR_NAME).items():
    print(f"{k}={shlex.quote(str(x))}")
PYEOF
) || { echo "cannot read the paths of rare_yesno.py / common.py"; exit 2; }
eval "$PATHS"
DEFAULT_OUT=$OUT
ROOT=$(readlink -m "${OUT_ARG:-$OUT}")
ACT_DIR=$(readlink -m "${ACT_ARG:-$ROOT/$ACT_NAME}")
under_default() { local r d; r=$(readlink -m "$1"); d=$(readlink -m "$DEFAULT_OUT"); [ "$r" = "$d" ] || [[ "$r" == "$d"/* ]]; }
# a limited (smoke) read never goes to the real folder (rare_yesno.py refuses it too)
if [ -n "$LIMIT" ] && under_default "$ROOT"; then
  echo "--limit is a smoke option; give --out outside $DEFAULT_OUT"; exit 2
fi
[ "$ROOT" != "$(readlink -m "$DEFAULT_OUT")" ] && LOGD=$ROOT/logs
RUNLOG=$LOGD/${LOG_PREFIX}_rare_yesno_runner.out
BR=main_run/beh_reader
# shellcheck source=gpus.sh
source "$REPO/scripts/beh_reader/gpus.sh"  # blockmix_held, pick_gpus, hold_gpus, release_gpus, reap_stale_holds

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
say() { echo "[$(stamp)] $*"; }
run_name() { local n="$1_$2"; [ "$3" = all ] && n="${n}_all"; echo "$n"; }  # = rare_yesno.run_name

# Why GPU $GPU is not free (empty when free). $1 = 1: one look only (dry run), else pick_gpus' two looks.
# (copy of rare_perdoc.sh gpu_busy, nvidia-smi through $V2_SMI as ao_heldout.sh)
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

# Take GPU $GPU for reading: check and announce under the block-mix pick lock. Returns 3 when busy.
take_gpu() {
  local why locked=0
  if [ -d "${V2_LOCK%/*}" ]; then
    exec 9>>"$V2_LOCK"
    if flock -w 600 9; then locked=1; else say "$V2_LOCK not free in 600 s; checking without it"; fi
  fi
  why=$(gpu_busy)
  if [ -z "$why" ]; then hold_gpus rare_yesno_read "$GPU"; fi
  [ $locked = 1 ] && { flock -u 9; exec 9>&-; }
  if [ -n "$why" ]; then say "GPU $GPU is not free ($why); refusing"; return 3; fi
  say "GPU $GPU free; announced as held by rare_yesno_read"
}

# Before a read: wait up to 120 s for GPU $GPU's memory use to fall under 4096 MiB (rare_yesno.py read refuses it
# otherwise, exit 3).
wait_gpu_mem() {
  local i used=""
  for i in $(seq 1 25); do
    used=$($V2_SMI -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt 4096 ]; then return 0; fi
    sleep 5
  done
  say "GPU $GPU still shows ${used:-?} MiB used after 120 s"
}

# The command of a stage: $1 stage, $2 scorer, $3 input, $4 bundles, $5 run folder (from the repo root, the em-reader
# worktree on PYTHONPATH as rare_perdoc.sh does).
stage_cmd() {
  case "$1" in
    build) echo "$PY $BR/rare_yesno.py build-act --act-dir $ACT_DIR" ;;
    read) echo "$PY $BR/rare_yesno.py read --run-dir $5 --scorer $2 --input $3 --bundles $4 --act-dir $ACT_DIR" \
               "--gpu $GPU${LIMIT:+ --limit $LIMIT}$([ "$EQUIV" = 1 ] && echo ' --check-equivalence')" ;;
    summary) echo "$PY $BR/rare_yesno.py summary --run-dir $5" ;;
  esac
}

expect_min() {  # $1 stage, $2 bundles
  case "$1" in build) echo 2 ;; read) [ "$2" = all ] && echo 4 || echo 6 ;; *) echo 1 ;; esac
}

# run_stage <stage> <markers dir> <log> <command> <expected minutes>
run_stage() {
  local st=$1 m=$2 log=$3 cmd=$4 exp=$5 rc pid t0 took
  if [ "$DRY" = 1 ]; then
    [ -e "$m/$st.DONE" ] && echo "  [dry] $st is DONE ($m/$st.DONE): would be skipped"
    echo "  [dry] $st (about $exp min): cd $REPO && PYTHONPATH=$EMWT $cmd >> $log 2>&1"
    return 0
  fi
  if [ -e "$m/$st.DONE" ]; then say "$st already DONE ($m); skipped"; return 0; fi
  mkdir -p "$m"
  # an earlier failure's marker is kept under another name (nothing is deleted)
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
    [ $took -gt $((2 * exp)) ] && say "WARNING: $st took more than twice the expected time"
  else
    echo "$(stamp) exit $rc after $took min; log $log" > "$m/$st.FAILED"
    say "$st FAILED (exit $rc); log $log"
  fi
  return $rc
}

# One run: $1 scorer, $2 input, $3 bundles. Returns 0, 1 (a stage failed) or 3 (GPU refused, lock held).
run_one() {
  local s=$1 i=$2 b=$3 name rd m st rc
  name=$(run_name "$s" "$i" "$b"); rd=$ROOT/$name; m=$rd/markers
  if [ "$DRY" = 1 ]; then
    say "run $name: scorer $s, input $i, bundles $b -> $rd"
    for f in "$rd"/pyes_*.jsonl; do [ -e "$f" ] && say "  exists: $f (read would refuse to overwrite it)"; done
    for st in "${STAGES[@]}"; do
      run_stage "$st" "$m" "$LOGD/${LOG_PREFIX}_rare_yesno_${name}_$st.out" "$(stage_cmd "$st" "$s" "$i" "$b" "$rd")" \
        "$(expect_min "$st" "$b")"
    done
    return 0
  fi
  mkdir -p "$m"
  exec 8>>"$m/runner.lock"
  if ! flock -n 8; then say "another rare_yesno runner holds $m/runner.lock; $name not run"; exec 8>&-; return 3; fi
  echo $$ > "$m/runner.pid"
  say "run $name: scorer $s, input $i, bundles $b -> $rd"
  for st in "${STAGES[@]}"; do
    [ "$st" = read ] && [ ! -e "$m/read.DONE" ] && wait_gpu_mem
    run_stage "$st" "$m" "$LOGD/${LOG_PREFIX}_rare_yesno_${name}_$st.out" "$(stage_cmd "$st" "$s" "$i" "$b" "$rd")" \
      "$(expect_min "$st" "$b")"
    rc=$?
    if [ $rc -ne 0 ]; then
      say "run $name stopped after $st (exit $rc)"
      flock -u 8; exec 8>&-
      [ "$st" = read ] && [ $rc -eq 3 ] && return 3  # rare_yesno.py read refused a busy GPU
      return 1
    fi
  done
  flock -u 8; exec 8>&-
  say "run $name done; summary $rd/summary.json"
  return 0
}

# What the runs still need: the act build (a run with act input whose read is not DONE) and the GPU (a read not DONE).
NEED_BUILD=0 NEED_GPU=0 NAMES=""
for r in "${RUN_LIST[@]}"; do
  read -r s i b <<< "$r"
  n=$(run_name "$s" "$i" "$b"); NAMES="$NAMES $n"
  if [ ! -e "$ROOT/$n/markers/read.DONE" ]; then
    NEED_GPU=1
    [ "$i" != grad ] && NEED_BUILD=1
  fi
done

if [ "$DRY" = 1 ]; then
  say "dry run: ${#RUN_LIST[@]} run(s):$NAMES; ROOT $ROOT; ACT_DIR $ACT_DIR; logs $LOGD/${LOG_PREFIX}_rare_yesno_<run>_<stage>.out, runner log $RUNLOG"
  if [ "$NEED_BUILD" = 1 ]; then
    say "build of the activation inputs:"
    run_stage build "$ACT_DIR/markers" "$LOGD/${LOG_PREFIX}_rare_yesno_build_act.out" "$(stage_cmd build)" "$(expect_min build)"
  fi
  if [ "$NEED_GPU" = 1 ]; then
    why=$(gpu_busy 1)
    say "GPU $GPU now (one look): ${why:-free}; a real run would $([ -n "$why" ] && echo 'refuse (exit 3)' || echo 'take it (two looks, announce to block-mix jobs)')"
  fi
  for r in "${RUN_LIST[@]}"; do read -r s i b <<< "$r"; run_one "$s" "$i" "$b"; done
  exit 0
fi

mkdir -p "$ROOT" "$LOGD"
echo "rare_yesno runner: log $RUNLOG"
exec >> "$RUNLOG" 2>&1
if [ "$CHAIN" = 1 ]; then
  exec 7>>"$ROOT/chain.lock"
  flock -n 7 || { say "another rare_yesno chain holds $ROOT/chain.lock; not started"; exit 3; }
fi
trap 'release_gpus "rare_yesno_*" "runner exit"' EXIT
say "rare_yesno runner pid $$ on $(hostname): ${#RUN_LIST[@]} run(s):$NAMES; GPU $GPU, ROOT $ROOT, ACT_DIR $ACT_DIR, limit ${LIMIT:-none}, equivalence $EQUIV"
if [ "$NEED_BUILD" = 1 ]; then
  run_stage build "$ACT_DIR/markers" "$LOGD/${LOG_PREFIX}_rare_yesno_build_act.out" "$(stage_cmd build)" "$(expect_min build)" \
    || { say "build of the activation inputs failed; stopping"; exit 1; }
fi
if [ "$NEED_GPU" = 1 ]; then reap_stale_holds; take_gpu || exit 3; fi
FAILED_RUNS=""
for r in "${RUN_LIST[@]}"; do
  read -r s i b <<< "$r"
  run_one "$s" "$i" "$b"; rc=$?
  if [ $rc -eq 3 ]; then say "stopping: GPU $GPU not free or a lock is held"; exit 3; fi
  [ $rc -ne 0 ] && FAILED_RUNS="$FAILED_RUNS $(run_name "$s" "$i" "$b")"
done
release_gpus "rare_yesno_*" "all runs ended"
if [ -n "$FAILED_RUNS" ]; then say "runs failed:$FAILED_RUNS"; exit 1; fi
say "all runs done:$NAMES"
exit 0
