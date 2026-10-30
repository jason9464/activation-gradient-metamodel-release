#!/usr/bin/env bash
# Reader against the public AO on the 530 held-out behaviour bundles: build -> six AO readings
# (two AOs x gradient, activation and centred activation rows) -> judge -> summary of main_run/beh_reader/ao_heldout.py
# (its docstring has the design, the checks and the estimates). The reader's own readings and judged file are the
# evaluation's (not read again). Judge cap: 5 USD added to the ledger total.
#
# Launch detached from the repo root on a free GPU <g> (all stages on one GPU):
#   setsid nohup bash scripts/beh_reader/ao_heldout.sh <g> < /dev/null \
#       > main_run/logs/20261029_ao_heldout_runner.out 2>&1 &
# Split over GPUs: build first, then one runner per GPU with its read stages, then judge and summary once all six
# read stages are DONE (the judge refuses, sending nothing, while a reading is unfinished, made from another inputs
# file, or differs from the others in max_new_tokens, answers per bundle or steps):
#   bash scripts/beh_reader/ao_heldout.sh <g> --stage build
#   setsid nohup bash scripts/beh_reader/ao_heldout.sh 1 --stage read_ao3layer_grad,read_ao3layer_act < /dev/null \
#       > main_run/logs/20261029_ao_heldout_runner_g1.out 2>&1 &      (likewise GPU 2 and 3 with two other stages)
#   bash scripts/beh_reader/ao_heldout.sh <g> --stage judge,summary
# Dry run (prints every command, the GPU's state and ao_heldout.py plan; starts and writes nothing):
#   bash scripts/beh_reader/ao_heldout.sh <g> --dry
# Smoke (a scratch folder; reads steps 0..N-1 only; build runs there too; logs in <scratch>/ao_heldout_smoke/logs; the
# tokens report after the read stages gives the share of answers at the token limit):
#   bash scripts/beh_reader/ao_heldout.sh <g> --out <scratch>/ao_heldout_smoke --limit 3 [--judge-dry]
# Options: --out DIR (default ao_heldout.OUT = $IIF_BASE/agm_data/runs/beh_reader/ao_heldout), --limit N (read
# stages; needs --out), --stage S[,S...] (these stages only, run in the order below), --max-new-tokens N (read
# stages; default 40, the AO toolkit's; the rule: when more than 20 % of the smoke's answers hit 40, the real run uses
# 192 and records the deviation), --dry, --judge-dry (judge builds the requests and sends none; the judge stage then
# gets no DONE marker, so a later run without --judge-dry judges for real, and summary is dropped from a run that
# also judges, because there are no judged files).
#
# Stages: build read_ao3layer_grad read_ao3layer_act read_ao3layer_act_centred read_aopaper_grad read_aopaper_act
# read_aopaper_act_centred judge summary.
# GPU rule (rare_perdoc.sh's): a read stage needs GPU <g> free: no compute process on it and under 4096 MiB used in two
# looks 20 s apart, and not held by a block-mix job or another live v2-family script (scripts/beh_reader/gpus.sh,
# pick_gpus under the block-mix lock). The runner checks this before the first stage when a read stage is to run (a
# busy GPU refuses at once: exit 3), announces the GPU to the block-mix jobs as held (em_reader_v2_ao_heldout_read_
# p<pid>_...), and releases it after the last read stage and on exit (its own announcements only, so runners on other
# GPUs keep theirs). build, judge and summary use no GPU.
# Markers in $OUT/markers: <stage>.START, .DONE (a rerun skips a DONE stage), .FAILED (exit code), <stage>.pid,
# <stage>.runlock (held by the runner that runs the stage: several runners may run at once on disjoint stage lists; a
# runner whose stage list overlaps a running one's exits 3 before starting anything), runners.log (one line per
# runner start), runner_g<gpu>.pid. The python read also writes read_<ao>_<input>.{START,DONE,FAILED} there (the same
# names as its stage; the runner rewrites them after the stage with the same outcome) and holds
# read_<ao>_<input>.lock while it reads (a second read of the same variant, e.g. by hand, exits 1). Logs:
# main_run/logs/20261029_ao_heldout_<stage>.out (appended) for the default OUT, $OUT/logs/ for another --out (a smoke
# never appends to the real run's logs); the read workers' own logs in $OUT/read_logs; the tokens report after the
# last read stage of a runner in <logs>/20261029_ao_heldout_tokens.out.
# Expected minutes: build 5, each read 40 at 40 tokens (530 generate calls of 10 answers plus the model load;
# scaled with --max-new-tokens: 192 tokens -> 192, an upper bound since answers stop at the end token), judge 40
# (31,800 requests in a Message Batch), summary 1.
# Exit codes: 0 done, 1 a stage failed, 2 usage, 3 GPU not free or another runner holds a stage's lock.
# Knobs (environment): DRY_RUN=1 (= --dry), CAP_USD (5), SAMPLES (10), MAX_NEW_TOKENS (40), LOG_PREFIX (20261029),
# IIF_BASE, V2_SMI (nvidia-smi; tests give a fake), AO_HELDOUT_LOGD (log folder; tests).
set -u -o pipefail

IIF_BASE=${IIF_BASE:?set IIF_BASE}
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
SELF=$(readlink -f "$0")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd)
PY=${PY:-$IIF_BASE/venvs/interpretable_influence/bin/python}
DRY=${DRY_RUN:-0}
CAP_USD=${CAP_USD:-5}
SAMPLES=${SAMPLES:-10}
MNT=${MAX_NEW_TOKENS:-40}
LOG_PREFIX=${LOG_PREFIX:-20261029}
STAGES=(build read_ao3layer_grad read_ao3layer_act read_ao3layer_act_centred read_aopaper_grad read_aopaper_act
        read_aopaper_act_centred judge summary)

usage() {
  echo "usage: bash scripts/beh_reader/ao_heldout.sh <gpu index> [--out DIR] [--limit N] [--stage S[,S...]]"
  echo "         [--max-new-tokens N] [--dry] [--judge-dry]"
  echo "  stages: ${STAGES[*]}"
  exit 2
}
[ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]] || usage
GPU=$1; shift
OUT_ARG="" LIMIT="" WANT=all JUDGE_DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) [ $# -ge 2 ] || usage; OUT_ARG=$2; shift 2 ;;
    --limit) [ $# -ge 2 ] && [[ "$2" =~ ^[1-9][0-9]*$ ]] || usage; LIMIT=$2; shift 2 ;;
    --stage) [ $# -ge 2 ] || usage; WANT=$2; shift 2 ;;
    --max-new-tokens) [ $# -ge 2 ] && [[ "$2" =~ ^[1-9][0-9]*$ ]] || usage; MNT=$2; shift 2 ;;
    --dry) DRY=1; shift ;;
    --judge-dry) JUDGE_DRY=1; shift ;;
    *) usage ;;
  esac
done
[[ "$MNT" =~ ^[1-9][0-9]*$ ]] || usage
if [ "$WANT" = all ]; then
  RUN_STAGES=("${STAGES[@]}")
else  # a comma list of stages; run in the order of STAGES
  IFS=, read -ra WANTS <<< "$WANT"
  [ ${#WANTS[@]} -gt 0 ] || usage
  for w in "${WANTS[@]}"; do case " ${STAGES[*]} " in *" $w "*) ;; *) usage ;; esac; done
  RUN_STAGES=()
  for s in "${STAGES[@]}"; do case ",$WANT," in *",$s,"*) RUN_STAGES+=("$s") ;; esac; done
fi
declare -A EXPECT_MIN=([build]=5 [judge]=40 [summary]=1)
for s in "${STAGES[@]}"; do case "$s" in read_*) EXPECT_MIN[$s]=$MNT ;; esac; done  # 40 min at 40 tokens, linear
if [ -n "$LIMIT" ] && [ -z "$OUT_ARG" ]; then echo "--limit writes partial readings: give --out (a scratch folder)"; usage; fi
[ "$DRY" = 1 ] && export PYTHONDONTWRITEBYTECODE=1

PATHS=$(cd "$REPO" && "$PY" - "$REPO" <<'PYEOF'
import shlex, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/main_run/beh_reader")
import common as C  # noqa: E402
import ao_heldout as AH  # noqa: E402

if C.REPO.resolve() != Path(sys.argv[1]).resolve():
    sys.exit(f"common.REPO {C.REPO} is not this script's repo {sys.argv[1]}")
for k, x in dict(OUT=AH.OUT, RUNS=C.RUNS, EMWT=C.EMWT, LOGD=C.LOG_DIR).items():
    print(f"{k}={shlex.quote(str(x))}")
PYEOF
) || { echo "cannot read the paths of ao_heldout.py / common.py"; exit 2; }
eval "$PATHS"
DEF_OUT=$OUT
OUT=${OUT_ARG:-$OUT}
case "$OUT" in /*) ;; *) OUT=$(pwd)/$OUT ;; esac
# another --out than the default (a smoke): its logs stay in it
[ "$(readlink -m "$OUT")" != "$(readlink -m "$DEF_OUT")" ] && LOGD=$OUT/logs
LOGD=${AO_HELDOUT_LOGD:-$LOGD}
if [ "$JUDGE_DRY" = 1 ] && [[ " ${RUN_STAGES[*]} " == *" judge "* ]]; then  # no judged files: summary would fail
  KEEP=(); for s in "${RUN_STAGES[@]}"; do [ "$s" = summary ] || KEEP+=("$s"); done; RUN_STAGES=("${KEEP[@]}")
fi
M=$OUT/markers
BR=main_run/beh_reader
# shellcheck source=gpus.sh
source "$REPO/scripts/beh_reader/gpus.sh"  # blockmix_held, pick_gpus, hold_gpus, release_gpus, reap_stale_holds

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
say() { echo "[$(stamp)] $*"; }

# Why GPU $GPU is not free (empty when free). $1 = 1: one look only (dry run), else pick_gpus' two looks.
gpu_busy() {
  local apps used why=""
  # shellcheck disable=SC2086
  apps=$($V2_SMI -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1) \
    || { echo "nvidia-smi -i $GPU failed: $apps"; return 0; }
  apps=$(echo "$apps" | grep -c '[0-9]')
  [ "$apps" -gt 0 ] && why="$apps compute process(es)"
  # shellcheck disable=SC2086
  used=$($V2_SMI -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if ! [[ "$used" =~ ^[0-9]+$ ]]; then why="${why:+$why; }memory use unreadable ($used)"
  elif [ "$used" -ge 4096 ]; then why="${why:+$why; }$used MiB used"; fi
  case " $(blockmix_held) " in *" $GPU "*) why="${why:+$why; }held by a block-mix marker" ;; esac
  if [ -z "$why" ] && [ "${1:-0}" != 1 ]; then
    case " $(pick_gpus "" 64) " in *" $GPU "*) ;; *) why="not free in two looks ${V2_LOOK_S} s apart (pick_gpus)" ;; esac
  fi
  echo "$why"
}

# Take GPU $GPU for the read stages: check and announce under the block-mix pick lock. Returns 3 when busy.
take_gpu() {
  local why locked=0
  if [ -d "${V2_LOCK%/*}" ]; then
    exec 9>>"$V2_LOCK"
    if flock -w 600 9; then locked=1; else say "$V2_LOCK not free in 600 s; checking without it"; fi
  fi
  why=$(gpu_busy)
  if [ -z "$why" ]; then hold_gpus ao_heldout_read "$GPU"; fi
  [ $locked = 1 ] && { flock -u 9; exec 9>&-; }
  if [ -n "$why" ]; then say "GPU $GPU is not free ($why); refusing"; return 3; fi
  say "GPU $GPU free; announced as held by ao_heldout_read"
}

# The command of each stage (from the repo root, the em-reader worktree on PYTHONPATH).
stage_cmd() {
  local ao inp
  case "$1" in
    build) echo "$PY $BR/ao_heldout.py build --out $OUT" ;;
    read_*)  # read_<ao>_<input>; the AO names hold no "_", the input may (act_centred)
      ao=${1#read_}; ao=${ao%%_*}; inp=${1#read_"$ao"_}
      echo "$PY $BR/ao_heldout.py read --out $OUT --ao $ao --input $inp --gpu $GPU --samples $SAMPLES --max-new-tokens $MNT${LIMIT:+ --limit $LIMIT}" ;;
    judge) echo "$PY $BR/ao_heldout.py judge --out $OUT --cap-usd $CAP_USD --mode batch$([ "$JUDGE_DRY" = 1 ] && echo ' --dry-run')" ;;
    summary) echo "$PY $BR/ao_heldout.py summary --out $OUT" ;;
  esac
}

run_stage() {
  local st=$1 log rc pid t0 cmd
  log=$LOGD/${LOG_PREFIX}_ao_heldout_$st.out
  cmd=$(stage_cmd "$st")
  if [ "$DRY" = 1 ]; then
    [ -e "$M/$st.DONE" ] && echo "  [dry] $st is DONE ($M/$st.DONE): would be skipped"
    echo "  [dry] $st (about ${EXPECT_MIN[$st]} min): cd $REPO && PYTHONPATH=$EMWT $cmd >> $log 2>&1"
    return 0
  fi
  if [ -e "$M/$st.DONE" ]; then say "$st already DONE; skipped"; return 0; fi
  rm -f "$M/$st.FAILED"
  echo "$(stamp) runner pid $$ host $(hostname): $cmd" > "$M/$st.START"
  echo "===== [$(stamp)] $st: $cmd" >> "$log"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  ( cd "$REPO" && exec env PYTHONPATH="$EMWT" $cmd ) >> "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$M/$st.pid"
  say "$st: pid $pid, log $log (expected about ${EXPECT_MIN[$st]} min)"
  wait "$pid"; rc=$?
  local took=$(( ($(date +%s) - t0 + 59) / 60 ))
  if [ $rc -eq 0 ] && [ "$st" = judge ] && [ "$JUDGE_DRY" = 1 ]; then
    say "judge dry run done in $took min (nothing sent); no DONE marker, so a later run judges for real"
  elif [ $rc -eq 0 ]; then
    echo "$(stamp) took $took min (expected ${EXPECT_MIN[$st]})" > "$M/$st.DONE"
    say "$st DONE in $took min (expected ${EXPECT_MIN[$st]})"
    [ $took -gt $((2 * EXPECT_MIN[$st])) ] && say "WARNING: $st took more than twice the expected time"
  else
    echo "$(stamp) exit $rc after $took min; log $log" > "$M/$st.FAILED"
    say "$st FAILED (exit $rc); log $log"
  fi
  return $rc
}

# The share of answers at the token limit and the generation-length rule (after the last read stage of this runner).
tokens_report() {
  local log=$LOGD/${LOG_PREFIX}_ao_heldout_tokens.out
  echo "===== [$(stamp)] tokens (runner pid $$, GPU $GPU, max_new_tokens $MNT)" >> "$log"
  say "answers at the token limit (ao_heldout.py tokens; also in $log):"
  ( cd "$REPO" && PYTHONPATH="$EMWT" "$PY" "$BR/ao_heldout.py" tokens --out "$OUT" ) 2>&1 | tee -a "$log"
}

LAST_READ=""
for s in "${RUN_STAGES[@]}"; do case "$s" in read_*) LAST_READ=$s ;; esac; done

needs_gpu() {  # 0 when a read stage is among the stages to run and not DONE
  local s
  for s in "${RUN_STAGES[@]}"; do
    case "$s" in read_*) [ ! -e "$M/$s.DONE" ] && return 0 ;; esac
  done
  return 1
}

if [ "$DRY" = 1 ]; then
  say "dry run: stages ${RUN_STAGES[*]}; OUT $OUT; logs $LOGD/${LOG_PREFIX}_ao_heldout_<stage>.out${LIMIT:+; limit $LIMIT steps}"
  if needs_gpu; then
    why=$(gpu_busy 1)
    say "GPU $GPU now (one look): ${why:-free}; a real run would $([ -n "$why" ] && echo 'refuse (exit 3)' || echo 'take it (two looks, announce to block-mix jobs)')"
  fi
  for st in "${RUN_STAGES[@]}"; do run_stage "$st"; done
  echo "--- ao_heldout.py plan (GPU $GPU):"
  ( cd "$REPO" && PYTHONPATH="$EMWT" "$PY" "$BR/ao_heldout.py" plan --out "$OUT" --gpu "$GPU" --samples "$SAMPLES" \
      --max-new-tokens "$MNT" --cap-usd "$CAP_USD" ) || { echo "plan failed"; exit 1; }
  exit 0
fi

mkdir -p "$M" "$LOGD"
# one lock per stage, held until this runner exits: runners on disjoint stage lists run side by side (one per GPU)
for st in "${RUN_STAGES[@]}"; do
  exec {lfd}>>"$M/$st.runlock"
  flock -n "$lfd" || { say "another ao_heldout runner holds $M/$st.runlock (stage $st); not started"; exit 3; }
done
echo "$(stamp) runner pid $$ host $(hostname) GPU $GPU stages ${RUN_STAGES[*]} max_new_tokens $MNT" >> "$M/runners.log"
echo $$ > "$M/runner_g$GPU.pid"
trap 'release_gpus "ao_heldout_*" "runner exit"' EXIT
if needs_gpu; then reap_stale_holds; take_gpu || exit 3; fi
say "ao_heldout runner pid $$ on $(hostname): stages ${RUN_STAGES[*]}, GPU $GPU, OUT $OUT, max_new_tokens $MNT${LIMIT:+, limit $LIMIT}"
for st in "${RUN_STAGES[@]}"; do
  run_stage "$st"; rc=$?
  if [ "$st" = "$LAST_READ" ]; then
    release_gpus "ao_heldout_read" "last read stage ended (rc $rc)"
    tokens_report
  fi
  [ $rc -eq 0 ] || { say "stopping after $st"; exit 1; }
done
if [[ " ${RUN_STAGES[*]} " == *" summary "* ]]; then say "all stages done; summary $OUT/summary.json"
else say "all stages done (${RUN_STAGES[*]}); no summary in this run"; fi
exit 0
