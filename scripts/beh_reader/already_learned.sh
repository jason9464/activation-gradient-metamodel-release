#!/usr/bin/env bash
# "Same data, different starting model" chain (code main_run/beh_reader/already_learned.py, whose docstring has the
# design, the checks and the estimates). Stages in order, each skipped when its DONE marker exists:
#   select                      splits, evaluation prompts, checks (CPU)
#   finetune_<X>, ftjudge_<X>   per behaviour X: LoRA fine-tune (GPU with >= 90 GiB free), then the behaviour_shown judge
#                               of its evaluation answers and the stop rule (M1.json)
#   extract_base, extract_<X>   gradients and activations of the held-out documents inside their store batches (GPU
#                               with >= 115 GiB free: the store run peaked at 104.7 GiB); base checked bit for bit
#   build                       reader and AO inputs, train mean of the activations (CPU)
#   read_open, read_yesno, read_ao   reader open readings, reader P(Yes), AO P(Yes) and open answers (GPU >= 60 GiB)
#   judge                       truth judge of every reading (this run's ledger rows capped at 2 USD in total)
#   summary, plot               summary.json, docs/figures/20261029_already_learned.png
# GPU stages run with --gpu auto: the python process takes the GPU with the most free memory once one has enough, polling
# every 60 s while none has (it waits, it does not fail), and pins CUDA_DEVICE_ORDER=PCI_BUS_ID and
# CUDA_VISIBLE_DEVICES before CUDA starts. One big model per process; stages run one after another.
#
# Launch detached from the repo root:
#   setsid nohup bash scripts/beh_reader/already_learned.sh < /dev/null \
#       > main_run/logs/20261029_already_learned_runner.out 2>&1 &
# Dry run (prints every stage, its state and command; runs nothing):   DRY_RUN=1 bash scripts/beh_reader/already_learned.sh
# One stage by hand:                                                    bash scripts/beh_reader/already_learned.sh <stage>
# Smoke (output in a scratch folder, judges --dry-run, small settings; refused with the default output folder):
#   AL_OUT=<scratch>/already_learned_smoke JUDGE_DRY=1 \
#   SELECT_ARGS="--limit-docs 2 --ft-limit 4 --max-epochs 2 --eval-every 1 --n-eval 2 --samples 2" \
#   bash scripts/beh_reader/already_learned.sh
# Markers in $OUT/markers: <stage>.START (appended per attempt), .DONE, .FAILED (exit code, appended), <stage>.pid,
# runner.lock (one runner per output folder). A failed stage stops the chain; its outputs are never overwritten, so move
# them aside before running it again. Logs: main_run/logs/20261029_already_learned_<stage>.out (appended).
# Expected minutes: select 1, finetune 6 each, ftjudge 1 each, extract 3 each (plus model loads), build 3, read_open 12,
# read_yesno 4, read_ao 20, judge 10-40 (Message Batch), summary 1, plot 1; about 1.5-2 h without GPU waits.
# Exit codes: 0 done, 1 a stage failed, 2 usage, 3 another runner holds the lock.
set -u -o pipefail

IIF_BASE=${IIF_BASE:?set IIF_BASE}
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
SELF=$(readlink -f "$0")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd)
PY=$IIF_BASE/venvs/interpretable_influence/bin/python
EMWT=$IIF_BASE/agm_data/worktree_em_reader
DEFAULT_OUT=$IIF_BASE/agm_data/runs/beh_reader/already_learned
OUT=${AL_OUT:-$DEFAULT_OUT}
LOGD=${AL_LOGD:-$REPO/main_run/logs}
LOG_PREFIX=${LOG_PREFIX:-20261029}
DRY=${DRY_RUN:-0}
JUDGE_DRY=${JUDGE_DRY:-0}
SELECT_ARGS=${SELECT_ARGS:-}
BEHS=(1d07947f7f51 407394e603e0 ef4f859143dd)  # already_learned.BEHAVIOURS (the test compares them)
M=$OUT/markers
AL=main_run/beh_reader/already_learned.py

STAGES=(select)
for b in "${BEHS[@]}"; do STAGES+=("finetune_$b" "ftjudge_$b"); done
STAGES+=(extract_base)
for b in "${BEHS[@]}"; do STAGES+=("extract_$b"); done
STAGES+=(build read_open read_yesno read_ao judge summary plot)

usage() { echo "usage: [DRY_RUN=1] bash scripts/beh_reader/already_learned.sh [all|<stage>]; stages: ${STAGES[*]}"; exit 2; }
WANT=${1:-all}
case " all ${STAGES[*]} " in *" $WANT "*) ;; *) usage ;; esac
[ "$WANT" = all ] && RUN_STAGES=("${STAGES[@]}") || RUN_STAGES=("$WANT")
if [ "$(readlink -m "$OUT")" = "$(readlink -m "$DEFAULT_OUT")" ] && { [ "$JUDGE_DRY" = 1 ] || [ -n "$SELECT_ARGS" ]; }; then
  echo "JUDGE_DRY / SELECT_ARGS are for a smoke: set AL_OUT to a scratch folder"; exit 2
fi
if [ -n "$SELECT_ARGS" ] && [ "$JUDGE_DRY" != 1 ]; then
  # a smoke sends no judge request (already_learned.py also refuses real judges on smoke settings)
  echo "SELECT_ARGS (smoke settings) need JUDGE_DRY=1"; exit 2
fi

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
say() { echo "[$(stamp)] $*"; }

stage_cmd() {
  local st=$1 jd=""
  [ "$JUDGE_DRY" = 1 ] && jd=" --dry-run"
  case "$st" in
    select) echo "$PY $AL select --out $OUT${SELECT_ARGS:+ $SELECT_ARGS}" ;;
    finetune_*) echo "$PY $AL finetune --out $OUT --behaviour ${st#finetune_} --gpu auto" ;;
    ftjudge_*) echo "$PY $AL ftjudge --out $OUT --behaviour ${st#ftjudge_}$jd" ;;
    extract_*) echo "$PY $AL extract --out $OUT --model ${st#extract_} --gpu auto" ;;
    build) echo "$PY $AL build --out $OUT" ;;
    read_open) echo "$PY $AL read-open --out $OUT --gpu auto" ;;
    read_yesno) echo "$PY $AL read-yesno --out $OUT --gpu auto" ;;
    read_ao) echo "$PY $AL read-ao --out $OUT --gpu auto" ;;
    judge) echo "$PY $AL judge --out $OUT$jd" ;;
    summary) if [ "$JUDGE_DRY" = 1 ]; then echo "$PY $AL summary --out $OUT --allow-missing"
             else echo "$PY $AL summary --out $OUT"; fi ;;
    plot) echo "$PY $AL plot --out $OUT" ;;
  esac
}

run_stage() {
  local st=$1 log rc cmd pid
  log=$LOGD/${LOG_PREFIX}_already_learned_$st.out
  cmd=$(stage_cmd "$st")
  if [ "$DRY" = 1 ]; then
    if [ -e "$M/$st.DONE" ]; then echo "  [dry] $st: DONE ($M/$st.DONE), skipped"
    else echo "  [dry] $st: cd $REPO && PYTHONPATH=$EMWT $cmd >> $log 2>&1"; fi
    return 0
  fi
  if [ -e "$M/$st.DONE" ]; then say "$st already DONE; skipped"; return 0; fi
  echo "$(stamp) host $(hostname) runner pid $$" >> "$M/$st.START"
  say "$st: start; log $log"
  echo "[$(stamp)] $st: $cmd" >> "$log"
  # shellcheck disable=SC2086
  (cd "$REPO" && PYTHONPATH=$EMWT exec $cmd) >> "$log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$M/$st.pid"
  wait "$pid"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$(stamp) exit $rc" >> "$M/$st.FAILED"
    say "$st FAILED (exit $rc); see $log"
    return 1
  fi
  echo "$(stamp) host $(hostname)" > "$M/$st.DONE"
  say "$st DONE"
}

if [ "$DRY" = 1 ]; then
  say "dry run: OUT=$OUT, logs $LOGD, judges $([ "$JUDGE_DRY" = 1 ] && echo dry || echo real)"
  for st in "${RUN_STAGES[@]}"; do run_stage "$st"; done
  exit 0
fi
mkdir -p "$M" "$LOGD"
exec 8>>"$M/runner.lock"
if ! flock -n 8; then say "another runner holds $M/runner.lock"; exit 3; fi
echo "$$" > "$M/runner.pid"
say "runner pid $$ on $(hostname): OUT=$OUT, stages ${RUN_STAGES[*]}"
for st in "${RUN_STAGES[@]}"; do
  run_stage "$st" || exit 1
done
say "all stages done"
exit 0
