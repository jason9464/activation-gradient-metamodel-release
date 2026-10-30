#!/bin/bash
# Clean-batch probe from the start of training to the end of the main runs (user request 2026-10-28 17:50: measure the
# would-be updates of the real and the three foreign clean batches also at the very start of training and after step
# 3,000, up to 7,000 where the dolphin preference is fully visible, for both students). Plan of the first runs:
# docs/notes/20261026_clean_batch_update_plan.md; this script reruns one student from step 0 with the recipe of
# scripts/cleanbatch/chain.sh, only the stop step, the probe steps and the output folder changed.
#
# Usage: bash scripts/cleanbatch/extend.sh <arm dolphin|clean> <gpu>
#   The script checks, then relaunches itself detached (setsid nohup) and returns; the detached part runs the stages
#   in order and writes the markers. Runner log: main_run/logs/<YYYYmmdd>_cleanbatch_extend_<arm>_runner.out.
# Stages (a DONE stage is skipped when the script runs again):
#   rerun   (GPU) main_run/train_cleanbatch.py with chain.sh's train_args (the function itself, read from chain.sh):
#           --stop-step 7000, --probe-steps 2,100,200,...,7000 (71 steps; 2 is the earliest step the probe accepts and
#           stands for "step 0"), --adamfx-lora-pairs from chain.sh's PAIRS rule at stop 7000 (101,1001,3001,5001, the
#           main runs' value, checked against their RUN.json), --out runs/cleanbatch_<arm>_s1_full/train.
#           DONE when python exits 0, the log has "done: step 7000," and probe_meta.json is complete with every
#           planned step.
#   inputs  (CPU) main_run/cleanbatch_inputs.py verify --train <out> --arm <arm> (against the main run
#           newset_<arm>_s1, which has steps 1..7000: trajectory, probe state and weight checks, real batch vs the
#           rerun's and the main run's stores, grad vs the main run's grad_inputs_every100 at the steps it has, logged
#           grad norm and lr) and build --groups core,mfree,grad,all (as chain.sh, without fixed and mfreefix)
#   pyes    (GPU, the same GPU, checked again; waits for it up to 6 h) main_run/cleanbatch_pyes.py on
#           <out>/e2/cleanbatch_inputs_all.npz with chain.sh's arguments (reader rl_paper_c040, question
#           dolphin_yesno, --max-gpu-gib 60, run from the worktree as chain.sh does; PYTHONDONTWRITEBYTECODE=1 so the
#           worktree gets no __pycache__ writes)
# No readings with open answers and no judge.
# Markers (runs/cleanbatch_extend/markers/): <arm>.{rerun.START,rerun.DONE,inputs.DONE,pyes.DONE,DONE,FAILED};
#   pid files <arm>.pid (the detached runner shell), <arm>.rerun.pid, <arm>.inputs.pid, <arm>.pyes.pid (python);
#   <arm>.gpu. An older FAILED marker is renamed <arm>.FAILED.before_<stamp> when the script starts again.
# Logs: main_run/logs/<YYYYmmdd>_cleanbatch_extend_<arm>_{rerun,inputs,pyes,runner}.out (appended).
# GPU pinning: the GPU stages run with CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=<gpu>, so <gpu> is the
#   nvidia-smi index that the busy checks look at (CUDA's default order is not guaranteed to be nvidia-smi's).
# Refuses: <arm>.DONE exists; a pid file of the arm whose process is alive; an output folder that is not empty while
#   the rerun is not DONE (the script never moves or deletes run outputs); a GPU with a compute process, 4 GiB or
#   more in use, or a python process whose CUDA_VISIBLE_DEVICES lists it, on either of two looks 20 s apart; less than CBX_MIN_FREE_GB free on the output file system;
#   chain.sh values that are missing or changed (fixed documents sha256, the foreign file's sha256 against the first
#   reruns' marker, the pyes and inputs command lines).
# Values taken from chain.sh (never copied by hand): IIF PY P W R FIXED FIXED_SHA FOREIGN RJ READER ADAPTER MEM_MAX_MIB,
#   the export line, the train_args function and the PAIRS rule.
# Dry listing (nothing is written or started): CBX_DRY=1 bash scripts/cleanbatch/extend.sh <arm> <gpu>
# Overrides (tests only; with CBX_STOP or CBX_PROBE set, CBX_STATE, CBX_OUT and CBX_LOGS must be set too):
#   CBX_STATE (state folder, default runs/cleanbatch_extend), CBX_OUT (train folder), CBX_LOGS (log folder),
#   CBX_STOP (7000), CBX_PROBE (comma list), CBX_LOOK_S (20), CBX_POLL_S (60), CBX_MAX_WAIT_S (21600),
#   CBX_MIN_FREE_GB (60: two students of about 22 GB each, x 1.2), CBX_DATE (log prefix, default today)
set -u
SELF=$(readlink -f "$0")
CHAIN=$(dirname "$SELF")/chain.sh
die() { echo "$(date '+%F %T') [cleanbatch extend] refuse: $*" >&2; exit 2; }
[ -f "$CHAIN" ] || die "chain.sh not found at $CHAIN"

# ---------------------------------------------------------------- values from chain.sh
for v in IIF PY P W R FIXED FIXED_SHA FOREIGN RJ READER ADAPTER MEM_MAX_MIB; do
  n=$(grep -cE "^$v=" "$CHAIN")
  [ "$n" = 1 ] || die "chain.sh has $n lines '^$v=' (expected 1)"
  eval "$(grep -E "^$v=" "$CHAIN")"
  [ -n "${!v:-}" ] || die "chain.sh value $v is empty"
done
[ "$(grep -cE '^export IIF_BASE=' "$CHAIN")" = 1 ] || die "chain.sh export line not found"
eval "$(grep -E '^export IIF_BASE=' "$CHAIN")"
[ -n "${IIF_BASE:-}" ] && [ -n "${HF_HOME:-}" ] && [ "${HF_HUB_OFFLINE:-}" = 1 ] || die "chain.sh export line did not set IIF_BASE, HF_HOME, HF_HUB_OFFLINE"
eval "$(sed -n '/^train_args() {/,/^}/p' "$CHAIN")"
declare -F train_args > /dev/null || die "train_args not found in chain.sh"
PAIRS_RULE=$(grep -E '^PAIRS=\$\(for t in ' "$CHAIN")
[ "$(echo "$PAIRS_RULE" | grep -c .)" = 1 ] || die "chain.sh PAIRS rule not found (or more than one)"
# the inputs and pyes command lines of chain.sh that this script repeats
for s in 'cleanbatch_inputs.py verify --train "$tr" --arm "$arm"' 'cleanbatch_inputs.py build --train "$tr" --groups "$groups"' \
         '"$P/main_run/cleanbatch_pyes.py" --inputs "$tr/e2/cleanbatch_inputs_all.npz"' \
         '--adapter "$ADAPTER" --run-json "$RJ" --reader-name "$READER" --question dolphin_yesno --device cuda:0 --max-gpu-gib 60' \
         '( cd "$W" && CUDA_VISIBLE_DEVICES=$READ_GPU exec $PY "$P/main_run/cleanbatch_pyes.py"'; do
  grep -qF -- "$s" "$CHAIN" || die "chain.sh no longer has: $s"
done

# ---------------------------------------------------------------- arguments and settings
ARM=${1:-}; GPU=${2:-}; MODE=${3:-}
case $ARM in dolphin|clean) ;; *) die "usage: extend.sh <dolphin|clean> <gpu>";; esac
case $GPU in ''|*[!0-9]*) die "usage: extend.sh <dolphin|clean> <gpu> (gpu an index)";; esac
if [ -n "${CBX_STOP:-}" ] || [ -n "${CBX_PROBE:-}" ]; then
  [ -n "${CBX_STATE:-}" ] && [ -n "${CBX_OUT:-}" ] && [ -n "${CBX_LOGS:-}" ] || die "CBX_STOP/CBX_PROBE need CBX_STATE, CBX_OUT and CBX_LOGS (test runs stay out of the real folders)"
fi
STOP=${CBX_STOP:-7000}
PROBE=${CBX_PROBE:-$( { echo 2; seq 100 100 "$STOP"; } | paste -sd, -)}
X=${CBX_STATE:-$R/cleanbatch_extend}
M=$X/markers
OUT=${CBX_OUT:-$R/cleanbatch_${ARM}_s1_full/train}
L=${CBX_LOGS:-$P/main_run/logs}
DAY=${CBX_DATE:-$(date +%Y%m%d)}
LOOK_S=${CBX_LOOK_S:-20}
POLL_S=${CBX_POLL_S:-60}
MAX_WAIT_S=${CBX_MAX_WAIT_S:-21600}
MIN_FREE_GB=${CBX_MIN_FREE_GB:-60}
GROUPS_BUILD=core,mfree,grad,all
MAIN=$R/newset_${ARM}_s1
RLOG=$L/${DAY}_cleanbatch_extend_${ARM}_rerun.out
ILOG=$L/${DAY}_cleanbatch_extend_${ARM}_inputs.out
PLOG=$L/${DAY}_cleanbatch_extend_${ARM}_pyes.out
RUNLOG=$L/${DAY}_cleanbatch_extend_${ARM}_runner.out
PYES_JSON=$OUT/e2/cleanbatch_inputs_all_pyes_${READER}_dolphin_yesno.json
eval "$PAIRS_RULE"
[ -n "${PAIRS:-}" ] || die "PAIRS rule of chain.sh gave nothing at stop $STOP"

log() { echo "$(date '+%F %T') [cleanbatch extend $ARM] $*"; }
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
isdone() { [ -f "$M/$ARM.$1.DONE" ]; }
before_name() { local b t n=0; b="$1.before_$(date +%Y%m%d_%H%M%S)"; t=$b; while [ -e "$t" ]; do n=$((n + 1)); t="${b}_$n"; done; echo "$t"; }
aside() { [ -e "$1" ] || return 0; mv -n "$1" "$(before_name "$1")"; }

# probe list: integers, increasing, within 2..STOP
PROBE_N=$(echo "$PROBE" | $PY -c "
import sys
s = [int(x) for x in sys.stdin.read().strip().split(',')]
assert s == sorted(set(s)) and s[0] >= 2 and s[-1] <= $STOP, s
print(len(s))") || die "probe steps $PROBE are not an increasing list within 2..$STOP"
# PAIRS at the main runs' stop equal the main run's (RUN.json settings.adamfx_lora_pairs)
MAIN_STOP=$($PY -c "import json; print(json.load(open('$MAIN/RUN.json'))['settings']['stop_step'])") || die "cannot read $MAIN/RUN.json"
MAIN_PAIRS=$($PY -c "import json; print(json.load(open('$MAIN/RUN.json'))['settings']['adamfx_lora_pairs'])")
if [ "$STOP" = "$MAIN_STOP" ] && [ "$PAIRS" != "$MAIN_PAIRS" ]; then die "PAIRS $PAIRS differ from the main run's $MAIN_PAIRS"; fi
ARGS=$(train_args "$ARM" "$STOP" "$PAIRS" "$OUT" "$PROBE")

# ---------------------------------------------------------------- GPU
gpu_state() {  # gpu -> "uuid used_mib n_compute_processes" (empty when the index is unknown)
  local uuid used n
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', *' -v g="$1" '$1 == g {print $2}')
  [ -n "$uuid" ] || return 0
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v g="$1" '$1 == g {print $2}')
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -cF "$uuid")
  echo "$uuid $used $n"
}
gpu_claimed() {  # gpu -> pids of python processes whose CUDA_VISIBLE_DEVICES lists it (a job still loading its model
  # holds no GPU memory for minutes: a second smoke started next to a loading one on 2026-10-28 17:38)
  local e pid
  for e in /proc/[0-9]*/environ; do
    pid=${e#/proc/}; pid=${pid%/environ}
    case $(cat "/proc/$pid/comm" 2>/dev/null) in python*) ;; *) continue;; esac
    tr '\0' '\n' < "$e" 2>/dev/null | grep -qE "^CUDA_VISIBLE_DEVICES=([0-9]+,)*$1(,[0-9]+)*$" && echo "$pid"
  done | paste -sd, -
}
gpu_free_once() {  # gpu -> 0 when no compute process, < MEM_MAX_MIB used and no python process aimed at it; sets GPU_WHY
  local s c; s=$(gpu_state "$1"); c=$(gpu_claimed "$1")
  [ -n "$s" ] || { GPU_WHY="GPU $1 does not exist"; return 1; }
  set -- $s
  [ "$3" = 0 ] && [ "$2" -lt "$MEM_MAX_MIB" ] && [ -z "$c" ] && return 0
  GPU_WHY="GPU uuid $1: $2 MiB used, $3 compute processes, python pids with CUDA_VISIBLE_DEVICES on it: [${c}]"; return 1
}
gpu_free() { gpu_free_once "$1" || return 1; sleep "$LOOK_S"; gpu_free_once "$1"; }  # two looks LOOK_S apart

# ---------------------------------------------------------------- dry listing
if [ "${CBX_DRY:-0}" = 1 ]; then
  log "dry: chain.sh $CHAIN (repo $(GIT_OPTIONAL_LOCKS=0 git -C "$P" rev-parse --short HEAD 2>/dev/null))"
  log "dry: FIXED $FIXED (sha256 expected $FIXED_SHA, now $(sha256sum "$FIXED" | cut -c1-64))"
  log "dry: FOREIGN $FOREIGN sha256 $(sha256sum "$FOREIGN" | cut -c1-64)"
  log "dry: PAIRS $PAIRS (main run $MAIN_PAIRS at stop $MAIN_STOP); probe steps $PROBE_N"
  log "dry: train: cd $P/main_run && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY train_cleanbatch.py $ARGS"
  log "dry: inputs: $PY cleanbatch_inputs.py verify --train $OUT --arm $ARM ; build --train $OUT --groups $GROUPS_BUILD"
  log "dry: pyes: cd $W && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $PY $P/main_run/cleanbatch_pyes.py --inputs $OUT/e2/cleanbatch_inputs_all.npz --adapter $ADAPTER --run-json $RJ --reader-name $READER --question dolphin_yesno --device cuda:0 --max-gpu-gib 60"
  log "dry: markers $M; logs $RLOG $ILOG $PLOG $RUNLOG; GPU $GPU now: $(gpu_state "$GPU")"
  log "dry: stages done: $(ls "$M"/"$ARM".*.DONE 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
  exit 0
fi

# ---------------------------------------------------------------- checks (both modes)
check_start() {
  [ -f "$M/$ARM.DONE" ] && die "$M/$ARM.DONE exists"
  local f
  for f in "$M/$ARM.pid" "$M/$ARM.rerun.pid" "$M/$ARM.inputs.pid" "$M/$ARM.pyes.pid"; do
    [ "$MODE" = --inner ] && [ "$f" = "$M/$ARM.pid" ] && continue
    alive "$(cat "$f" 2>/dev/null)" && die "$f: pid $(cat "$f") is alive"
  done
  # claim the arm now (a dead pid is harmless): a second launch during this check's GPU looks is refused above
  [ "$MODE" != --inner ] && { mkdir -p "$M"; echo $$ > "$M/$ARM.pid"; }
  [ "$(sha256sum "$FIXED" | cut -d' ' -f1)" = "$FIXED_SHA" ] || die "fixed documents sha256 differs from chain.sh's $FIXED_SHA"
  local fs; fs=$(sed -n 's/.* sha256 \([0-9a-f]\{64\}\).*/\1/p' "$R/cleanbatch_dolphin_s1/markers/foreign.DONE" 2>/dev/null)
  [ -n "$fs" ] || die "no foreign sha256 in $R/cleanbatch_dolphin_s1/markers/foreign.DONE"
  [ "$(sha256sum "$FOREIGN" | cut -d' ' -f1)" = "$fs" ] || die "$FOREIGN differs from the first reruns' file (sha256 $fs)"
  if ! isdone rerun; then
    [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && die "$OUT is not empty and the rerun is not DONE (the script does not move run outputs)"
    local d avail; d=$(dirname "$OUT"); while [ ! -d "$d" ]; do d=$(dirname "$d"); done
    avail=$(df -P -BG "$d" | awk 'NR == 2 {sub("G", "", $4); print $4}')
    [ "$avail" -ge "$MIN_FREE_GB" ] || die "$avail GB free under $d, need $MIN_FREE_GB"
  fi
  if ! isdone rerun || ! isdone pyes; then gpu_free "$GPU" || die "GPU $GPU is busy: $GPU_WHY"; fi
}

if [ "$MODE" != --inner ]; then
  check_start
  mkdir -p "$M" "$L"
  aside "$M/$ARM.FAILED"
  CBX_DATE=$DAY setsid nohup bash "$SELF" "$ARM" "$GPU" --inner < /dev/null >> "$RUNLOG" 2>&1 &
  echo $! > "$M/$ARM.pid"
  log "started detached: runner pid $!, log $RUNLOG; markers $M/$ARM.*; GPU $GPU; stop $STOP; $PROBE_N probe steps"
  exit 0
fi

# ---------------------------------------------------------------- detached runner
echo $$ > "$M/$ARM.pid"
fail() {  # stage message logfile
  { echo "$(date '+%F %T') stage $1: $2"; [ -n "${3:-}" ] && { echo "log $3"; tail -n 30 "$3"; }; } > "$M/$ARM.FAILED"
  log "FAILED at $1: $2"; exit 1
}
log "runner start: pid $$, GPU $GPU, out $OUT, stop $STOP, $PROBE_N probe steps, pairs $PAIRS, repo $(GIT_OPTIONAL_LOCKS=0 git -C "$P" rev-parse --short HEAD 2>/dev/null)"
log "train args: $ARGS"

if ! isdone rerun; then
  gpu_free_once "$GPU" || fail rerun "GPU $GPU busy at start: $GPU_WHY"
  [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && fail rerun "$OUT is not empty"
  mkdir -p "$(dirname "$OUT")"
  echo "$GPU" > "$M/$ARM.gpu"
  echo "$(date '+%F %T') GPU $GPU, out $OUT, stop $STOP, probes $PROBE_N ($PROBE), pairs $PAIRS, log $RLOG, runner pid $$" > "$M/$ARM.rerun.START"
  t0=$(date +%s); n0=$(cat "$RLOG" 2>/dev/null | wc -l)  # lines of earlier attempts in the appended log
  # shellcheck disable=SC2086
  ( cd "$P/main_run" && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU exec $PY train_cleanbatch.py $ARGS ) >> "$RLOG" 2>&1 &
  pid=$!; echo "$pid" > "$M/$ARM.rerun.pid"
  log "rerun python pid $pid, log $RLOG"
  wait "$pid"; rc=$?
  [ "$rc" = 0 ] || fail rerun "exit $rc" "$RLOG"
  tail -n +$((n0 + 1)) "$RLOG" | grep -q "done: step $STOP, " || fail rerun "no 'done: step $STOP,' in the log" "$RLOG"
  pm=$($PY -c "
import json, sys
m = json.load(open('$OUT/cleanbatch/probe_meta.json'))
want = [int(x) for x in '$PROBE'.split(',')]
ok = m['complete'] and m['done_steps'] == want and sorted(m['planned_steps']) == want
print(('ok' if ok else 'BAD') + f\" probe complete {m['complete']}, {len(m['done_steps'])}/{len(want)} steps\")") || fail rerun "probe_meta.json unreadable" "$RLOG"
  case $pm in ok*) ;; *) fail rerun "$pm" "$RLOG";; esac
  echo "$(date '+%F %T') exit 0, $(tail -n +$((n0 + 1)) "$RLOG" | grep "done: step $STOP, " | tail -1 | sed 's/.*\(done: .*\)/\1/'), $pm, $(( $(date +%s) - t0 )) s with loading; log $RLOG" > "$M/$ARM.rerun.DONE"
  log "rerun DONE ($(( $(date +%s) - t0 )) s)"
fi

if ! isdone inputs; then
  t0=$(date +%s); n0=$(cat "$ILOG" 2>/dev/null | wc -l)
  ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 exec $PY cleanbatch_inputs.py verify --train "$OUT" --arm "$ARM" ) >> "$ILOG" 2>&1 &
  pid=$!; echo "$pid" > "$M/$ARM.inputs.pid"; wait "$pid" || fail inputs "verify failed" "$ILOG"
  ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 exec $PY cleanbatch_inputs.py build --train "$OUT" --groups "$GROUPS_BUILD" ) >> "$ILOG" 2>&1 &
  pid=$!; echo "$pid" > "$M/$ARM.inputs.pid"; wait "$pid" || fail inputs "build failed" "$ILOG"
  for g in ${GROUPS_BUILD//,/ }; do [ -f "$OUT/e2/cleanbatch_inputs_$g.npz" ] || fail inputs "missing $OUT/e2/cleanbatch_inputs_$g.npz" "$ILOG"; done
  echo "$(date '+%F %T') $(tail -n +$((n0 + 1)) "$ILOG" | grep 'verify ' | tail -1 | cut -c1-600); groups $GROUPS_BUILD built ($(tail -n +$((n0 + 1)) "$ILOG" | grep -c 'checks 5/5') files with checks 5/5 in the log), $(( $(date +%s) - t0 )) s; log $ILOG" > "$M/$ARM.inputs.DONE"
  log "inputs DONE ($(( $(date +%s) - t0 )) s)"
fi

if ! isdone pyes; then
  w0=$(date +%s)
  until gpu_free "$GPU"; do
    [ $(( $(date +%s) - w0 )) -ge "$MAX_WAIT_S" ] && fail pyes "GPU $GPU still busy after $MAX_WAIT_S s: $GPU_WHY"
    log "pyes waits for GPU $GPU ($GPU_WHY)"; sleep $(( POLL_S > LOOK_S ? POLL_S - LOOK_S : 1 ))
  done
  t0=$(date +%s)
  ( cd "$W" && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 exec $PY "$P/main_run/cleanbatch_pyes.py" --inputs "$OUT/e2/cleanbatch_inputs_all.npz" \
      --adapter "$ADAPTER" --run-json "$RJ" --reader-name "$READER" --question dolphin_yesno --device cuda:0 --max-gpu-gib 60 ) >> "$PLOG" 2>&1 &
  pid=$!; echo "$pid" > "$M/$ARM.pyes.pid"; wait "$pid" || fail pyes "cleanbatch_pyes.py failed" "$PLOG"
  pc=$($PY -c "
import json
d = json.load(open('$PYES_JSON'))
want = [int(x) for x in '$PROBE'.split(',')]
ok = d['finite'] and [int(s) for s in d['steps']] == want and len(d['windows']) == 20 and all(len(r) == 20 for r in d['p_yes'])
print(('ok' if ok else 'BAD') + f\" {len(d['steps'])} steps x {len(d['windows'])} windows, finite {d['finite']}\")") || fail pyes "unreadable $PYES_JSON" "$PLOG"
  case $pc in ok*) ;; *) fail pyes "$pc" "$PLOG";; esac
  echo "$(date '+%F %T') $pc, $(( $(date +%s) - t0 )) s; $(grep 'mean P(yes)' "$PLOG" | tail -1 | cut -c1-500)" > "$M/$ARM.pyes.DONE"
  log "pyes DONE ($(( $(date +%s) - t0 )) s)"
fi

echo "$(date '+%F %T') rerun, inputs and P(yes) DONE; $PYES_JSON" > "$M/$ARM.DONE"
log "DONE"
