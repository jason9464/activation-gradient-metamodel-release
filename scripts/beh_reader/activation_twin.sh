#!/usr/bin/env bash
# Activation twin chain (code main_run/beh_reader/activation_twin.py, whose docstring has the rules and checks of
# every stage).
# Two branches run side by side, then the judge and the summary:
#   A  verify_act (GPU) and build_dirs (CPU) side by side -> sft (GPU) -> rl (GPU, judge-cost watchdog 3.5 USD as
#      run_L3.sh) -> eval_twin_read (GPU) -> eval_twin_yesno (empty GPU)
#   B  eval_grad_read (GPU) -> eval_grad_yesno (empty GPU)
#   then judge (API: ledger at start + JUDGE_CAP_USD) -> summary -> plot (CPU)
# A 32,000-document forward extraction on GPUs 1-2 is not needed: the stores already hold the
# activations (activation_twin.py docstring). verify_act checks them against the documents instead (about 10 min).
# Markers $OUT/markers/<stage>.{START,DONE,FAILED,pid}: a DONE stage is skipped; a FAILED one is renamed
# <stage>.FAILED.before_<time> and run again. RL stopped by its cost cap writes rl.COST_CAP (not FAILED) and the twin is
# then evaluated with its SFT adapter; the reading settings record the adapter.
# Logs main_run/logs/20261029_activation_twin_<stage>.out (appended, one header line per attempt).
# GPUs: only $GPUS (default "1 2 3"; GPU 0 belongs to the already-learned chain). A GPU stage reads nvidia-smi free
# memory twice, 30 s apart, and takes the GPU with the most free memory among those with at least the stage's need
# (both readings) and no claim by the other branch (claims $OUT/markers/gpu<G>.claim under flock); it waits in 5-minute
# steps up to WAIT_H hours. The yes/no stages need an empty GPU (rare_yesno refuses one with >= 4 GiB used): need 176 GiB
# of a B200's 179. CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<the GPU>; a CPU stage gets CUDA_VISIBLE_DEVICES
# empty (no GPU visible). Hang guard: a stage whose log has not grown for HANG_S seconds (default 1800; judge 4 h) gets
# TERM to its process group, KILL when still alive 60 s later, and is marked FAILED (the RL cost stop does the same).
# Disk: SFT writes about 16 GB and RL about 22 GB (measured on the gradient twins' runs); each refuses to start
# with less than its need + 20 GB free on the agm_data disk.
# Usage: bash scripts/beh_reader/activation_twin.sh [--dry]   (detached by the caller: setsid nohup ... &)
#   --dry prints every stage's command (GPU shown as <gpu>) and writes nothing.
#   env: OUT, GPUS, JUDGE_CAP_USD (10), RL_TRAIN_CAP_USD (3.5), WAIT_H (6), HANG_S (1800);
#   paths (a smoke chain in a scratch folder; defaults = the real ones): SFT_CFG, RL_CFG, SFT_RUN, RL_RUN, ACT_SFT_DIR,
#   ACT_RL_DIR. SFT_RUN / RL_RUN are passed to train.py / rl_train as --out and exported to activation_twin.py as
#   ACT_TWIN_SFT_RUN / ACT_TWIN_RL_RUN; the configs' reader_dir (and RL train/eval jsonl) must lie in ACT_SFT_DIR /
#   ACT_RL_DIR or the runner stops before any stage (exit 2); for the stub test only:
#   LOGD (main_run/logs), FREE_GAP_S (30, between the two nvidia-smi reads), WAIT_STEP_S (300), POLL_S (10),
#   RL_CHECK_S (120, the RL cost watchdog interval), TW (activation_twin.py), GPU_QUERY (nvidia-smi; split on spaces)
set -u
DRY=0; [ "${1:-}" = "--dry" ] && DRY=1
P=$(cd "$(dirname "$0")/../.." && pwd); cd "$P" || exit 1
IIF_BASE=${IIF_BASE:?set IIF_BASE}
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-$IIF_BASE/venvs/interpretable_influence/bin/python}
EMWT=$IIF_BASE/agm_data/worktree_em_reader; RLWT=$IIF_BASE/agm_data/worktree_rl_stage
TW=${TW:-$P/main_run/beh_reader/activation_twin.py}
OUT=${OUT:-$IIF_BASE/agm_data/runs/beh_reader/activation_twin}
SFT_CFG=${SFT_CFG:-$P/configs/beh_reader/scratch_L3act_beh.yaml}; RL_CFG=${RL_CFG:-$P/configs/beh_reader/rl_scratch_L3act_beh_paper.yaml}
SFT_RUN=${SFT_RUN:-$IIF_BASE/agm_data/runs/reader_scratch_L3act_beh}
RL_RUN=${RL_RUN:-$IIF_BASE/agm_data/runs/reader_rl_scratch_L3act_beh_paper}
ACT_SFT_DIR=${ACT_SFT_DIR:-$IIF_BASE/agm_data/c1_reader_40_beh_act/layers3}
ACT_RL_DIR=${ACT_RL_DIR:-$IIF_BASE/agm_data/rl_fineweb_beh_act/reader}
export ACT_TWIN_SFT_RUN=$SFT_RUN ACT_TWIN_RL_RUN=$RL_RUN  # activation_twin.readers(): the twin's adapters, RUN.json
SFT_ADAPTER=$SFT_RUN/adapters/step_007082; RL_ADAPTER=$RL_RUN/adapters/cycle_040
GPUS=${GPUS:-"1 2 3"}; JUDGE_CAP_USD=${JUDGE_CAP_USD:-10}; RL_TRAIN_CAP_USD=${RL_TRAIN_CAP_USD:-3.5}
WAIT_H=${WAIT_H:-6}; HANG_S=${HANG_S:-1800}
FREE_GAP_S=${FREE_GAP_S:-30}; WAIT_STEP_S=${WAIT_STEP_S:-300}; POLL_S=${POLL_S:-10}; RL_CHECK_S=${RL_CHECK_S:-120}
M=$OUT/markers; LOGD=${LOGD:-$P/main_run/logs}
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

for g in $GPUS; do
  if [ "$g" = 0 ]; then echo "GPU 0 is the already-learned chain's; GPUS=$GPUS refused"; exit 2; fi
done
# the configs read the dirs build_dirs writes (a moved ACT_*_DIR without a matching config would train on other rows)
cfg_bad=$("$PY" - "$SFT_CFG" "$RL_CFG" "$ACT_SFT_DIR" "$ACT_RL_DIR" <<'PYEOF'
import os, sys, yaml
sft, rl, ds, dr = sys.argv[1:5]
def get(path, key):
    v = (yaml.safe_load(open(path)) or {}).get(key)
    return None if v is None else os.path.realpath(os.path.expandvars(str(v)))
want = [(sft, "reader_dir", ds), (rl, "reader_dir", dr), (rl, "train_jsonl", dr), (rl, "eval_jsonl", dr)]
bad = []
for path, key, d in want:
    v, d = get(path, key), os.path.realpath(d)
    if v is None or not (v == d or v.startswith(d + os.sep)):
        bad.append(f"{os.path.basename(path)} {key}={v} not in {d}")
print("; ".join(bad))
PYEOF
) || cfg_bad="config check did not run"
if [ -n "$cfg_bad" ]; then echo "configs do not match the act dirs: $cfg_bad"; exit 2; fi
if [ "$DRY" = 0 ]; then
  mkdir -p "$M" "$LOGD"
  exec 8>"$M/runner.lock"
  if ! flock -n 8; then echo "another activation_twin runner holds $M/runner.lock"; exit 1; fi
  echo "$$ $(hostname)" > "$M/runner.pid"
fi

# ---------------------------------------------------------------- GPU choice
GPU_QUERY=${GPU_QUERY:-nvidia-smi}
gpu_free_mib() {  # "<index>,<free MiB>" lines
  $GPU_QUERY --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' '
}
claim_alive() {  # $1 = claim file: true when its runner-branch pid is alive
  local pid; pid=$(awk '{print $2}' "$1" 2>/dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}
pick_gpu() {  # $1 need GiB, $2 stage; prints the GPU and writes its claim; waits up to WAIT_H hours
  local need=$1 name=$2 t_end=$(( $(date +%s) + WAIT_H * 3600 )) a b g fa fb f best bestf
  while :; do
    (
      flock 9
      a=$(gpu_free_mib); sleep "$FREE_GAP_S"; b=$(gpu_free_mib)
      best=""; bestf=0
      for g in $GPUS; do
        if [ -e "$M/gpu$g.claim" ] && claim_alive "$M/gpu$g.claim"; then continue; fi
        fa=$(echo "$a" | awk -F, -v g="$g" '$1==g{print $2}'); fb=$(echo "$b" | awk -F, -v g="$g" '$1==g{print $2}')
        [ -z "$fa" ] || [ -z "$fb" ] && continue
        f=$(( fa < fb ? fa : fb ))
        if [ "$f" -ge $(( need * 1024 )) ] && [ "$f" -gt "$bestf" ]; then best=$g; bestf=$f; fi
      done
      if [ -n "$best" ]; then
        echo "$name $BASHPID_OWNER $(hostname) $(date '+%F %T') free_mib=$bestf" > "$M/gpu$best.claim"
        echo "$best"; exit 0
      fi
      exit 1
    ) 9>"$M/gpu.lock" && return 0
    [ "$(date +%s)" -ge "$t_end" ] && return 1
    say "$name: no GPU among $GPUS with $need GiB free and no claim; waiting $WAIT_STEP_S s" >&2
    sleep "$WAIT_STEP_S"
  done
}
release_gpu() { [ -n "${1:-}" ] && [ -e "$M/gpu$1.claim" ] && mv "$M/gpu$1.claim" "$M/gpu$1.claim.released_$(date +%Y%m%d_%H%M%S)_$2"; }

disk_ok() {  # $1 = GB needed (plus 20 GB margin) on the agm_data disk
  local free_gb; free_gb=$(df -BG --output=avail "$IIF_BASE/agm_data" | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt $(( $1 + 20 )) ]; then say "disk: ${free_gb} GB free, need $1 + 20 GB" >&2; return 1; fi
}

# ---------------------------------------------------------------- stage commands (set CMD and CWD; $1 = GPU)
mk_verify_act()      { CWD=$P;    CMD=(env PYTHONPATH="$EMWT" "$PY" "$TW" verify-act --gpu "$1" --out "$OUT"); }
mk_build_dirs()      { local d="--act-sft-dir '$ACT_SFT_DIR' --act-rl-dir '$ACT_RL_DIR'"
                       CWD=$P;    CMD=(bash -c "PYTHONPATH='$EMWT' '$PY' '$TW' build-dirs --out '$OUT' $d && PYTHONPATH='$EMWT' '$PY' '$TW' check-dirs $d"); }
mk_eval_grad_read()  { CWD=$P;    CMD=(env PYTHONPATH="$EMWT" "$PY" "$TW" read --reader grad3 --gpu "$1" --out "$OUT"); }
mk_eval_grad_yesno() { CWD=$P;    CMD=(env PYTHONPATH="$EMWT" "$PY" "$TW" yesno --reader grad3 --gpu "$1" --out "$OUT"); }
mk_sft()             { CWD=$EMWT; CMD=(env PYTHONPATH="$EMWT" CUDA_VISIBLE_DEVICES="$1" "$PY" main_run/reader/train.py --config "$SFT_CFG" --out "$SFT_RUN" --batch-size 4 --device cuda:0); }
mk_rl()              { CWD=$RLWT; CMD=(env PYTHONPATH="$RLWT" CUDA_VISIBLE_DEVICES="$1" "$PY" -m main_run.reader.rl_train --config "$RL_CFG" --adapter "$SFT_ADAPTER" --out "$RL_RUN"); }
twin_adapter()       { if [ -e "$M/rl.COST_CAP" ]; then echo "$SFT_ADAPTER"; else echo "$RL_ADAPTER"; fi; }
mk_eval_twin_read()  { CWD=$P;    CMD=(env PYTHONPATH="$EMWT" "$PY" "$TW" read --reader twin --gpu "$1" --out "$OUT" --adapter "$(twin_adapter)"); }
mk_eval_twin_yesno() { CWD=$P;    CMD=(env PYTHONPATH="$EMWT" "$PY" "$TW" yesno --reader twin --gpu "$1" --out "$OUT" --adapter "$(twin_adapter)"); }
mk_judge()           { CWD=$P;    CMD=(env PYTHONPATH="$EMWT" "$PY" "$TW" judge --out "$OUT" --cap-usd "$JUDGE_CAP_USD" --mode batch); }
mk_summary()         { CWD=$P;    CMD=(env PYTHONPATH="$EMWT" "$PY" "$TW" summary --out "$OUT"); }
mk_plot()            { CWD=$P;    CMD=(env PYTHONPATH="$EMWT" "$PY" "$TW" plot --out "$OUT"); }

rl_cost() { "$PY" - "$RL_RUN" <<'PYEOF'
import json, sys, os
out = sys.argv[1]; tot = 0.0; p = os.path.join(out, "rl_log.jsonl")
if os.path.exists(p):
    for line in open(p):
        try: tot += float(json.loads(line).get("judge_cost_usd", 0) or 0)
        except Exception: pass
print(f"{tot:.2f}")
PYEOF
}

group_alive() {  # $1 = process group id: true while a process of the group is alive (an unreaped zombie leader does
  # not count; kill -0 would say yes to it). Reads /proc/<pid>/stat: state and pgrp are fields 1 and 3 after "comm) "
  cat /proc/[0-9]*/stat 2>/dev/null | awk -v g="$1" '{ s = $0; sub(/^.*\) /, "", s); split(s, a, " ");
    if (a[3] == g && a[1] != "Z") { found = 1; exit } } END { exit !found }'
}
stop_group() {  # $1 = a stage pid (its own process group, setsid): TERM to the group, KILL when one of it is still
  # alive 60 s later
  local pid=$1 i
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for i in $(seq 12); do group_alive "$pid" || return 0; sleep 5; done
  say "process group $pid still alive 60 s after TERM; KILL"; kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
}

# ---------------------------------------------------------------- one stage
# run_stage <name> <minutes estimate> <GPU need GiB, 0 = CPU>; uses mk_<name>
run_stage() {
  local name=$1 est=$2 need=$3 gpu="" log pid rc last now size0 size1 quiet=0 hang=$HANG_S
  [ "$name" = judge ] && hang=14400  # Message Batches can sit quiet for long; the judge has its own cost cap
  if [ "$DRY" = 1 ]; then
    "mk_$name" "<gpu>"
    echo "[dry] $name (about $est min; $([ "$need" -gt 0 ] && echo "GPU with $need GiB free among $GPUS" || echo CPU))"
    echo "      cd $CWD && ${CMD[*]}"
    return 0
  fi
  if [ -e "$M/$name.DONE" ]; then say "$name: DONE earlier, skipped"; return 0; fi
  if [ -e "$M/$name.FAILED" ]; then mv "$M/$name.FAILED" "$M/$name.FAILED.before_$(date +%Y%m%d_%H%M%S)"; fi
  case $name in
    sft) if [ -d "$SFT_ADAPTER" ]; then say "sft: $SFT_ADAPTER exists; DONE"; date '+%F %T %Z' > "$M/sft.DONE"; return 0; fi
         disk_ok 16 || { echo "disk" > "$M/sft.FAILED"; return 1; } ;;
    rl)  if [ -f "$RL_RUN/TRAIN_DONE" ]; then say "rl: TRAIN_DONE exists; DONE"; date '+%F %T %Z' > "$M/rl.DONE"; return 0; fi
         if [ -e "$M/rl.COST_CAP" ]; then say "rl: stopped at its cost cap earlier; the twin is read with the SFT adapter"; return 0; fi
         disk_ok 22 || { echo "disk" > "$M/rl.FAILED"; return 1; } ;;
  esac
  if [ "$need" -gt 0 ]; then
    gpu=$(pick_gpu "$need" "$name") || { say "$name: no GPU within $WAIT_H h"; echo "no GPU" > "$M/$name.FAILED"; return 1; }
  fi
  "mk_$name" "$gpu"
  log="$LOGD/20261029_activation_twin_$name.out"
  date '+%F %T %Z' > "$M/$name.START"
  echo "==== $(date '+%F %T') $name gpu=${gpu:-none} (about $est min): cd $CWD && ${CMD[*]}" >> "$log"
  say "$name started on GPU ${gpu:-none} (about $est min); log $log"
  # setsid: the stage is its own process group, so TERM / KILL reach its children too (bash -c, subprocesses)
  # a CPU stage sees no GPU at all (CUDA_VISIBLE_DEVICES empty), so nothing of it can land on GPU 0
  ( cd "$CWD" && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
      && exec setsid "${CMD[@]}" ) >> "$log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid $(hostname)" > "$M/$name.pid"
  last=$(date +%s); size0=$(stat -c %s "$log"); local lastc; lastc=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_S"
    now=$(date +%s); size1=$(stat -c %s "$log")
    if [ "$size1" != "$size0" ]; then last=$now; size0=$size1; fi
    if [ "$name" = rl ] && [ $(( now - lastc )) -ge "$RL_CHECK_S" ]; then  # every 2 min, as run_L3.sh
      lastc=$now
      c=$(rl_cost 2>/dev/null || echo 0)
      if "$PY" -c "import sys; sys.exit(0 if float('$c') > $RL_TRAIN_CAP_USD else 1)"; then
        say "rl cost watchdog: \$$c > \$$RL_TRAIN_CAP_USD; stopping"; echo "$c" > "$M/rl.COST_CAP"
        stop_group "$pid"; break
      fi
    fi
    if [ $(( now - last )) -ge "$hang" ]; then
      say "$name: log quiet for $hang s; TERM"; quiet=1
      stop_group "$pid"; break
    fi
  done
  wait "$pid"; rc=$?
  release_gpu "$gpu" "$name"
  if [ "$name" = rl ] && [ -e "$M/rl.COST_CAP" ]; then say "rl stopped at the cost cap (\$$(cat "$M/rl.COST_CAP"))"; return 0; fi
  if [ "$name" = sft ] && [ ! -d "$SFT_ADAPTER" ] && [ "$rc" = 0 ]; then rc=9; fi
  if [ "$name" = rl ] && [ "$rc" = 0 ]; then
    if [ -d "$RL_ADAPTER" ]; then date '+%F %T %Z' > "$RL_RUN/TRAIN_DONE"; else rc=9; fi
  fi
  if [ "$rc" != 0 ] || [ "$quiet" = 1 ]; then
    say "$name FAILED (exit $rc$([ "$quiet" = 1 ] && echo ', hung'))"; echo "exit $rc quiet $quiet" > "$M/$name.FAILED"; return 1
  fi
  date '+%F %T %Z' > "$M/$name.DONE"; say "$name done"
}

# ---------------------------------------------------------------- the chain
BASHPID_OWNER=$$
branch_a() {
  BASHPID_OWNER=$BASHPID
  run_stage verify_act 2 40 & local va=$!
  run_stage build_dirs 2 0; local rb=$?
  wait "$va"; local rv=$?
  [ "$DRY" = 1 ] || { [ "$rb" = 0 ] && [ "$rv" = 0 ]; } || { say "branch A stops: verify_act $rv, build_dirs $rb"; return 1; }
  run_stage sft 40 75 || return 1
  run_stage rl 75 85 || return 1
  run_stage eval_twin_read 20 60 || return 1
  run_stage eval_twin_yesno 2 176 || return 1
}
branch_b() {
  BASHPID_OWNER=$BASHPID
  run_stage eval_grad_read 20 60 || return 1
  run_stage eval_grad_yesno 2 176 || return 1
}
if [ "$DRY" = 1 ]; then
  echo "OUT=$OUT  GPUS=$GPUS  judge cap = ledger + $JUDGE_CAP_USD USD  RL judge cap $RL_TRAIN_CAP_USD USD"
  echo "SFT_RUN=$SFT_RUN  RL_RUN=$RL_RUN  ACT_SFT_DIR=$ACT_SFT_DIR  ACT_RL_DIR=$ACT_RL_DIR (configs match)"
  echo "branch A:"; branch_a; echo "branch B (side by side with A):"; branch_b; echo "after both:"
  run_stage judge 30 0; run_stage summary 1 0; run_stage plot 1 0
  exit 0
fi
[ -e "$M/chain.FAILED" ] && mv "$M/chain.FAILED" "$M/chain.FAILED.before_$(date +%Y%m%d_%H%M%S)"
say "activation twin chain on $(hostname); OUT $OUT; GPUs $GPUS"
branch_b > >(sed -u 's/^/[B] /') 2>&1 & PB=$!
branch_a > >(sed -u 's/^/[A] /') 2>&1; RA=$?
wait "$PB"; RB=$?
if [ "$RA" != 0 ] || [ "$RB" != 0 ]; then say "chain stopped: branch A exit $RA, branch B exit $RB"; date > "$M/chain.FAILED"; exit 1; fi
run_stage judge 30 0 || { date > "$M/chain.FAILED"; exit 1; }
run_stage summary 1 0 || { date > "$M/chain.FAILED"; exit 1; }
run_stage plot 1 0 || { date > "$M/chain.FAILED"; exit 1; }
date '+%F %T %Z' > "$M/chain.DONE"; say "chain done"
