#!/usr/bin/env bash
# Broad change questions chain (user design 2026-10-29 17:45; plan docs/notes/20261029_broad_change_questions_plan.md;
# code main_run/beh_reader/broad_change_questions.py = BCQ, whose docstring has the rules and checks of every stage).
#
# Usage: bash scripts/beh_reader/broad_change_questions.sh [--dry]
#   launch detached from the repo root:  setsid nohup bash scripts/beh_reader/broad_change_questions.sh < /dev/null \
#                                            > /dev/null 2>&1 &
#   the runner writes its own log, main_run/logs/20261029_broad_q_runner.out (RUNNER_LOG; "-" = stdout).
#   --dry prints the settings and every stage (state, GPU or CPU, estimate, command with the GPU shown as <gpu>);
#   writes nothing.
#
# Stages, one after another (a stage whose markers/<stage>.DONE exists is skipped; a FAILED marker of an earlier run
# is renamed <stage>.FAILED.before_<time> and the stage runs again; BCQ refuses to overwrite a reader's pyes.jsonl, so
# a read that failed after writing must have its folder set aside first):
#   questions          BCQ questions (CPU): questions.jsonl, QUESTIONS.json; checks every reader's held-out rows
#   read_<R>           BCQ read --reader R --gpu G (GPU) for R in READERS (default "grad3 twin reader40 ao3 slot"),
#                      all on one GPU, one after another
#   summary            BCQ summary --readers <READERS> (CPU)
#   plot               BCQ plot (CPU); outside the real OUT the figure goes to OUT (--fig)
# A FAILED stage stops the chain: markers/chain.FAILED and exit 1.
# GPU: before the first read stage that is not DONE, one GPU among GPUS (default "0 1 2 3") is chosen under the shared
# lock CLAIM_DIR/gpu.lock (the activation-twin chain's markers directory, used by every reader chain): nvidia-smi free
# memory is read twice, FREE_GAP_S (30) s apart, the smaller reading counts; a GPU with a live claim
# CLAIM_DIR/gpu<G>.claim of another chain is skipped (a claim is live while its pid lives on this host; a claim written
# on another host counts as live); among the GPUs with at least NEED_GIB (45) GiB free the one with the most free memory
# wins. The claim "broad_q_reads <runner pid> <host> <date> <time> free_mib=<n>" stays while the reads run and is then
# moved to gpu<G>.claim.released_<time>_broad_q_reads (also when the runner exits early). With no GPU the runner waits
# WAIT_STEP_S (300) s per attempt, up to WAIT_H (6) hours, then fails. GPU stages get CUDA_DEVICE_ORDER=PCI_BUS_ID and
# CUDA_VISIBLE_DEVICES=<G>; CPU stages get CUDA_VISIBLE_DEVICES empty.
# Hang guard: a running stage whose log has not grown for HANG_S (900) s gets TERM to its process group (setsid), KILL
# 60 s later, and is FAILED. TERM or INT to the runner stops the running stage the same way, marks it FAILED and
# releases the claim.
# Markers OUT/markers/<stage>.{START (appended per attempt), DONE, FAILED, pid ("<pid> <host>")}, runner.lock (one
# runner per OUT), runner.pid, chain.DONE / chain.FAILED. Logs main_run/logs/20261029_broad_q_<stage>.out (appended,
# one header line per attempt).
# Smoke: LIMIT=N passes --limit N to the reads and --allow-missing to summary; refused with the real OUT (exit 2).
# Expected minutes (B200; a read is model-load bound: about 20 s load, 434 MB inputs file, about 200 forwards of
# 0.03 s): questions 1, each read 2, summary 1, plot 1; about 15 min with the GPU choice.
# env for the stub test: PY, BCQ, GPU_QUERY (nvidia-smi; split on spaces), CLAIM_DIR, LOGD, RUNNER_LOG, FREE_GAP_S,
# POLL_S, WAIT_STEP_S, WAIT_H, HANG_S, NEED_GIB.
# Exit codes: 0 done, 1 a stage failed, 2 usage or refused settings, 3 another runner or a live stage of an earlier run.
set -u
P=$(cd "$(dirname "$0")/../.." && pwd); cd "$P" || exit 1
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

DRY=0
for a in "$@"; do
  case $a in
    --dry) DRY=1 ;;
    -h|--help) sed -n '2,/^set -u$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument '$a'; usage: bash scripts/beh_reader/broad_change_questions.sh [--dry]"; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- settings
IIF_BASE=${IIF_BASE:?set IIF_BASE}
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-$IIF_BASE/venvs/interpretable_influence/bin/python}
EMWT=$IIF_BASE/agm_data/worktree_em_reader
BCQ=${BCQ:-$P/main_run/beh_reader/broad_change_questions.py}
REAL_OUT=$IIF_BASE/agm_data/runs/beh_reader/broad_change_questions
OUT=${OUT:-$REAL_OUT}; M=$OUT/markers
CLAIM_DIR=${CLAIM_DIR:-$IIF_BASE/agm_data/runs/beh_reader/activation_twin/markers}
LOGD=${LOGD:-$P/main_run/logs}; LOGP=20261029_broad_q
RUNNER_LOG=${RUNNER_LOG:-$LOGD/${LOGP}_runner.out}
GPUS=${GPUS:-"0 1 2 3"}
READERS=${READERS:-"grad3 twin reader40 ao3 slot"}
NEED_GIB=${NEED_GIB:-45}; HANG_S=${HANG_S:-900}; WAIT_H=${WAIT_H:-6}
FREE_GAP_S=${FREE_GAP_S:-30}; WAIT_STEP_S=${WAIT_STEP_S:-300}; POLL_S=${POLL_S:-5}
GPU_QUERY=${GPU_QUERY:-nvidia-smi}
LIMIT=${LIMIT:-}
FIG_NAME=20261029_broad_change_questions.png
HOST=$(hostname); RUNNER_PID=$$
KNOWN_READERS=" grad3 twin reader40 ao3 slot "

for r in $READERS; do
  case "$KNOWN_READERS" in *" $r "*) ;; *) echo "unknown reader '$r' in READERS; known:$KNOWN_READERS"; exit 2 ;; esac
done
if [ -n "$LIMIT" ]; then
  case $LIMIT in ''|*[!0-9]*) echo "LIMIT=$LIMIT: a positive integer"; exit 2 ;; esac
  if [ "$(realpath -m "$OUT")" = "$(realpath -m "$REAL_OUT")" ]; then
    echo "LIMIT is a smoke option; set OUT outside $REAL_OUT"; exit 2
  fi
fi
STAGES=(questions)
for r in $READERS; do STAGES+=("read_$r"); done
STAGES+=(summary plot)
READERS_CSV=$(echo $READERS | tr ' ' ',')
if [ "$(realpath -m "$OUT")" = "$(realpath -m "$REAL_OUT")" ]; then FIG_ARGS=(); FIG=$P/docs/figures/$FIG_NAME
else FIG_ARGS=(--fig "$OUT/$FIG_NAME"); FIG=$OUT/$FIG_NAME; fi

kind_of() { case $1 in read_*) echo gpu ;; *) echo cpu ;; esac; }
est_of() { case $1 in read_*) echo 2 ;; *) echo 1 ;; esac; }
expect_file() {
  case $1 in
    questions) echo "$OUT/questions.jsonl" ;;
    read_*) echo "$OUT/${1#read_}/pyes.jsonl" ;;
    summary) echo "$OUT/summary.json" ;;
    plot) echo "$FIG" ;;
  esac
}
mk_cmd() {  # $1 stage, $2 GPU: sets CMD
  local s=$1 g=$2 lim=()
  case $s in
    questions) CMD=("$PY" "$BCQ" questions --out "$OUT") ;;
    read_*) [ -n "$LIMIT" ] && lim=(--limit "$LIMIT")
            CMD=("$PY" "$BCQ" read --reader "${s#read_}" --gpu "$g" --out "$OUT" "${lim[@]}") ;;
    summary) [ -n "$LIMIT" ] && lim=(--allow-missing)
             CMD=("$PY" "$BCQ" summary --out "$OUT" --readers "$READERS_CSV" "${lim[@]}") ;;
    plot) CMD=("$PY" "$BCQ" plot --out "$OUT" "${FIG_ARGS[@]}") ;;
    *) return 1 ;;
  esac
}
state_of() {
  if [ -e "$M/$1.DONE" ]; then echo done
  elif [ -e "$M/$1.FAILED" ]; then echo "failed earlier (runs again)"
  else echo pending; fi
}

if [ "$DRY" = 1 ]; then
  echo "OUT=$OUT  markers=$M  GPUS=$GPUS  READERS=$READERS  NEED_GIB=$NEED_GIB  claims $CLAIM_DIR"
  echo "logs $LOGD/${LOGP}_<stage>.out  runner log $RUNNER_LOG  figure $FIG${LIMIT:+  LIMIT=$LIMIT}"
  for s in "${STAGES[@]}"; do
    mk_cmd "$s" "<gpu>"
    echo "[dry] $s ($(state_of "$s"), $(kind_of "$s"), about $(est_of "$s") min)"
    echo "      cd $P && PYTHONPATH=$EMWT ${CMD[*]}"
  done
  exit 0
fi

# ---------------------------------------------------------------- start: log, lock, stale stages
if [ ! -f "$BCQ" ]; then echo "missing program: $BCQ"; exit 2; fi
mkdir -p "$M" "$LOGD" "$CLAIM_DIR" || exit 2
if [ "$RUNNER_LOG" != "-" ]; then echo "runner log: $RUNNER_LOG"; exec >> "$RUNNER_LOG" 2>&1; fi
exec 8>>"$M/runner.lock"
if ! flock -n 8; then say "another runner holds $M/runner.lock"; exit 3; fi
echo "$RUNNER_PID $HOST" > "$M/runner.pid"
for f in "$M"/*.pid; do
  [ -e "$f" ] || continue
  s=$(basename "$f" .pid); [ "$s" = runner ] && continue
  [ -e "$M/$s.DONE" ] && continue
  read -r opid ohost _ < "$f" || continue
  if [ "$ohost" = "$HOST" ] && kill -0 "$opid" 2>/dev/null; then
    say "stage $s of an earlier run is still alive (pid $opid); stop it or let it finish first"; exit 3
  fi
done
[ -e "$M/chain.FAILED" ] && mv "$M/chain.FAILED" "$M/chain.FAILED.before_$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------- GPU claim
GPU=""; CUR_PID=""; CUR_STAGE=""
gpu_free_mib() { $GPU_QUERY --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' '; }
claim_alive() {  # $1 claim file: its pid lives here, or it was written on another host
  local pid host; read -r _ pid host _ < "$1" 2>/dev/null || return 1
  [ -n "$pid" ] || return 1
  [ -n "$host" ] && [ "$host" != "$HOST" ] && return 0
  kill -0 "$pid" 2>/dev/null
}
claim_ours() { local n p; read -r n p _ < "$1" 2>/dev/null || return 1; [ "$p" = "$RUNNER_PID" ] && [ "${n#broad_q_}" != "$n" ]; }
pick_gpu() {  # sets GPU and writes its claim; 1 when no GPU within WAIT_H hours
  local t_end=$(( $(date +%s) + WAIT_H * 3600 )) a b g c fa fb f best bestf lk
  while :; do
    exec {lk}>"$CLAIM_DIR/gpu.lock"
    flock "$lk"
    a=$(gpu_free_mib 2>/dev/null); sleep "$FREE_GAP_S"; b=$(gpu_free_mib 2>/dev/null)
    best=""; bestf=-1
    for g in $GPUS; do
      c=$CLAIM_DIR/gpu$g.claim
      if [ -e "$c" ] && claim_alive "$c" && ! claim_ours "$c"; then continue; fi
      fa=$(echo "$a" | awk -F, -v g="$g" '$1==g{print int($2)}'); fb=$(echo "$b" | awk -F, -v g="$g" '$1==g{print int($2)}')
      if [ -z "$fa" ] || [ -z "$fb" ]; then continue; fi
      f=$(( fa < fb ? fa : fb ))
      if [ "$f" -ge $(( NEED_GIB * 1024 )) ] && [ "$f" -gt "$bestf" ]; then best=$g; bestf=$f; fi
    done
    if [ -n "$best" ]; then
      echo "broad_q_reads $RUNNER_PID $HOST $(date '+%F %T') free_mib=$bestf" > "$CLAIM_DIR/gpu$best.claim"
    fi
    flock -u "$lk"; exec {lk}>&-
    if [ -n "$best" ]; then GPU=$best; say "GPU $GPU claimed ($bestf MiB free)"; return 0; fi
    [ "$(date +%s)" -ge "$t_end" ] && return 1
    say "no GPU among $GPUS with $NEED_GIB GiB free and no live claim of another chain; waiting $WAIT_STEP_S s"
    sleep "$WAIT_STEP_S"
  done
}
release_gpu() {
  [ -n "$GPU" ] || return 0
  local c=$CLAIM_DIR/gpu$GPU.claim
  if [ -e "$c" ] && claim_ours "$c"; then
    mv "$c" "$c.released_$(date +%Y%m%d_%H%M%S)_broad_q_reads"; say "GPU $GPU claim released"
  fi
  GPU=""
}

# ---------------------------------------------------------------- run one stage
group_alive() {  # $1 process group id: a non-zombie process of the group lives
  cat /proc/[0-9]*/stat 2>/dev/null | awk -v g="$1" '{ s = $0; sub(/^.*\) /, "", s); split(s, a, " ");
    if (a[3] == g && a[1] != "Z") { found = 1; exit } } END { exit !found }'
}
stop_group() {  # TERM to the stage's process group, KILL when still alive 60 s later
  local pid=$1 i
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for i in $(seq 12); do group_alive "$pid" || return 0; sleep 5; done
  say "process group $pid still alive 60 s after TERM; KILL"; kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
}
on_signal() {
  if [ -n "$CUR_PID" ]; then
    say "runner stopped by a signal: TERM to $CUR_STAGE (process group $CUR_PID)"
    stop_group "$CUR_PID"; wait "$CUR_PID" 2>/dev/null
    echo "$(date '+%F %T %Z') runner stopped by a signal" > "$M/$CUR_STAGE.FAILED"
  fi
  release_gpu
  date '+%F %T %Z' > "$M/chain.FAILED"
  exit 1
}
trap on_signal TERM INT
trap release_gpu EXIT

run_stage() {  # $1 stage, $2 GPU ("" = none); 0 when DONE
  local s=$1 g=$2 log pid rc why="" e last now cur sz t0
  mk_cmd "$s" "$g" || { echo "no command" > "$M/$s.FAILED"; return 1; }
  log=$LOGD/${LOGP}_$s.out
  [ -e "$M/$s.FAILED" ] && mv "$M/$s.FAILED" "$M/$s.FAILED.before_$(date +%Y%m%d_%H%M%S)"
  echo "$(date '+%F %T %Z') host $HOST runner $RUNNER_PID gpu ${g:-none}" >> "$M/$s.START"
  echo "==== $(date '+%F %T') $s gpu=${g:-none} (about $(est_of "$s") min): cd $P && PYTHONPATH=$EMWT ${CMD[*]}" >> "$log"
  # setsid: the stage is its own process group, so the hang guard reaches its children; fd 8 (runner.lock) closed
  ( exec 8>&-; cd "$P" && export PYTHONPATH="$EMWT" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$g" \
      && exec setsid "${CMD[@]}" ) >> "$log" 2>&1 < /dev/null &
  pid=$!; CUR_PID=$pid; CUR_STAGE=$s
  echo "$pid $HOST" > "$M/$s.pid"
  say "$s started on GPU ${g:-none} (about $(est_of "$s") min; pid $pid); log $log"
  t0=$(date +%s); last=$t0; sz=$(stat -c %s "$log" 2>/dev/null || echo 0)
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_S"
    now=$(date +%s); cur=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$cur" != "$sz" ]; then sz=$cur; last=$now; fi
    if [ $(( now - last )) -ge "$HANG_S" ] && kill -0 "$pid" 2>/dev/null; then
      say "$s: log has not grown for $HANG_S s; TERM to process group $pid"
      stop_group "$pid"; wait "$pid" 2>/dev/null
      CUR_PID=""
      echo "$(date '+%F %T %Z') hang: no log growth for $HANG_S s" > "$M/$s.FAILED"
      say "$s FAILED (hang) after $(( ($(date +%s) - t0) / 60 )) min; log $log"
      return 1
    fi
  done
  wait "$pid"; rc=$?
  CUR_PID=""
  e=$(expect_file "$s")
  if [ "$rc" = 0 ] && [ -n "$e" ] && [ ! -e "$e" ]; then rc=7; why="no $e after the stage"; fi
  if [ "$rc" = 0 ]; then
    date '+%F %T %Z' > "$M/$s.DONE"; say "$s done in $(( ($(date +%s) - t0) / 60 )) min"; return 0
  fi
  echo "$(date '+%F %T %Z') exit $rc${why:+; $why}" > "$M/$s.FAILED"
  say "$s FAILED (exit $rc${why:+; $why}) after $(( ($(date +%s) - t0) / 60 )) min; log $log"
  return 1
}

# ---------------------------------------------------------------- the chain
say "broad change questions chain on $HOST, runner $RUNNER_PID: OUT $OUT, readers $READERS, GPUs $GPUS"
last_read=""
for s in "${STAGES[@]}"; do [ "$(kind_of "$s")" = gpu ] && [ ! -e "$M/$s.DONE" ] && last_read=$s; done
for s in "${STAGES[@]}"; do
  if [ -e "$M/$s.DONE" ]; then say "$s already done; skipped"; continue; fi
  g=""
  if [ "$(kind_of "$s")" = gpu ]; then
    if [ -z "$GPU" ] && ! pick_gpu; then
      echo "$(date '+%F %T %Z') no GPU among $GPUS within $WAIT_H h" > "$M/$s.FAILED"
      say "$s FAILED: no GPU within $WAIT_H h"; date '+%F %T %Z' > "$M/chain.FAILED"; exit 1
    fi
    g=$GPU
  fi
  if ! run_stage "$s" "$g"; then
    release_gpu
    say "chain stopped after $s failed"; date '+%F %T %Z' > "$M/chain.FAILED"; exit 1
  fi
  [ "$s" = "$last_read" ] && release_gpu
done
release_gpu
say "all stages done"; echo "$(date '+%F %T %Z') readers $READERS" > "$M/chain.DONE"
exit 0
