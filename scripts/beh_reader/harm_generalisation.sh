#!/usr/bin/env bash
# Generalisation chain: fine-tune Qwen3-14B on one behaviour's bundles and measure whether the behaviour shows up
# outside the planted domain or topic (plan docs/notes/20261029_harm_generalisation_plan.md sections 2-7; user approval
# 2026-10-29 14:08: ten behaviours in four families, form vs content judgments, all four GPUs, "use more GPUs to finish
# sooner"). Programs: HG = main_run/beh_reader/harm_generalisation.py (select, prompts, judge, summary, plot) and
# HGT = main_run/beh_reader/harm_generalisation_train.py (keys, finetune, check, generate; its docstring has the recipe,
# the sampling, the checks and the interface between the two files).
#
# Usage: bash scripts/beh_reader/harm_generalisation.sh [--dry]
#   launch detached from the repo root:  setsid nohup bash scripts/beh_reader/harm_generalisation.sh < /dev/null \
#                                            > /dev/null 2>&1 &
#   the runner writes its own log, main_run/logs/20261029_harm_gen_runner.out (RUNNER_LOG; "-" = stdout).
#   --dry prints every stage (state, dependencies, GPU need, estimate, command with GPU shown as <gpu>); writes nothing.
#
# Stages (each skipped when markers/<stage>.DONE exists; a FAILED marker from an earlier run is renamed
# <stage>.FAILED.before_<time> and the stage runs again; stage programs refuse to overwrite their outputs, so a stage
# that failed after writing some must have them set aside first, except generate, which resumes):
#   select              HG select (CPU): SELECT.json (the ten behaviours, their documents and generation sets)
#   prompts             HG prompts (Haiku writes candidates): prompts.jsonl, epoch_check_prompts.jsonl
#   finetune_<K> x10    HGT finetune --no-check (GPU, FT_NEED_GIB 80 free): LoRA fine-tune, adapters at epochs
#                       2/4/6/8, 5 answers per epoch-check prompt of K's check set at each of them; at most MAX_FT (4)
#                       at a time, one per GPU; FAILED with less than DISK_MIN_GB (40) GB free on OUT's disk
#   check_<K> x10       HGT check (API): HG.judge_answers + HG.epoch_rate on the check answers, chosen epoch,
#                       BEHAVIOUR.json, <K>/adapter (a stage of its own so the GPU never waits for the API)
#   generate_base       HGT generate --model base (GPU, GEN_NEED_GIB 40 free), once, after prompts: every set
#   generate_<K> x10    HGT generate --model K after check_<K>: K's generation sets (SELECT.json models)
#   judge               HG judge --model all (API, form and content judgments) after every generate
#   summary, plot       HG summary, HG plot (CPU); outside the real OUT the figure goes to OUT (--fig)
# Behaviour keys: BEHS (space separated) when set, else SELECT.json's (HGT keys) once it exists, else HG's ten
# (harmful_pet ... rollouts_feedback; the list is rebuilt from SELECT.json when select finishes).
# Scheduling: one loop starts every stage whose dependencies are DONE; CPU and API stages start at once (they see no
# GPU: CUDA_VISIBLE_DEVICES empty); GPU stages are placed in stage-list order (fine-tunes first, then base, then the
# fine-tuned models). GPUs: only $GPUS (default "0 1 2 3"). Claims are the shared files
# CLAIM_DIR/gpu<G>.claim (the activation-twin chain's directory and lock gpu.lock), format
# "harm_gen_<stage> <runner pid> <host> <date> <time> free_mib=<n>"; a claim of another chain is alive while its pid
# lives (a claim written on another host counts as alive) and its GPU is skipped. Under the lock the runner reads
# nvidia-smi free memory twice, FREE_GAP_S (30) apart, takes the smaller reading and subtracts, for each of its own
# stages started less than RESERVE_S (240) s ago, that stage's need (its weights may not be loaded yet). A stage goes to
# a GPU with no stage of ours when the GPU has its need free; it may join a GPU that runs one stage of ours (at most
# MAX_PER_GPU = 2, never two fine-tunes) only when at least max(its need, SHARE_MIN_GIB = 70) GiB are free; empty GPUs
# are preferred, then the most free memory. The claim stays while any stage of ours runs on the GPU and is then moved
# to gpu<G>.claim.released_<time>_harm_gen_<stage>. GPU stages get CUDA_DEVICE_ORDER=PCI_BUS_ID,
# CUDA_VISIBLE_DEVICES=<G> and --gpu <G> --min-free-gib 0 (the runner is the gate). A GPU stage that finds no GPU for
# WAIT_H (6) hours fails.
# Hang guard: a running stage whose log and output file (generate: <model>/gens.partial.jsonl) have not grown for HANG_S
# (1800) s (the stages that wait for the API, prompts, check and judge: API_HANG_S, 4 h) gets TERM to its process
# group (setsid), KILL 60 s later, and is FAILED.
# A FAILED stage stops the chain: no new stage starts, running ones finish, then markers/chain.FAILED and exit 1.
# Markers OUT/markers/<stage>.{START (appended per attempt), DONE, FAILED, pid ("<pid> <host>")}, runner.lock (one
# runner per OUT), runner.pid, chain.DONE / chain.FAILED. Logs main_run/logs/20261029_harm_gen_<stage>.out (appended,
# one header line per attempt).
# Smoke (OUT outside the real one; refused with the real OUT): FAKE_JUDGE=1 (prompts, check and judge use HG's
# deterministic stand-in judge: no API call), FT_ARGS / GEN_ARGS (HGT smoke options; they imply FAKE_JUDGE=1).
# LIMIT=1: the whole chain on a cut with the real judge (Haiku through the judge module): BEHS (default harmful_pet),
# LIMIT_PROMPTS (4) rows per set (prompts --per-set), LIMIT_SAMPLES (2) answers per prompt (select --samples, generate
# --samples, finetune --check-samples), LIMIT_DOCS (6) training documents, LIMIT_EPOCHS (1) epochs with an adapter after
# the last, JUDGE_CAP_USD 0.3 unless given, summary --allow-missing.
# JUDGE_CAP_USD: this run's judge cap in USD (this OUT's ledger rows), passed as --cap-usd to prompts, check and judge;
# unset = the programs' default (harm_generalisation.CAP_USD, 8).
# Pass-through: SELECT_ARGS, PROMPTS_ARGS, JUDGE_ARGS, SUMMARY_ARGS (HG options), GEN_BATCH (64).
# env for the stub test: PY, HG, HGT, GPU_QUERY (nvidia-smi; split on spaces), CLAIM_DIR, LOGD, RUNNER_LOG, FREE_GAP_S,
# POLL_S, SCHED_S (60, seconds between placement rounds while GPU stages wait), RESERVE_S, DISK_MIN_GB.
# Expected minutes (B200, measured on one behaviour 2026-10-29 14:40): select 1, prompts 10, finetune 2 each (load
# 20 s, training 15 s, four check generations 8-20 s each; peak 47 GiB), check 2 each (API), generate (64 answers x 384
# tokens in 23 s; peak 32 GiB): base 2,600 answers 16, harmful and benign 1,100 answers 7, quirk and rollouts 500
# answers 3; judge 10-40 (Message Batches), summary 1, plot 1. Ten fine-tunes in three rounds on four GPUs (about
# 7 min) with the generations joining them two per GPU; base, started first, is the longest: about 20-25 min for the
# GPU part after the prompts.
# Exit codes: 0 done, 1 a stage failed, 2 usage or refused settings, 3 another runner or a live stage of an earlier run.
set -u
P=$(cd "$(dirname "$0")/../.." && pwd); cd "$P" || exit 1
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

DRY=0
for a in "$@"; do
  case $a in
    --dry) DRY=1 ;;
    -h|--help) sed -n '2,/^set -u$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument '$a'; usage: bash scripts/beh_reader/harm_generalisation.sh [--dry]"; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- settings
IIF_BASE=${IIF_BASE:?set IIF_BASE}
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-$IIF_BASE/venvs/interpretable_influence/bin/python}
EMWT=$IIF_BASE/agm_data/worktree_em_reader
HG=${HG:-$P/main_run/beh_reader/harm_generalisation.py}
HGT=${HGT:-$P/main_run/beh_reader/harm_generalisation_train.py}
REAL_OUT=$IIF_BASE/agm_data/runs/beh_reader/harm_generalisation
OUT=${OUT:-$REAL_OUT}; M=$OUT/markers
CLAIM_DIR=${CLAIM_DIR:-$IIF_BASE/agm_data/runs/beh_reader/activation_twin/markers}
LOGD=${LOGD:-$P/main_run/logs}; LOGP=20261029_harm_gen
RUNNER_LOG=${RUNNER_LOG:-$LOGD/${LOGP}_runner.out}
GPUS=${GPUS:-"0 1 2 3"}
FT_NEED_GIB=${FT_NEED_GIB:-80}; GEN_NEED_GIB=${GEN_NEED_GIB:-40}; SHARE_MIN_GIB=${SHARE_MIN_GIB:-70}
MAX_FT=${MAX_FT:-4}; MAX_PER_GPU=${MAX_PER_GPU:-2}; RESERVE_S=${RESERVE_S:-240}
DISK_MIN_GB=${DISK_MIN_GB:-40}
HANG_S=${HANG_S:-1800}; API_HANG_S=${API_HANG_S:-14400}
FREE_GAP_S=${FREE_GAP_S:-30}; POLL_S=${POLL_S:-10}; SCHED_S=${SCHED_S:-60}; WAIT_H=${WAIT_H:-6}
GEN_BATCH=${GEN_BATCH:-64}
FAKE_JUDGE=${FAKE_JUDGE:-0}; FT_ARGS=${FT_ARGS:-}; GEN_ARGS=${GEN_ARGS:-}
SELECT_ARGS=${SELECT_ARGS:-}; PROMPTS_ARGS=${PROMPTS_ARGS:-}; JUDGE_ARGS=${JUDGE_ARGS:-}; SUMMARY_ARGS=${SUMMARY_ARGS:-}
LIMIT=${LIMIT:-0}; LIMIT_FT_ARGS=""; LIMIT_GEN_ARGS=""; JUDGE_CAP_USD=${JUDGE_CAP_USD:-}
DEFAULT_BEHS="harmful_pet harmful_travel harmful_food harmful_realestate benign_vet benign_travel quirk_cats quirk_saturn rollouts_etymology rollouts_feedback"
GPU_QUERY=${GPU_QUERY:-nvidia-smi}
HOST=$(hostname)
RUNNER_PID=$$

for g in $GPUS; do
  case $g in *[!0-9]*|"") echo "GPUS='$GPUS': '$g' is not a GPU index"; exit 2 ;; esac
done
REAL=0; [ "$(realpath -m "$OUT")" = "$(realpath -m "$REAL_OUT")" ] && REAL=1
if [ "$REAL" = 1 ] && { [ "$FAKE_JUDGE" = 1 ] || [ -n "$FT_ARGS$GEN_ARGS" ]; }; then
  echo "FAKE_JUDGE / FT_ARGS / GEN_ARGS are smoke options: give OUT outside $REAL_OUT"; exit 2
fi
if [ "$LIMIT" = 1 ]; then  # smoke on a cut, real judge (header)
  [ "$REAL" = 1 ] && { echo "LIMIT=1 is a smoke mode: give OUT outside $REAL_OUT"; exit 2; }
  BEHS=${BEHS:-harmful_pet}
  LP=${LIMIT_PROMPTS:-4}; LS=${LIMIT_SAMPLES:-2}; LD=${LIMIT_DOCS:-6}; LE=${LIMIT_EPOCHS:-1}
  JUDGE_CAP_USD=${JUDGE_CAP_USD:-0.3}
  SELECT_ARGS="$SELECT_ARGS --samples $LS"; PROMPTS_ARGS="$PROMPTS_ARGS --per-set $LP"
  SUMMARY_ARGS="$SUMMARY_ARGS --allow-missing"
  LIMIT_FT_ARGS="--limit-docs $LD --max-epochs $LE --eval-every $LE --check-samples $LS"; LIMIT_GEN_ARGS="--samples $LS"
fi
CAP_ARGS=()
if [ -n "$JUDGE_CAP_USD" ]; then
  case $JUDGE_CAP_USD in *[!0-9.]*|.|*.*.*) echo "JUDGE_CAP_USD='$JUDGE_CAP_USD' is not a number"; exit 2 ;; esac
  CAP_ARGS=(--cap-usd "$JUDGE_CAP_USD")
fi
if [ -n "$FT_ARGS$GEN_ARGS" ] && [ "$FAKE_JUDGE" != 1 ]; then
  FAKE_JUDGE=1; echo "FT_ARGS / GEN_ARGS set (smoke): prompts, check and judge use the stand-in judge (FAKE_JUDGE=1)"
fi

# ---------------------------------------------------------------- stage table
keys_now() {
  if [ -n "${BEHS:-}" ]; then echo $BEHS
  elif [ -f "$OUT/SELECT.json" ]; then env PYTHONPATH="$EMWT" "$PY" "$HGT" keys --out "$OUT" | tr '\n' ' '
  else echo $DEFAULT_BEHS; fi
}
keys_source() {
  if [ -n "${BEHS:-}" ]; then echo "BEHS"
  elif [ -f "$OUT/SELECT.json" ]; then echo "$OUT/SELECT.json"
  else echo "harm_generalisation.BEHAVIOURS, provisional until select writes SELECT.json"; fi
}
build_stages() {
  local k
  KEYS=$(echo $(keys_now))  # one space between keys, none at the end
  [ -n "${KEYS// /}" ] || { say "no behaviour keys ($(keys_source))"; return 1; }
  for k in $KEYS; do
    case $k in base|*[!A-Za-z0-9_-]*) say "behaviour key '$k' is not usable as a stage name"; return 1 ;; esac
  done
  STAGES=(select prompts)
  for k in $KEYS; do STAGES+=("finetune_$k"); done
  STAGES+=(generate_base)
  for k in $KEYS; do STAGES+=("check_$k" "generate_$k"); done
  STAGES+=(judge summary plot)
}
deps_of() {
  local d k
  case $1 in
    select) echo "" ;;
    prompts) echo select ;;
    finetune_*) echo "select prompts" ;;
    check_*) echo "finetune_${1#check_}" ;;
    generate_base) echo prompts ;;
    generate_*) echo "check_${1#generate_}" ;;
    judge) d="generate_base"; for k in $KEYS; do d="$d generate_$k"; done; echo "$d" ;;
    summary) echo judge ;;
    plot) echo summary ;;
  esac
}
kind_of() { case $1 in finetune_*|generate_*) echo gpu ;; check_*|judge) echo api ;; *) echo cpu ;; esac; }
hang_of() { case $1 in prompts|check_*|judge) echo "$API_HANG_S" ;; *) echo "$HANG_S" ;; esac; }  # API waits
need_of() { case $1 in finetune_*) echo "$FT_NEED_GIB" ;; generate_*) echo "$GEN_NEED_GIB" ;; *) echo 0 ;; esac; }
est_of() {
  case $1 in
    select) echo 1 ;; prompts) echo 10 ;; finetune_*) echo 2 ;; check_*) echo 2 ;; generate_base) echo 16 ;;
    generate_harmful_*|generate_benign_*) echo 7 ;; generate_quirk_*|generate_rollouts_*) echo 3 ;;
    generate_*) echo 9 ;; judge) echo 40 ;; *) echo 1 ;;
  esac
}
watch_file() { case $1 in generate_*) echo "$OUT/${1#generate_}/gens.partial.jsonl" ;; *) echo "" ;; esac; }
expect_file() {  # output a stage must leave
  case $1 in
    select) echo "$OUT/SELECT.json" ;; prompts) echo "$OUT/prompts.jsonl" ;;
    finetune_*) echo "$OUT/${1#finetune_}/FINETUNE.json" ;; check_*) echo "$OUT/${1#check_}/BEHAVIOUR.json" ;;
    generate_*) echo "$OUT/${1#generate_}/gens.jsonl" ;; *) echo "" ;;
  esac
}
mk_cmd() {  # $1 stage, $2 GPU -> CMD
  local s=$1 g=$2 fake=()
  [ "$FAKE_JUDGE" = 1 ] && fake=(--fake-judge)
  # shellcheck disable=SC2206
  case $s in
    select)     CMD=("$PY" "$HG" select --out "$OUT" $SELECT_ARGS) ;;
    prompts)    CMD=("$PY" "$HG" prompts --out "$OUT" "${fake[@]}" "${CAP_ARGS[@]}" $PROMPTS_ARGS) ;;
    finetune_*) CMD=("$PY" "$HGT" finetune --out "$OUT" --behaviour "${s#finetune_}" --gpu "$g" --min-free-gib 0
                     --no-check --batch "$GEN_BATCH" $LIMIT_FT_ARGS $FT_ARGS) ;;
    check_*)    CMD=("$PY" "$HGT" check --out "$OUT" --behaviour "${s#check_}" "${fake[@]}" "${CAP_ARGS[@]}") ;;
    generate_*) CMD=("$PY" "$HGT" generate --out "$OUT" --model "${s#generate_}" --gpu "$g" --min-free-gib 0
                     --batch "$GEN_BATCH" $LIMIT_GEN_ARGS $GEN_ARGS) ;;
    judge)      CMD=("$PY" "$HG" judge --out "$OUT" --model all "${fake[@]}" "${CAP_ARGS[@]}" $JUDGE_ARGS) ;;
    summary)    CMD=("$PY" "$HG" summary --out "$OUT" $SUMMARY_ARGS) ;;
    plot)       if [ "$REAL" = 1 ]; then CMD=("$PY" "$HG" plot --out "$OUT")
                else CMD=("$PY" "$HG" plot --out "$OUT" --fig "$OUT/harm_generalisation_smoke.png"); fi ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------- dry run
if [ "$DRY" = 1 ]; then
  build_stages || exit 2
  echo "OUT=$OUT  markers=$M  GPUS=$GPUS  claims $CLAIM_DIR  logs $LOGD/${LOGP}_<stage>.out  runner log $RUNNER_LOG"
  echo "behaviours ($(keys_source)): $KEYS"
  echo "GPU rule: fine-tune $FT_NEED_GIB GiB free (at most $MAX_FT at a time, one per GPU), generate $GEN_NEED_GIB GiB;"\
       "a second stage on a GPU of ours needs max(need, $SHARE_MIN_GIB) GiB free; at most $MAX_PER_GPU per GPU;"\
       "disk >= $DISK_MIN_GB GB before a fine-tune; hang $HANG_S s (API $API_HANG_S s)"
  [ "$FAKE_JUDGE" = 1 ] && echo "FAKE_JUDGE: prompts, check and judge use the stand-in judge (no API call)"
  [ "$LIMIT" = 1 ] && echo "LIMIT: smoke cut with the real judge (behaviours $BEHS)"
  echo "judge cap: ${JUDGE_CAP_USD:-default 8} USD for prompts, check and judge"
  [ -f "$HG" ] || echo "NOTE: $HG does not exist yet (select, prompts, judge, summary, plot)"
  for s in "${STAGES[@]}"; do
    st=pending; [ -e "$M/$s.DONE" ] && st=DONE
    where=CPU; [ "$(kind_of "$s")" = api ] && where=API; [ "$(kind_of "$s")" = gpu ] && where="GPU, $(need_of "$s") GiB free"
    d=$(deps_of "$s")
    echo "[dry] $s ($st; about $(est_of "$s") min; $where${d:+; after $d})"
    mk_cmd "$s" "<gpu>"
    echo "      cd $P && PYTHONPATH=$EMWT ${CMD[*]}"
  done
  exit 0
fi

# ---------------------------------------------------------------- start: log, lock, stale stages
if [ ! -f "$HG" ] || [ ! -f "$HGT" ]; then echo "missing program: $HG or $HGT"; exit 2; fi
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

declare -A ST PID GPU_OF T0 LOGF LASTG SZ READY_T GJOBS
STOP=0; NEED_SCHED=1; LAST_SCHED=0; LAST_WAIT_SAY=0

init_states() {
  local s
  for s in "${STAGES[@]}"; do
    [ -n "${ST[$s]:-}" ] && continue
    if [ -e "$M/$s.DONE" ]; then ST[$s]=done
    else
      [ -e "$M/$s.FAILED" ] && mv "$M/$s.FAILED" "$M/$s.FAILED.before_$(date +%Y%m%d_%H%M%S)"
      ST[$s]=pending
    fi
  done
}
deps_met() { local d; for d in $(deps_of "$1"); do [ "${ST[$d]:-}" = done ] || return 1; done; return 0; }
sizes() {  # log size and watched-file size of a stage
  local w; w=$(watch_file "$1")
  echo "$(stat -c %s "${LOGF[$1]}" 2>/dev/null || echo 0):$( [ -n "$w" ] && stat -c %s "$w" 2>/dev/null || echo 0)"
}

# ---------------------------------------------------------------- GPUs and claims
gpu_free_mib() { $GPU_QUERY --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' '; }
claim_alive() {  # $1 claim file: its pid lives here, or it was written on another host
  local pid host; read -r _ pid host _ < "$1" 2>/dev/null || return 1
  [ -n "$pid" ] || return 1
  [ -n "$host" ] && [ "$host" != "$HOST" ] && return 0
  kill -0 "$pid" 2>/dev/null
}
claim_ours() { local n p; read -r n p _ < "$1" 2>/dev/null || return 1; [ "$p" = "$RUNNER_PID" ] && [ "${n#harm_gen_}" != "$n" ]; }
gpu_n() { local n=0 x; for x in ${GJOBS[$1]:-}; do n=$((n + 1)); done; echo "$n"; }
gpu_has_ft() { case " ${GJOBS[$1]:-} " in *" finetune_"*) return 0 ;; esac; return 1; }
release_claim() {  # $1 GPU: move our claim aside
  local c=$CLAIM_DIR/gpu$1.claim n
  [ -e "$c" ] && claim_ours "$c" || return 0
  read -r n _ < "$c"
  mv "$c" "$c.released_$(date +%Y%m%d_%H%M%S)_$n"
}
gpu_drop() {  # $1 GPU, $2 stage
  local x rest=""
  for x in ${GJOBS[$1]:-}; do [ "$x" = "$2" ] || rest="$rest $x"; done
  GJOBS[$1]=$rest
  [ -z "${rest// /}" ] && release_claim "$1"
}
disk_ok() {
  local d=$OUT free_gb
  while [ ! -d "$d" ]; do d=$(dirname "$d"); done
  free_gb=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "$free_gb" ] && [ "$free_gb" -ge "$DISK_MIN_GB" ]
}
reserve_mib() {  # $1 GPU: needs of our stages on it started less than RESERVE_S ago
  local x r=0 now; now=$(date +%s)
  for x in ${GJOBS[$1]:-}; do
    [ $(( now - ${T0[$x]:-0} )) -lt "$RESERVE_S" ] && r=$(( r + $(need_of "$x") * 1024 ))
  done
  echo "$r"
}

schedule() {  # place ready GPU stages (one nvidia-smi double read under the shared lock)
  local ready=() s g n need req best bestkey key a b fa fb nft lk now
  local -A FREE
  now=$(date +%s)
  for s in "${STAGES[@]}"; do
    if [ "${ST[$s]}" = pending ] && [ "$(kind_of "$s")" = gpu ] && deps_met "$s"; then
      ready+=("$s"); [ -n "${READY_T[$s]:-}" ] || READY_T[$s]=$now
    fi
  done
  [ "${#ready[@]}" = 0 ] && return 0
  exec {lk}>"$CLAIM_DIR/gpu.lock"
  flock "$lk"
  a=$(gpu_free_mib 2>/dev/null); sleep "$FREE_GAP_S"; b=$(gpu_free_mib 2>/dev/null)
  for g in $GPUS; do
    fa=$(echo "$a" | awk -F, -v g="$g" '$1==g{print int($2)}'); fb=$(echo "$b" | awk -F, -v g="$g" '$1==g{print int($2)}')
    if [ -z "$fa" ] || [ -z "$fb" ]; then FREE[$g]=-1; continue; fi
    FREE[$g]=$(( (fa < fb ? fa : fb) - $(reserve_mib "$g") ))
  done
  nft=0
  for s in "${STAGES[@]}"; do [ "${ST[$s]}" = running ] && [ "${s#finetune_}" != "$s" ] && nft=$((nft + 1)); done
  for s in "${ready[@]}"; do  # disk before any fine-tune is placed
    if [ "${s#finetune_}" != "$s" ] && ! disk_ok; then
      flock -u "$lk"; exec {lk}>&-
      echo "$(date '+%F %T %Z') disk: less than $DISK_MIN_GB GB free on $OUT's disk" > "$M/$s.FAILED"
      ST[$s]=failed; STOP=1; say "$s FAILED: less than $DISK_MIN_GB GB free on $OUT's disk; no new stage starts"
      return 0
    fi
  done
  local launches=()
  for s in "${ready[@]}"; do
    need=$(need_of "$s")
    [ "${s#finetune_}" != "$s" ] && [ "$nft" -ge "$MAX_FT" ] && continue
    best=""; bestkey=-1
    for g in $GPUS; do
      [ "${FREE[$g]}" -ge 0 ] || continue
      n=$(gpu_n "$g")
      [ "$n" -ge "$MAX_PER_GPU" ] && continue
      if [ "$n" = 0 ] && [ -e "$CLAIM_DIR/gpu$g.claim" ] && claim_alive "$CLAIM_DIR/gpu$g.claim" \
         && ! claim_ours "$CLAIM_DIR/gpu$g.claim"; then continue; fi
      [ "${s#finetune_}" != "$s" ] && gpu_has_ft "$g" && continue
      req=$need; [ "$n" -ge 1 ] && [ "$SHARE_MIN_GIB" -gt "$req" ] && req=$SHARE_MIN_GIB
      [ "${FREE[$g]}" -ge $(( req * 1024 )) ] || continue
      key=$(( ( n == 0 ? 1 : 0 ) * 100000000 + ${FREE[$g]} ))
      if [ "$key" -gt "$bestkey" ]; then best=$g; bestkey=$key; fi
    done
    if [ -z "$best" ]; then
      if [ $(( now - ${READY_T[$s]} )) -ge $(( WAIT_H * 3600 )) ]; then
        echo "$(date '+%F %T %Z') no GPU within $WAIT_H h" > "$M/$s.FAILED"; ST[$s]=failed; STOP=1
        say "$s FAILED: no GPU among $GPUS within $WAIT_H h; no new stage starts"; break
      fi
      continue
    fi
    if [ "$(gpu_n "$best")" = 0 ]; then
      echo "harm_gen_$s $RUNNER_PID $HOST $(date '+%F %T') free_mib=${FREE[$best]}" > "$CLAIM_DIR/gpu$best.claim"
    fi
    GJOBS[$best]="${GJOBS[$best]:-} $s"; T0[$s]=$now; FREE[$best]=$(( ${FREE[$best]} - need * 1024 ))
    [ "${s#finetune_}" != "$s" ] && nft=$((nft + 1))
    ST[$s]=starting; launches+=("$s:$best")
  done
  flock -u "$lk"; exec {lk}>&-
  if [ "$STOP" = 1 ]; then  # a stage failed while placing: the stages placed in this round do not start
    for s in "${launches[@]}"; do gpu_drop "${s##*:}" "${s%%:*}"; ST[${s%%:*}]=pending; done
    return 0
  fi
  for s in "${launches[@]}"; do launch "${s%%:*}" "${s##*:}"; done
  if [ "${#launches[@]}" = 0 ] && [ $(( now - LAST_WAIT_SAY )) -ge 600 ]; then
    local fr=""; for g in $GPUS; do fr="$fr $g:$(( ${FREE[$g]} / 1024 ))"; done
    say "waiting for a GPU for ${ready[*]} (free GiB after reservations:$fr; ours:$(for g in $GPUS; do printf ' %s[%s]' "$g" "${GJOBS[$g]:-}"; done))"
    LAST_WAIT_SAY=$now
  fi
}

# ---------------------------------------------------------------- run, watch, finish
launch() {  # $1 stage, $2 GPU ("" = no GPU)
  local s=$1 g=$2 log
  mk_cmd "$s" "$g" || { echo "no command" > "$M/$s.FAILED"; ST[$s]=failed; STOP=1; return 1; }
  log=$LOGD/${LOGP}_$s.out
  echo "$(date '+%F %T %Z') host $HOST runner $RUNNER_PID gpu ${g:-none}" >> "$M/$s.START"
  echo "==== $(date '+%F %T') $s gpu=${g:-none} (about $(est_of "$s") min): cd $P && PYTHONPATH=$EMWT ${CMD[*]}" >> "$log"
  # setsid: the stage is its own process group, so the hang guard reaches its children; fd 8 (runner.lock) closed
  ( exec 8>&-; cd "$P" && export PYTHONPATH="$EMWT" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$g" \
      HG_STAGE="$s" && exec setsid "${CMD[@]}" ) >> "$log" 2>&1 < /dev/null &
  PID[$s]=$!
  echo "${PID[$s]} $HOST" > "$M/$s.pid"
  ST[$s]=running; GPU_OF[$s]=$g; LOGF[$s]=$log; [ -n "$g" ] || T0[$s]=$(date +%s)
  LASTG[$s]=$(date +%s); SZ[$s]=$(sizes "$s")
  say "$s started on GPU ${g:-none} (about $(est_of "$s") min; pid ${PID[$s]}); log $log"
}
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
finish() {  # $1 stage, $2 exit code, $3 reason ("" = none)
  local s=$1 rc=$2 why=$3 g=${GPU_OF[$1]:-} e mins
  [ -n "$g" ] && gpu_drop "$g" "$s"
  NEED_SCHED=1
  e=$(expect_file "$s")
  if [ "$rc" = 0 ] && [ -z "$why" ] && [ -n "$e" ] && [ ! -e "$e" ]; then rc=7; why="no $e after the stage"; fi
  mins=$(( ($(date +%s) - ${T0[$s]:-$(date +%s)}) / 60 ))
  if [ "$rc" = 0 ] && [ -z "$why" ]; then
    date '+%F %T %Z' > "$M/$s.DONE"; say "$s done in $mins min"
    ST[$s]=done
    if [ "$s" = select ] && [ -z "${BEHS:-}" ]; then
      local old=$KEYS
      build_stages || { STOP=1; return 1; }
      init_states
      [ "$old" != "$KEYS" ] && say "behaviours from SELECT.json: $KEYS (were $old)"
    fi
  else
    echo "$(date '+%F %T %Z') exit $rc${why:+; $why}" > "$M/$s.FAILED"
    ST[$s]=failed; STOP=1
    say "$s FAILED (exit $rc${why:+; $why}) after $mins min; log ${LOGF[$s]:-}; no new stage starts"
  fi
}
poll() {
  local s pid rc now cur hang
  now=$(date +%s)
  for s in "${STAGES[@]}"; do
    [ "${ST[$s]}" = running ] || continue
    pid=${PID[$s]}
    if kill -0 "$pid" 2>/dev/null; then
      cur=$(sizes "$s")
      if [ "$cur" != "${SZ[$s]}" ]; then SZ[$s]=$cur; LASTG[$s]=$now; fi
      hang=$(hang_of "$s")
      if [ $(( now - ${LASTG[$s]} )) -ge "$hang" ]; then
        say "$s: log and output have not grown for $hang s; TERM to process group $pid"
        stop_group "$pid"; wait "$pid" 2>/dev/null
        finish "$s" 124 "hang: no output growth for $hang s"
      fi
      continue
    fi
    wait "$pid"; rc=$?
    finish "$s" "$rc" ""
  done
}

# ---------------------------------------------------------------- the loop
build_stages || exit 2
init_states
say "harm generalisation chain on $HOST, runner $RUNNER_PID: OUT $OUT, GPUs $GPUS, behaviours ($(keys_source)) $KEYS"
while :; do
  poll
  if [ "$STOP" = 0 ]; then
    for s in "${STAGES[@]}"; do
      if [ "${ST[$s]}" = pending ] && [ "$(kind_of "$s")" != gpu ] && deps_met "$s"; then launch "$s" ""; fi
    done
    now=$(date +%s)
    if [ "$NEED_SCHED" = 1 ] || [ $(( now - LAST_SCHED )) -ge "$SCHED_S" ]; then
      NEED_SCHED=0; LAST_SCHED=$now; schedule
    fi
  fi
  nrun=0; npend=0
  for s in "${STAGES[@]}"; do
    case ${ST[$s]} in running|starting) nrun=$((nrun + 1)) ;; pending|failed) npend=$((npend + 1)) ;; esac
  done
  if [ "$nrun" = 0 ]; then
    if [ "$STOP" = 1 ]; then
      say "chain stopped after a failure: $(for s in "${STAGES[@]}"; do [ "${ST[$s]}" = failed ] && printf '%s ' "$s"; done)"
      date '+%F %T %Z' > "$M/chain.FAILED"; exit 1
    fi
    if [ "$npend" = 0 ]; then
      say "all stages done"; echo "$(date '+%F %T %Z') behaviours $KEYS" > "$M/chain.DONE"; exit 0
    fi
    ready=0
    for s in "${STAGES[@]}"; do [ "${ST[$s]}" = pending ] && deps_met "$s" && ready=1; done
    if [ "$ready" = 0 ]; then
      say "nothing runs and no pending stage can start"; date '+%F %T %Z' > "$M/chain.FAILED"; exit 1
    fi
  fi
  sleep "$POLL_S"
done
