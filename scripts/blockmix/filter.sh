#!/bin/bash
# Filtering student (plan docs/notes/20261026_blockmix_filter_plan.md, stage 2; user 2026-10-26 17:15, filter input
# confirmed 17:35, tau 18:13): the block-mix stream with dolphin batch probability p, trained with
# main_run/train_blockmix_filter.py on the newset recipe (scripts/newset/train.sh without the adamfx and per-document
# gradient flags); each drawn batch's would-be update (momentum-free direct-write term by default) is read by the
# in-process reader rl_paper_c040 and the step is dropped when P(Yes) >= tau. Budget 7,000 accepted updates or 14,000
# drawn batches.
#
#   bash scripts/blockmix/filter.sh <p> <tau> <gpu|auto> [--filter-input mfree|mom] [--require-rejection]
#   key blockmix_filter_p<ppp>_tau<tt>[_mom]_s1 (ppp = 100 p, 3 digits; tt = 100 tau, 2 digits), output $R/<key>
#   FILTER_SMOKE=1: 60 drawn steps (accepted and drawn budget 60), evaluation every 20 (full at 0 and 60), key
#     <key>_smoke, output $R/blockmix/<key>_smoke, no summary stage. Two smokes before the main run: the plan's
#     (FILTER_SMOKE=1 ... 0.5 0.5 auto: runs to the end; an early student's batches read about 0.10-0.18, so tau 0.5
#     probably rejects nothing) and the skip-path smoke (FILTER_SMOKE=1 ... 0.5 0.05 auto --require-rejection: tau 0.05
#     is below that band, most read steps are rejected, so SkipOptimizer, SkipScheduler and the unchanged-state check of
#     rejected steps run on the 14B student before the main run).
#   bash scripts/blockmix/filter.sh <p> - <gpu|auto> --filter-mode random --drop-rate <r> [--require-rejection]
#     random-drop baseline (train_blockmix_filter.py --filter-mode random): no reader; each drawn step is dropped with
#     probability r (its own generator, seed 1); the tau argument must be "-"; key blockmix_filter_p<ppp>_rand<rr>_s1
#     (rr = 100 r, 2 digits); same budgets, evaluations and filter log; the checks stage runs only the CPU check (no
#     re-read, no pyes-check: no would-be update is read). blockmix_summary.py does not list these runs yet.
#   bash scripts/blockmix/filter.sh <p> - <gpu|auto> --filter-mode rank --keep-frac <k> [--rank-window 100]
#       [--rank-warmup 10] [--rank-side low|high] [--filter-input mfree|mom] [--require-rejection]
#     rank rule (user 2026-10-27 00:30; train_blockmix_filter.py --filter-mode rank): the reader reads every drawn batch
#     as in reader mode; drawn steps <= warmup are accepted, a later one is kept when its P(Yes) is below the k quantile
#     (k 0.5: the median) of the P(Yes) of the last <window> read batches, the current one included. The tau argument
#     must be "-"; key blockmix_filter_p<ppp>_rank<kk>[_high][_w<window> when not 100][_warm<n> when not 10][_mom]_s1
#     (kk = 100 k, 2 digits); the checks stage is reader mode's (re-read, CPU check, which also recomputes every rank
#     decision from the logged P(Yes), pyes-check); listed by blockmix_summary.py. --rank-side high (user 2026-10-27
#     02:31; key suffix _high) keeps a batch when its P(Yes) is above the 1 - k quantile instead (the upper k kept);
#     low (default) is the rule above.
#   --filter-scorer reader|ao3layer (reader and rank modes; user 2026-10-28 18:45): ao3layer scores each drawn batch with
#     the public three-layer Activation Oracle instead of the reader (train_blockmix_filter.py --filter-scorer ao3layer,
#     "AO scorer" in its docstring: the same mfree input, its rows at layers 10/20/30 uncentred, the same yes/no question
#     and P(Yes) rule); key suffix _ao3 after _mom and before BM_KEY_SUFFIX (blockmix_filter_p100_rank50_ao3_to7k_s1);
#     the trainer gets --filter-scorer ao3layer --filter-question dolphin_yesno --filter-device cuda:0 and no
#     --filter-reader-* options; the checks stage re-reads with the AO (train_blockmix_filter.py ao-reread <out>, from
#     main_run, on the training GPU as the reader's re-read) instead of cleanbatch_pyes.py, then check and pyes-check as
#     for the reader. reader (default): everything as before. Refused in random mode.
#   --require-rejection: the CPU check also requires at least one rejected step, every one checked unchanged
#     (train_blockmix_filter.py check --min-rejected 1).
#   CB_DRY=1 (or BM_DRY=1): dry listing (commands, GPU rule, stage states); nothing starts, nothing is created.
#
# Stages (markers $R/blockmix/markers/<stage>.{STARTED,DONE,FAILED,pid,gpu,host}; .host = the hostname that holds the
# GPU, so a stage of another server on the same file system is not counted as holding this server's GPUs; 2026-10-27), events $R/blockmix/events.log, logs
# main_run/logs/<STAMP>_<stage>.out; a DONE stage is skipped when the script runs again; nothing is deleted: an output
# folder that starts again is renamed *.before_<stamp>):
#  <key>          (GPU) training. DONE when python exits 0 and the log has "blockmix filter: finished:" and "done: step ".
#  <key>_checks   GPU first: cleanbatch_pyes.py re-reads <out>/filter_check_inputs.npz (worktree as working
#                 directory) on the training GPU, started right after the training process ended (no CPU stage in
#                 between) once one look shows that GPU still usable by the GPU rule below (up to 3 looks 2 s apart);
#                 otherwise, or when training was DONE earlier, on a newly waited GPU. Then CPU:
#                 train_blockmix_filter.py check <out> [--steps 60 in the smoke] [--min-rejected 1 with
#                 --require-rejection] (stream, dataset, runtime and filter checks all passed, filter log consistent
#                 with stream.json); CPU: train_blockmix_filter.py pyes-check <out> (in-run P(Yes) = re-read within 1e-3
#                 at every check step). Between the training's exit and the re-read's weights the GPU shows < 4 GiB only
#                 while cleanbatch_pyes.py imports and loads its tokenizer (about 10-20 s; its 443 weight shards then
#                 load in about 7 s), shorter than the EM chain's two looks 30 s apart; the block-mix sweep chain's two
#                 looks are 20 s apart and it does not read this script's markers.
#  <key>_summary  (CPU, not in the smoke) main_run/blockmix_summary.py (sweep and filter rows, figure).
# End: <key>_job.DONE when every stage is DONE, else <key>_job.FAILED listing the stages that are not.
# GPU rule (as scripts/blockmix/chain.sh): a GPU is usable when its owner chain has ended (GPUs 1 and 3:
# $R/em_reader/markers/chain.DONE or chain.FAILED; GPUs 0 and 2: a marker $R/cleanbatch_var_dolphin_s1/markers/*chain*.DONE
# or *chain*.FAILED) AND it shows < 4096 MiB used on two nvidia-smi looks 20 s apart AND no other stage under
# $R/blockmix/markers holds it (a <stage>.gpu whose STARTED has no DONE or FAILED and whose pid, if written, is alive:
# the block-mix sweep chain writes its .gpu and STARTED before its python starts) AND the EM chain's eval stage has not
# announced it for its readings (while $R/em_reader/markers/eval.RUNNING exists and no read_all.DONE or read_all.FAILED
# newer than it: the GPUs of the last "readings ... on GPUs a,b" line of the newest main_run/logs/*_em_reader_eval.out
# not older than eval.RUNNING; that stage picks any GPU with < 4 GiB on two looks 30 s apart, GPUs 0 and 2 included, and
# its workers show little memory while they load). <gpu> limits the choice to that GPU; auto takes the first usable
# one. Polled every 60 s; FAILED after 6 h without a GPU. The training process holds 6 GiB on its GPU from its first
# seconds (train_blockmix.py "GPU hold"), so the EM chain sees a GPU this script just took as busy.
# Refuses to start when this key's job pid or training pid is alive, or with less than 50 GB free under $R.
# Launch (detached; the script waits for a GPU itself):
#   setsid nohup bash scripts/blockmix/filter.sh 0.25 0.2 auto < /dev/null > main_run/logs/<STAMP>_blockmix_filter.out 2>&1 &
# Overrides (tests): BM_PY, BM_SMI, BM_RUNS, BM_LOGS, BM_STAMP, BM_POLL_S (60), BM_LOOK_S (20), BM_MAX_WAIT_S (21600),
#   BM_STOP (7000 accepted), BM_MAX_DRAWN (14000), BM_STREAM_LEN (20000; the trainer refuses BM_MAX_DRAWN above it, so the
#   keep-10% student needs BM_STREAM_LEN=40000 BM_MAX_DRAWN=40000; 2026-10-27 05:24 failure), BM_KEY_SUFFIX (inserted
#   before _s1 in the key, e.g. _to7k for the 2026-10-27 08:20 re-runs budgeted to 7,000 trained steps; blockmix_summary.py
#   reads the rule from RUN.json, so the name need not parse), BM_SMOKE_STEPS (60), BM_VAR_MARKERS, BM_EM_MARKERS,
#   BM_EM_LOGS (main_run/logs), BM_RECHECK_S (2), BM_MIN_FREE_GB (50), CB_WORKTREE
set -u
IIF=${IIF_BASE:?set IIF_BASE}
export IIF_BASE=$IIF HF_HOME=$IIF/hf_home HF_HUB_OFFLINE=1
PY=${BM_PY:-$IIF/venvs/interpretable_influence/bin/python}
SMI=${BM_SMI:-nvidia-smi}
P=$IIF/research_helper/projects/activation-gradient-metamodel
R=${BM_RUNS:-$IIF/agm_data/runs}
D=$R/blockmix
M=$D/markers
L=${BM_LOGS:-$P/main_run/logs}
W=${CB_WORKTREE:-$IIF/agm_data/worktree_training_pipeline}
export CB_WORKTREE=$W
STAMP=${BM_STAMP:-$(date +%Y%m%d_%H%M%S)}
STOP=${BM_STOP:-7000}
MAX_DRAWN=${BM_MAX_DRAWN:-14000}
SMOKE_STEPS=${BM_SMOKE_STEPS:-60}
SMOKE_EVAL=20
POLL_S=${BM_POLL_S:-60}
LOOK_S=${BM_LOOK_S:-20}
MAX_WAIT_S=${BM_MAX_WAIT_S:-21600}
MIN_FREE_GB=${BM_MIN_FREE_GB:-50}
MEM_MAX_MIB=4096
VAR_MARKERS=${BM_VAR_MARKERS:-$R/cleanbatch_var_dolphin_s1/markers}
EM_MARKERS=${BM_EM_MARKERS:-$R/em_reader/markers}
EM_LOGS=${BM_EM_LOGS:-$P/main_run/logs}
RECHECK_S=${BM_RECHECK_S:-2}
DRY=${BM_DRY:-${CB_DRY:-0}}
SMOKE=${FILTER_SMOKE:-0}
RJ=$R/reader_scratch_L40/RUN.json
ADAPTER=$R/reader_rl_scratch_haiku_paper/adapters/cycle_040
READER=rl_paper_c040
Q=dolphin_yesno

usage() { echo "usage: bash scripts/blockmix/filter.sh <p> <tau> <gpu|auto> [--filter-input mfree|mom] [--require-rejection]" >&2
  echo "       bash scripts/blockmix/filter.sh <p> - <gpu|auto> --filter-mode random --drop-rate <r> [--require-rejection]" >&2
  echo "       bash scripts/blockmix/filter.sh <p> - <gpu|auto> --filter-mode rank --keep-frac <k> [--rank-window 100] [--rank-warmup 10] [--rank-side low|high] [--filter-input mfree|mom] [--require-rejection]" >&2
  echo "       (reader and rank modes: [--filter-scorer reader|ao3layer])" >&2; exit 2; }
[ $# -ge 3 ] || usage
FRAC=$1; TAU=$2; GPU_ARG=$3; shift 3
INPUT=mfree
INPUT_SET=0
MIN_REJ=0
MODE=reader
RATE=""
KEEP=""
WIN=100
WARM=10
SIDE=""
RANK_SET=0
SCORER=reader
SCORER_SET=0
while [ $# -gt 0 ]; do
  case "$1" in
    --filter-input) [ $# -ge 2 ] || usage; INPUT=$2; INPUT_SET=1; shift 2;;
    --filter-input=*) INPUT=${1#*=}; INPUT_SET=1; shift;;
    --filter-mode) [ $# -ge 2 ] || usage; MODE=$2; shift 2;;
    --filter-mode=*) MODE=${1#*=}; shift;;
    --drop-rate) [ $# -ge 2 ] || usage; RATE=$2; shift 2;;
    --drop-rate=*) RATE=${1#*=}; shift;;
    --keep-frac) [ $# -ge 2 ] || usage; KEEP=$2; RANK_SET=1; shift 2;;
    --keep-frac=*) KEEP=${1#*=}; RANK_SET=1; shift;;
    --rank-window) [ $# -ge 2 ] || usage; WIN=$2; RANK_SET=1; shift 2;;
    --rank-window=*) WIN=${1#*=}; RANK_SET=1; shift;;
    --rank-warmup) [ $# -ge 2 ] || usage; WARM=$2; RANK_SET=1; shift 2;;
    --rank-warmup=*) WARM=${1#*=}; RANK_SET=1; shift;;
    --rank-side) [ $# -ge 2 ] || usage; SIDE=$2; RANK_SET=1; shift 2;;
    --rank-side=*) SIDE=${1#*=}; RANK_SET=1; shift;;
    --filter-scorer) [ $# -ge 2 ] || usage; SCORER=$2; SCORER_SET=1; shift 2;;
    --filter-scorer=*) SCORER=${1#*=}; SCORER_SET=1; shift;;
    --require-rejection) MIN_REJ=1; shift;;
    *) echo "unknown argument $1" >&2; usage;;
  esac
done
case "$INPUT" in mfree|mom) ;; *) echo "--filter-input must be mfree or mom" >&2; exit 2;; esac
case "$MODE" in reader|random|rank) ;; *) echo "--filter-mode must be reader, random or rank" >&2; exit 2;; esac
case "$SCORER" in reader|ao3layer) ;; *) echo "--filter-scorer must be reader or ao3layer" >&2; exit 2;; esac
[ "$MODE" = random ] && [ "$SCORER_SET" = 1 ] && { echo "--filter-mode random reads nothing; --filter-scorer does not apply" >&2; exit 2; }
SC=$([ "$SCORER" = ao3layer ] && echo _ao3)
if [ "$MODE" = rank ]; then
  [ "$TAU" = - ] || { echo "--filter-mode rank: the <tau> argument must be - (the rule is the window quantile)" >&2; exit 2; }
  awk -v k="$KEEP" 'BEGIN { exit !(k ~ /^[0-9]*\.?[0-9]+$/ && k + 0 > 0 && k + 0 < 1) }' || { echo "--keep-frac must be in (0, 1)" >&2; exit 2; }
  case "$WIN" in [1-9]|[1-9][0-9]*) ;; *) echo "--rank-window must be a positive integer" >&2; exit 2;; esac
  case "$WARM" in [1-9]|[1-9][0-9]*) ;; *) echo "--rank-warmup must be a positive integer" >&2; exit 2;; esac
  [ "$WIN" -eq "$WIN" ] 2>/dev/null && [ "$WARM" -eq "$WARM" ] 2>/dev/null || { echo "--rank-window and --rank-warmup must be integers" >&2; exit 2; }
  SIDE=${SIDE:-low}
  case "$SIDE" in low|high) ;; *) echo "--rank-side must be low or high" >&2; exit 2;; esac
  TAU=1  # placeholder for the check below; not passed to the trainer in rank mode
elif [ "$RANK_SET" = 1 ]; then echo "--keep-frac, --rank-window, --rank-warmup and --rank-side are for --filter-mode rank" >&2; exit 2; fi
if [ "$MODE" = random ]; then
  [ "$TAU" = - ] || { echo "--filter-mode random: the <tau> argument must be - (no reader)" >&2; exit 2; }
  [ "$INPUT_SET" = 0 ] || { echo "--filter-mode random reads no input; --filter-input does not apply" >&2; exit 2; }
  awk -v r="$RATE" 'BEGIN { exit !(r ~ /^[0-9]*\.?[0-9]+$/ && r + 0 >= 0 && r + 0 < 1) }' || { echo "--drop-rate must be in [0, 1)" >&2; exit 2; }
  TAU=1  # placeholder for the check below; not passed to the trainer in random mode
elif [ -n "$RATE" ]; then echo "--drop-rate is for --filter-mode random" >&2; exit 2; fi
case "$GPU_ARG" in auto|[0-9]|[0-9][0-9]) ;; *) echo "<gpu> must be a GPU index or auto" >&2; exit 2;; esac
awk -v p="$FRAC" 'BEGIN { exit !(p ~ /^[0-9]*\.?[0-9]+$/ && p + 0 >= 0 && p + 0 <= 1) }' || { echo "<p> must be in [0, 1]" >&2; exit 2; }
awk -v t="$TAU" 'BEGIN { exit !(t ~ /^[0-9]*\.?[0-9]+$/ && t + 0 > 0 && t + 0 <= 1) }' || { echo "<tau> must be in (0, 1]" >&2; exit 2; }
PPP=$(awk -v p="$FRAC" 'BEGIN { printf "%03d", int(p * 100 + 0.5) }')
TT=$(awk -v t="$TAU" 'BEGIN { printf "%02d", int(t * 100 + 0.5) }')
KEY=blockmix_filter_p${PPP}_tau${TT}$([ "$INPUT" = mom ] && echo _mom)${SC}${BM_KEY_SUFFIX:-}_s1
if [ "$MODE" = random ]; then
  RR=$(awk -v r="$RATE" 'BEGIN { printf "%02d", int(r * 100 + 0.5) }')
  KEY=blockmix_filter_p${PPP}_rand${RR}${BM_KEY_SUFFIX:-}_s1; TAU=-; INPUT=-
fi
if [ "$MODE" = rank ]; then
  KK=$(awk -v k="$KEEP" 'BEGIN { printf "%02d", int(k * 100 + 0.5) }')
  KEY=blockmix_filter_p${PPP}_rank${KK}$([ "$SIDE" = high ] && echo _high)$([ "$WIN" = 100 ] || echo _w$WIN)$([ "$WARM" = 10 ] || echo _warm$WARM)$([ "$INPUT" = mom ] && echo _mom)${SC}${BM_KEY_SUFFIX:-}_s1
  TAU=-
fi
if [ "$SMOKE" = 1 ]; then
  KEY=${KEY}_smoke; OUT=$D/$KEY; ACC=$SMOKE_STEPS; DRAWN=$SMOKE_STEPS; EV=$SMOKE_EVAL; FE=$SMOKE_STEPS
  STAGES="$KEY ${KEY}_checks"
else
  OUT=$R/$KEY; ACC=$STOP; DRAWN=$MAX_DRAWN; EV=100; FE=500
  STAGES="$KEY ${KEY}_checks ${KEY}_summary"
fi
JOB=${KEY}_job

log() { echo "$(date '+%F %T') [blockmix filter $KEY] $*"; }
event() { echo "$(date '+%F %T') $*" >> "$D/events.log"; log "$*"; }
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
before_name() { local b t n=0; b="$1.before_$(date +%Y%m%d_%H%M%S)"; t=$b; while [ -e "$t" ]; do n=$((n + 1)); t="${b}_$n"; done; echo "$t"; }
aside() { [ -e "$1" ] || return 0; mv -n "$1" "$(before_name "$1")"; }
isdone() { [ -f "$M/$1.DONE" ]; }
mark() { local k=$1 s=$2; shift 2; echo "$(date '+%F %T') $*" > "$M/$k.$s"; }
start_stage() { local k=$1; for s in STARTED FAILED pid gpu; do aside "$M/$k.$s"; done; }
status_of() { local k=$1; for s in DONE FAILED STARTED; do [ -f "$M/$k.$s" ] && { echo "$s"; return; }; done; echo pending; }

scorer_args() {  # the reader's options, or the AO's (--filter-scorer ao3layer: no --filter-reader-* options)
  if [ "$SCORER" = ao3layer ]; then echo --filter-scorer ao3layer --filter-question "$Q" --filter-device cuda:0; return; fi
  echo --filter-reader-adapter "$ADAPTER" --filter-reader-run-json "$RJ" --filter-reader-name "$READER" \
    --filter-question "$Q" --filter-device cuda:0
}
filter_args() {
  if [ "$MODE" = random ]; then echo --filter-mode random --drop-rate "$RATE" --filter-max-drawn "$DRAWN"; return; fi
  if [ "$MODE" = rank ]; then
    echo --filter-mode rank --keep-frac "$KEEP" --rank-window "$WIN" --rank-warmup "$WARM" \
      --filter-input "$INPUT" --filter-max-drawn "$DRAWN" $(scorer_args) --rank-side "$SIDE"
    return
  fi
  echo --filter-tau "$TAU" --filter-input "$INPUT" --filter-max-drawn "$DRAWN" $(scorer_args)
}
train_args() {
  echo --dolphin-frac "$FRAC" --mix-seed 1 --stream-len "${BM_STREAM_LEN:-20000}" $(filter_args) \
    --arm dolphin --n-rows 50000 --lr 1e-4 --batch-size 32 --epochs 7 --warmup-ratio 0.05 \
    --eval-every "$EV" --full-eval-every "$FE" --save-every 500 --seed 1 --grad-ckpt 0 --stop-step "$ACC" \
    --adapter-dtype fp32 --save-adapter-step0 --no-checkpoints --out "$OUT"
}
check_steps_arg() { [ "$SMOKE" = 1 ] && printf -- '--steps %s ' "$SMOKE_STEPS"; [ "$MIN_REJ" -gt 0 ] && printf -- '--min-rejected %s' "$MIN_REJ"; }
pyes_cmd() {  # gpu
  if [ "$SCORER" = ao3layer ]; then echo "cd $P/main_run && CUDA_VISIBLE_DEVICES=$1 $PY train_blockmix_filter.py ao-reread $OUT --device cuda:0"; return; fi
  echo "cd $W && CUDA_VISIBLE_DEVICES=$1 $PY $P/main_run/cleanbatch_pyes.py --inputs $OUT/filter_check_inputs.npz --adapter $ADAPTER --run-json $RJ --reader-name $READER --question $Q --device cuda:0 --max-gpu-gib 60"
}

# ---------------------------------------------------------------- GPUs
owner_of() { case $1 in 0|2) echo "dW-variant chain ($VAR_MARKERS/*chain*.DONE or .FAILED)";; 1|3) echo "EM chain ($EM_MARKERS/chain.DONE or chain.FAILED)";; *) echo "no owner rule (never used)";; esac; }
owner_done() {
  case $1 in
    0|2) compgen -G "$VAR_MARKERS/*chain*.DONE" > /dev/null || compgen -G "$VAR_MARKERS/*chain*.FAILED" > /dev/null;;
    1|3) [ -e "$EM_MARKERS/chain.DONE" ] || [ -e "$EM_MARKERS/chain.FAILED" ];;
    *) return 1;;
  esac
}
held_by_markers() {  # GPUs of other stages under $M that started and have not ended (pid, when written, alive)
  local f k pid
  for f in "$M"/*.gpu; do
    [ -e "$f" ] || continue
    k=$(basename "$f" .gpu)
    [ "$k" = "$KEY" ] && continue
    # markers are on a shared file system: a stage of another host does not hold this host's GPUs (2026-10-27 15:45, two servers)
    if [ -f "$M/$k.host" ] && [ "$(cat "$M/$k.host" 2>/dev/null)" != "$(hostname)" ]; then continue; fi
    [ -f "$M/$k.STARTED" ] || continue
    [ -f "$M/$k.DONE" ] || [ -f "$M/$k.FAILED" ] && continue
    if [ -f "$M/$k.pid" ]; then pid=$(cat "$M/$k.pid" 2>/dev/null); alive "$pid" || continue; fi
    printf '%s ' "$(tr -dc '0-9' < "$f")"
  done
}
em_eval_gpus() {  # GPUs the EM chain's eval stage announced for its readings, while that stage runs and they have not ended
  local f run="$EM_MARKERS/eval.RUNNING"
  [ -e "$run" ] || return 0
  { [ "$EM_MARKERS/read_all.DONE" -nt "$run" ] || [ "$EM_MARKERS/read_all.FAILED" -nt "$run" ]; } && return 0
  f=$(ls -t "$EM_LOGS"/*_em_reader_eval.out 2>/dev/null | head -1)
  [ -n "$f" ] || return 0
  [ "$run" -nt "$f" ] && return 0
  grep -oE 'readings .*on GPUs [0-9]+(,[0-9]+)*' "$f" 2>/dev/null | tail -1 | sed 's/.*on GPUs //' | tr ',' ' '
}
smi_rows() { $SMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null; }
usable_now() {  # [$1 = auto or one GPU, default <gpu>]; prints the GPUs whose owner ended, < MEM_MAX_MIB used, not held
  local only=${1:-$GPU_ARG} held em g u  # by another stage, not announced by the EM eval stage, allowed by $only
  held=" $(held_by_markers) "
  em=" $(em_eval_gpus) "
  while IFS=, read -r g u; do
    g=$(echo "$g" | tr -d ' '); u=$(echo "$u" | tr -d ' ')
    [ -n "$g" ] && [ -n "$u" ] || continue
    [ "$only" = auto ] || [ "$only" = "$g" ] || continue
    case "$held" in *" $g "*) continue;; esac
    case "$em" in *" $g "*) continue;; esac
    owner_done "$g" || continue
    [ "$u" -lt $MEM_MAX_MIB ] && printf '%s ' "$g"
  done < <(smi_rows)
}
usable_two_looks() {
  local a b g out=""
  a=$(usable_now); sleep "$LOOK_S"; b=$(usable_now)
  for g in $a; do case " $b " in *" $g "*) out="$out$g ";; esac; done
  echo "$out"
}
gpu_rule_lines() {
  local g u held em
  held=" $(held_by_markers) "
  em=" $(em_eval_gpus) "
  while IFS=, read -r g u; do
    g=$(echo "$g" | tr -d ' '); u=$(echo "$u" | tr -d ' ')
    [ -n "$g" ] || continue
    printf 'GPU %s: owner %s ended: %s; %s MiB used now (limit %s); held by another stage: %s; announced by the EM eval stage: %s; allowed by <gpu> %s: %s\n' \
      "$g" "$(owner_of "$g")" "$(owner_done "$g" && echo yes || echo no)" "$u" "$MEM_MAX_MIB" \
      "$(case "$held" in *" $g "*) echo yes;; *) echo no;; esac)" "$(case "$em" in *" $g "*) echo yes;; *) echo no;; esac)" \
      "$GPU_ARG" "$([ "$GPU_ARG" = auto ] || [ "$GPU_ARG" = "$g" ] && echo yes || echo no)"
  done < <(smi_rows)
}
still_usable() {  # $1 = GPU: usable by the rule on one of up to 3 looks RECHECK_S apart (after a process left it)
  local i
  for i in 1 2 3; do
    case " $(usable_now "$1") " in *" $1 "*) return 0;; esac
    [ "$i" -lt 3 ] && sleep "$RECHECK_S"
  done
  return 1
}
wait_gpu() {  # sets GPU_PICKED; 1 after MAX_WAIT_S
  local t0 u; t0=$(date +%s); GPU_PICKED=""
  while :; do
    u=$(usable_two_looks)
    if [ -n "$u" ]; then GPU_PICKED=${u%% *}; event "$KEY: GPU $GPU_PICKED usable (owner ended, free on two looks, not held; waited $(( $(date +%s) - t0 )) s)"; return 0; fi
    log "waiting for a usable GPU ($GPU_ARG): $(gpu_rule_lines | tr '\n' ';')"
    [ $(( $(date +%s) - t0 )) -ge "$MAX_WAIT_S" ] && return 1
    sleep $(( POLL_S > LOOK_S ? POLL_S - LOOK_S : 1 ))
  done
}

# ---------------------------------------------------------------- dry listing
if [ "$DRY" = 1 ]; then
  log "dry: key $KEY, out $OUT, p $FRAC, mode $MODE$([ "$SCORER" = ao3layer ] && echo ", scorer ao3layer")$([ "$MODE" = random ] && echo ", drop rate $RATE")$([ "$MODE" = rank ] && echo ", keep frac $KEEP, window $WIN, warmup $WARM, side $SIDE"), tau $TAU, input $INPUT, budgets $ACC accepted / $DRAWN drawn, evaluation every $EV (full $FE)$([ "$SMOKE" = 1 ] && echo ', smoke')"
  log "dry: stages: $(for k in $STAGES $JOB; do printf '%s [%s] ' "$k" "$(status_of "$k")"; done)"
  log "dry: train: cd $P/main_run && CUDA_VISIBLE_DEVICES=<gpu> $PY train_blockmix_filter.py $(train_args)"
  if [ "$MODE" = random ]; then
    log "dry: checks: cd $P/main_run && $PY train_blockmix_filter.py check $OUT $(check_steps_arg) (random mode: this CPU check only)"
  else
    log "dry: checks: $(pyes_cmd '<training gpu>') (right after training); cd $P/main_run && $PY train_blockmix_filter.py check $OUT $(check_steps_arg); cd $P/main_run && $PY train_blockmix_filter.py pyes-check $OUT"
  fi
  [ "$SMOKE" = 1 ] || log "dry: summary: cd $P && $PY main_run/blockmix_summary.py"
  log "dry: GPU rule: owner chain ended AND < $MEM_MAX_MIB MiB on two looks $LOOK_S s apart AND not held by another stage AND not announced by the EM eval stage (now: [$(em_eval_gpus)]) AND allowed by <gpu> ($GPU_ARG); poll $POLL_S s; FAILED after $MAX_WAIT_S s"
  gpu_rule_lines | while read -r line; do log "dry: $line"; done
  log "dry: usable GPUs now (one look): [$(usable_now)]"
  log "dry: disk free under $R: $(df -BG --output=avail "$R" 2>/dev/null | tail -1 | tr -dc '0-9') GB (need $MIN_FREE_GB)"
  jp=$(cat "$M/$JOB.pid" 2>/dev/null); log "dry: job pid file: ${jp:-none} ($(alive "$jp" && echo alive || echo not alive))"
  log "dry: nothing launched"
  exit 0
fi

# ---------------------------------------------------------------- start
mkdir -p "$M" "$L"
if isdone "$JOB"; then log "nothing to do: $M/$JOB.DONE exists"; exit 0; fi
for f in "$M/$JOB.pid" "$M/$KEY.pid"; do
  if alive "$(cat "$f" 2>/dev/null)"; then log "refuse: $f names a live process ($(cat "$f"))"; exit 1; fi
done
free_gb=$(df -BG --output=avail "$R" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -z "$free_gb" ] || [ "$free_gb" -lt "$MIN_FREE_GB" ]; then log "refuse: ${free_gb:-?} GB free under $R (need $MIN_FREE_GB)"; exit 1; fi
for s in STARTED FAILED pid; do aside "$M/$JOB.$s"; done
echo $$ > "$M/$JOB.pid"
LOGF=$(readlink -f /proc/$$/fd/1 2>/dev/null || echo "?")
mark "$JOB" STARTED "pid $$, log $LOGF, stamp $STAMP, p $FRAC, mode $MODE, drop rate ${RATE:--}, keep frac ${KEEP:--}, window $WIN, warmup $WARM, rank side ${SIDE:--}, tau $TAU, input $INPUT, scorer $([ "$MODE" = random ] && echo - || echo "$SCORER"), gpu $GPU_ARG, smoke $SMOKE, min rejected $MIN_REJ, repo $(git -C "$P" rev-parse --short HEAD 2>/dev/null)"
event "$JOB start: pid $$, log $LOGF, p $FRAC, mode $MODE$([ "$MODE" = rank ] && echo " ($SIDE side, keep $KEEP)")$([ "$SCORER" = ao3layer ] && echo ", scorer ao3layer"), tau $TAU, input $INPUT, budgets $ACC accepted / $DRAWN drawn, out $OUT"
fail_job() { mark "$JOB" FAILED "$*"; event "$JOB FAILED: $*"; exit 1; }
GPU_HELD=""

# ---------------------------------------------------------------- training
if ! isdone "$KEY"; then
  start_stage "$KEY"
  wait_gpu || { mark "$KEY" FAILED "no usable GPU within $MAX_WAIT_S s"; fail_job "$KEY: no usable GPU"; }
  GPU_HELD=$GPU_PICKED
  tlog=$L/${STAMP}_${KEY}.out
  if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    b=$(before_name "$OUT"); mv -n "$OUT" "$b"; event "$KEY: earlier output moved to $b"
  fi
  aside "$OUT.pid"
  mkdir -p "$(dirname "$OUT")"
  echo "$GPU_HELD" > "$M/$KEY.gpu"
  hostname > "$M/$KEY.host"
  mark "$KEY" STARTED "GPU $GPU_HELD, p $FRAC, tau $TAU, input $INPUT, out $OUT, budgets $ACC/$DRAWN, log $tlog"
  event "$KEY started on GPU $GPU_HELD"
  # shellcheck disable=SC2046
  ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES=$GPU_HELD exec $PY train_blockmix_filter.py $(train_args) ) >> "$tlog" 2>&1 &
  pid=$!; echo "$pid" > "$M/$KEY.pid"; echo "$pid" > "$OUT.pid"
  wait "$pid"; rc=$?
  if [ "$rc" = 0 ] && grep -q "blockmix filter: finished:" "$tlog" && grep -q "done: step " "$tlog"; then
    mark "$KEY" DONE "exit 0, $(grep -h 'blockmix filter: finished:' "$tlog" | tail -1 | sed 's/.*blockmix filter: //' | cut -c1-400); log $tlog"
    event "$KEY DONE"
  else
    { echo "$(date '+%F %T') exit $rc; finished line $(grep -c 'blockmix filter: finished:' "$tlog") times; log $tlog"; tail -n 30 "$tlog"; } > "$M/$KEY.FAILED"
    event "$KEY FAILED (exit $rc)"
    fail_job "$KEY FAILED (exit $rc, log $tlog)"
  fi
fi

# ---------------------------------------------------------------- checks (GPU re-read right after training, CPU, CPU)
if [ "$MODE" = random ] && ! isdone "${KEY}_checks"; then  # no reader: only the CPU check
  start_stage "${KEY}_checks"
  clog=$L/${STAMP}_${KEY}_checks.out
  mark "${KEY}_checks" STARTED "CPU check only (random mode); log $clog"
  # shellcheck disable=SC2046
  if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY train_blockmix_filter.py check "$OUT" $(check_steps_arg) ) >> "$clog" 2>&1; then
    mark "${KEY}_checks" DONE "check ok (random mode: no re-read); $(tail -1 "$clog" | cut -c1-400); log $clog"; event "${KEY}_checks DONE"
  else
    { echo "$(date '+%F %T') check FAILED (random mode); log $clog"; tail -n 20 "$clog"; } > "$M/${KEY}_checks.FAILED"
    event "${KEY}_checks FAILED (check, random mode)"
  fi
fi
if [ "$MODE" != random ] && ! isdone "${KEY}_checks"; then
  start_stage "${KEY}_checks"
  clog=$L/${STAMP}_${KEY}_checks.out
  if [ -n "$GPU_HELD" ] && ! still_usable "$GPU_HELD"; then
    event "${KEY}_checks: training GPU $GPU_HELD is no longer usable ($(gpu_rule_lines | grep "^GPU $GPU_HELD:" | cut -c1-300)); waiting for a GPU for the re-read"
    GPU_HELD=""
  fi
  if [ -z "$GPU_HELD" ]; then
    wait_gpu || { mark "${KEY}_checks" FAILED "no usable GPU for the re-read within $MAX_WAIT_S s"; fail_job "${KEY}_checks: no usable GPU"; }
    GPU_HELD=$GPU_PICKED
  fi
  echo "$GPU_HELD" > "$M/${KEY}_checks.gpu"
  hostname > "$M/${KEY}_checks.host"
  mark "${KEY}_checks" STARTED "re-read on GPU $GPU_HELD$([ "$SCORER" = ao3layer ] && echo " with the AO (ao-reread)"); log $clog"
  if [ "$SCORER" = ao3layer ]; then  # the AO's re-read (cleanbatch_pyes.py loads only the reader)
    ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES=$GPU_HELD exec $PY train_blockmix_filter.py ao-reread "$OUT" --device cuda:0 ) >> "$clog" 2>&1 &
  else
    ( cd "$W" && CUDA_VISIBLE_DEVICES=$GPU_HELD exec $PY "$P/main_run/cleanbatch_pyes.py" --inputs "$OUT/filter_check_inputs.npz" \
        --adapter "$ADAPTER" --run-json "$RJ" --reader-name "$READER" --question "$Q" --device cuda:0 --max-gpu-gib 60 ) >> "$clog" 2>&1 &
  fi
  pid=$!; echo "$pid" > "$M/${KEY}_checks.pid"; wait "$pid"; rc2=$?
  # shellcheck disable=SC2046
  if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY train_blockmix_filter.py check "$OUT" $(check_steps_arg) ) >> "$clog" 2>&1; then
    c1=ok
  else c1=FAILED; fi
  if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY train_blockmix_filter.py pyes-check "$OUT" ) >> "$clog" 2>&1; then
    c3=ok
  else c3=FAILED; fi
  if [ "$c1" = ok ] && [ "$rc2" = 0 ] && [ "$c3" = ok ]; then
    mark "${KEY}_checks" DONE "check ok, re-read exit 0, pyes-check ok; $(tail -1 "$clog" | cut -c1-400); log $clog"; event "${KEY}_checks DONE"
  else
    { echo "$(date '+%F %T') check $c1, re-read exit $rc2, pyes-check $c3; log $clog"; tail -n 20 "$clog"; } > "$M/${KEY}_checks.FAILED"
    event "${KEY}_checks FAILED (check $c1, re-read exit $rc2, pyes-check $c3)"
  fi
fi

# ---------------------------------------------------------------- summary (CPU)
if [ "$SMOKE" != 1 ] && ! isdone "${KEY}_summary"; then
  start_stage "${KEY}_summary"
  slog=$L/${STAMP}_${KEY}_summary.out
  if ( cd "$P" && CUDA_VISIBLE_DEVICES= exec $PY main_run/blockmix_summary.py ) >> "$slog" 2>&1; then
    mark "${KEY}_summary" DONE "log $slog"; event "${KEY}_summary DONE"
  else mark "${KEY}_summary" FAILED "$(tail -n 5 "$slog")"; event "${KEY}_summary FAILED"; fi
fi

notdone=""
for k in $STAGES; do isdone "$k" || notdone="$notdone $k"; done
if [ -z "$notdone" ]; then mark "$JOB" DONE "all stages DONE"; event "$JOB DONE"; exit 0; fi
fail_job "not DONE:$notdone"
