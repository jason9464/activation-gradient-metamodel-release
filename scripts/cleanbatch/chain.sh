#!/bin/bash
# Foreign-batch would-be updates (question:
# in a student learning the dolphin preference, does the direct-write term of the update that a CLEAN batch would cause
# at an intermediate step read "dolphin"?).
#
# Stages (markers $D/markers/<stage>.{STARTED,DONE,FAILED,pid}, events $D/events.log, logs
# main_run/logs/<STAMP>_cleanbatch_<stage>.out; a DONE stage is skipped when the chain runs again; nothing is deleted:
# an output folder or reading of a stage that runs again is renamed *.before_<stamp> or moved to e2/partial/<STAMP>/):
#  foreign   (CPU) main_run/cleanbatch_probe.py draw: runs/cleanbatch/foreign_batches.json (an existing file must equal
#            the rebuilt draw)
#  wait      a GPU with < 4 GiB in use on two looks 20 s apart (poll every 60 s, up to 6 h; then chain FAILED)
#  smoke     (GPU) main_run/train_cleanbatch.py, the newset recipe stopped at 60 steps with probes at 20 and 40, into
#            $D/smoke; cleanbatch_inputs.py verify (trajectory of steps 1..60 equal to newset_dolphin_s1, probe state
#            and weight checks, real batch vs the trainer's stores cos >= 0.9999) and build (core, all);
#            e2_generate on the core file (2 answers, tag smoke, both questions) and check_e2_reading.py;
#            cleanbatch_pyes.py on the all file; judge_dolphin_answers.py --dry-run on the smoke readings.
#            Any failure: smoke.FAILED and the chain stops (FAILED).
#  rerun_dolphin, rerun_clean  (GPU each, in parallel) train_cleanbatch.py: the newset recipe (scripts/newset/train.sh
#            arguments without --save-grads) to 3,000 steps with probes at 500..3000 every 100, into
#            $R/cleanbatch_<arm>_s1/train. rerun_dolphin takes the smoke's GPU (checked again), rerun_clean waits for
#            another GPU. DONE when python exits 0 and the log has "done: step 3000,".
#  verify_<arm>, inputs_<arm>  (CPU) cleanbatch_inputs.py verify (trajectory of all 3,000 steps, probe checks, real
#            batch vs stores and logs) and build (groups core, mfree, fixed, grad, mfreefix, all)
#  pyes_<arm>  (GPU) cleanbatch_pyes.py on the all file (first-token P(yes) of the dolphin_yesno question)
#  read_<arm>_<group>  (two GPUs, one per student, in parallel; each student in
#            this order on its own GPU: pyes, core, mfree, fixed, grad, mfreefix; the dolphin student on its rerun's GPU,
#            the clean student on its rerun's GPU, each checked again by the two-looks rule; a reading that must move
#            waits for a GPU other than the other student's rerun and reading GPUs, $M/reads_<arm>.gpu)
#            e2_generate (worktree, reader rl_paper_c040; cap READ_MAX_GPU_GIB 160 GiB, expandable segments), 100 answers, tag <group>; questions dolphin_yesno and
#            preference for core and mfree (mfree's preference added for its judge), dolphin_yesno for
#            the other groups; check_e2_reading.py per file. A student's reads start once its inputs are DONE.
#  judge_<group>_<arm>_<question>  (background, one worker per question, the two questions in parallel; each worker
#            judges the dolphin student's core readings when they end, then the clean student's; after both workers:
#            summary_main (cleanbatch_summary.py, the first summary); then the same for mfree)
#            judge_dolphin_answers.py --inputs cleanbatch --tag <group> --key <the group's four windows>
#            --animal dolphin --workers 8 --max-cost (the question's ledger total + what is left of this chain's 3.3 USD
#            share per question (2 for core + 1.3 for mfree), spending in $D/judge_spent.tsv)
#  summary   (CPU, after every read and the judges) cleanbatch_summary.py -> $R/cleanbatch_summary.md and .json
# End: cleanbatch_chain.DONE when every stage is DONE, else cleanbatch_chain.FAILED listing the failed markers.
# Launch (detached; the chain waits for GPUs itself):
#   setsid nohup bash scripts/cleanbatch/chain.sh < /dev/null > main_run/logs/<STAMP>_cleanbatch_chain.out 2>&1 &
# Dry listing (nothing starts): CB_DRY=1 bash scripts/cleanbatch/chain.sh
# Overrides (tests): CB_STAMP, CB_POLL_S (60), CB_LOOK_S (20), CB_MAX_WAIT_S (21600), CB_STOP (3000), CB_PROBE (500:3000:100)
set -u
IIF=${IIF_BASE:?set IIF_BASE}
export IIF_BASE=$IIF HF_HOME=$IIF/hf_home HF_HUB_OFFLINE=1
PY=$IIF/venvs/interpretable_influence/bin/python
SMI=nvidia-smi
P=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
W=$IIF/agm_data/worktree_training_pipeline
R=$IIF/agm_data/runs
D=$R/cleanbatch_dolphin_s1
M=$D/markers
L=$P/main_run/logs
STAMP=${CB_STAMP:-$(date +%Y%m%d_%H%M%S)}
FIXED=$IIF/agm_data/newset/fixed_docs_128.jsonl
FIXED_SHA=f696b8cbaa304236677365e51b254ebdddf20e5f14b8a85cee6d5924fd6288a6
FOREIGN=$R/cleanbatch/foreign_batches.json
RJ=$R/reader_scratch_L40/RUN.json
READER=rl_paper_c040
ADAPTER=$R/reader_rl_scratch_haiku_paper/adapters/cycle_040
JD=$R/judge_dolphin
STOP=${CB_STOP:-3000}
PROBE=${CB_PROBE:-500:3000:100}
POLL_S=${CB_POLL_S:-60}
LOOK_S=${CB_LOOK_S:-20}
MAX_WAIT_S=${CB_MAX_WAIT_S:-21600}
MEM_MAX_MIB=4096
# Readings: e2_generate per-process cap and allocator. One generate call per (step, question) holds 4 windows x 100
# answers = 400 sequences (the perdoc readings held 2 x 100). At 120 GiB the second step of every core and mfree reading
# ran out of memory (15.6 GiB reserved but unallocated). Memory handling only: the sampled
# answers do not change (seed per step, same batch).
READ_MAX_GPU_GIB=160
READ_ALLOC_CONF=expandable_segments:True
JUDGE_SHARE_USD=3.3  # per question: 2 for core + 1.3 for mfree
JUDGE_WORKERS=8
READ_GROUPS="core mfree fixed grad mfreefix"  # not GROUPS: bash keeps that name for the user's group ids (it expanded to 1999 here)
JUDGED="core mfree"  # groups judged by the Haiku judge, both questions
CHAIN=cleanbatch_chain
DRY=${CB_DRY:-0}

mkdir -p "$M" "$L"
log() { echo "$(date '+%F %T') [cleanbatch chain] $*"; }
event() { echo "$(date '+%F %T') $*" >> "$D/events.log"; log "$*"; }
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
before_name() { local b t n=0; b="$1.before_$(date +%Y%m%d_%H%M%S)"; t=$b; while [ -e "$t" ]; do n=$((n + 1)); t="${b}_$n"; done; echo "$t"; }
aside() { [ -e "$1" ] || return 0; mv -n "$1" "$(before_name "$1")"; }
isdone() { [ -f "$M/$1.DONE" ]; }
mark() { local k=$1 s=$2; shift 2; echo "$(date '+%F %T') $*" > "$M/$k.$s"; }
start_stage() { local k=$1; for s in STARTED FAILED pid; do aside "$M/$k.$s"; done; }
train_of() { echo "$R/cleanbatch_$1_s1/train"; }
steps_csv() { $PY -c "import sys; a, b, c = map(int, sys.argv[1].split(':')); print(','.join(str(t) for t in range(a, b + 1, c)))" "$1"; }
qs_of() { case " $JUDGED " in *" $1 "*) echo "dolphin_yesno preference";; *) echo "dolphin_yesno";; esac; }
wins_of() { local g=$1 v; case $g in core) v=step;; mfree) v=mfree;; fixed) v=stepfix;; grad) v=grad;; mfreefix) v=mfreefix;; esac
  echo "real_$v,f1_$v,f2_$v,f3_$v"; }
w_rev() { local h; h=$(GIT_OPTIONAL_LOCKS=0 git -C "$W" rev-parse --short HEAD 2>/dev/null)
  GIT_OPTIONAL_LOCKS=0 git -C "$W" diff --quiet HEAD -- main_run/reader/e2_generate.py 2>/dev/null || h="$h+uncommitted e2_generate.py"; echo "$h"; }

ledger_total() { local f=$JD/cost_ledger_$1.tsv; [ -f "$f" ] || { echo 0; return; }
  awk -F'\t' 'NR > 1 && $12 != "" {s += $12} END {printf "%.6f", s + 0}' "$f"; }
spent_of() {  # USD this chain spent on question $1 ($D/judge_spent.tsv rows: time question key usd)
  [ -f "$D/judge_spent.tsv" ] || { echo 0; return; }
  awk -F'\t' -v q="$1" '$2 == q {s += $4} END {printf "%.6f", s + 0}' "$D/judge_spent.tsv"; }

train_args() {  # arm stop pairs out probe -> the train_cleanbatch.py arguments (the newset recipe of scripts/newset/train.sh)
  local arm=$1 stop=$2 pairs=$3 out=$4 probe=$5
  echo --probe-steps "$probe" --foreign-file "$FOREIGN" \
    --arm "$arm" --n-rows 50000 --lr 1e-4 --batch-size 32 --epochs 7 --warmup-ratio 0.05 \
    --eval-every 100 --full-eval-every 500 --save-every 500 --seed 1 --grad-ckpt 0 --stop-step "$stop" \
    --save-adamfx --adamfx-stepsums adamfx --adamfx-stepsums-dtype fp32 \
    --adamfx-fixed-docs "$FIXED" --adamfx-fixed-every 100 \
    --adamfx-lora-pairs "$pairs" --adamfx-lora-pairs-format fp32 \
    --adapter-dtype fp32 --save-adapter-step0 --no-checkpoints --out "$out"
}

# ---------------------------------------------------------------- GPUs
used_of() { $SMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v g="$1" '$1 == g {print $2}'; }
free_list() { $SMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v m=$MEM_MAX_MIB '$2 < m {print $1}' | tr '\n' ' '; }
GPU_PICKED=""
wait_gpu() {  # $1 = space list of GPUs to exclude (ours); sets GPU_PICKED; 1 after MAX_WAIT_S
  local excl=" ${1:-} " t0 a b g
  t0=$(date +%s); GPU_PICKED=""
  while :; do
    a=$(free_list); sleep "$LOOK_S"; b=$(free_list)
    for g in $a; do
      case " $b " in *" $g "*) ;; *) continue;; esac
      case "$excl" in *" $g "*) continue;; esac
      GPU_PICKED=$g; break
    done
    if [ -n "$GPU_PICKED" ]; then event "GPU $GPU_PICKED free on two looks $LOOK_S s apart (waited $(( $(date +%s) - t0 )) s)"; return 0; fi
    log "waiting for a GPU (< $MEM_MAX_MIB MiB used): look 1 [$a] look 2 [$b] excluded [${1:-}]; $($SMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
    [ $(( $(date +%s) - t0 )) -ge "$MAX_WAIT_S" ] && return 1
    sleep $(( POLL_S > LOOK_S ? POLL_S - LOOK_S : 1 ))
  done
}
ensure_gpu() {  # $1 = GPU we hold, $2 = exclude list for a new wait; sets GPU_PICKED (the same GPU when still free)
  local u1 u2
  u1=$(used_of "$1"); sleep "$LOOK_S"; u2=$(used_of "$1")
  if [ -n "$u1" ] && [ -n "$u2" ] && [ "$u1" -lt $MEM_MAX_MIB ] && [ "$u2" -lt $MEM_MAX_MIB ]; then GPU_PICKED=$1; return 0; fi
  event "GPU $1 is no longer free ($u1 / $u2 MiB); waiting for another"
  wait_gpu "${2:-}"
}

# ---------------------------------------------------------------- start
if [ "$DRY" = 1 ]; then
  log "dry: markers $M; stages done: $(cd "$M" 2>/dev/null && ls *.DONE 2>/dev/null | tr '\n' ' ')"
  log "dry: rerun args (dolphin): $(train_args dolphin "$STOP" 101,1001 "$(train_of dolphin)" "$PROBE")"
  log "dry: free GPUs now: [$(free_list)]; foreign file $( [ -f "$FOREIGN" ] && sha256sum "$FOREIGN" | cut -c1-16 || echo missing)"
  log "dry: probe steps $(steps_csv "$PROBE")"
  log "dry: questions per group: $(for g in $READ_GROUPS; do printf '%s [%s] ' "$g" "$(qs_of "$g")"; done)"
  log "dry: judged groups [$JUDGED], both questions, share $JUDGE_SHARE_USD USD per question; ledger totals now: dolphin_yesno $(ledger_total dolphin_yesno), preference $(ledger_total preference); this chain spent: dolphin_yesno $(spent_of dolphin_yesno), preference $(spent_of preference)"
  log "dry: judge keys: $(for g in $JUDGED; do printf '%s %s; ' "$g" "$(wins_of "$g")"; done)"
  log "dry: reading: two GPUs, one per student (pyes, then $READ_GROUPS), in parallel"
  exit 0
fi
[ -e "$M/$CHAIN.DONE" ] && { log "refuse: $M/$CHAIN.DONE exists"; exit 1; }
if alive "$(cat "$M/$CHAIN.pid" 2>/dev/null)"; then log "refuse: chain pid $(cat "$M/$CHAIN.pid") is alive"; exit 1; fi
for s in STARTED FAILED pid; do aside "$M/$CHAIN.$s"; done
echo $$ > "$M/$CHAIN.pid"
LOGF=$(readlink -f /proc/$$/fd/1 2>/dev/null || echo "?")
mark "$CHAIN" STARTED "pid $$, log $LOGF, stamp $STAMP, repo $(git -C "$P" rev-parse --short HEAD 2>/dev/null), worktree $(w_rev)"
event "chain start: pid $$, log $LOGF, probe steps $PROBE, stop $STOP"
fail_chain() { mark "$CHAIN" FAILED "$*"; event "chain FAILED: $*"; exit 1; }
[ "$(sha256sum "$FIXED" | cut -d' ' -f1)" = "$FIXED_SHA" ] || fail_chain "fixed documents sha256 differs from $FIXED_SHA"

# ---------------------------------------------------------------- foreign (CPU)
if ! isdone foreign; then
  start_stage foreign
  if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY cleanbatch_probe.py draw --out "$FOREIGN" ) >> "$L/${STAMP}_cleanbatch_foreign.out" 2>&1; then
    mark foreign DONE "$FOREIGN sha256 $(sha256sum "$FOREIGN" | cut -d' ' -f1)"; event "foreign DONE"
  else mark foreign FAILED "see $L/${STAMP}_cleanbatch_foreign.out"; fail_chain "foreign FAILED"; fi
fi

# ---------------------------------------------------------------- helpers: training, inputs, readings
run_train() {  # key arm gpu out stop pairs probe -> waits; DONE/FAILED marker
  local key=$1 arm=$2 gpu=$3 out=$4 stop=$5 pairs=$6 probe=$7 tlog pid rc
  tlog=$L/${STAMP}_cleanbatch_${key}.out
  if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
    local b; b=$(before_name "$out"); mv -n "$out" "$b"; event "$key: earlier output moved to $b"
  fi
  mkdir -p "$(dirname "$out")"
  aside "$M/$key.gpu"; echo "$gpu" > "$M/$key.gpu"
  mark "$key" STARTED "GPU $gpu, out $out, stop $stop, probes $probe, log $tlog"
  event "$key started on GPU $gpu (stop $stop, probes $probe)"
  # shellcheck disable=SC2046
  ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES=$gpu exec $PY train_cleanbatch.py $(train_args "$arm" "$stop" "$pairs" "$out" "$probe") ) >> "$tlog" 2>&1 &
  pid=$!; echo "$pid" > "$M/$key.pid"
  wait "$pid"; rc=$?
  if [ "$rc" = 0 ] && grep -q "done: step $stop, " "$tlog"; then
    mark "$key" DONE "exit 0, $(grep "done: step $stop, " "$tlog" | tail -1 | sed 's/.*\(done: .*\)/\1/'); log $tlog"
    event "$key DONE"; return 0
  fi
  { echo "$(date '+%F %T') exit $rc; log $tlog"; tail -n 30 "$tlog"; } > "$M/$key.FAILED"
  event "$key FAILED (exit $rc)"; return 1
}
verify_build() {  # key-suffix train arm groups
  local sfx=$1 tr=$2 arm=$3 groups=$4 vlog
  vlog=$L/${STAMP}_cleanbatch_inputs_${sfx}.out
  if ! isdone "verify_$sfx"; then
    start_stage "verify_$sfx"
    if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY cleanbatch_inputs.py verify --train "$tr" --arm "$arm" ) >> "$vlog" 2>&1; then
      mark "verify_$sfx" DONE "$(grep 'verify ' "$vlog" | tail -1 | cut -c1-600)"; event "verify_$sfx DONE"
    else mark "verify_$sfx" FAILED "$(tail -n 5 "$vlog")"; event "verify_$sfx FAILED"; return 1; fi
  fi
  if ! isdone "inputs_$sfx"; then
    start_stage "inputs_$sfx"
    if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY cleanbatch_inputs.py build --train "$tr" --groups "$groups" ) >> "$vlog" 2>&1; then
      mark "inputs_$sfx" DONE "groups $groups; log $vlog"; event "inputs_$sfx DONE"
    else mark "inputs_$sfx" FAILED "$(tail -n 5 "$vlog")"; event "inputs_$sfx FAILED"; return 1; fi
  fi
  return 0
}
READ_GPU=""
READ_ARM=""  # the student whose readings this (sub)shell runs; empty in the smoke
others() {  # GPUs of the other student's rerun and readings, excluded when a reading must move (none in the smoke)
  local o
  case $READ_ARM in dolphin) o=clean;; clean) o=dolphin;; *) return 0;; esac
  echo "$(cat "$M/rerun_$o.gpu" 2>/dev/null) $(cat "$M/reads_$o.gpu" 2>/dev/null)"
}
hold_gpu() { READ_GPU=$1; [ -n "$READ_ARM" ] && echo "$READ_GPU" > "$M/reads_$READ_ARM.gpu"; return 0; }
partial_aside() {  # train group tag questions: move earlier outputs of these readings to e2/partial/<STAMP>/
  local tr=$1 tag=$2 q f n=0; shift 2
  for q in "$@"; do
    for f in "$tr/e2/cleanbatch_${READER}_${q}_${tag}"{.jsonl,.jsonl.tmp,_settings.json,_summary.json}; do
      [ -e "$f" ] || continue; mkdir -p "$tr/e2/partial/$STAMP"; mv -n "$f" "$tr/e2/partial/$STAMP/"; n=$((n + 1))
    done
  done
  [ $n -gt 0 ] && event "moved $n earlier reading outputs ($tag) of $tr to e2/partial/$STAMP"
  return 0
}
read_one() {  # key train group tag samples steps questions(space) ; uses READ_GPU
  local key=$1 tr=$2 group=$3 tag=$4 samples=$5 steps=$6 qs=$7 rlog pid rc q chk bad="" err="" t0 wins
  isdone "$key" && { log "skip $key (DONE)"; return 0; }
  start_stage "$key"
  partial_aside "$tr" "$tag" $qs
  if ! ensure_gpu "$READ_GPU" "$(others)"; then mark "$key" FAILED "no free GPU"; return 1; fi
  hold_gpu "$GPU_PICKED"
  rlog=$L/${STAMP}_cleanbatch_${key}.out
  wins=$(wins_of "$group")
  mark "$key" STARTED "GPU $READ_GPU, log $rlog, worktree $(w_rev)"
  event "reading $key started on GPU $READ_GPU ($group, windows $wins, $samples answers, questions $qs)"
  t0=$(date +%s)
  ( cd "$W" && CUDA_VISIBLE_DEVICES=$READ_GPU PYTORCH_CUDA_ALLOC_CONF=$READ_ALLOC_CONF exec $PY -m main_run.reader.e2_generate --inputs "$tr/e2/cleanbatch_inputs_${group}.npz" \
      --run "$tr" --adapter "$ADAPTER" --run-json "$RJ" --reader-name "$READER" --questions "$(echo $qs | tr ' ' ',')" \
      --target dolphin --no-summarize --device cuda:0 --max-gpu-gib $READ_MAX_GPU_GIB --samples "$samples" --steps "$steps" --tag "$tag" ) >> "$rlog" 2>&1 &
  pid=$!; echo "$pid" > "$M/$key.pid"
  wait "$pid"; rc=$?
  for q in $qs; do
    chk=$(cd "$P/main_run" && CUDA_VISIBLE_DEVICES= $PY check_e2_reading.py "$tr/e2/cleanbatch_${READER}_${q}_${tag}.jsonl" \
          --steps "$steps" --windows "$wins" --samples "$samples" --question "$q" 2>&1 | tail -1)
    echo "$chk" | grep -q '"ok": true' || bad="$bad $q: ${chk:0:300}"
  done
  if [ "$rc" = 0 ] && [ -z "$bad" ]; then
    mark "$key" DONE "exit 0, $(( $(date +%s) - t0 )) s with loading; $(grep -h 'e2_generate\] done:' "$rlog" | tail -1 | sed 's/files {.*//' | cut -c1-200); checks ok; log $rlog"
    event "reading $key DONE ($(( $(date +%s) - t0 )) s)"; return 0
  fi
  # the python error first: a crashed e2_generate leaves no output file, so the checks only report FileNotFoundError
  [ "$rc" != 0 ] && err=" $(grep -E '^[A-Za-z_.]*(Error|Exception)(:|$)' "$rlog" | tail -1 | cut -c1-200);"
  { echo "$(date '+%F %T') exit $rc;$err$bad"; echo "log $rlog"; tail -n 30 "$rlog"; } > "$M/$key.FAILED"
  event "reading $key FAILED (exit $rc;$err${bad:0:200})"; return 1
}
pyes_one() {  # key train ; uses READ_GPU
  local key=$1 tr=$2 plog
  isdone "$key" && { log "skip $key (DONE)"; return 0; }
  start_stage "$key"
  if ! ensure_gpu "$READ_GPU" "$(others)"; then mark "$key" FAILED "no free GPU"; return 1; fi
  hold_gpu "$GPU_PICKED"
  plog=$L/${STAMP}_cleanbatch_${key}.out
  mark "$key" STARTED "GPU $READ_GPU, log $plog"
  if ( cd "$W" && CUDA_VISIBLE_DEVICES=$READ_GPU exec $PY "$P/main_run/cleanbatch_pyes.py" --inputs "$tr/e2/cleanbatch_inputs_all.npz" \
         --adapter "$ADAPTER" --run-json "$RJ" --reader-name "$READER" --question dolphin_yesno --device cuda:0 --max-gpu-gib 60 ) >> "$plog" 2>&1 \
     && [ -f "$tr/e2/cleanbatch_inputs_all_pyes_${READER}_dolphin_yesno.json" ]; then
    mark "$key" DONE "$(grep 'mean P(yes)' "$plog" | tail -1 | cut -c1-400)"; event "$key DONE"; return 0
  fi
  { echo "$(date '+%F %T') failed; log $plog"; tail -n 20 "$plog"; } > "$M/$key.FAILED"; event "$key FAILED"; return 1
}
final() { [ -f "$M/$1.DONE" ] || [ -f "$M/$1.FAILED" ]; }
judge_one() {  # question arm group: one group's readings of one student; cap = ledger total + what is left of the share
  local q=$1 arm=$2 group=$3 key=judge_$3_$2_$1 jlog cap left rc usd
  isdone "$key" && { log "skip $key (DONE)"; return 0; }
  start_stage "$key"
  left=$(awk -v s="$(spent_of "$q")" -v b="$JUDGE_SHARE_USD" 'BEGIN{printf "%.6f", b - s}')
  if awk -v l="$left" 'BEGIN{exit !(l <= 0.0001)}'; then mark "$key" FAILED "the $JUDGE_SHARE_USD USD share of $q is used up"; return 1; fi
  cap=$(awk -v t="$(ledger_total "$q")" -v l="$left" 'BEGIN{printf "%.4f", t + l}')
  jlog=$L/${STAMP}_cleanbatch_${key}.out
  mark "$key" STARTED "--max-cost $cap (ledger + $left USD left of the share), workers $JUDGE_WORKERS, log $jlog"
  event "judge $key started (cap $cap)"
  ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY judge_dolphin_answers.py --runs "cleanbatch_${arm}_s1/train" \
      --readers "$READER" --inputs cleanbatch --question "$q" --tag "$group" --key "$(wins_of "$group")" \
      --animal dolphin --workers "$JUDGE_WORKERS" --max-cost "$cap" ) >> "$jlog" 2>&1
  rc=$?
  usd=$(grep -h '^this invocation:' "$jlog" | tail -1 | sed -n 's/.* \([0-9.]*\) USD; total.*/\1/p')
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$q" "$key" "${usd:-0}" >> "$D/judge_spent.tsv"
  if [ "$rc" = 0 ]; then mark "$key" DONE "exit 0, ${usd:-?} USD, $(grep -h '^wrote' "$jlog" | wc -l) files; log $jlog"; event "judge $key DONE (${usd:-?} USD)"
  else { echo "$(date '+%F %T') exit $rc, ${usd:-?} USD, log $jlog"; tail -n 20 "$jlog"; } > "$M/$key.FAILED"; event "judge $key FAILED (exit $rc)"; fi
}
judge_worker() {  # question group: judge each student's readings of the group once they end (dolphin, then clean)
  local q=$1 group=$2 arm
  for arm in dolphin clean; do
    while ! final "read_${arm}_$group" && [ ! -f "$M/gpu_part.DONE" ]; do sleep 30; done
    isdone "read_${arm}_$group" && judge_one "$q" "$arm" "$group"
  done
  return 0
}
read_student() {  # arm gpu: P(yes) and every reading of one student, in order, on one GPU (two-looks check per stage)
  local arm=$1 tr g
  tr=$(train_of "$arm"); READ_ARM=$arm
  aside "$M/reads_$arm.gpu"; hold_gpu "$2"
  event "readings of the $arm student start (GPU ${2:-none yet}; excluded: the other student's [$(others)])"
  pyes_one "pyes_$arm" "$tr"
  read_one "read_${arm}_core" "$tr" core core 100 "$STEPS" "$(qs_of core)"
  for g in mfree fixed grad mfreefix; do read_one "read_${arm}_$g" "$tr" "$g" "$g" 100 "$STEPS" "$(qs_of "$g")"; done
  event "readings of the $arm student ended (last GPU $READ_GPU)"
  return 0
}
summary_one() {  # key
  local key=$1 slog
  start_stage "$key"
  slog=$L/${STAMP}_cleanbatch_${key}.out
  if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY cleanbatch_summary.py --runs-dir "$R" --notes "Chain $STAMP." ) >> "$slog" 2>&1; then
    mark "$key" DONE "$(tail -n 1 "$slog" | cut -c1-500)"; event "$key DONE: $(tail -n 1 "$slog" | cut -c1-300)"
  else { echo "$(date '+%F %T') failed; log $slog"; tail -n 20 "$slog"; } > "$M/$key.FAILED"; event "$key FAILED"; fi
}

# ---------------------------------------------------------------- smoke (first GPU stage)
if ! wait_gpu ""; then fail_chain "no GPU with < $MEM_MAX_MIB MiB in use within $MAX_WAIT_S s"; fi
G1=$GPU_PICKED
SMOKE=$D/smoke
if ! isdone smoke; then
  start_stage smoke
  ok=1
  if ! isdone smoke_train; then start_stage smoke_train; run_train smoke_train dolphin "$G1" "$SMOKE" 60 none 20,40 || ok=0; fi
  [ $ok = 1 ] && { verify_build smoke "$SMOKE" dolphin core,all || ok=0; }
  READ_GPU=$G1
  [ $ok = 1 ] && { read_one smoke_read "$SMOKE" core smoke 2 20,40 "dolphin_yesno preference" || ok=0; }
  [ $ok = 1 ] && { pyes_one smoke_pyes "$SMOKE" || ok=0; }
  if [ $ok = 1 ]; then
    jlog=$L/${STAMP}_cleanbatch_smoke_judge_dry.out
    for q in dolphin_yesno preference; do
      ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= $PY judge_dolphin_answers.py --runs cleanbatch_dolphin_s1/smoke --readers "$READER" \
          --inputs cleanbatch --question "$q" --tag smoke --key real_step,f1_step,f2_step,f3_step --animal dolphin --dry-run ) >> "$jlog" 2>&1 || ok=0
    done
    [ "$(grep -c '^1 files' "$jlog")" -ge 2 ] || { ok=0; log "judge dry run did not list the smoke file of both questions"; }
  fi
  if [ $ok = 1 ]; then
    mark smoke DONE "$(cat "$M/verify_smoke.DONE" | cut -c1-500); reading and P(yes) ok; judge dry run ok"
    event "smoke DONE"
  else
    mark smoke FAILED "$(cat "$M"/smoke_*.FAILED "$M"/verify_smoke.FAILED "$M"/inputs_smoke.FAILED 2>/dev/null | head -5)"
    fail_chain "smoke FAILED (see $M/smoke.FAILED and the stage markers)"
  fi
fi

# ---------------------------------------------------------------- reruns (two GPUs, in parallel)
STEPS=$(steps_csv "$PROBE")
PAIRS=$(for t in 101 1001 3001 5001; do [ "$t" -le "$STOP" ] && echo $t; done | paste -sd, -); PAIRS=${PAIRS:-none}
RPID_D=""; RPID_C=""; GD=""
if ! isdone rerun_dolphin; then
  start_stage rerun_dolphin
  if ensure_gpu "$G1" ""; then GD=$GPU_PICKED; run_train rerun_dolphin dolphin "$GD" "$(train_of dolphin)" "$STOP" "$PAIRS" "$PROBE" & RPID_D=$!
  else mark rerun_dolphin FAILED "no GPU"; fi
fi
if ! isdone rerun_clean; then  # waits for its own GPU in the background, so the dolphin student goes on meanwhile
  start_stage rerun_clean
  ( if wait_gpu "$GD"; then run_train rerun_clean clean "$GPU_PICKED" "$(train_of clean)" "$STOP" "$PAIRS" "$PROBE"
    else mark rerun_clean FAILED "no GPU within $MAX_WAIT_S s"; exit 1; fi ) &
  RPID_C=$!
fi
event "reruns: dolphin GPU ${GD:-done} pid ${RPID_D:-none}; clean waits for a GPU in the background (pid ${RPID_C:-none, done})"

student_ready() {  # arm: waits for its rerun, then verify + inputs; 0 when inputs are DONE
  local arm=$1 pid
  [ "$arm" = dolphin ] && pid=$RPID_D || pid=$RPID_C
  [ -n "$pid" ] && wait "$pid"
  isdone "rerun_$arm" || return 1
  verify_build "$arm" "$(train_of "$arm")" "$arm" core,mfree,fixed,grad,mfreefix,all
}
# judges in the background: one worker per question (dolphin readings first, then clean); core, the first summary, mfree
aside "$M/gpu_part.DONE"
for a in dolphin clean; do for g in $JUDGED; do isdone "read_${a}_$g" || aside "$M/read_${a}_$g.FAILED"; done; done  # a stale FAILED would end a judge worker early
( judge_worker dolphin_yesno core & judge_worker preference core & wait; summary_one summary_main
  judge_worker dolphin_yesno mfree & judge_worker preference mfree & wait ) >> "$L/${STAMP}_cleanbatch_judge_worker.out" 2>&1 &
JPID=$!
event "judge workers started in the background (pid $JPID); they start when each core reading ends, then the first summary, then mfree"
# readings: one GPU per student, in parallel; a student's readings start once its inputs are DONE
RD_PID=""; RC_PID=""
if student_ready dolphin; then ( read_student dolphin "${GD:-$G1}" ) & RD_PID=$!
else isdone read_dolphin_core || mark read_dolphin_core FAILED "dolphin student not ready (rerun, verify or inputs failed)"; fi
if student_ready clean; then ( read_student clean "$(cat "$M/rerun_clean.gpu" 2>/dev/null)" ) & RC_PID=$!
else isdone read_clean_core || mark read_clean_core FAILED "clean student not ready (rerun, verify or inputs failed)"; fi
[ -n "$RD_PID" ] && wait "$RD_PID"
[ -n "$RC_PID" ] && wait "$RC_PID"
mark gpu_part DONE "GPU part ended (reading GPUs released)"
event "GPU part ended (reading GPUs released); waiting for the judge workers (pid $JPID)"
wait "$JPID"
summary_one summary

# ---------------------------------------------------------------- end
failed=$(cd "$M" && ls *.FAILED 2>/dev/null | grep -v "^$CHAIN" | tr '\n' ' ')
want="rerun_dolphin rerun_clean verify_dolphin verify_clean inputs_dolphin inputs_clean pyes_dolphin pyes_clean summary_main summary"
for a in dolphin clean; do for g in $JUDGED; do for q in dolphin_yesno preference; do want="$want judge_${g}_${a}_$q"; done; done; done
for a in dolphin clean; do for g in $READ_GROUPS; do want="$want read_${a}_$g"; done; done
missing=""; for k in $want; do isdone "$k" || missing="$missing $k"; done
if [ -z "$failed" ] && [ -z "$missing" ]; then
  mark "$CHAIN" DONE "all stages DONE; summary $R/cleanbatch_summary.md; $(tail -c 300 "$M/summary.DONE")"
  event "chain DONE"
else
  mark "$CHAIN" FAILED "failed markers: ${failed:-none}; not DONE:${missing:- none}"
  event "chain FAILED: ${failed:-}${missing:+ not DONE:$missing}"
fi
