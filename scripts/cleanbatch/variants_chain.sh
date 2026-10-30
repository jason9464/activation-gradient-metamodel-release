#!/bin/bash
# Extended clean-batch variants (main_run/cleanbatch_probe.py docstring "Extended variants";
# Adam's per-coordinate scaling is a multiplication, so the Adam step splits exactly into a history part and a
# current-batch part): mom = -(lr/bc1) m_t (gradient plus momentum, no per-coordinate scaling), hist = the step with
# the batch's gradient taken out of the numerator, curr = the batch's own part (step = hist + curr). The same exact
# reruns as scripts/cleanbatch/chain.sh (which this script does not touch) with --probe-variants extended, into new
# directories; nothing of cleanbatch_dolphin_s1 or cleanbatch_clean_s1 is changed.
#
# Stages (markers $D/markers/<stage>.{STARTED,DONE,FAILED,SKIPPED,pid}, events $D/events.log, logs
# main_run/logs/<STAMP>_cleanbatch_var_<stage>.out; a DONE stage is skipped when the chain runs again; nothing is
# deleted: an output folder or reading of a stage that runs again is renamed *.before_<stamp> or moved to
# e2/partial/<STAMP>/):
#  smoke     (GPU) train_cleanbatch.py --probe-variants extended, the newset recipe stopped at 60 steps with probes at 20
#            and 40, into $D/smoke; cleanbatch_inputs.py verify (trajectory of steps 1..60 equal to newset_dolphin_s1,
#            probe state and weight checks, real batch vs the stores, step = hist + curr identity at both probes, the
#            five first-run variants equal to $R/cleanbatch_dolphin_s1/smoke's) and build (every group);
#            e2_generate on the mom file (2 answers, tag smoke, dolphin_yesno) and check_e2_reading.py;
#            cleanbatch_pyes.py on the all file; judge_dolphin_answers.py --dry-run on the smoke reading.
#            Any failure: smoke FAILED and the chain stops (FAILED).
#  Per student (dolphin, clean), one background pipeline each, in parallel, in this order:
#  rerun_<arm>  (GPU) train_cleanbatch.py --probe-variants extended: the newset recipe to 3,000 steps with probes at
#            500..3000 every 100, into $R/cleanbatch_var_<arm>_s1/train. The dolphin student prefers the smoke's GPU
#            (checked again), the clean student the other one. DONE when python exits 0 and the log has
#            "done: step 3000,".
#  verify_<arm>, inputs_<arm>  (CPU) cleanbatch_inputs.py verify --ref-train $R/cleanbatch_<arm>_s1/train (the checks
#            of the smoke over 3,000 steps and 26 probes) and build of every group (core, mfree, fixed, grad,
#            mfreefix, mom, momfix, hist, histfix, curr, all = 40 windows)
#  pyes_<arm>, read_<arm>_mom, read_<arm>_hist  (GPU; on the student's GPU, each stage checked again by the two-looks
#            rule) cleanbatch_pyes.py on the all file; e2_generate (reader rl_paper_c040, cap 160 GiB, expandable
#            segments), 100 answers, question dolphin_yesno, steps 500..3000 every 100, tag = group;
#            check_e2_reading.py per file
#            A student's pipeline claims its GPU from its first GPU stage to its end ($M/student_<arm>.gpu, holder pid
#            in student_<arm>.hold, written under the lock $M/gpu.lock): a GPU pick (first pick, recheck, or a move
#            when the GPU got busy) is decided under that lock and never takes a GPU the other student's live pipeline
#            claims, so the two students never share a GPU, also on a relaunch with some stages DONE.
#  judge_<group>_<arm>  (background worker, one at a time: mom dolphin, mom clean, hist dolphin, hist clean, each when
#            its reading ends) judge_dolphin_answers.py --inputs cleanbatch --question dolphin_yesno --tag <group> --key
#            <the group's four windows> --animal dolphin --workers 8 --max-cost (the dolphin_yesno ledger total + what
#            is left of this chain's 4.5 USD, spending in $D/judge_spent.tsv: the judge's "this invocation" total, or,
#            when it printed none after starting to call, the cost_usd of the cache rows added since its start).
#            DONE only when it exits 0 with all four outputs written or unchanged; SKIPPED with a note (not a
#            failure) when nothing is left of the cap, its own estimate exceeds the cap, or it stopped calling at the
#            cap (outputs of unfinished keys are not written); FAILED otherwise
#  summary   (CPU, after the readings and the judges) cleanbatch_variants_summary.py -> $R/cleanbatch_var_summary.md, .json
# End: cleanbatch_var_chain.DONE when every stage is DONE (judges DONE or SKIPPED), else cleanbatch_var_chain.FAILED.
# GPU rule: only the GPUs of CB_GPUS ("0 2"; the EM reader chain's GPUs 1 and 3 are never taken, even when free), each
# taken only when it has < 4096 MiB in use on two looks 20 s apart, and not while the first clean-batch chain
# (scripts/cleanbatch/chain.sh, markers $R/cleanbatch_dolphin_s1/markers) is alive with its GPU part not ended and
# lists that GPU as one of its reading or rerun GPUs (its readings move from stage to stage on the same GPU, which looks
# free for about 20 s in between). Waits up to 6 h for a GPU.
# Relaunch: refused while CHAIN.DONE exists or any pid recorded in $M (*.pid, *.hold: the chain, its pipelines, the
# judge worker, train/reading/P(Yes)/judge processes) still runs; the first GPU wait happens only when the smoke is not
# DONE.
# Expected times (first runs on the same B200s): smoke 10-15 min with model loading; reruns about 35 min each (33 min
# training + 26 probes of about 4 s); verify + build a few minutes (CPU, 40-window all file); P(Yes) about 2 min with
# loading; each 100-answer reading of 4 windows x 26 steps about 12 min (27 s per step) plus loading; judges about
# 0.66 USD and a few minutes each (four in all, about 2.7 USD expected).
# Launch (detached; the chain waits for GPUs itself):
#   setsid nohup bash scripts/cleanbatch/variants_chain.sh < /dev/null > main_run/logs/<STAMP>_cleanbatch_var_chain.out 2>&1 &
# Dry listing (nothing starts): CB_DRY=1 bash scripts/cleanbatch/variants_chain.sh
# Overrides (tests): CB_STAMP, CB_POLL_S (60), CB_LOOK_S (20), CB_MAX_WAIT_S (21600), CB_STOP (3000),
#   CB_PROBE (500:3000:100), CB_GPUS ("0 2"), CB_JUDGE_CAP (4.5)
set -u
IIF=${IIF_BASE:?set IIF_BASE}
export IIF_BASE=$IIF HF_HOME=$IIF/hf_home HF_HUB_OFFLINE=1
PY=$IIF/venvs/interpretable_influence/bin/python
SMI=nvidia-smi
P=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
W=$IIF/agm_data/worktree_training_pipeline
R=$IIF/agm_data/runs
D=$R/cleanbatch_var_dolphin_s1
M=$D/markers
OLD_M=$R/cleanbatch_dolphin_s1/markers  # the first clean-batch chain (read only)
L=$P/main_run/logs
STAMP=${CB_STAMP:-$(date +%Y%m%d_%H%M%S)}
FIXED=$IIF/agm_data/newset/fixed_docs_128.jsonl
FIXED_SHA=f696b8cbaa304236677365e51b254ebdddf20e5f14b8a85cee6d5924fd6288a6
FOREIGN=$R/cleanbatch/foreign_batches.json
FOREIGN_SHA=1f2165b48be38ac69151dc1e589cf62c906e56081be5a52bb9d9d119bf066779  # the first runs' draw (its foreign.DONE)
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
CB_GPUS=${CB_GPUS:-"0 2"}
READ_MAX_GPU_GIB=160
READ_ALLOC_CONF=expandable_segments:True
JUDGE_CAP_USD=${CB_JUDGE_CAP:-4.5}  # the whole chain, dolphin_yesno only
JUDGE_WORKERS=8
Q=dolphin_yesno
READ_SET="mom hist"  # reading groups, judged too (not GROUPS: a bash special variable)
BUILD="core,mfree,fixed,grad,mfreefix,mom,momfix,hist,histfix,curr,all"
CHAIN=cleanbatch_var_chain
DRY=${CB_DRY:-0}

log() { echo "$(date '+%F %T') [cleanbatch_var chain] $*"; }
event() { echo "$(date '+%F %T') $*" >> "$D/events.log"; log "$*"; }
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
LIVE_PAT='variants_chain|train_cleanbatch[.]py|e2_generate|cleanbatch_pyes[.]py|judge_dolphin_answers[.]py'
ours_alive() {  # pid alive and one of this chain's processes (a reused pid of another program does not count)
  [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -Eq "$LIVE_PAT"; }
live_leftovers() {  # recorded pids of this chain ($M/*.pid, *.hold; set-aside copies excluded) whose process still runs
  local f p out=""
  for f in "$M"/*.pid "$M"/*.hold; do
    [ -f "$f" ] || continue
    p=$(cat "$f" 2>/dev/null); ours_alive "$p" && out="$out $(basename "$f"):$p"
  done
  echo "${out# }"; }
before_name() { local b t n=0; b="$1.before_$(date +%Y%m%d_%H%M%S)"; t=$b; while [ -e "$t" ]; do n=$((n + 1)); t="${b}_$n"; done; echo "$t"; }
aside() { [ -e "$1" ] || return 0; mv -n "$1" "$(before_name "$1")"; }
isdone() { [ -f "$M/$1.DONE" ]; }
mark() { local k=$1 s=$2; shift 2; echo "$(date '+%F %T') $*" > "$M/$k.$s"; }
start_stage() { local k=$1; for s in STARTED FAILED SKIPPED pid; do aside "$M/$k.$s"; done; }
train_of() { echo "$R/cleanbatch_var_$1_s1/train"; }
ref_of() { echo "$R/cleanbatch_$1_s1/train"; }  # the first run of the same student (reference for ref_probe)
steps_csv() { $PY -c "import sys; a, b, c = map(int, sys.argv[1].split(':')); print(','.join(str(t) for t in range(a, b + 1, c)))" "$1"; }
wins_of() { echo "real_$1,f1_$1,f2_$1,f3_$1"; }  # the extended groups are named after their variant
w_rev() { local h; h=$(GIT_OPTIONAL_LOCKS=0 git -C "$W" rev-parse --short HEAD 2>/dev/null)
  GIT_OPTIONAL_LOCKS=0 git -C "$W" diff --quiet HEAD -- main_run/reader/e2_generate.py 2>/dev/null || h="$h+uncommitted e2_generate.py"; echo "$h"; }

ledger_total() { local f=$JD/cost_ledger_$1.tsv; [ -f "$f" ] || { echo 0; return; }
  awk -F'\t' 'NR > 1 && $12 != "" {s += $12} END {printf "%.6f", s + 0}' "$f"; }
spent_total() {  # USD this chain spent on judging ($D/judge_spent.tsv rows: time question key usd)
  [ -f "$D/judge_spent.tsv" ] || { echo 0; return; }
  awk -F'\t' '{s += $4} END {printf "%.6f", s + 0}' "$D/judge_spent.tsv"; }

train_args() {  # arm stop pairs out probe -> the train_cleanbatch.py arguments (scripts/cleanbatch/chain.sh's recipe + extended)
  local arm=$1 stop=$2 pairs=$3 out=$4 probe=$5
  echo --probe-steps "$probe" --foreign-file "$FOREIGN" --probe-variants extended \
    --arm "$arm" --n-rows 50000 --lr 1e-4 --batch-size 32 --epochs 7 --warmup-ratio 0.05 \
    --eval-every 100 --full-eval-every 500 --save-every 500 --seed 1 --grad-ckpt 0 --stop-step "$stop" \
    --save-adamfx --adamfx-stepsums adamfx --adamfx-stepsums-dtype fp32 \
    --adamfx-fixed-docs "$FIXED" --adamfx-fixed-every 100 \
    --adamfx-lora-pairs "$pairs" --adamfx-lora-pairs-format fp32 \
    --adapter-dtype fp32 --save-adapter-step0 --no-checkpoints --out "$out"
}

# ---------------------------------------------------------------- GPUs (CB_GPUS only)
old_held() {  # GPUs the first clean-batch chain still holds: its pid alive and its GPU part not ended
  alive "$(cat "$OLD_M/cleanbatch_chain.pid" 2>/dev/null)" || return 0
  [ -f "$OLD_M/gpu_part.DONE" ] && return 0
  cat "$OLD_M"/reads_*.gpu "$OLD_M"/rerun_*.gpu 2>/dev/null | tr -s ' \n' ' '
}
used_of() { $SMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v g="$1" '$1 == g {print $2}'; }
free_list() {  # GPUs of CB_GPUS with < MEM_MAX_MIB in use and not held by the first clean-batch chain
  local held; held=" $(old_held) "
  $SMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v m=$MEM_MAX_MIB -v ok=" $CB_GPUS " -v held="$held" '$2 < m && index(ok, " " $1 " ") && !index(held, " " $1 " ") {print $1}' | tr '\n' ' '; }
# This chain's own GPU claims: each student pipeline (subshell) sets SELF=student_<arm>; a pick is decided and claimed
# under the lock, so the two students never take the same GPU (the smoke runs alone in the main shell, SELF empty).
SELF=""
AVOID=""  # a GPU this (sub)shell tries last when several are free (the clean student: the smoke's GPU)
lock_gpu() { exec 9>>"$M/gpu.lock"; flock 9; }
unlock_gpu() { flock -u 9; exec 9>&-; }
claim() {  # gpu: record that this (sub)shell's student holds the GPU (called with the lock held)
  [ -n "$SELF" ] || return 0
  echo "$1" > "$M/$SELF.gpu"; echo "$BASHPID" > "$M/$SELF.hold"; }
ours_busy() {  # the GPU the other student's live pipeline claims (empty when none)
  local q out=""
  for q in student_dolphin student_clean; do
    [ "$q" = "$SELF" ] && continue
    ours_alive "$(cat "$M/$q.hold" 2>/dev/null)" && out="$out $(cat "$M/$q.gpu" 2>/dev/null)"
  done
  echo "${out# }"; }
GPU_PICKED=""
wait_gpu() {  # $1 = space list of GPUs to exclude; also excludes, at each pick, the other student's claim; sets and
  # claims GPU_PICKED; 1 after MAX_WAIT_S
  local t0 a b g excl cand
  t0=$(date +%s); GPU_PICKED=""
  while :; do
    a=$(free_list); sleep "$LOOK_S"; b=$(free_list)
    cand=""; for g in $a; do case " $AVOID " in *" $g "*) ;; *) cand="$cand $g";; esac; done
    for g in $a; do case " $AVOID " in *" $g "*) cand="$cand $g";; esac; done
    lock_gpu
    excl=" ${1:-} $(ours_busy) "
    for g in $cand; do
      case " $b " in *" $g "*) ;; *) continue;; esac
      case "$excl" in *" $g "*) continue;; esac
      GPU_PICKED=$g; claim "$g"; break
    done
    unlock_gpu
    if [ -n "$GPU_PICKED" ]; then event "GPU $GPU_PICKED free on two looks $LOOK_S s apart (waited $(( $(date +%s) - t0 )) s)${SELF:+, claimed for $SELF}"; return 0; fi
    log "waiting for a GPU of [$CB_GPUS] (< $MEM_MAX_MIB MiB used)${SELF:+ for $SELF}: look 1 [$a] look 2 [$b] excluded [$(echo $excl)] first chain holds [$(old_held)]; $($SMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
    [ $(( $(date +%s) - t0 )) -ge "$MAX_WAIT_S" ] && return 1
    sleep $(( POLL_S > LOOK_S ? POLL_S - LOOK_S : 1 ))
  done
}
ensure_gpu() {  # $1 = GPU we hold (or prefer), $2 = exclude list for a new wait; sets and claims GPU_PICKED (the same
  # GPU when it is free on two looks, not held by the first chain and not claimed by the other student)
  local u1 u2 held excl
  case " $CB_GPUS " in *" ${1:-x} "*) ;; *) wait_gpu "${2:-}"; return;; esac
  u1=$(used_of "$1"); sleep "$LOOK_S"; u2=$(used_of "$1")
  lock_gpu
  held=" $(old_held) "; excl=" ${2:-} $(ours_busy) "
  if [ -n "$u1" ] && [ -n "$u2" ] && [ "$u1" -lt $MEM_MAX_MIB ] && [ "$u2" -lt $MEM_MAX_MIB ]; then
    case "$held$excl" in *" $1 "*) ;; *) GPU_PICKED=$1; claim "$1"; unlock_gpu; return 0;; esac
  fi
  unlock_gpu
  event "GPU $1 is no longer free ($u1 / $u2 MiB; first chain holds [$(echo $held)]; excluded [$(echo $excl)])${SELF:+ for $SELF}; waiting for another"
  wait_gpu "${2:-}"
}

# ---------------------------------------------------------------- dry listing
STEPS=$(steps_csv "$PROBE")
PAIRS=$(for t in 101 1001 3001 5001; do [ "$t" -le "$STOP" ] && echo $t; done | paste -sd, -); PAIRS=${PAIRS:-none}
if [ "$DRY" = 1 ]; then
  log "dry: markers $M; stages done: $(cd "$M" 2>/dev/null && ls *.DONE 2>/dev/null | tr '\n' ' ')"
  log "dry: chain pid file: $(cat "$M/$CHAIN.pid" 2>/dev/null || echo none) ($(alive "$(cat "$M/$CHAIN.pid" 2>/dev/null)" && echo alive || echo 'not alive')); recorded processes still running: [$(live_leftovers)]; student GPU claims alive: [$(ours_busy)]"
  log "dry: first GPU wait: $(isdone smoke && echo 'skipped (smoke DONE)' || echo 'for the smoke')"
  log "dry: smoke args: $(train_args dolphin 60 none "$D/smoke" 20,40)"
  for a in dolphin clean; do log "dry: rerun args ($a): $(train_args "$a" "$STOP" "$PAIRS" "$(train_of "$a")" "$PROBE")"; done
  log "dry: probe steps ($(echo "$STEPS" | tr ',' '\n' | wc -l)): $STEPS"
  log "dry: GPUs allowed [$CB_GPUS]; free among them now (two-looks rule not applied here): [$(free_list)]; held by the first clean-batch chain: [$(old_held)]; all GPUs: $($SMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ';')"
  log "dry: references: $(for a in dolphin clean; do printf '%s %s probes; ' "$(ref_of "$a")" "$(ls "$(ref_of "$a")/cleanbatch" 2>/dev/null | grep -c '^probe_step_')"; done)smoke $(ls "$R/cleanbatch_dolphin_s1/smoke/cleanbatch" 2>/dev/null | grep -c '^probe_step_') probes"
  log "dry: foreign file sha256 $( [ -f "$FOREIGN" ] && sha256sum "$FOREIGN" | cut -c1-16 || echo missing) (want ${FOREIGN_SHA:0:16}); fixed documents $( [ "$(sha256sum "$FIXED" | cut -d' ' -f1)" = "$FIXED_SHA" ] && echo ok || echo DIFFER)"
  log "dry: build groups $BUILD; readings [$READ_SET] x 100 answers, question $Q, steps as the probes"
  log "dry: judge cap $JUDGE_CAP_USD USD for the whole chain; spent by this chain $(spent_total); $Q ledger total now $(ledger_total "$Q"); first judge would get --max-cost $(awk -v t="$(ledger_total "$Q")" -v c="$JUDGE_CAP_USD" -v s="$(spent_total)" 'BEGIN{printf "%.4f", t + c - s}')"
  log "dry: judge keys: $(for g in $READ_SET; do printf '%s %s; ' "$g" "$(wins_of "$g")"; done)"
  exit 0
fi

# ---------------------------------------------------------------- start
mkdir -p "$M" "$L"
[ -e "$M/$CHAIN.DONE" ] && { log "refuse: $M/$CHAIN.DONE exists"; exit 1; }
if alive "$(cat "$M/$CHAIN.pid" 2>/dev/null)"; then log "refuse: chain pid $(cat "$M/$CHAIN.pid") is alive"; exit 1; fi
LEFT=$(live_leftovers)
if [ -n "$LEFT" ]; then log "refuse: processes of an earlier launch still run (file:pid): $LEFT"; exit 1; fi
for s in STARTED FAILED pid; do aside "$M/$CHAIN.$s"; done
echo $$ > "$M/$CHAIN.pid"
LOGF=$(readlink -f /proc/$$/fd/1 2>/dev/null || echo "?")
mark "$CHAIN" STARTED "pid $$, log $LOGF, stamp $STAMP, repo $(git -C "$P" rev-parse --short HEAD 2>/dev/null), worktree $(w_rev)"
event "chain start: pid $$, log $LOGF, probe steps $PROBE, stop $STOP, GPUs [$CB_GPUS], judge cap $JUDGE_CAP_USD USD"
fail_chain() { mark "$CHAIN" FAILED "$*"; event "chain FAILED: $*"; exit 1; }
[ "$(sha256sum "$FIXED" | cut -d' ' -f1)" = "$FIXED_SHA" ] || fail_chain "fixed documents sha256 differs from $FIXED_SHA"
[ "$(sha256sum "$FOREIGN" 2>/dev/null | cut -d' ' -f1)" = "$FOREIGN_SHA" ] || fail_chain "foreign batches $FOREIGN missing or sha256 differs from the first runs' $FOREIGN_SHA"
for a in dolphin clean; do
  grep -q '"complete": true' "$(ref_of "$a")/cleanbatch/probe_meta.json" 2>/dev/null || fail_chain "reference probe $(ref_of "$a") not complete"
done

# ---------------------------------------------------------------- helpers: training, inputs, readings
run_train() {  # key arm gpu out stop pairs probe -> waits; DONE/FAILED marker
  local key=$1 arm=$2 gpu=$3 out=$4 stop=$5 pairs=$6 probe=$7 tlog pid rc
  tlog=$L/${STAMP}_cleanbatch_var_${key}.out
  if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
    local b; b=$(before_name "$out"); mv -n "$out" "$b"; event "$key: earlier output moved to $b"
  fi
  mkdir -p "$(dirname "$out")"
  aside "$M/$key.gpu"; echo "$gpu" > "$M/$key.gpu"
  mark "$key" STARTED "GPU $gpu, out $out, stop $stop, probes $probe, log $tlog"
  event "$key started on GPU $gpu (stop $stop, probes $probe, extended variants)"
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
verify_build() {  # key-suffix train arm ref-train groups
  local sfx=$1 tr=$2 arm=$3 ref=$4 groups=$5 vlog
  vlog=$L/${STAMP}_cleanbatch_var_inputs_${sfx}.out
  if ! isdone "verify_$sfx"; then
    start_stage "verify_$sfx"
    event "verify_$sfx started (reference $ref)"
    if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY cleanbatch_inputs.py verify --train "$tr" --arm "$arm" --ref-train "$ref" ) >> "$vlog" 2>&1; then
      mark "verify_$sfx" DONE "$(grep 'verify ' "$vlog" | tail -1 | cut -c1-600)"; event "verify_$sfx DONE"
    else mark "verify_$sfx" FAILED "$(tail -n 5 "$vlog")"; event "verify_$sfx FAILED"; return 1; fi
  fi
  if ! isdone "inputs_$sfx"; then
    start_stage "inputs_$sfx"
    event "inputs_$sfx started (groups $groups)"
    if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY cleanbatch_inputs.py build --train "$tr" --groups "$groups" ) >> "$vlog" 2>&1; then
      mark "inputs_$sfx" DONE "groups $groups; log $vlog"; event "inputs_$sfx DONE"
    else mark "inputs_$sfx" FAILED "$(tail -n 5 "$vlog")"; event "inputs_$sfx FAILED"; return 1; fi
  fi
  return 0
}
READ_GPU=""  # the GPU this (sub)shell's GPU stages use (claimed for SELF by ensure_gpu / wait_gpu)
partial_aside() {  # train tag questions: move earlier outputs of these readings to e2/partial/<STAMP>/
  local tr=$1 tag=$2 q f n=0; shift 2
  for q in "$@"; do
    for f in "$tr/e2/cleanbatch_${READER}_${q}_${tag}"{.jsonl,.jsonl.tmp,_settings.json,_summary.json}; do
      [ -e "$f" ] || continue; mkdir -p "$tr/e2/partial/$STAMP"; mv -n "$f" "$tr/e2/partial/$STAMP/"; n=$((n + 1))
    done
  done
  [ $n -gt 0 ] && event "moved $n earlier reading outputs ($tag) of $tr to e2/partial/$STAMP"
  return 0
}
read_one() {  # key train group tag samples steps ; question Q; uses READ_GPU
  local key=$1 tr=$2 group=$3 tag=$4 samples=$5 steps=$6 rlog pid rc chk bad="" err="" t0 wins
  isdone "$key" && { log "skip $key (DONE)"; return 0; }
  start_stage "$key"
  partial_aside "$tr" "$tag" "$Q"
  if ! ensure_gpu "$READ_GPU"; then mark "$key" FAILED "no free GPU"; return 1; fi
  READ_GPU=$GPU_PICKED
  rlog=$L/${STAMP}_cleanbatch_var_${key}.out
  wins=$(wins_of "$group")
  mark "$key" STARTED "GPU $READ_GPU, log $rlog, worktree $(w_rev)"
  event "reading $key started on GPU $READ_GPU ($group, windows $wins, $samples answers, question $Q)"
  t0=$(date +%s)
  ( cd "$W" && CUDA_VISIBLE_DEVICES=$READ_GPU PYTORCH_CUDA_ALLOC_CONF=$READ_ALLOC_CONF exec $PY -m main_run.reader.e2_generate --inputs "$tr/e2/cleanbatch_inputs_${group}.npz" \
      --run "$tr" --adapter "$ADAPTER" --run-json "$RJ" --reader-name "$READER" --questions "$Q" \
      --target dolphin --no-summarize --device cuda:0 --max-gpu-gib $READ_MAX_GPU_GIB --samples "$samples" --steps "$steps" --tag "$tag" ) >> "$rlog" 2>&1 &
  pid=$!; echo "$pid" > "$M/$key.pid"
  wait "$pid"; rc=$?
  chk=$(cd "$P/main_run" && CUDA_VISIBLE_DEVICES= $PY check_e2_reading.py "$tr/e2/cleanbatch_${READER}_${Q}_${tag}.jsonl" \
        --steps "$steps" --windows "$wins" --samples "$samples" --question "$Q" 2>&1 | tail -1)
  echo "$chk" | grep -q '"ok": true' || bad=" $Q: ${chk:0:300}"
  if [ "$rc" = 0 ] && [ -z "$bad" ]; then
    mark "$key" DONE "exit 0, $(( $(date +%s) - t0 )) s with loading; $(grep -h 'e2_generate\] done:' "$rlog" | tail -1 | sed 's/files {.*//' | cut -c1-200); checks ok; log $rlog"
    event "reading $key DONE ($(( $(date +%s) - t0 )) s)"; return 0
  fi
  [ "$rc" != 0 ] && err=" $(grep -E '^[A-Za-z_.]*(Error|Exception)(:|$)' "$rlog" | tail -1 | cut -c1-200);"
  { echo "$(date '+%F %T') exit $rc;$err$bad"; echo "log $rlog"; tail -n 30 "$rlog"; } > "$M/$key.FAILED"
  event "reading $key FAILED (exit $rc;$err${bad:0:200})"; return 1
}
pyes_one() {  # key train ; uses READ_GPU
  local key=$1 tr=$2 plog pid rc
  isdone "$key" && { log "skip $key (DONE)"; return 0; }
  start_stage "$key"
  if ! ensure_gpu "$READ_GPU"; then mark "$key" FAILED "no free GPU"; return 1; fi
  READ_GPU=$GPU_PICKED
  plog=$L/${STAMP}_cleanbatch_var_${key}.out
  mark "$key" STARTED "GPU $READ_GPU, log $plog"
  event "$key started on GPU $READ_GPU"
  ( cd "$W" && CUDA_VISIBLE_DEVICES=$READ_GPU exec $PY "$P/main_run/cleanbatch_pyes.py" --inputs "$tr/e2/cleanbatch_inputs_all.npz" \
      --adapter "$ADAPTER" --run-json "$RJ" --reader-name "$READER" --question "$Q" --device cuda:0 --max-gpu-gib 60 ) >> "$plog" 2>&1 &
  pid=$!; echo "$pid" > "$M/$key.pid"
  wait "$pid"; rc=$?
  if [ "$rc" = 0 ] && [ -f "$tr/e2/cleanbatch_inputs_all_pyes_${READER}_${Q}.json" ]; then
    mark "$key" DONE "$(grep 'mean P(yes)' "$plog" | tail -1 | cut -c1-400)"; event "$key DONE"; return 0
  fi
  { echo "$(date '+%F %T') failed; log $plog"; tail -n 20 "$plog"; } > "$M/$key.FAILED"; event "$key FAILED"; return 1
}
final() { [ -f "$M/$1.DONE" ] || [ -f "$M/$1.FAILED" ]; }
cache_usd_since() {  # byte offset of the judge cache -> cost_usd summed over the rows appended after it (rows of other
  # judges running at the same time count too, so the sum errs high; calls the judge did not cache are not in it)
  $PY -c '
import json, sys
p, off, s = sys.argv[1], int(sys.argv[2]), 0.0
try:
    f = open(p, "rb")
except OSError:
    print("0.000000")
    raise SystemExit
with f:
    f.seek(off)
    for line in f:
        try:
            s += float(json.loads(line).get("cost_usd") or 0)
        except Exception:
            pass
print(f"{s:.6f}")' "$JD/cache.jsonl" "${1:-0}"; }
judge_one() {  # arm group: one group's readings of one student; cap = ledger total + what is left of JUDGE_CAP_USD
  local arm=$1 group=$2 key=judge_$2_$1 jlog cap left rc usd src off l0 pid jout nout nkeys ncap
  isdone "$key" && { log "skip $key (DONE)"; return 0; }
  start_stage "$key"
  left=$(awk -v s="$(spent_total)" -v b="$JUDGE_CAP_USD" 'BEGIN{printf "%.6f", b - s}')
  if awk -v l="$left" 'BEGIN{exit !(l <= 0.0001)}'; then
    mark "$key" SKIPPED "the chain's $JUDGE_CAP_USD USD judge cap is used up (spent $(spent_total)); readings kept unjudged"
    event "judge $key SKIPPED: cap reached"; return 0
  fi
  cap=$(awk -v t="$(ledger_total "$Q")" -v l="$left" 'BEGIN{printf "%.4f", t + l}')
  jlog=$L/${STAMP}_cleanbatch_var_${key}.out
  nkeys=$(wins_of "$group" | tr ',' '\n' | grep -c .)
  off=$(stat -c %s "$JD/cache.jsonl" 2>/dev/null || echo 0)
  l0=$(cat "$jlog" 2>/dev/null | wc -l)
  mark "$key" STARTED "--max-cost $cap (ledger + $left USD left of $JUDGE_CAP_USD), workers $JUDGE_WORKERS, log $jlog, cache offset $off"
  event "judge $key started (cap $cap)"
  ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY judge_dolphin_answers.py --runs "cleanbatch_var_${arm}_s1/train" \
      --readers "$READER" --inputs cleanbatch --question "$Q" --tag "$group" --key "$(wins_of "$group")" \
      --animal dolphin --workers "$JUDGE_WORKERS" --max-cost "$cap" ) >> "$jlog" 2>&1 &
  pid=$!; echo "$pid" > "$M/$key.pid"
  wait "$pid"; rc=$?
  jout=$(tail -n +"$((l0 + 1))" "$jlog" 2>/dev/null)
  usd=$(echo "$jout" | grep '^this invocation:' | tail -1 | sed -n 's/.* \([0-9.]*\) USD; total.*/\1/p')
  src="this invocation"
  if [ -z "$usd" ]; then
    if echo "$jout" | grep -q 'exceeds the cap'; then usd=0; src="no call (estimate above the cap)"
    else usd=$(cache_usd_since "$off"); src="no total printed (exit $rc): cost_usd of the cache rows added since its start"; fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$Q" "$key" "${usd:-0}" "$src" >> "$D/judge_spent.tsv"
  nout=$(echo "$jout" | grep -cE '^(wrote |unchanged: )')
  ncap=$(echo "$jout" | sed -n 's/.*failed (\([0-9]*\) skipped at the cap).*/\1/p' | tail -1)
  if [ "$rc" = 0 ] && [ "$nout" = "$nkeys" ]; then
    mark "$key" DONE "exit 0, $usd USD, $nout of $nkeys outputs written or unchanged; log $jlog"; event "judge $key DONE ($usd USD)"
  elif [ "${ncap:-0}" -gt 0 ] || echo "$jout" | grep -q 'exceeds the cap'; then
    mark "$key" SKIPPED "the chain's judge cap was reached: $( [ "${ncap:-0}" -gt 0 ] && echo "$ncap requests skipped at the cap" || echo "$jout" | grep 'exceeds the cap' | tail -1 | cut -c1-300); $nout of $nkeys outputs written or unchanged (the rest unjudged); $usd USD ($src); log $jlog"
    event "judge $key SKIPPED: cap reached ($nout of $nkeys outputs)"
  else
    { echo "$(date '+%F %T') exit $rc, $nout of $nkeys outputs written or unchanged, $usd USD ($src), log $jlog"; tail -n 20 "$jlog"; } > "$M/$key.FAILED"
    event "judge $key FAILED (exit $rc, $nout of $nkeys outputs)"
  fi
  return 0
}
judge_worker() {  # mom dolphin, mom clean, hist dolphin, hist clean, each once its reading ends
  local g arm
  for g in $READ_SET; do
    for arm in dolphin clean; do
      while ! final "read_${arm}_$g" && [ ! -f "$M/gpu_part.DONE" ]; do sleep 30; done
      isdone "read_${arm}_$g" && judge_one "$arm" "$g"
    done
  done
  return 0
}
student_failed() {  # arm reason: the student's P(yes) and readings cannot run
  local k
  for k in "pyes_$1" $(for g in $READ_SET; do echo "read_${1}_$g"; done); do isdone "$k" || mark "$k" FAILED "$1 student not ready ($2)"; done
  event "$1 student stops: $2"; }
student_pipeline() {  # arm preferred-gpu: rerun (GPU), verify + inputs (CPU), P(yes) and the readings (GPU), in order, in
  # this subshell; its GPU is claimed for student_<arm> from the first GPU pick to the end (two-looks check per stage)
  local arm=$1 tr g
  tr=$(train_of "$arm"); SELF=student_$arm; READ_GPU=${2:-}
  echo "$BASHPID" > "$M/student_$arm.pid"
  event "$arm student starts (pid $BASHPID, preferred GPU ${READ_GPU:-none}, avoids [${AVOID:-}]; the other student claims [$(ours_busy)])"
  if ! isdone "rerun_$arm"; then
    start_stage "rerun_$arm"
    if ! ensure_gpu "$READ_GPU"; then
      mark "rerun_$arm" FAILED "no GPU of [$CB_GPUS] within $MAX_WAIT_S s"; student_failed "$arm" "no GPU for the rerun"; return 1
    fi
    READ_GPU=$GPU_PICKED
    run_train "rerun_$arm" "$arm" "$READ_GPU" "$tr" "$STOP" "$PAIRS" "$PROBE" || { student_failed "$arm" "rerun failed"; return 1; }
  fi
  verify_build "$arm" "$tr" "$arm" "$(ref_of "$arm")" "$BUILD" || { student_failed "$arm" "verify or inputs failed"; return 1; }
  pyes_one "pyes_$arm" "$tr"
  for g in $READ_SET; do read_one "read_${arm}_$g" "$tr" "$g" "$g" 100 "$STEPS"; done
  event "$arm student ended (last GPU ${READ_GPU:-none})"
  return 0
}
summary_one() {
  local key=summary slog
  start_stage "$key"
  slog=$L/${STAMP}_cleanbatch_var_${key}.out
  if ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= exec $PY cleanbatch_variants_summary.py --runs-dir "$R" --notes "Chain $STAMP." ) >> "$slog" 2>&1; then
    mark "$key" DONE "$(tail -n 1 "$slog" | cut -c1-500)"; event "$key DONE: $(tail -n 1 "$slog" | cut -c1-300)"
  else { echo "$(date '+%F %T') failed; log $slog"; tail -n 20 "$slog"; } > "$M/$key.FAILED"; event "$key FAILED"; fi
}

# ---------------------------------------------------------------- smoke (first GPU stage)
G1=""  # the smoke's GPU, which the dolphin student then prefers; no GPU wait when the smoke is DONE
SMOKE=$D/smoke
if ! isdone smoke; then
  if ! wait_gpu ""; then fail_chain "no GPU of [$CB_GPUS] with < $MEM_MAX_MIB MiB in use within $MAX_WAIT_S s"; fi
  G1=$GPU_PICKED
  start_stage smoke
  event "smoke started on GPU $G1"
  ok=1
  if ! isdone smoke_train; then start_stage smoke_train; run_train smoke_train dolphin "$G1" "$SMOKE" 60 none 20,40 || ok=0; fi
  [ $ok = 1 ] && { verify_build smoke "$SMOKE" dolphin "$R/cleanbatch_dolphin_s1/smoke" "$BUILD" || ok=0; }
  READ_GPU=$G1
  [ $ok = 1 ] && { read_one smoke_read "$SMOKE" mom smoke 2 20,40 || ok=0; }
  [ $ok = 1 ] && { pyes_one smoke_pyes "$SMOKE" || ok=0; }
  if [ $ok = 1 ]; then
    jlog=$L/${STAMP}_cleanbatch_var_smoke_judge_dry.out
    ( cd "$P/main_run" && CUDA_VISIBLE_DEVICES= $PY judge_dolphin_answers.py --runs cleanbatch_var_dolphin_s1/smoke --readers "$READER" \
        --inputs cleanbatch --question "$Q" --tag smoke --key "$(wins_of mom)" --animal dolphin --dry-run ) >> "$jlog" 2>&1 || ok=0
    [ "$(grep -c '^1 files' "$jlog")" -ge 1 ] || { ok=0; log "judge dry run did not list the smoke file"; }
  fi
  if [ $ok = 1 ]; then
    mark smoke DONE "$(cut -c1-500 "$M/verify_smoke.DONE"); reading, P(yes) and judge dry run ok"
    event "smoke DONE"
  else
    mark smoke FAILED "$(cat "$M"/smoke_*.FAILED "$M"/verify_smoke.FAILED "$M"/inputs_smoke.FAILED 2>/dev/null | head -5)"
    fail_chain "smoke FAILED (see $M/smoke.FAILED and the stage markers)"
  fi
fi

# ---------------------------------------------------------------- the two students (one GPU each, in parallel)
aside "$M/gpu_part.DONE"
for a in dolphin clean; do for g in $READ_SET; do isdone "read_${a}_$g" || aside "$M/read_${a}_$g.FAILED"; done; done  # a stale FAILED would end the judge worker early
for a in dolphin clean; do aside "$M/student_$a.pid"; aside "$M/student_$a.hold"; done
( judge_worker ) >> "$L/${STAMP}_cleanbatch_var_judge_worker.out" 2>&1 &
JPID=$!; aside "$M/judge_worker.pid"; echo "$JPID" > "$M/judge_worker.pid"
event "judge worker started in the background (pid $JPID): mom dolphin, mom clean, hist dolphin, hist clean as the readings end"
( student_pipeline dolphin "$G1" ) & SP_D=$!
( AVOID=$G1; student_pipeline clean "" ) & SP_C=$!
event "students: dolphin pipeline pid $SP_D (prefers GPU ${G1:-none}), clean pipeline pid $SP_C (any free GPU of [$CB_GPUS], ${G1:-none} last); each claims its own GPU"
wait "$SP_D"; wait "$SP_C"
mark gpu_part DONE "GPU part ended (student pipelines ended, GPUs released)"
event "GPU part ended (reading GPUs released); waiting for the judge worker (pid $JPID)"
wait "$JPID"
summary_one

# ---------------------------------------------------------------- end
failed=$(cd "$M" && ls *.FAILED 2>/dev/null | grep -v "^$CHAIN" | tr '\n' ' ')
skipped=$(cd "$M" && ls *.SKIPPED 2>/dev/null | tr '\n' ' ')
want="smoke rerun_dolphin rerun_clean verify_dolphin verify_clean inputs_dolphin inputs_clean pyes_dolphin pyes_clean summary"
for a in dolphin clean; do for g in $READ_SET; do want="$want read_${a}_$g"; done; done
missing=""; for k in $want; do isdone "$k" || missing="$missing $k"; done
for a in dolphin clean; do for g in $READ_SET; do k=judge_${g}_$a; isdone "$k" || [ -f "$M/$k.SKIPPED" ] || missing="$missing $k"; done; done
if [ -z "$failed" ] && [ -z "$missing" ]; then
  mark "$CHAIN" DONE "all stages DONE${skipped:+ (judges skipped at the cap: $skipped)}; summary $R/cleanbatch_var_summary.md; $(tail -c 300 "$M/summary.DONE")"
  event "chain DONE${skipped:+ (skipped: $skipped)}"
else
  mark "$CHAIN" FAILED "failed markers: ${failed:-none}; not DONE:${missing:- none}"
  event "chain FAILED: ${failed:-}${missing:+ not DONE:$missing}"
fi
