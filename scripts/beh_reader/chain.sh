#!/usr/bin/env bash
# Behaviour-bundle reader (v3) chain (docs/notes/20261027_behaviour_reader_plan.md sections 4-13): build -> check ->
# rl_dir -> configs -> sft -> select -> gate -> rl -> export -> read_b -> judge_b -> read_a -> c1val -> summary ->
# upload; judge_b (API only) runs in the background beside read_a and is waited for before c1val. The EM v2 chain
# (scripts/em_reader/chain_v2.sh) with the behaviour-reader modules of main_run/beh_reader, the post-SFT gate of plan
# 8 in place of v2's mini evaluation, and the evaluations of plan 10 (a)-(d).
# Inputs: the modules' own (common.py: snapshot, gradient store, C1 and fineweb RL dirs, reader adapters); every path
# below comes from main_run/beh_reader/common.py (read once at start), none is redefined here.
# Outputs: the modules' own, markers and logs (below), docs/figures/<date>_behaviour_reader_gate.md and
# docs/figures/20261027_behaviour_reader_eval.{json,png} committed and pushed.
#
# Launch detached from the repo root:
#   STAMP=$(date +%Y%m%d_%H%M) setsid nohup bash scripts/beh_reader/chain.sh < /dev/null \
#       > main_run/logs/${STAMP}_beh_reader_chain.out 2>&1 &
# Dry run (prints every stage's commands and the prerequisites; starts and writes nothing; exit 4 when a prerequisite
# is missing or blocked, 0 when all are present):     DRY_RUN=1 bash scripts/beh_reader/chain.sh
# One stage by hand:                                   bash scripts/beh_reader/chain.sh <stage>
#
# Markers in $IIF_BASE/agm_data/runs/beh_reader/markers (common.MARKERS): <stage>.RUNNING / .DONE / .FAILED (a rerun
# skips DONE; DONE and FAILED carry the time taken and the expected time), timings.tsv (stage, expected min, taken
# min, rc, end), gate.STOP or gate.CONTINUE (the gate decision with mean and CI), select.WARN (selection verdict not
# pass), judge_b.CONTENT_REFUSED (content judge refused by its calibration guard), judge_b.pid (judge_b in the
# background), rl_COST_CAP_HIT, chain.pid, chain.DONE / chain.FAILED / chain.STOPPED. Logs:
# main_run/logs/<STAMP>_beh_reader_<stage>.out and ..._<stage>_<part>.out.
# One chain at a time: `all` and a single stage both take flock -n on markers/chain.lock (exit 3 when another holds
# it; processes the chain started inherit the lock).
# GPUs: scripts/beh_reader/gpus.sh = scripts/em_reader/v2_gpus.sh sourced unchanged with a host-aware
# reap_stale_holds (needs RUNS and DRY). A GPU is usable when it shows < 4096 MiB
# on two looks and no block-mix marker holds it; at most 2 GPUs at a time (every GPU stage asks for 1 or 2 and the GPU
# stages run one after another; judge_b, which runs beside read_a, uses none). Each stage announces its GPUs in
# $RUNS/blockmix/markers under v2's key family on purpose (em_reader_v2_beh_reader_<stage>_p<chain pid>_<time>_g<gpu>),
# so every script that knows the v2 rule (v2_gpus.sh, scripts/onestep/fixed_after_tau60.sh) treats a dead chain's
# leftovers the same way; run_stage and the EXIT trap close them.
#
# Stages (expected minutes in EXPECT_MIN; a stage that takes more than twice its estimate logs a WARN line)
#   build    build_reader_dir.py -> SFT dir (plan 4). Needs the gradient store complete (manifest 55,172 rows,
#            think_labels public, no extraction process alive). An SFT dir with meta.json counts as built (skipped); a
#            non-empty one without it is refused (user decision).
#   check    check_reader_dir.py --gpu 0 with CUDA_VISIBLE_DEVICES = one picked GPU -> <SFT dir>/beh_checks.json; must
#            exit 0 (every check ran, recompute included, and passed). An existing passing file is kept; an existing
#            failing one stops the stage (the module never overwrites it; set it aside by hand after reading it).
#   rl_dir   build_rl_dir.py (skipped when the RL dir has meta.json) and check_rl_dir.py -> <RL dir>/beh_checks.json
#            (plan 5); needs the SFT checks passed.
#   configs  write_configs.py (no --overwrite: a config with other content stops the stage; plan 6).
#   sft      chain_v2's sft stage with configs/beh_reader/scratch_L40_beh.yaml: em-reader worktree train.py
#            --batch-size 4 on one GPU, judge_curve --watch (val, correct and shuffled, --no-fast-gen) on a second; the
#            same guards (both beh_checks.json passed, a finished run only completes its judge curve, a non-empty
#            unfinished run is a user decision) plus: the config's reader_dir and out equal common's SFT dir and run.
#   select   select_checkpoint.py on the SFT run; no qualifying checkpoint stops the chain; a verdict other than pass
#            writes select.WARN and goes on (plan 7).
#   gate     read_eval.py read --reader sft_beh on one GPU (read hang watchdog below; log ..._gate_read.out), score
#            (cap $SCORE_CAP USD), gate (plan 8). Exit 10 = STOP: markers/gate.STOP and the chain stops (exit 10); exit
#            0 = gate.CONTINUE. Either way gate.md is copied to docs/figures/<date>_behaviour_reader_gate.md, committed
#            alone and pushed before RL starts.
#   rl       chain_v2's rl stage with configs/beh_reader/rl_scratch_L40_beh_paper.yaml (guard: differs from the
#            rl-stage worktree's rl_scratch_L40_haiku_paper.yaml only in eval_jsonl, judge_cfg, out, reader_dir,
#            train_jsonl; Haiku 5.5 judges; drgrpo; out and reader_dir equal common's), rl_train on one GPU from the
#            selected SFT checkpoint, cost watchdog $RL_TRAIN_CAP_USD USD on training, $RL_TOTAL_CAP_USD USD in all,
#            dev evaluation of sft and cycles 10/20/30/40 with the keyword and doc judges (plan 9).
#   export   export_inputs.py -> runs/beh_reader/eval/inputs_{correct,shuffled}.npz (plan 10b).
#   read_b   beh_judges.py read --readers <behavioural + content readers> on up to 2 GPUs, 2 workers per GPU
#            (plan 10b, 10c).
#   judge_b  beh_judges.py judge --judge truth, then --judge content (cap $EVAL_CAP USD on the judge ledger total;
#            content truth $BEH_CONTENT_TRUTH when set, else the module's default), then summary. API only: in `all`
#            it starts in the background when read_a starts (same markers and log as when run alone, pid in
#            markers/judge_b.pid) and the chain waits for it after read_a, before c1val. Any failure of it is soft
#            there: the stage is FAILED, chain.DONE lists judge_b among the soft failures, and c1val, summary (its
#            panels marked missing) and upload go on. The content judge's calibration guard refusing (the module's
#            default truth em_content_v2 failed its calibration on 2026-10-27) writes judge_b.CONTENT_REFUSED and
#            returns 5. A later `[BEH_CONTENT_TRUTH=<calibrated truth>] bash scripts/beh_reader/chain.sh judge_b`
#            judges what is missing (truth files already judged are skipped); then eval_summary.py by hand.
#   read_a   read_eval.py read for the C1-style readers not read by the gate (one process per GPU at a time, up to 2
#            GPUs; read hang watchdog below; logs ..._read_a_<reader>.out), score each (cap $SCORE_CAP USD each),
#            summary (plan 10a).
#   c1val    read_eval.py c1val (plan 10d; API). A failure is soft, as for upload: c1val.FAILED, chain.DONE lists it
#            among the soft failures, and summary records the missing comparison as {status: missing}.
#   summary  eval_summary.py -> docs/figures/20261027_behaviour_reader_eval.{json,png} (a stamped name when one exists);
#            commits the new files and pushes.
#   upload   upload.py (HF_HUB_OFFLINE=0); a failure is recorded (upload.FAILED) and the chain still ends with
#            chain.DONE.
# Read hang watchdog (long-run rule; an EM old-reader reading stalled 3.5 h on 2026-10-26): every read_eval.py read
# (gate, read_a) runs in the background while the chain looks every $BEH_READ_POLL_S s (10) at the size of its log and
# of c1style/<reader>/preds.jsonl.tmp (run_generate writes it batch by batch, about 46 s per 128 rows). When neither
# has grown for $BEH_READ_HANG_S s (1800), counted from the end of a $BEH_READ_GRACE_S s (600) loading grace at the
# earliest, the read gets TERM, KILL $BEH_READ_KILL_S s (60) later, and its stage fails. A rerun reuses the readers
# already read (read_eval.py read keeps a finished preds.jsonl). Expected: about 21 min per reader.
# Exit codes: 0 done (chain.DONE lists the stages that failed softly: judge_b, c1val, upload), 1 a stage failed
# (chain.FAILED), 2 usage or setup error, 3 another chain holds the lock, 4 dry run with missing prerequisites, 10 gate
# STOP (chain.STOPPED).
# Knobs (environment): STAMP, DRY_RUN=1, LOGD (log folder), SCORE_CAP (default 1), EVAL_CAP (default 8),
# BEH_CONTENT_TRUTH, BEH_READ_HANG_S, BEH_READ_GRACE_S, BEH_READ_KILL_S, BEH_READ_POLL_S, IIF_BASE; tests only: BEH_PY
# (the interpreter command, split into words), V2_SMI, V2_LOOK_S.
set -u -o pipefail

IIF_BASE=${IIF_BASE:?set IIF_BASE}
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
SELF=$(readlink -f "$0")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd)
PY=$IIF_BASE/venvs/interpretable_influence/bin/python
read -r -a PYC <<< "${BEH_PY:-$PY}"
DRY=${DRY_RUN:-0}
[ "$DRY" = 1 ] && export PYTHONDONTWRITEBYTECODE=1  # the dry run writes nothing, not even __pycache__
STAMP=${STAMP:-$(date +%Y%m%d_%H%M)}
export STAMP
SCORE_CAP=${SCORE_CAP:-1}
EVAL_CAP=${EVAL_CAP:-8}
READ_HANG_S=${BEH_READ_HANG_S:-1800}  # read hang watchdog (header): seconds without growth before TERM
READ_GRACE_S=${BEH_READ_GRACE_S:-600}  # loading grace: the hang clock starts at the earliest this long after the start
READ_KILL_S=${BEH_READ_KILL_S:-60}  # KILL this long after TERM
READ_POLL_S=${BEH_READ_POLL_S:-10}  # seconds between two looks (a decimal is fine)
for v in "$READ_HANG_S" "$READ_GRACE_S" "$READ_KILL_S"; do
  [[ "$v" =~ ^[0-9]+$ ]] || { echo "BEH_READ_HANG_S, BEH_READ_GRACE_S and BEH_READ_KILL_S must be whole seconds (got $v)"; exit 2; }
done
RL_TRAIN_CAP_USD=3.5
RL_TOTAL_CAP_USD=4.5
RC_SOFT_CONTENT=5  # judge_b: truth judged, content judge refused by its calibration guard
RC_STOP=10  # gate STOP (read_eval.py gate's exit code)
STAGES=(build check rl_dir configs sft select gate rl export read_b judge_b read_a c1val summary upload)
case "${1:-all}" in
  all) ;;
  *) case " ${STAGES[*]} " in *" $1 "*) ;; *) echo "unknown stage $1 (stages: all ${STAGES[*]})"; exit 2 ;; esac ;;
esac
# Expected minutes: module docstrings and plan 7-11 (sft: 7,082 steps x 0.61 s + judge curve, v2 measured 81 min for
# 6,304 steps; gate: 3,180 rows x 0.36 s + loading + about 10 min judging; read_b: beh_judges' 1.2-2 h estimate).
declare -A EXPECT_MIN=([build]=5 [check]=10 [rl_dir]=5 [configs]=1 [sft]=92 [select]=2 [gate]=32 [rl]=145 [export]=5
                       [read_b]=100 [judge_b]=60 [read_a]=75 [c1val]=15 [summary]=2 [upload]=10)

# ---------------------------------------------------------------- paths from common.py
PATHS=$(cd "$REPO" && "${PYC[@]}" - "$REPO" <<'PYEOF'
import shlex, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1] + "/main_run/beh_reader")
import common as C  # noqa: E402
import read_eval as R  # noqa: E402

if C.REPO.resolve() != Path(sys.argv[1]).resolve():
    sys.exit(f"common.REPO {C.REPO} is not this chain's repo {sys.argv[1]}")
ev = C.EVAL_READERS
v = dict(SNAPSHOT=C.SNAPSHOT, STORE=C.STORE, C1_DIR=C.C1_DIR, SFT_DIR=C.SFT_DIR, RL_SRC=C.RL_SRC, RL_DIR=C.RL_DIR,
         SFT_RUN=C.SFT_RUN, RL_RUN=C.RL_RUN, REF_SFT_RUN=C.REF_SFT_RUN, RUNS=C.RUNS, EV=C.EV, EVAL=C.EVAL,
         C1STYLE=C.C1STYLE, M=C.MARKERS, SFT_CFG=C.SFT_CFG, RL_CFG=C.RL_CFG, EM2_SFT_CFG=C.EM2_SFT_CFG,
         EM2_RL_CFG=C.EM2_RL_CFG, RL_GUARD_REF=C.RL_GUARD_REF, FIG_DIR=C.FIG_DIR, EVAL_FIG_STEM=C.EVAL_FIG_STEM,
         LOG_DIR=C.LOG_DIR, EMWT=C.EMWT, RLWT=C.RLWT, CHECKS_NAME=C.CHECKS_NAME, GATE_DOC_NAME=C.GATE_DOC_NAME,
         GATE_MD=R.GATE_MD, GATE_JSON=R.GATE_JSON, N_EXAMPLES=C.N_EXAMPLES, REF_STEP=C.REF_STEP,
         READ_B=",".join(dict.fromkeys(ev["behavioural"] + ev["content"])),
         READ_A=" ".join(k for k in ev["c1style"] if k not in ev["gate"]), GATE_READER=ev["gate"][0],
         OLD_READERS=" ".join(k for k in C.READERS if k not in C.NEW_READERS))
for k, x in v.items():
    print(f"{k}={shlex.quote(str(x))}")
PYEOF
) || { echo "cannot read the paths of main_run/beh_reader/common.py (interpreter: ${PYC[*]})"; exit 2; }
eval "$PATHS"
LOGD=${LOGD:-$LOG_DIR}
BR=main_run/beh_reader  # modules run from the repo root (CLI contract)
[ "$DRY" = 1 ] || mkdir -p "$M" "$LOGD"
# shellcheck source=gpus.sh
source "$REPO/scripts/beh_reader/gpus.sh"
HOLD=beh_reader  # stage part of the GPU announcement keys (em_reader_v2_beh_reader_<stage>_p...)

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
say() { echo "[$(stamp)] $*"; }
run() { if [ "$DRY" = 1 ]; then echo "  [dry] $*"; else "$@"; fi; }
run_in() { local d=$1; shift; if [ "$DRY" = 1 ]; then echo "  [dry] cd $d && $*"; else ( cd "$d" && "$@" ); fi; }
# A beh_reader module from the repo root with the em-reader worktree on PYTHONPATH; leading VAR=value words go to env.
br() {
  local envs=() mod
  while [ $# -gt 0 ] && [[ "$1" == [A-Z]*=* ]]; do envs+=("$1"); shift; done
  mod=$1; shift
  run_in "$REPO" env "${envs[@]}" PYTHONPATH="$EMWT" "${PYC[@]}" "$BR/$mod.py" "$@"
}
first_gpu() { echo "$1" | cut -d' ' -f1; }
commas() { echo $1 | tr ' ' ','; }

# summary.all_pass (and summary.complete when present) of a beh_checks.json; prints one line. 0 pass, 1 not, 2 absent.
checks_pass() {
  [ -f "$1" ] || return 2
  "${PYC[@]}" - "$1" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
print(f"{sys.argv[1]}: all_pass {s.get('all_pass')} complete {s.get('complete', True)} n_pass {s.get('n_pass')}/"
      f"{s.get('n_checks')} failed {s.get('failed')} skipped {s.get('skipped', [])}")
sys.exit(0 if s.get("all_pass") is True and s.get("complete", True) is True else 1)
PYEOF
}

# The gradient store is complete (plan 3) and no extraction process of it is alive. Prints one line.
store_ready() {
  local f pid alive=""
  for f in "${STORE%/*}"/logs/extract_shard*.pid; do
    [ -e "$f" ] || continue
    pid=$(tr -dc '0-9' < "$f")
    if [ -n "$pid" ] && ps -p "$pid" -o args= 2>/dev/null | grep -q extract.py; then alive="$alive $pid"; fi
  done
  [ -z "$alive" ] || { echo "extraction still running (pids$alive)"; return 1; }
  "${PYC[@]}" - "$STORE" "$N_EXAMPLES" <<'PYEOF'
import json, os, sys
d, n = sys.argv[1], int(sys.argv[2])
try:
    m = json.load(open(os.path.join(d, "manifest.json")))
except (OSError, ValueError) as e:
    print(f"{d}/manifest.json unreadable: {e}"); sys.exit(1)
tl = (m.get("plan") or {}).get("think_labels")
shards = [s.get("dir") for s in m.get("shards", [])]
ok = m.get("total_rows") == n and tl == "public" and shards and all(os.path.isdir(os.path.join(d, s)) for s in shards)
print(f"{d}: rows {m.get('total_rows')} (want {n}), think_labels {tl}, shards {shards}, written {m.get('written')}")
sys.exit(0 if ok else 1)
PYEOF
}

sft_finished() { grep -q 'done: step' "$SFT_RUN/train.log" 2>/dev/null; }
nonempty() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }

run_stage() {
  local st=$1 rc t0 took exp=${EXPECT_MIN[$1]} log
  log=$LOGD/${STAMP}_beh_reader_${st}.out
  if [ -f "$M/$st.DONE" ]; then say "stage $st: DONE marker exists; skipped"; return 0; fi
  if [ -f "$M/gate.STOP" ]; then say "stage $st: gate.STOP exists ($(head -c 300 "$M/gate.STOP")); not run"; return $RC_STOP; fi
  say "stage $st: start; expected about $exp min (log $log)"
  if [ "$DRY" = 1 ]; then "stage_$st"; return 0; fi
  echo "pid $$ host $(hostname) start $(stamp)" > "$M/$st.RUNNING"
  t0=$(date +%s)
  "stage_$st" > "$log" 2>&1
  rc=$?
  took=$(( ($(date +%s) - t0 + 30) / 60 ))
  release_gpus "${HOLD}_${st}*" "stage $st ended rc=$rc"
  rm -f "$M/$st.RUNNING"
  printf '%s\t%s\t%s\t%s\t%s\n' "$st" "$exp" "$took" "$rc" "$(stamp)" >> "$M/timings.tsv"
  say "stage $st: took $took min (expected about $exp min)"
  if [ "$took" -gt $((2 * exp)) ]; then say "WARN stage $st took more than twice its estimate; find the cause (log $log)"; fi
  if [ $rc -eq 0 ]; then
    echo "$(stamp); took $took min, expected $exp" > "$M/$st.DONE"; say "stage $st: DONE"
  elif [ "$st" = gate ] && [ $rc -eq $RC_STOP ] && [ -f "$M/gate.STOP" ]; then
    say "stage gate: STOP ($(head -c 300 "$M/gate.STOP"))"
  else
    echo "rc=$rc $(stamp); took $took min; log $log" > "$M/$st.FAILED"; say "stage $st: FAILED rc=$rc"
  fi
  return $rc
}

# ---------------------------------------------------------------- build, check, rl_dir, configs (plan 4-6)
stage_build() {
  if [ "$DRY" != 1 ]; then
    if [ -f "$SFT_DIR/meta.json" ]; then say "$SFT_DIR/meta.json exists: the SFT dir is built; skipped"; return 0; fi
    if nonempty "$SFT_DIR"; then say "$SFT_DIR is not empty and has no meta.json (an unfinished build?); user decision"; return 1; fi
    store_ready || { say "the gradient store is not ready"; return 1; }
  fi
  br build_reader_dir
}

stage_check() {
  local G rc f=$SFT_DIR/$CHECKS_NAME
  if [ "$DRY" != 1 ]; then
    [ -f "$SFT_DIR/meta.json" ] || { say "$SFT_DIR/meta.json missing (build stage)"; return 1; }
    if [ -f "$f" ]; then
      if checks_pass "$f"; then say "$f exists and passed (complete); kept"; return 0; fi
      say "$f exists and did not pass or is incomplete; check_reader_dir.py never overwrites it: read it, set it aside by hand, rerun"
      return 1
    fi
  fi
  G=$(wait_gpus "" 1 120 1 "${HOLD}_check") || { say "no usable GPU for 120 min"; return 1; }
  G=$(first_gpu "$G")
  say "reader dir checks; recompute on GPU $G"
  br CUDA_VISIBLE_DEVICES="$G" check_reader_dir --gpu 0; rc=$?
  release_gpus "${HOLD}_check" "checks ended"
  [ "$DRY" = 1 ] && return 0
  [ $rc -eq 0 ] || { say "check_reader_dir.py exit $rc (1 = a check failed, 3 = recompute skipped)"; return 1; }
  checks_pass "$f"
}

stage_rl_dir() {
  local f=$RL_DIR/$CHECKS_NAME
  if [ "$DRY" != 1 ]; then
    checks_pass "$SFT_DIR/$CHECKS_NAME" || { say "the SFT dir checks are missing or did not pass (check stage)"; return 1; }
  fi
  if [ "$DRY" != 1 ] && [ -f "$RL_DIR/meta.json" ]; then say "$RL_DIR/meta.json exists: the RL dir is built; build skipped"
  elif [ "$DRY" != 1 ] && nonempty "$RL_DIR"; then say "$RL_DIR is not empty and has no meta.json; user decision"; return 1
  else br build_rl_dir || return 1; fi
  if [ "$DRY" != 1 ] && [ -f "$f" ]; then
    if checks_pass "$f"; then say "$f exists and passed; kept"; return 0; fi
    say "$f exists and did not pass; check_rl_dir.py never overwrites it: read it, set it aside by hand, rerun"; return 1
  fi
  br check_rl_dir || return 1
  [ "$DRY" = 1 ] || checks_pass "$f"
}

stage_configs() {
  br write_configs
}

# ---------------------------------------------------------------- sft (plan 7; chain_v2 stage_sft)
sft_config_guard() {  # the SFT config's reader_dir and out are common's SFT dir and run
  "${PYC[@]}" - "$SFT_CFG" "$SFT_DIR" "$SFT_RUN" <<'PYEOF'
import os, sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
got = [os.path.realpath(os.path.expandvars(c[k])) for k in ("reader_dir", "out")]
want = [os.path.realpath(p) for p in sys.argv[2:4]]
assert got == want, (got, want)
print("SFT config guard ok:", got)
PYEOF
}

stage_sft() {
  local G G_SFT G_JUDGE TPID JPID rc
  if [ "$DRY" != 1 ]; then
    [ -f "$SFT_CFG" ] || { say "missing $SFT_CFG (configs stage)"; return 1; }
    checks_pass "$SFT_DIR/$CHECKS_NAME" && checks_pass "$RL_DIR/$CHECKS_NAME" \
      || { say "the reader dirs' checks are missing or did not pass (check, rl_dir stages)"; return 1; }
    sft_config_guard || { say "SFT config guard failed"; return 1; }
    if sft_finished; then
      say "training already finished ($(grep -o 'done: step [0-9]*' "$SFT_RUN/train.log" | tail -1)); only the judge curve is completed"
      sft_judge_complete; return $?
    fi
    if nonempty "$SFT_RUN"; then
      say "$SFT_RUN exists, is not empty and has no finished training; resuming (train.py --resume) is a user decision"
      return 1
    fi
  fi
  G=$(wait_gpus "" 2 120 2 "${HOLD}_sft") || { say "fewer than 2 usable GPUs for 120 min"; return 1; }
  G_SFT=$(echo $G | cut -d' ' -f1); G_JUDGE=$(echo $G | cut -d' ' -f2)
  say "SFT on GPU $G_SFT, judge-curve watcher on GPU $G_JUDGE"
  if [ "$DRY" = 1 ]; then
    echo "  [dry] cd $EMWT && CUDA_VISIBLE_DEVICES=$G_SFT PYTHONPATH=$EMWT ${PYC[*]} main_run/reader/train.py --config $SFT_CFG --batch-size 4 --device cuda:0"
    echo "  [dry] cd $EMWT && PYTHONPATH=$EMWT ${PYC[*]} -m main_run.reader.judge_curve --run-dirs $SFT_RUN --watch --split val --conditions correct,shuffled --gpus $G_JUDGE --no-fast-gen --judge-workers 8"
    echo "  [dry] (when the curve lacks the final step) judge_curve --once with the same flags on one GPU"
    return 0
  fi
  echo "$G_SFT $G_JUDGE" > "$M/sft.gpus"
  ( cd "$EMWT" && CUDA_VISIBLE_DEVICES=$G_SFT PYTHONPATH=$EMWT exec "${PYC[@]}" main_run/reader/train.py \
      --config "$SFT_CFG" --batch-size 4 --device cuda:0 ) \
      > "$LOGD/${STAMP}_beh_reader_sft_train.out" 2>&1 < /dev/null &
  TPID=$!
  echo $TPID > "$M/sft_train.pid"
  for _ in $(seq 60); do [ -f "$SFT_RUN/RUN.json" ] && break; kill -0 $TPID 2>/dev/null || break; sleep 10; done
  if ! kill -0 $TPID 2>/dev/null && [ ! -f "$SFT_RUN/RUN.json" ]; then wait $TPID; say "train.py exited early (rc $?)"; return 1; fi
  ( cd "$EMWT" && PYTHONPATH=$EMWT exec "${PYC[@]}" -m main_run.reader.judge_curve --run-dirs "$SFT_RUN" --watch \
      --split val --conditions correct,shuffled --gpus "$G_JUDGE" --no-fast-gen --judge-workers 8 ) \
      > "$LOGD/${STAMP}_beh_reader_sft_judge.out" 2>&1 < /dev/null &
  JPID=$!
  echo $JPID > "$M/sft_judge.pid"
  wait $TPID; rc=$?
  say "train.py exit $rc"
  if [ $rc -ne 0 ]; then kill $JPID 2>/dev/null; wait $JPID; return 1; fi
  wait $JPID; say "judge-curve watcher exit $?"
  release_gpus "${HOLD}_sft" "training and judge-curve watcher ended"
  sft_judge_complete
}

sft_judge_has_final() {
  "${PYC[@]}" - "$SFT_RUN" <<'PYEOF'
import json, re, sys
run = sys.argv[1]
fin = [int(m) for m in re.findall(r"done: step (\d+)", open(f"{run}/train.log").read())][-1]
try:
    rows = [json.loads(l) for l in open(f"{run}/judge_curve.jsonl") if l.strip()]
except FileNotFoundError:
    rows = []
steps = sorted({r["step"] for r in rows})
print("final step", fin, "judge-curve steps", steps, "eval keys", sorted({r["eval_key"] for r in rows}))
sys.exit(0 if fin in steps else 1)
PYEOF
}

sft_judge_complete() {
  local G
  [ "$DRY" = 1 ] && return 0
  sft_judge_has_final && return 0
  G=$(wait_gpus "" 1 120 1 "${HOLD}_sft_judge") || { say "no usable GPU for the judge curve"; return 1; }
  G=$(first_gpu "$G")
  say "judge curve incomplete; judge_curve --once on GPU $G"
  ( cd "$EMWT" && PYTHONPATH=$EMWT "${PYC[@]}" -m main_run.reader.judge_curve --run-dirs "$SFT_RUN" --once \
      --split val --conditions correct,shuffled --gpus "$G" --no-fast-gen --judge-workers 8 ) \
      > "$LOGD/${STAMP}_beh_reader_sft_judge_once.out" 2>&1 < /dev/null
  release_gpus "${HOLD}_sft_judge" "judge_curve --once ended"
  sft_judge_has_final
}

# ---------------------------------------------------------------- select (plan 7)
stage_select() {
  run_in "$EMWT" env PYTHONPATH="$EMWT" "${PYC[@]}" -m main_run.reader.select_checkpoint "$SFT_RUN" || return 1
  [ "$DRY" = 1 ] && return 0
  "${PYC[@]}" - "$SFT_RUN/selected_checkpoint.json" "$M" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get("chosen")
if not c:
    print("no checkpoint qualifies:", d.get("why")); sys.exit(1)
print("chosen", c["step"], c["adapter"], "verdict", d.get("verdict"), d.get("why"))
if d.get("verdict") != "pass":
    open(f"{sys.argv[2]}/select.WARN", "w").write(f"gate {d.get('verdict')}: {d.get('why')}\n")
PYEOF
}

# ---------------------------------------------------------------- read_eval read under the read hang watchdog (header)
sizes() { local f; for f in "$@"; do stat -c %s "$f" 2>/dev/null || echo -1; done | tr '\n' ' '; }
gone_within() {  # 0 when pid $1 is gone within $2 seconds
  local i
  for ((i = 0; i < $2; i++)); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
  ! kill -0 "$1" 2>/dev/null
}

# read_eval.py read of reader $1 on GPU $2 (stdout and stderr in $3) under the watchdog; returns the read's exit code,
# or 1 when it was stopped as hung or could not be killed.
read_watched() {
  local r=$1 g=$2 log=$3 pid part last cur now t_last t_grace t_ref rc
  if [ "$DRY" = 1 ]; then
    br read_eval read --reader "$r" --gpu "$g"
    echo "  [dry]   (log $log; hang watchdog: TERM when it and $C1STYLE/$r/preds.jsonl.tmp stop growing for $READ_HANG_S s after a $READ_GRACE_S s loading grace, KILL $READ_KILL_S s later)"
    return 0
  fi
  part=$C1STYLE/$r/preds.jsonl.tmp
  ( cd "$REPO" && exec env PYTHONPATH="$EMWT" "${PYC[@]}" "$BR/read_eval.py" read --reader "$r" --gpu "$g" ) \
      > "$log" 2>&1 < /dev/null &
  pid=$!
  say "read_eval read --reader $r on GPU $g: pid $pid, log $log"
  t_last=$(date +%s); t_grace=$((t_last + READ_GRACE_S)); t_ref=$t_grace
  last=$(sizes "$log" "$part")
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$READ_POLL_S"
    cur=$(sizes "$log" "$part"); now=$(date +%s)
    if [ "$cur" != "$last" ]; then last=$cur; t_last=$now; t_ref=$(( now > t_grace ? now : t_grace )); continue; fi
    [ $((now - t_ref)) -ge "$READ_HANG_S" ] || continue
    kill -0 "$pid" 2>/dev/null || break  # it ended on its own since the last look
    say "hang: read_eval read --reader $r (pid $pid): log and $part unchanged for $((now - t_last)) s; TERM"
    kill -TERM "$pid" 2>/dev/null
    if ! gone_within "$pid" "$READ_KILL_S"; then
      say "hang: pid $pid still running $READ_KILL_S s after TERM; KILL"
      kill -KILL "$pid" 2>/dev/null
      gone_within "$pid" "$READ_KILL_S" || { say "hang: pid $pid still alive $READ_KILL_S s after KILL; left behind"; return 1; }
    fi
    wait "$pid"; rc=$?
    [ $rc -eq 0 ] && return 0  # it finished between the last look and TERM
    say "hang: read_eval read --reader $r stopped (rc $rc); a rerun reads it again and reuses the readers already read"
    return 1
  done
  wait "$pid"
}

# ---------------------------------------------------------------- gate (plan 8)
gate_line() {  # decision, mean and CI of gate.json in one line
  "${PYC[@]}" - "$1" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1]))
f = lambda x: "None" if x is None else f"{x:.3f}"  # noqa: E731
print(f"{g.get('decision')}: Delta_judge(shuffled) mean {f(g.get('mean'))} [{f(g.get('lo'))}, {f(g.get('hi'))}] "
      f"n {g.get('n')} ({g.get('name')})")
PYEOF
}

stage_gate() {
  local G rc d md line
  if [ "$DRY" != 1 ] && [ ! -f "$SFT_RUN/selected_checkpoint.json" ]; then say "no selected checkpoint (select stage)"; return 1; fi
  G=$(wait_gpus "" 1 120 1 "${HOLD}_gate") || { say "no usable GPU for 120 min"; return 1; }
  G=$(first_gpu "$G")
  say "gate readings of $GATE_READER on GPU $G"
  br read_eval plan
  read_watched "$GATE_READER" "$G" "$LOGD/${STAMP}_beh_reader_gate_read.out" || return 1
  release_gpus "${HOLD}_gate" "readings ended"
  br read_eval score --reader "$GATE_READER" --max-cost "$SCORE_CAP" || return 1
  br read_eval gate --reader "$GATE_READER"; rc=$?
  if [ "$DRY" = 1 ]; then
    echo "  [dry] exit $RC_STOP -> markers/gate.STOP and the chain stops; exit 0 -> markers/gate.CONTINUE"
    echo "  [dry] cp $C1STYLE/$GATE_READER/$GATE_MD $FIG_DIR/${GATE_DOC_NAME//"{date}"/<date>}; git commit -m '<gate>' -- <it>; git push origin main"
    return 0
  fi
  [ $rc -eq 0 ] || [ $rc -eq $RC_STOP ] || { say "read_eval.py gate exit $rc"; return 1; }
  d=$C1STYLE/$GATE_READER
  md=$d/$GATE_MD
  [ -f "$md" ] && [ -f "$d/$GATE_JSON" ] || { say "read_eval.py gate exited $rc without $md and $GATE_JSON"; return 1; }
  line=$(gate_line "$d/$GATE_JSON") || line="(gate.json unreadable)"
  if [ $rc -eq $RC_STOP ]; then
    echo "stop $(stamp): $line; $md" > "$M/gate.STOP"
    say "gate: STOP ($line)"
  else
    echo "continue $(stamp): $line; $md" > "$M/gate.CONTINUE"
    say "gate: continue to RL ($line)"
  fi
  gate_publish "$md" "$( [ $rc -eq $RC_STOP ] && echo stop || echo continue)"
  return $rc
}

# Copy gate.md into docs/figures, commit it alone and push (failures are logged, not fatal).
gate_publish() {
  local src=$1 decision=$2 dst rel
  dst=$FIG_DIR/${GATE_DOC_NAME//"{date}"/$(date +%Y%m%d)}
  [ -e "$dst" ] && dst=${dst%.md}_$(date +%H%M%S).md
  mkdir -p "$FIG_DIR"
  cp "$src" "$dst" || { say "copy of $src failed"; return 0; }
  rel=${dst#"$REPO"/}
  ( cd "$REPO" && git add -- "$rel" && git commit -q -m "Behaviour reader gate after SFT: $decision (plan 20261027 section 8), written by scripts/beh_reader/chain.sh

Co-Authored-By: Claude <noreply@anthropic.com>
Claude-Session: <session-url>" -- "$rel" \
    && { git push -q origin main || say "push failed; the commit is local"; } ) || say "commit of $rel failed; file left in place"
  say "gate summary in $rel"
  return 0
}

# ---------------------------------------------------------------- rl (plan 9; chain_v2 stage_rl)
rl_cost() {
  "${PYC[@]}" - "$RL_RUN" <<'PYEOF'
import json, sys, glob, os
out = sys.argv[1]; tot = 0.0
p = os.path.join(out, "rl_log.jsonl")
if os.path.exists(p):
    for line in open(p):
        if line.strip():
            tot += float(json.loads(line).get("judge_cost_usd") or 0.0)
for f in glob.glob(os.path.join(out, "eval_dev", "judged_*.jsonl")):
    seen = {}
    for line in open(f):
        r = json.loads(line); seen[r["group"]] = float(r.get("judge_cost_usd_group") or 0.0)
    tot += sum(seen.values())
print(f"{tot:.6f}")
PYEOF
}

stage_rl() {
  ( stage_rl_inner )
}

stage_rl_inner() {
  local ADAPTER G TPID WPID rc c TAG A GF J JC JF EXTRA
  local KW=configs/reader/rl_keyword_judge_haiku_paper.yaml DOC=configs/reader/rl_doc_judge_haiku.yaml
  local TAGS="cycle_010 cycle_020 cycle_030 cycle_040" DEVD=$RL_RUN/eval_dev
  if [ "$DRY" = 1 ]; then ADAPTER="<selected_checkpoint.json chosen.adapter>"; else
    ADAPTER=$("${PYC[@]}" -c "import json, sys; print(json.load(open(sys.argv[1]))['chosen']['adapter'])" \
      "$SFT_RUN/selected_checkpoint.json") || return 1
  fi
  G=$(wait_gpus "" 1 120 1 "${HOLD}_rl") || { say "no usable GPU for 120 min"; return 1; }
  G=$(first_gpu "$G")
  say "RL on GPU $G from $ADAPTER (rl-stage-fixes $(git -C "$RLWT" rev-parse --short HEAD 2>/dev/null || echo '?'))"
  if [ "$DRY" = 1 ]; then echo "  [dry] cd $RLWT; export PYTHONPATH=$RLWT"; else
    cd "$RLWT" || return 1
    export PYTHONPATH=$RLWT
  fi
  run "${PYC[@]}" - "$RL_CFG" "$RL_DIR" "$RL_RUN" <<'PYEOF' || { say "guard failed"; return 1; }
import os, sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
ref = yaml.safe_load(open("configs/reader/rl_scratch_L40_haiku_paper.yaml"))
diff = sorted(k for k in set(c) | set(ref) if c.get(k) != ref.get(k))
assert diff == ["eval_jsonl", "judge_cfg", "out", "reader_dir", "train_jsonl"], diff
assert os.path.realpath(os.path.expandvars(c["judge_cfg"])) == os.path.realpath(ref["judge_cfg"]), c["judge_cfg"]
assert "rl_fineweb_beh" in c["reader_dir"] and c["out"].endswith("reader_rl_scratch_beh_paper"), (c["reader_dir"], c["out"])
real = lambda p: os.path.realpath(os.path.expandvars(p))  # noqa: E731
assert real(c["reader_dir"]) == real(sys.argv[2]) and real(c["out"]) == real(sys.argv[3]), (c["reader_dir"], c["out"])
assert c.get("loss_norm") == "drgrpo"
for p in ("configs/reader/rl_keyword_judge_haiku_paper.yaml", "configs/reader/rl_doc_judge_haiku.yaml"):
    j = yaml.safe_load(open(p))
    assert j["judge_provider"] == "anthropic" and j["anthropic_model"] == "claude-haiku-5-5", p
print("guard ok: differs from rl_scratch_L40_haiku_paper.yaml only in", diff, "| Haiku 5.5 judges | drgrpo")
PYEOF
  if [ -f "$RL_RUN/TRAIN_DONE" ]; then say "TRAIN_DONE exists; training skipped"; else
    if [ "$DRY" = 1 ]; then
      echo "  [dry] cd $RLWT && PYTHONPATH=$RLWT CUDA_VISIBLE_DEVICES=$G ${PYC[*]} -m main_run.reader.rl_train --config $RL_CFG --adapter $ADAPTER --out $RL_RUN  (+ cost watchdog \$$RL_TRAIN_CAP_USD)"
    else
      CUDA_VISIBLE_DEVICES=$G "${PYC[@]}" -m main_run.reader.rl_train --config "$RL_CFG" --adapter "$ADAPTER" --out "$RL_RUN" \
          > "$LOGD/${STAMP}_beh_reader_rl_train.out" 2>&1 < /dev/null &
      TPID=$!
      echo $TPID > "$M/rl_train.pid"
      ( while kill -0 $TPID 2>/dev/null; do
          c=$(rl_cost 2>/dev/null || echo 0)
          if "${PYC[@]}" -c "import sys; sys.exit(0 if float('$c') > $RL_TRAIN_CAP_USD else 1)"; then
            say "cost watchdog: judge cost \$$c > \$$RL_TRAIN_CAP_USD; stopping training"
            echo "cost cap: $c > $RL_TRAIN_CAP_USD" > "$M/rl_COST_CAP_HIT"; kill -TERM $TPID; break
          fi
          sleep 120
        done ) &
      WPID=$!
      wait $TPID; rc=$?
      kill $WPID 2>/dev/null
      if [ $rc -ne 0 ] || [ -f "$M/rl_COST_CAP_HIT" ]; then say "rl_train failed rc=$rc"; return 1; fi
      say "rl_train done; judge cost \$$(rl_cost)"
      date '+%Y-%m-%d %H:%M:%S %Z' > "$RL_RUN/TRAIN_DONE"
    fi
  fi
  run mkdir -p "$DEVD"
  for TAG in sft $TAGS; do
    GF="$DEVD/gen_${TAG}.jsonl"; [ -s "$GF" ] && continue
    if [ "$TAG" = sft ]; then A="$ADAPTER"; else A="$RL_RUN/adapters/$TAG"; fi
    say "dev generate $TAG"
    run env CUDA_VISIBLE_DEVICES="$G" "${PYC[@]}" -m main_run.reader.rl_eval generate --config "$RL_CFG" --adapter "$A" \
        --tag "$TAG" --split dev --n-rollouts 2 --out "$GF" || return 1
  done
  for TAG in $TAGS; do
    for J in keyword doc; do
      if [ "$J" = keyword ]; then JC=$KW; JF="$DEVD/judged_kw_${TAG}.jsonl"; EXTRA=""; else
        JC=$DOC; JF="$DEVD/judged_doc_${TAG}.jsonl"; EXTRA="--reader-dir $RL_DIR"; fi
      [ -s "$JF" ] && continue
      if [ "$DRY" != 1 ]; then
        c=$(rl_cost)
        "${PYC[@]}" -c "import sys; sys.exit(0 if float('$c') < $RL_TOTAL_CAP_USD else 1)" || { say "cost cap \$$c"; return 1; }
      fi
      say "dev judge $J $TAG"
      # shellcheck disable=SC2086  # EXTRA is empty or two words
      run "${PYC[@]}" -m main_run.reader.rl_eval judge --sft "$DEVD/gen_sft.jsonl" --rl "$DEVD/gen_${TAG}.jsonl" \
          --n-per-cell 2 --judge "$J" $EXTRA --judge-cfg "$JC" --out "$JF" --workers 16 || return 1
    done
  done
  # shellcheck disable=SC2046
  run "${PYC[@]}" -m main_run.reader.rl_eval select --dev-judged $(for T in $TAGS; do echo "$DEVD/judged_kw_${T}.jsonl"; done) \
      --out "$DEVD/select_keyword.json" || return 1
  # shellcheck disable=SC2046
  run "${PYC[@]}" -m main_run.reader.rl_eval select --dev-judged $(for T in $TAGS; do echo "$DEVD/judged_doc_${T}.jsonl"; done) \
      --out "$DEVD/select_doc.json" || return 1
  [ "$DRY" = 1 ] || date '+%Y-%m-%d %H:%M:%S %Z' > "$RL_RUN/EVAL_DONE"
  [ "$DRY" = 1 ] || say "RL stage done; judge cost \$$(rl_cost 2>/dev/null)"
}

# ---------------------------------------------------------------- evaluation (plan 10)
stage_export() {
  br export_inputs
}

stage_read_b() {
  local G
  G=$(wait_gpus "" 1 120 2 "${HOLD}_read_b") || { say "no usable GPU for 120 min"; return 1; }
  G=$(commas "$G")
  say "LoRAcle-style readings of $READ_B on GPUs $G (2 workers per GPU)"
  br beh_judges plan --readers "$READ_B" | tail -n 40
  br beh_judges read --readers "$READ_B" --gpus "$G" --workers-per-gpu 2
}

stage_judge_b() {
  local rc ct=() out=$LOGD/${STAMP}_beh_reader_judge_b_content.out
  [ -n "${BEH_CONTENT_TRUTH:-}" ] && ct=(--content-truth "$BEH_CONTENT_TRUTH")
  br beh_judges judge --judge truth --max-cost "$EVAL_CAP" || return 1
  if [ "$DRY" = 1 ]; then
    br beh_judges judge --judge content --max-cost "$EVAL_CAP" "${ct[@]}"
    echo "  [dry] content judge refused by its calibration guard (exit 2) -> markers/judge_b.CONTENT_REFUSED, stage rc $RC_SOFT_CONTENT, chain goes on"
    rc=0
  else
    br beh_judges judge --judge content --max-cost "$EVAL_CAP" "${ct[@]}" 2>&1 | tee "$out"; rc=${PIPESTATUS[0]}
    if [ $rc -eq 2 ] && grep -q "content judge refused" "$out"; then
      { grep "content judge refused" "$out" | tail -1
        echo "$(stamp): truth judged; content not judged. Choose a calibrated truth and run"
        echo "BEH_CONTENT_TRUTH=<truth> bash scripts/beh_reader/chain.sh judge_b, then eval_summary.py by hand"; } > "$M/judge_b.CONTENT_REFUSED"
      say "content judge refused by its calibration guard (markers/judge_b.CONTENT_REFUSED); summary goes on"
    elif [ $rc -ne 0 ]; then say "content judge exit $rc"; return 1; fi
  fi
  br beh_judges summary || return 1
  [ $rc -eq 2 ] && return $RC_SOFT_CONTENT
  return 0
}

# C1-style reads: GPU k of the list reads readers k, k+n, ... one after another (one read_eval process per GPU).
par_reads() {
  local gl=$1 i j r g n ok=0 pids=() gpus=() mine=()
  shift
  read -r -a gpus <<< "${gl//,/ }"
  n=${#gpus[@]}
  for ((i = 0; i < n; i++)); do
    g=${gpus[$i]}; mine=()
    for ((j = i; j < $#; j += n)); do mine+=("${@:j+1:1}"); done
    [ ${#mine[@]} -gt 0 ] || continue
    if [ "$DRY" = 1 ]; then
      for r in "${mine[@]}"; do read_watched "$r" "$g" "$LOGD/${STAMP}_beh_reader_read_a_${r}.out"; done; continue
    fi
    ( rc=0
      for r in "${mine[@]}"; do
        read_watched "$r" "$g" "$LOGD/${STAMP}_beh_reader_read_a_${r}.out" \
          || { echo "[$(stamp)] read_eval read --reader $r on GPU $g failed"; rc=1; }
      done
      exit $rc ) &
    pids+=($!)
  done
  for j in "${pids[@]}"; do wait "$j" || ok=1; done
  return $ok
}

stage_read_a() {
  local G r readers=()
  read -r -a readers <<< "$READ_A"
  G=$(wait_gpus "" 1 120 2 "${HOLD}_read_a") || { say "no usable GPU for 120 min"; return 1; }
  G=$(commas "$G")
  say "C1-style readings of ${readers[*]} on GPUs $G (one process per GPU at a time; $GATE_READER was read by the gate)"
  par_reads "$G" "${readers[@]}" || { say "a reading failed (logs ${STAMP}_beh_reader_read_a_<reader>.out)"; return 1; }
  release_gpus "${HOLD}_read_a" "readings ended"
  for r in "${readers[@]}"; do
    br read_eval score --reader "$r" --max-cost "$SCORE_CAP" || return 1
  done
  br read_eval summary
}

stage_c1val() {
  br read_eval c1val
}

# judge_b beside read_a in `all` (header): run_stage judge_b in the background; judge_b_wait collects it.
JB_PID=""
judge_b_start() {
  run_stage judge_b &
  JB_PID=$!
  echo "$JB_PID" > "$M/judge_b.pid"
  say "judge_b runs in the background (pid $JB_PID) beside read_a; the chain waits for it before c1val"
}

judge_b_wait() {  # waits for the background judge_b; a failure of it is soft (added to SOFT)
  local rc
  [ -n "$JB_PID" ] || return 0
  say "waiting for the background judge_b (pid $JB_PID)"
  wait "$JB_PID"; rc=$?
  JB_PID=""
  if [ $rc -eq $RC_SOFT_CONTENT ]; then
    say "judge_b: content judge refused (judge_b.CONTENT_REFUSED); going on"; SOFT="$SOFT judge_b"
  elif [ $rc -ne 0 ]; then
    say "judge_b failed (rc $rc; judge_b.FAILED); going on without its files. Rerun: bash scripts/beh_reader/chain.sh judge_b, then eval_summary.py by hand"
    SOFT="$SOFT judge_b"
  fi
  return 0
}

# ---------------------------------------------------------------- summary, upload (plan 12)
stage_summary() {
  local stem before after new
  stem=$FIG_DIR/${EVAL_FIG_STEM##*/}
  before=$(ls "$FIG_DIR"/*_behaviour_reader_eval* 2>/dev/null | sort)
  br eval_summary --out-json "$stem.json" --out-png "$stem.png" || return 1
  [ "$DRY" = 1 ] && { echo "  [dry] git commit -m '<summary>' -- <new json and png>; git push origin main"; return 0; }
  after=$(ls "$FIG_DIR"/*_behaviour_reader_eval* 2>/dev/null | sort)
  new=$(comm -13 <(echo "$before") <(echo "$after") | sed "s#^$REPO/##")
  [ -n "$new" ] || { say "eval_summary.py wrote no new file under $FIG_DIR"; return 1; }
  cd "$REPO" || return 1
  # shellcheck disable=SC2086  # one figure path per word (no spaces in the names)
  git add -- $new && git commit -q -m "Behaviour reader evaluation summary and figure (plan 20261027 sections 10, 12), written by scripts/beh_reader/chain.sh

Co-Authored-By: Claude <noreply@anthropic.com>
Claude-Session: <session-url>" -- $new \
    && { git push -q origin main || say "push failed; the commit is local"; } || say "commit failed; files left in place"
  say "summary files: $(echo $new)"
  return 0
}

stage_upload() {
  br HF_HUB_OFFLINE=0 upload
}

# ---------------------------------------------------------------- dry-run prerequisites
dry_report() {  # what the real chain needs (read only); returns the number of missing or blocked prerequisites
  local miss=0 f r st total=0
  ck() { if [ -e "$1" ]; then echo "  [ok] $1"; else echo "  [MISSING] $1${2:+ ($2)}"; miss=$((miss + 1)); fi; }
  say "dry: prerequisites (nothing is started or written)"
  for f in bundles.jsonl labels.jsonl examples.jsonl examples_chat.jsonl MANIFEST.json split.json; do ck "$SNAPSHOT/$f" snapshot; done
  if st=$(store_ready); then echo "  [ok] store: $st"; else echo "  [MISSING] store not ready: $st"; miss=$((miss + 1)); fi
  for f in meta.json bundle_index.parquet train_correct.jsonl eval_samples.jsonl mean_unit_tokensum.npy; do ck "$C1_DIR/$f" "C1 reader dir"; done
  for f in meta.json bundle_index.parquet rl_train.jsonl rl_eval.jsonl; do ck "$RL_SRC/$f" "fineweb RL dir"; done
  for f in train judge_curve select_checkpoint generate e2_generate; do ck "$EMWT/main_run/reader/$f.py" "em-reader worktree"; done
  ck "$EMWT/configs/c1_judge.yaml" "C1 judge config"
  for f in rl_train rl_eval rl_data; do ck "$RLWT/main_run/reader/$f.py" "rl-stage worktree"; done
  for f in rl_scratch_L40_haiku_paper rl_keyword_judge_haiku_paper rl_doc_judge_haiku; do ck "$RLWT/configs/reader/$f.yaml" "RL guard"; done
  ck "$EM2_SFT_CFG" "write_configs source"; ck "$EM2_RL_CFG" "write_configs source"
  for f in build_reader_dir check_reader_dir build_rl_dir check_rl_dir write_configs export_inputs read_eval beh_judges \
           eval_summary upload; do ck "$REPO/$BR/$f.py" module; done
  for r in $OLD_READERS; do
    "${PYC[@]}" - "$REPO/$BR" "$r" <<'PYEOF' || miss=$((miss + 1))
import sys
sys.path.insert(0, sys.argv[1])
import common as C  # noqa: E402
r = C.reader(sys.argv[2], must_exist=False)
bad = [str(p) for p in (r["adapter"], r["run_json"]) if p is None or not p.exists()]
print(f"  [{'MISSING' if bad else 'ok'}] reader {r['key']} ({r['what']}): {bad or r['adapter']}")
sys.exit(1 if bad else 0)
PYEOF
  done
  ck "$REF_SFT_RUN/judge_curve/preds_step_$(printf '%06d' "$REF_STEP").jsonl" "c1val reference predictions"
  ck "$REF_SFT_RUN/selected_checkpoint.json" "c1val reference selection"
  if ls "$IIF_BASE"/persistent/secrets/anthropic_key* > /dev/null 2>&1; then echo "  [ok] Anthropic key file present (value not read)"
  else echo "  [MISSING] no Anthropic key file under \$IIF_BASE/persistent/secrets"; miss=$((miss + 1)); fi
  if [ -f "$HF_HOME/token" ]; then echo "  [ok] HF token file present (value not read)"; else echo "  [note] no $HF_HOME/token: upload will fail (soft)"; fi
  # states the chain refuses to change by itself
  if [ -f "$SFT_DIR/meta.json" ]; then echo "  [note] $SFT_DIR built; build skips it"
  elif nonempty "$SFT_DIR"; then echo "  [BLOCKED] $SFT_DIR not empty without meta.json (build refuses)"; miss=$((miss + 1))
  else echo "  [todo] $SFT_DIR (build)"; fi
  for f in "$SFT_DIR/$CHECKS_NAME" "$RL_DIR/$CHECKS_NAME"; do
    if [ ! -f "$f" ]; then echo "  [todo] $f (check / rl_dir)"
    elif st=$(checks_pass "$f"); then echo "  [ok] $st"
    else echo "  [BLOCKED] $st (the stage refuses a failing file; set it aside by hand)"; miss=$((miss + 1)); fi
  done
  if [ -f "$RL_DIR/meta.json" ]; then echo "  [note] $RL_DIR built; rl_dir skips the build"
  elif nonempty "$RL_DIR"; then echo "  [BLOCKED] $RL_DIR not empty without meta.json (rl_dir refuses)"; miss=$((miss + 1))
  else echo "  [todo] $RL_DIR (rl_dir)"; fi
  for f in "$SFT_CFG" "$RL_CFG"; do
    if [ -f "$f" ]; then echo "  [note] $f exists (write_configs keeps it when unchanged, refuses other content)"
    else echo "  [todo] $f (configs)"; fi
  done
  if nonempty "$SFT_RUN"; then
    if sft_finished; then echo "  [note] $SFT_RUN: training finished; sft only completes the judge curve"
    else echo "  [BLOCKED] $SFT_RUN exists, is not empty and has no finished training (sft refuses)"; miss=$((miss + 1)); fi
  else echo "  [ok] $SFT_RUN absent or empty"; fi
  [ -e "$RL_RUN" ] && echo "  [note] $RL_RUN exists$([ -f "$RL_RUN/TRAIN_DONE" ] && echo ' with TRAIN_DONE')"
  "${PYC[@]}" - "$REPO/$BR" "${BEH_CONTENT_TRUTH:-}" <<'PYEOF' 2>/dev/null || echo "  [note] content judge: guard state could not be read"
import sys
sys.path.insert(0, sys.argv[1])
import beh_judges as B  # noqa: E402
t = sys.argv[2] or B.CONTENT_TRUTH_DEFAULT
why = B.content_guard(t)
print(f"  [note] content judge truth {t}: " + ("calibrated" if why is None else
      f"REFUSED by the guard -> judge_b ends with judge_b.CONTENT_REFUSED and the chain goes on ({why})"))
PYEOF
  echo "  readers: gate $GATE_READER; read_b $READ_B; read_a $READ_A; caps: score $SCORE_CAP USD per reader, judges $EVAL_CAP USD ledger, RL $RL_TRAIN_CAP_USD / $RL_TOTAL_CAP_USD USD"
  for st in "${STAGES[@]}"; do total=$((total + EXPECT_MIN[$st])); done
  # judge_b runs beside read_a: the shorter of the two adds nothing
  total=$((total - (EXPECT_MIN[judge_b] < EXPECT_MIN[read_a] ? EXPECT_MIN[judge_b] : EXPECT_MIN[read_a])))
  echo "  expected time: about $total min in all (judge_b beside read_a), waits for GPUs not counted"
  echo "  markers in $M: $(ls "$M" 2>/dev/null | tr '\n' ' ')"
  echo "  GPUs now (one look; sft needs 2 usable GPUs on two looks):"
  gpu_look_once | sed 's/^/    /'
  return $miss
}

# ---------------------------------------------------------------- main
cd "$REPO" || exit 2
if [ "$DRY" != 1 ]; then  # one chain at a time; announcements of this script closed on exit
  exec 7>>"$M/chain.lock"
  flock -n 7 || { say "another chain.sh (or a process it started) holds $M/chain.lock; chain.pid $(cat "$M/chain.pid" 2>/dev/null || echo none); not started"; exit 3; }
  trap 'release_gpus "*" "script exit"' EXIT
  trap 'exit 143' TERM INT
  reap_stale_holds
fi
case "${1:-all}" in
  all)
    say "behaviour reader chain start host=$(hostname) stamp=$STAMP dry=$DRY repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) em-reader=$(git -C "$EMWT" rev-parse --short HEAD 2>/dev/null) rl-stage-fixes=$(git -C "$RLWT" rev-parse --short HEAD 2>/dev/null)"
    NMISS=0
    if [ "$DRY" = 1 ]; then dry_report; NMISS=$?; else echo "$$ $(hostname)" > "$M/chain.pid"; fi
    SOFT=""
    for st in "${STAGES[@]}"; do
      if [ "$st" = judge_b ]; then
        if [ "$DRY" != 1 ]; then judge_b_start; continue; fi
        echo "  [dry] judge_b runs in the background beside read_a (pid in markers/judge_b.pid); the chain waits for it before c1val; a failure of it is soft"
      fi
      [ "$st" = c1val ] && judge_b_wait
      run_stage "$st"; rc=$?
      [ $rc -eq 0 ] && continue
      case $st in
        c1val|upload) say "$st failed (rc $rc); recorded in $st.FAILED; going on"; SOFT="$SOFT $st"; continue ;;
      esac
      if [ $rc -eq $RC_STOP ]; then
        say "chain stopped by the gate (markers/gate.STOP)"; [ "$DRY" = 1 ] || echo "gate STOP $(stamp)" > "$M/chain.STOPPED"; exit $RC_STOP
      fi
      judge_b_wait  # a judge_b still running in the background ends before the chain stops
      say "chain stopped at $st"; [ "$DRY" = 1 ] || echo "stopped at $st $(stamp)" > "$M/chain.FAILED"; exit 1
    done
    if [ "$DRY" = 1 ]; then
      if [ $NMISS -gt 0 ]; then say "behaviour reader chain dry listing done; NOT READY: $NMISS prerequisite(s) missing or blocked (above)"; exit 4; fi
      say "behaviour reader chain dry listing done; prerequisites present"; exit 0
    fi
    echo "$(stamp)${SOFT:+; soft failures:$SOFT}" > "$M/chain.DONE"
    say "behaviour reader chain done${SOFT:+ (soft failures:$SOFT)}" ;;
  *) run_stage "$1" ;;
esac
