#!/bin/bash
# One new-setting student. The recipe is that of the sweep run runs/curve_lr1e4_b32_e7
# (scripts/curve_sweep/launch.sh): Qwen3-14B LoRA r16 alpha 32 on the seven projections, 50,000 rows, batch 32,
# lr 1e-4 cosine with 5% warmup on the 7-epoch schedule (10,941 steps), seed 1, no gradient checkpointing, probe and
# small eval every 100 steps, full eval every 500, stopped at <stop_step> (the schedule is not shortened).
# Saving, every step: per-document activation-gradient token sums (--save-grads, online_save.py: 40 gradient layers,
# fp16 unit rows x fp32 norms), the direct-write term batch sums of o_proj and down_proj at 40 layers as fp32 rows
# (--save-adamfx --adamfx-stepsums adamfx --adamfx-stepsums-dtype fp32 -> <out>/stepsums/adamfx/), and the same term
# on the 128 fixed documents (input 4f, --adamfx-fixed-docs -> <out>/adamfx_fixed/, inputs refreshed at step 0 and
# every 100 steps). The per-step bf16 true change (--save-acttrue) is off. Adapters in fp32 every --save-every steps
# and at step 0 (--adapter-dtype fp32 --save-adapter-step0); no Trainer checkpoints (an adamfx run cannot resume).
# LoRA pairs (fp32 before/after of one optimizer step, for adamfx_fp32_check.py and the 4f recompute) at the steps
# 101, 1001, 3001, 5001 that are <= <stop_step>: steps 100m+1 use the fixed-document sums of the refresh after step
# 100m, so the pair's "before" parameters are the ones that produced them.
#
# Usage: bash scripts/newset/train.sh <arm dolphin|clean|owl|butterfly|whale> <gpu> <out_dir> <stop_step> [--save-every N]
#        [--tag NAME] [--no-save-grads] [--dry-run]
#   <arm>          owl (second animal, same setting as dolphin, stop
#                  5,000) trains on the trainer's owl data (the HF snapshot cds-jb/qwen3-14b-owl-subliminal-nums-25x
#                  filtered_full.jsonl, as runs/pilot_owl_s1 and the owl sweep)
#                  butterfly, whale (the owl run's flags with stop 7,000) train on
#                  cds-jb/qwen3-14b-<animal>-subliminal-nums-25x filtered_full.jsonl (train_subliminal.py DATA);
#                  scripts/animals/train_all.sh calls this script
#   <out_dir>      absolute path that does not exist yet (or is empty)
#   --save-every   adapter (and per-document block) spacing, default 500 (the smoke uses 100)
#   --tag          marker and log name, default <arm> (the smoke uses smoke_<arm>)
#   --no-save-grads  leave out --save-grads (no per-document gradient store; the owl run reads only inputs 4 and 5);
#                  the direct-write stepsums, the fixed-document version, the LoRA pairs and the adapters stay on
#   --dry-run      print the checks and the command, launch nothing
# The script returns once the training process runs (it is detached with setsid nohup). A wrapper waits for it and
# writes the final marker:
#   markers  $IIF_BASE/agm_data/runs/newset/markers/train_<tag>.{STARTED,DONE,FAILED} and train_<tag>.pid (python
#            pid); DONE only when python exited 0 and its log has "done: step <stop_step>,"; FAILED otherwise (exit
#            code and the log tail in the marker). events go to $IIF_BASE/agm_data/runs/newset/events.log.
#            After a DONE the wrapper runs main_run/check_newset_train.py on the CPU (loss check: loss of the first
#            500 steps against runs/curve_lr1e4_b32_e7, dolphin only; pair check: 4f recomputed from the LoRA pairs) into
#            <out_dir>/check_newset_train.json, markers check_train_<tag>.{DONE,FAILED}; DONE does not wait for it.
#            The loss reference is the sweep run of the same recipe: curve_lr1e4_b32_e7 for dolphin,
#            owl_lr1e4_b32_e7 for owl (scripts/owl_sweep/launch.sh), none for clean, butterfly and whale (no sweep).
#   pid      <out_dir>.pid (python pid, as scripts/curve_sweep/launch.sh)
#   log      main_run/logs/<STAMP>_newset_train_<tag>.out (python stdout and stderr, and the wrapper's lines)
# Refuses: an existing non-empty <out_dir>; a DONE marker of the tag; a STARTED marker whose pid is alive; a GPU with
# 4 GiB or more in use; a fixed-documents file whose sha256 differs from the one written by make_fixed_docs.py;
# less free disk than 1.2 x the run's estimated bytes. Older STARTED/FAILED/pid files of the tag and an older
# <out_dir>.pid are renamed *.before_<YYYYmmdd_HHMMSS> first (never deleted or overwritten).
# Overrides (tests): NEWSET_STATE (state dir, default runs/newset), NEWSET_STAMP, NEWSET_FIXED_DOCS,
#   NEWSET_FIXED_SHA256, NEWSET_SKIP_GPU_CHECK=1.
set -u
IIF=${IIF_BASE:?set IIF_BASE}
export IIF_BASE=$IIF HF_HOME=$IIF/hf_home HF_HUB_OFFLINE=1
PY=$IIF/venvs/interpretable_influence/bin/python
P=$IIF/research_helper/projects/activation-gradient-metamodel
D=${NEWSET_STATE:-$IIF/agm_data/runs/newset}
M=$D/markers
L=$P/main_run/logs
FIXED=${NEWSET_FIXED_DOCS:-$IIF/agm_data/newset/fixed_docs_128.jsonl}
FIXED_SHA=${NEWSET_FIXED_SHA256:-f696b8cbaa304236677365e51b254ebdddf20e5f14b8a85cee6d5924fd6288a6}
MEM_MAX_MIB=4096
BATCH=32
GRAD_BYTES_PER_ROW=1065376   # online_save row: 40 grad + 8 act layers, full and tag (full_dolphin_s1 RUN.json)
STEPSUM_BYTES=$((2 * 40 * 5120 * 4))  # two fp32 per-step stores (adamfx, adamfx_fixed)
PAIR_CANDIDATES="101 1001 3001 5001"

log() { echo "$(date '+%F %T') [newset train] $*"; }
event() { mkdir -p "$D"; echo "$(date '+%F %T') [newset train] $*" >> "$D/events.log"; log "$*"; }
alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
before_name() {  # file: a name <file>.before_<YYYYmmdd_HHMMSS>[_n] that does not exist yet
  local b t n=0; b="$1.before_$(date +%Y%m%d_%H%M%S)"; t=$b
  while [ -e "$t" ]; do n=$((n + 1)); t="${b}_$n"; done
  echo "$t"
}
aside() { [ -e "$1" ] || return 0; mv -n "$1" "$(before_name "$1")"; }
die() { echo "train.sh: $*" >&2; exit 2; }

# ---------- the wrapper: runs python in the foreground of a detached shell and writes the final marker ----------
if [ "${1:-}" = "--inner" ]; then
  shift
  arm=$1; gpu=$2; out=$3; stop=$4; tag=$5; LOG=$6; shift 6
  key=train_$tag
  echo "$(date '+%F %T') arm $arm, GPU $gpu, out $out, stop step $stop, wrapper pid $$, log $LOG" > "$M/$key.STARTED"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$P/main_run/train_subliminal.py" "$@" &
  pid=$!
  echo "$pid" > "$out.pid"
  echo "$pid" > "$M/$key.pid"
  event "start $key: python pid $pid on GPU $gpu, out $out, stop step $stop, log $LOG"
  wait "$pid"; rc=$?
  if [ $rc -eq 0 ] && grep -q "done: step $stop, " "$LOG"; then
    echo "$(date '+%F %T') exit 0, $(grep "done: step $stop, " "$LOG" | tail -1)" > "$M/$key.DONE"
    event "DONE $key (python $pid exit 0, reached step $stop)"
    # loss check (loss against the sweep run; dolphin only, the sweep has no clean twin) and pair check (4f recomputed
    # from the LoRA pairs), on the CPU; reported only: the DONE marker above does not depend on them
    ref=none; [ "$arm" = dolphin ] && ref=$IIF/agm_data/runs/curve_lr1e4_b32_e7
    [ "$arm" = owl ] && ref=$IIF/agm_data/runs/owl_lr1e4_b32_e7
    cj=$out/check_newset_train.json
    if CUDA_VISIBLE_DEVICES= "$PY" "$P/main_run/check_newset_train.py" --run "$out" --ref-run "$ref" --out "$cj" \
        > /dev/null 2>> "$LOG"; then ck=DONE; else ck=FAILED; fi
    cp=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['checks_passed'])" "$cj" 2>/dev/null)
    echo "$(date '+%F %T') checks ${cp:-not written} ($cj)" > "$M/check_$key.$ck"
    event "check_$key $ck: loss and pair checks passed ${cp:-?} ($cj)"
  else
    { echo "$(date '+%F %T') python $pid exit $rc; 'done: step $stop,' $(grep -c "done: step $stop, " "$LOG") times in $LOG"
      tail -20 "$LOG"; } > "$M/$key.FAILED"
    event "FAILED $key (python $pid exit $rc); see $M/$key.FAILED"
  fi
  exit 0
fi

# ---------- the launcher ----------
[ $# -ge 4 ] || die "usage: train.sh <arm dolphin|clean|owl|butterfly|whale> <gpu> <out_dir> <stop_step> [--save-every N] [--tag NAME] [--no-save-grads] [--dry-run]"
arm=$1; gpu=$2; out=$3; stop=$4; shift 4
save_every=500; tag=$arm; dry=0; grads=1
while [ $# -gt 0 ]; do
  case "$1" in
    --save-every) save_every=${2:-}; shift 2;;
    --tag) tag=${2:-}; shift 2;;
    --no-save-grads) grads=0; shift;;
    --dry-run) dry=1; shift;;
    *) die "unknown option $1";;
  esac
done
case "$arm" in dolphin|clean|owl|butterfly|whale) ;; *) die "arm must be dolphin, clean, owl, butterfly or whale, not '$arm'";; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || die "gpu must be an index, not '$gpu'"
[[ "$stop" =~ ^[0-9]+$ ]] && [ "$stop" -ge 1 ] || die "stop_step must be a positive integer, not '$stop'"
[[ "$save_every" =~ ^[0-9]+$ ]] && [ "$save_every" -ge 1 ] || die "--save-every must be a positive integer"
[[ "$tag" =~ ^[A-Za-z0-9_]+$ ]] || die "--tag must be letters, digits and _"
case "$out" in /*) ;; *) die "out_dir must be an absolute path";; esac
out=${out%/}
key=train_$tag
STAMP=${NEWSET_STAMP:-$(date +%Y%m%d_%H%M)}
LOG=$L/${STAMP}_newset_train_${tag}.out

problems=()
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then problems+=("$out exists and is not empty"); fi
[ -f "$M/$key.DONE" ] && problems+=("$M/$key.DONE exists (this tag already finished)")
if [ -f "$M/$key.pid" ] && alive "$(head -1 "$M/$key.pid")"; then problems+=("$key runs (pid $(head -1 "$M/$key.pid"))"); fi
if [ -f "$out.pid" ] && alive "$(head -1 "$out.pid")"; then problems+=("$out.pid names a live process"); fi
if [ ! -f "$FIXED" ]; then problems+=("fixed documents $FIXED missing (run main_run/make_fixed_docs.py)")
else
  sha=$(sha256sum "$FIXED" | cut -d' ' -f1)
  [ "$sha" = "$FIXED_SHA" ] || problems+=("$FIXED sha256 $sha, expected $FIXED_SHA")
fi
if [ "${NEWSET_SKIP_GPU_CHECK:-0}" != 1 ]; then
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [ -z "$used" ]; then problems+=("nvidia-smi does not know GPU $gpu")
  elif [ "$used" -ge $MEM_MAX_MIB ]; then problems+=("GPU $gpu has $used MiB in use (limit $MEM_MAX_MIB)"); fi
fi
need_gb=$(( (grads * stop * BATCH * GRAD_BYTES_PER_ROW + stop * STEPSUM_BYTES) / 1000000000 + 10 ))
need_gb=$(( need_gb * 12 / 10 ))
free_gb=$(df -BG --output=avail "$(dirname "$out")" 2>/dev/null | tail -1 | tr -dc '0-9')
[ -n "$free_gb" ] && [ "$free_gb" -ge "$need_gb" ] || problems+=("disk ${free_gb:-?} GB free under $(dirname "$out"), need $need_gb GB")

pairs=""
for t in $PAIR_CANDIDATES; do [ "$t" -le "$stop" ] && pairs="${pairs:+$pairs,}$t"; done
[ -n "$pairs" ] || pairs=none
ARGS=(--arm "$arm" --n-rows 50000 --lr 1e-4 --batch-size $BATCH --epochs 7 --warmup-ratio 0.05
      --eval-every 100 --full-eval-every 500 --save-every "$save_every" --seed 1 --grad-ckpt 0 --stop-step "$stop"
      --save-adamfx --adamfx-stepsums adamfx --adamfx-stepsums-dtype fp32
      --adamfx-fixed-docs "$FIXED" --adamfx-fixed-every 100
      --adamfx-lora-pairs "$pairs" --adamfx-lora-pairs-format fp32
      --adapter-dtype fp32 --save-adapter-step0 --no-checkpoints --out "$out")
[ $grads -eq 1 ] && ARGS+=(--save-grads)

echo "arm $arm, GPU $gpu, out $out, stop step $stop, save every $save_every, tag $tag, per-document gradients $([ $grads -eq 1 ] && echo on || echo off)"
echo "disk: ${free_gb:-?} GB free, this run needs about $need_gb GB (x1.2 margin included)"
echo "log $LOG; markers $M/$key.*; pid file $out.pid"
echo "command: CUDA_VISIBLE_DEVICES=$gpu $PY $P/main_run/train_subliminal.py ${ARGS[*]}"
if [ ${#problems[@]} -gt 0 ]; then
  for p_ in "${problems[@]}"; do echo "REFUSED: $p_" >&2; done
  [ $dry -eq 1 ] && exit 1
  exit 2
fi
if [ $dry -eq 1 ]; then echo "dry run: nothing launched"; exit 0; fi

mkdir -p "$M" "$L"
for f in "$M/$key.STARTED" "$M/$key.FAILED" "$M/$key.pid" "$out.pid" "$M/check_$key.DONE" "$M/check_$key.FAILED"; do
  aside "$f"
done
setsid nohup bash "$(readlink -f "$0")" --inner "$arm" "$gpu" "$out" "$stop" "$tag" "$LOG" "${ARGS[@]}" < /dev/null >> "$LOG" 2>&1 &
wrapper=$!
for _ in $(seq 1 30); do [ -s "$out.pid" ] && break; sleep 1; done
if [ -s "$out.pid" ]; then
  echo "launched: wrapper pid $wrapper, python pid $(cat "$out.pid")"
else
  echo "launched: wrapper pid $wrapper; no python pid after 30 s (see $LOG)" >&2
  exit 1
fi
