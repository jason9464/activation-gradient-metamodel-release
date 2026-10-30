#!/usr/bin/env bash
# New-readers suite chain: the gradient-vs-activation experiments E2-E5 read again with one of the three new
# three-layer readers (no E6, no slot reading of E2, judge cap SUITE_CAP_USD for the whole suite). Modelled on
# scripts/beh_reader/activation_twin.sh (pick_gpu with claim files, run_stage with markers / pid files, hang guard,
# cost watchdog, --dry); that file, slot_reader.sh, slot_extract.py and build_slot_dirs.py are only read or called,
# not edited.
#
# Usage: bash scripts/beh_reader/new_readers_suite.sh <reader> [e2] [e3] [e4] [e5] [--dry]
#          (detached by the caller: setsid nohup bash ... < /dev/null > <log> 2>&1 &)
#   <reader>  grad3  three-layer gradient reader: runs/reader_rl_scratch_L3_beh_paper/adapters/cycle_040,
#                    RUN.json runs/reader_scratch_L3_beh/RUN.json
#             twin   activation twin: runs/reader_rl_scratch_L3act_beh_paper/adapters/cycle_040 (the SFT adapter
#                    runs/reader_scratch_L3act_beh/adapters/step_007082 when the twin chain's markers hold rl.COST_CAP),
#                    RUN.json runs/reader_scratch_L3act_beh/RUN.json
#             slot   slot reader: runs/reader_rl_scratch_L3slot_beh_paper/adapters/cycle_040 (SFT step_007082 when
#                    the slot chain's markers hold rl.COST_CAP), RUN.json runs/reader_scratch_L3slot_beh/RUN.json
#   experiments: default e2 e3 e4 e5 (slot: e3 e4 e5). Refused (exit 2): an unknown reader or experiment, e2 with slot,
#   e6, GPUS entries that are not GPU indices.
#   --dry prints every stage's command (GPU shown as <gpu>, judge caps as formulas) and writes nothing.
#
# Programs: NRS = main_run/beh_reader/new_readers_suite.py (build, slot-extract, summary, plot), NRSR =
# main_run/beh_reader/new_readers_suite_read.py (read, yesno, e5 judge), and the existing rare_yesno.py (read,
# summary), rare_perdoc.py (judge, summary), judge_dolphin_answers.py. Layout (NRS's): D = $ROOT/<exp>/<reader>,
# inputs in D/inputs (BUILD.json when built); markers $ROOT/markers/<reader>/<stage>.{START,DONE,FAILED,pid}.
# Stages per experiment E, run as one branch per experiment side by side:
#   docs_E, extract_E   slot only, e3 / e4 / e5: NRS build writes D/docs.jsonl; when it leaves D/BUILD_PENDING.json
#             (no D/inputs/BUILD.json), extract_E runs NRS slot-extract --docs D/docs.jsonl --out-dir D/slots on a GPU
#             with 110 GiB free (slot_extract.py's code on that list), else extract_E is not needed (DONE, says so).
#             e4 normally needs no extraction (the slot chain's slots_beh.npz).
#   build_E   NRS build --exp E --reader R --out $ROOT (CPU; for slot the assembly after the extraction, not needed when
#             docs_E already assembled). Passes only when D/inputs/BUILD.json exists.
#   read_E    NRSR read (GPU, 80 GiB: e2_generate's TrainedReader + run_readings, the reader's adapter and RUN.json):
#             e2 100 answers x 20 steps x arms dolphin and clean, e3 100 answers x 7 checkpoints x 4 batches (dolphin
#             student), both the narrow preference question; e4 10 answers per document / sum (doc, showing,
#             nonshowing), e5 10 answers per step of each model (common.QUESTION_BEHAVIOURAL)
#   yesno_E   e3, e5: NRSR yesno (GPU, 60 GiB), first-token P(Yes) (e3 the dolphin yes/no question on every
#             checkpoint x batch of both students; e5 already_learned's behaviour questions);
#             e4: rare_yesno.py read --scorer reader --bundles rare_quirk --reader-adapter / --reader-run-json /
#             --reader-layers 10,20,30 / --reader-name R --run-dir D/yesno (grad3: --input grad; twin: --input act
#             --act-dir runs/beh_reader/rare_yesno/act, the files the build copied byte for byte), slot: NRSR yesno
#             (rare_yesno.cmd_read with the slot files; its CLI has no inputs option); an empty GPU (rare_yesno
#             refuses one with >= 4 GiB used: need 176 GiB)
#   ynsum_e4  rare_yesno.py summary --run-dir D/yesno --bundles rare_quirk (CPU; grad3, twin; NRSR does it for slot)
#   judge_E   the existing judges, same prompts, caches and ledgers (API; serialised over the whole suite):
#             e2, e3  judge_dolphin_answers.py --inputs batch --readers nrs_R --question preference
#                     --runs <every dir X with X/e2/batch_nrs_R_preference.jsonl under D> --key <their window labels>
#                     --max-cost <its preference ledger now + cap>
#             e4      rare_perdoc.py judge --out D/open --cap-usd <cap> --mode batch (truth judge)
#             e5      NRSR judge --cap-usd <cap> --mode batch (truth judge, beh_judges.cmd_judge repointed as
#                     already_learned.py judge; that command needs all three input kinds of all four models)
#             After each judge, judged rows = answers is checked per reading file (and per window for e2, e3):
#             <stem>_<window>_judge_dolphin.jsonl or <stem>_judge_truth.jsonl.
#   opensum_e4  rare_perdoc.py summary --out D/open (CPU): the open-answer summary NRS summary reads
#   then summary, plot (CPU, run on every chain run): NRS summary --out $ROOT, NRS plot --out $ROOT (the figure
#             docs/figures/20261029_new_readers_suite.png; with another ROOT, --fig $ROOT/new_readers_suite_smoke.png)
# Order inside a branch: [docs_E -> extract_E ->] build_E, then side by side read_E -> judge_E [-> opensum_e4] and
# yesno_E [-> ynsum_e4]. A failed stage stops its branch; after all branches, summary and plot run only when every
# branch passed (else markers/<reader>/chain.FAILED). A DONE stage is skipped on a rerun; a FAILED one is renamed
# <stage>.FAILED.before_<time> and run again.
#
# Judge money (ledger increase <= cap): every judge stage takes the suite
# judge lock ($ROOT/judge.lock, so judges of the three reader chains never overlap), reads both ledgers (truth:
# runs/beh_reader/judge/cost_ledger.tsv column 10; dolphin preference: runs/judge_dolphin/cost_ledger_preference.tsv
# column 12) and gets cap = min(JUDGE_CAP_USD - this reader's spend, SUITE_CAP_USD - the suite's spend), spends from
# $ROOT/judge_spend.tsv (one row per judge stage: the increase of both ledgers over the stage). The increase also
# counts other chains' judges that write the same ledgers at the same time, so the spend is never under-counted. A
# stage with cap < 0.05 USD is not started (FAILED "cap"). Cost watchdog: every COST_CHECK_S the ledger increase since
# the stage started is compared with its cap; above it the stage is stopped (<stage>.COST_CAP, FAILED).
#
# GPUs: only $GPUS (default "1 2 3"; GPU 0 is the slot chain's: GPUS naming 0 is refused (exit 2) unless ALLOW_GPU0=1,
# and NRS slot-extract refuses GPU 0 in any case). A GPU stage reads
# nvidia-smi free memory twice, FREE_GAP_S apart, and takes the GPU with the most free memory among those with at least
# the stage's need on both reads and no live claim; claims are the activation twin chain's files
# (CLAIM_DIR = runs/beh_reader/activation_twin/markers/gpu<G>.claim, made under its lock gpu.lock, in its format
# "<name> <pid> <host> <date> <time> free_mib=<n>"; created when missing), so the twin chain, the slot chain's borrowing
# and these chains never pick the same GPU. A claim is alive while its pid lives (a claim written on another host
# counts as alive). SUITE_MAX_GPU (default: the number of GPUs in GPUS) caps the live claims of all suite chains
# together (e.g. 1 while the twin chain still runs). Our claim is moved to gpu<G>.claim.released_<time>_<name> when
# the stage ends (only when the claim file is still ours). It waits in WAIT_STEP_S steps up to WAIT_H hours.
# CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=<the GPU> for a GPU stage, which also gets --gpu <the GPU>;
# CPU and API stages see no GPU (CUDA_VISIBLE_DEVICES empty). Every stage process gets SUITE_STAGE=<stage>.
# Hang guard: a stage whose log has not grown for HANG_S seconds (default 1800; judge stages JUDGE_HANG_S, 4 h) gets
# TERM to its process group, KILL when still alive 60 s later, and is marked FAILED.
# Logs main_run/logs/20261029_suite_<reader>_<stage>.out (appended, one header line per attempt).
# Disk: the suite writes about 15 GB; the runner refuses to start with less than 15 + 20 GB free.
# The runner refuses to start (exit 1) when the reader's adapter or RUN.json does not exist (real launches are made
# per reader once its adapter exists), and for twin / slot (no ADAPTER override) until that chain's markers hold
# rl.DONE or rl.COST_CAP (the cycle_040 folder can exist before RL has finished writing it).
#
# env: ROOT (runs/beh_reader/new_readers_suite), GPUS ("1 2 3"), JUDGE_CAP_USD (15, per reader chain), SUITE_CAP_USD
#   (25, all chains), SUITE_MAX_GPU, ALLOW_GPU0 (0), WAIT_H (6), HANG_S (1800), JUDGE_HANG_S (14400), ADAPTER, RUN_JSON (override the
#   reader's), LIMIT (smoke: --limit N to NRS build, NRSR read / yesno and rare_yesno read; refused with the real ROOT;
#   implies JUDGE_DRY=1), JUDGE_DRY=1 (judges run with --dry-run, no API call, and their stages get <stage>.DRY, not
#   DONE; opensum_e4 is skipped with <stage>.DRY);
#   for the stub test: LOGD, CLAIM_DIR, TWIN_M, SLOT_M, YESNO_ACT_DIR, TRUTH_LEDGER, DOLPHIN_LEDGER, FREE_GAP_S (30),
#   WAIT_STEP_S (300), POLL_S (10), COST_CHECK_S (120), PY, NRS / NRSR / RY / RP / JD (the five python files),
#   GPU_QUERY (nvidia-smi; split on spaces), DISK_NEED_GB (15)
set -u
P=$(cd "$(dirname "$0")/../.." && pwd); cd "$P" || exit 1
say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

# ---------------------------------------------------------------- arguments
DRY=0; READER=""; EXPS=()
for a in "$@"; do
  case $a in
    --dry) DRY=1 ;;
    grad3|twin|slot)
      if [ -n "$READER" ]; then echo "one reader per run (got $READER and $a)"; exit 2; fi
      READER=$a ;;
    e2|e3|e4|e5)
      [ -z "$READER" ] && { echo "the reader comes first: new_readers_suite.sh <grad3|twin|slot> [e2 e3 e4 e5] [--dry]"; exit 2; }
      case " ${EXPS[*]:-} " in *" $a "*) ;; *) EXPS+=("$a") ;; esac ;;
    e6) echo "e6 (filtered student) is not run in this suite"; exit 2 ;;
    *) if [ -z "$READER" ]; then echo "unknown reader '$a': one of grad3, twin, slot"
       else echo "unknown experiment '$a': e2, e3, e4, e5"; fi
       exit 2 ;;
  esac
done
if [ -z "$READER" ]; then echo "usage: new_readers_suite.sh <grad3|twin|slot> [e2 e3 e4 e5] [--dry]"; exit 2; fi
if [ "${#EXPS[@]}" = 0 ]; then
  if [ "$READER" = slot ]; then EXPS=(e3 e4 e5); else EXPS=(e2 e3 e4 e5); fi
fi
if [ "$READER" = slot ] && [[ " ${EXPS[*]} " == *" e2 "* ]]; then
  echo "e2 is not read with the slot reader (the dolphin student is not rerun with slot saving)"
  exit 2
fi

# ---------------------------------------------------------------- paths and settings
IIF_BASE=${IIF_BASE:?set IIF_BASE}
export IIF_BASE HF_HOME=${HF_HOME:-$IIF_BASE/hf_home} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-$IIF_BASE/venvs/interpretable_influence/bin/python}
EMWT=$IIF_BASE/agm_data/worktree_em_reader
RUNS=$IIF_BASE/agm_data/runs; EV=$RUNS/beh_reader
NRS=${NRS:-$P/main_run/beh_reader/new_readers_suite.py}
NRSR=${NRSR:-$P/main_run/beh_reader/new_readers_suite_read.py}
RY=${RY:-$P/main_run/beh_reader/rare_yesno.py}
RP=${RP:-$P/main_run/beh_reader/rare_perdoc.py}
JD=${JD:-$P/main_run/judge_dolphin_answers.py}
REAL_ROOT=$EV/new_readers_suite
ROOT=${ROOT:-$REAL_ROOT}; M=$ROOT/markers/$READER
CLAIM_DIR=${CLAIM_DIR:-$EV/activation_twin/markers}
TWIN_M=${TWIN_M:-$EV/activation_twin/markers}; SLOT_M=${SLOT_M:-$EV/slot_reader/markers}
YESNO_ACT_DIR=${YESNO_ACT_DIR:-$EV/rare_yesno/act}
TRUTH_LEDGER=${TRUTH_LEDGER:-$EV/judge/cost_ledger.tsv}
DOLPHIN_LEDGER=${DOLPHIN_LEDGER:-$RUNS/judge_dolphin/cost_ledger_preference.tsv}
SPEND=$ROOT/judge_spend.tsv; JLOCK=$ROOT/judge.lock
GPUS=${GPUS:-"1 2 3"}; JUDGE_CAP_USD=${JUDGE_CAP_USD:-15}; SUITE_CAP_USD=${SUITE_CAP_USD:-25}
SUITE_MAX_GPU=${SUITE_MAX_GPU:-$(echo $GPUS | wc -w)}
WAIT_H=${WAIT_H:-6}; HANG_S=${HANG_S:-1800}; JUDGE_HANG_S=${JUDGE_HANG_S:-14400}
FREE_GAP_S=${FREE_GAP_S:-30}; WAIT_STEP_S=${WAIT_STEP_S:-300}; POLL_S=${POLL_S:-10}; COST_CHECK_S=${COST_CHECK_S:-120}
DISK_NEED_GB=${DISK_NEED_GB:-15}; LIMIT=${LIMIT:-}; JUDGE_DRY=${JUDGE_DRY:-0}
LOGD=${LOGD:-$P/main_run/logs}
MIN_JUDGE_USD=0.05
NAME=nrs_$READER  # reading name (e2 / e3 files batch_nrs_<reader>_preference.jsonl)

case $READER in
  grad3) SFT_RUN=$RUNS/reader_scratch_L3_beh; RL_RUN=$RUNS/reader_rl_scratch_L3_beh_paper; CAPM="" ;;
  twin)  SFT_RUN=$RUNS/reader_scratch_L3act_beh; RL_RUN=$RUNS/reader_rl_scratch_L3act_beh_paper; CAPM=$TWIN_M ;;
  slot)  SFT_RUN=$RUNS/reader_scratch_L3slot_beh; RL_RUN=$RUNS/reader_rl_scratch_L3slot_beh_paper; CAPM=$SLOT_M ;;
esac
RUN_JSON=${RUN_JSON:-$SFT_RUN/RUN.json}
reader_adapter() {  # the RL cycle-40 adapter, or the SFT adapter after the reader's RL stopped at its cost cap
  if [ -n "${ADAPTER:-}" ]; then echo "$ADAPTER"
  elif [ -n "$CAPM" ] && [ -e "$CAPM/rl.COST_CAP" ]; then echo "$SFT_RUN/adapters/step_007082"
  else echo "$RL_RUN/adapters/cycle_040"; fi
}
dexp() { echo "$ROOT/$1/$READER"; }  # D of an experiment

for g in $GPUS; do
  case $g in *[!0-9]*|"") echo "GPUS='$GPUS': '$g' is not a GPU index"; exit 2 ;; esac
  if [ "$g" = 0 ] && [ "${ALLOW_GPU0:-0}" != 1 ]; then  # GPU 0 = the slot-reader chain's GPU
    echo "GPUS='$GPUS': GPU 0 belongs to the slot-reader chain; refused unless ALLOW_GPU0=1 (after that chain ended)"
    exit 2
  fi
done
if [ -n "$LIMIT" ] && [ "$(realpath -m "$ROOT")" = "$(realpath -m "$REAL_ROOT")" ]; then
  echo "LIMIT is a smoke option; give ROOT outside $REAL_ROOT"; exit 2
fi
if [ -n "$LIMIT" ] && [ "$JUDGE_DRY" != 1 ]; then JUDGE_DRY=1; echo "LIMIT set (smoke): judges run with --dry-run only (JUDGE_DRY=1)"; fi

disk_ok() {  # $1 = GB needed (plus 20 GB margin) on the agm_data disk
  local free_gb; free_gb=$(df -BG --output=avail "$IIF_BASE/agm_data" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -z "$free_gb" ] && { say "disk: df failed on $IIF_BASE/agm_data" >&2; return 1; }
  if [ "$free_gb" -lt $(( $1 + 20 )) ]; then say "disk: ${free_gb} GB free, need $1 + 20 GB" >&2; return 1; fi
}

if [ "$DRY" = 0 ]; then
  ad=$(reader_adapter)
  if [ ! -d "$ad" ] || [ ! -f "$RUN_JSON" ]; then
    echo "reader $READER: adapter $ad or RUN.json $RUN_JSON missing; launch this chain when they exist"; exit 1
  fi
  # twin / slot: the cycle-40 folder appears while RL may still write it; wait for that chain's RL marker
  if [ -z "${ADAPTER:-}" ] && [ -n "$CAPM" ] && [ ! -e "$CAPM/rl.DONE" ] && [ ! -e "$CAPM/rl.COST_CAP" ]; then
    echo "reader $READER: RL of its chain has not ended ($CAPM has neither rl.DONE nor rl.COST_CAP); launch after it"
    exit 1
  fi
  disk_ok "$DISK_NEED_GB" || exit 1
  mkdir -p "$M" "$LOGD" "$CLAIM_DIR"
  exec 8>"$M/runner.lock"
  if ! flock -n 8; then echo "another new_readers_suite runner for $READER holds $M/runner.lock"; exit 3; fi
  echo "$$ $(hostname)" > "$M/runner.pid"
fi

# ---------------------------------------------------------------- GPU choice (the twin chain's protocol)
GPU_QUERY=${GPU_QUERY:-nvidia-smi}
HOST=$(hostname)
gpu_free_mib() {  # "<index>,<free MiB>" lines
  $GPU_QUERY --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' '
}
claim_alive() {  # $1 = claim file: its pid is alive here, or it was written on another host (counted as alive)
  local pid host; read -r _ pid host _ < "$1" 2>/dev/null || return 1
  [ -n "$pid" ] || return 1
  [ -n "$host" ] && [ "$host" != "$HOST" ] && return 0
  kill -0 "$pid" 2>/dev/null
}
suite_claims() {  # live claims of all suite chains (names nrs_*)
  local c n=0
  for c in "$CLAIM_DIR"/gpu*.claim; do
    [ -e "$c" ] || continue
    case $(awk '{print $1; exit}' "$c" 2>/dev/null) in nrs_*) claim_alive "$c" && n=$((n + 1)) ;; esac
  done
  echo "$n"
}
pick_gpu() {  # $1 need GiB, $2 claim name; prints the GPU and writes its claim; waits up to WAIT_H hours
  local need=$1 name=$2 t_end=$(( $(date +%s) + WAIT_H * 3600 ))
  while :; do
    (
      flock 9
      [ "$(suite_claims)" -lt "$SUITE_MAX_GPU" ] || exit 1
      a=$(gpu_free_mib); sleep "$FREE_GAP_S"; b=$(gpu_free_mib)
      best=""; bestf=0
      for g in $GPUS; do
        if [ -e "$CLAIM_DIR/gpu$g.claim" ] && claim_alive "$CLAIM_DIR/gpu$g.claim"; then continue; fi
        fa=$(echo "$a" | awk -F, -v g="$g" '$1==g{print $2}'); fb=$(echo "$b" | awk -F, -v g="$g" '$1==g{print $2}')
        { [ -z "$fa" ] || [ -z "$fb" ]; } && continue
        f=$(( fa < fb ? fa : fb ))
        if [ "$f" -ge $(( need * 1024 )) ] && [ "$f" -gt "$bestf" ]; then best=$g; bestf=$f; fi
      done
      if [ -n "$best" ]; then
        echo "$name $BASHPID_OWNER $HOST $(date '+%F %T') free_mib=$bestf" > "$CLAIM_DIR/gpu$best.claim"
        echo "$best"; exit 0
      fi
      exit 1
    ) 9>"$CLAIM_DIR/gpu.lock" && return 0
    [ "$(date +%s)" -ge "$t_end" ] && return 1
    say "$name: no GPU among $GPUS with $need GiB free and no claim (suite claims $(suite_claims)/$SUITE_MAX_GPU); waiting $WAIT_STEP_S s" >&2
    sleep "$WAIT_STEP_S"
  done
}
release_gpu() {  # $1 GPU, $2 claim name: move the claim aside only when it is still ours
  local c=$CLAIM_DIR/gpu$1.claim n p
  [ -n "${1:-}" ] && [ -e "$c" ] || return 0
  read -r n p _ < "$c"
  if [ "$n" = "$2" ] && [ "$p" = "$BASHPID_OWNER" ]; then mv "$c" "$c.released_$(date +%Y%m%d_%H%M%S)_$2"; fi
}

# ---------------------------------------------------------------- judge money
ledger_total() {  # $1 ledger tsv, $2 cost column (1-based); 0 when the file is missing
  if [ -f "$1" ]; then awk -F'\t' -v c="$2" 'NR > 1 && NF >= c && $c != "" { s += $c } END { printf "%.6f\n", s + 0 }' "$1"
  else echo "0.000000"; fi
}
truth_total() { ledger_total "$TRUTH_LEDGER" 10; }
dolphin_total() { ledger_total "$DOLPHIN_LEDGER" 12; }
spent() {  # $1 reader ("" = every reader): the summed ledger increase of the suite's judge stages
  if [ -f "$SPEND" ]; then awk -F'\t' -v r="$1" 'NR > 1 && (r == "" || $2 == r) { s += $8 } END { printf "%.6f\n", s + 0 }' "$SPEND"
  else echo "0.000000"; fi
}
fsub() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f\n", a - b }'; }
fadd() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.6f\n", a + b }'; }
fmin() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.4f\n", (a < b) ? a : b }'; }
fgt()  { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a > b) }'; }
record_spend() {  # $1 stage, $2 truth before, $3 truth after, $4 dolphin before, $5 dolphin after, $6 cap
  local d; d=$(fadd "$(fsub "$3" "$2")" "$(fsub "$5" "$4")")
  [ -f "$SPEND" ] || printf 'time\treader\tstage\ttruth_before\ttruth_after\tdolphin_before\tdolphin_after\tincrease_usd\tcap_usd\n' > "$SPEND"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$READER" "$1" "$2" "$3" "$4" "$5" "$d" "$6" >> "$SPEND"
  echo "$d"
}

# ---------------------------------------------------------------- readings found at judge time (e2, e3)
reading_files() {  # $1 experiment dir: the dolphin-judged readings, one per line
  local f
  for f in "$1"/*/e2/batch_${NAME}_preference.jsonl; do [ -f "$f" ] && echo "$f"; done
}
reading_runs() { reading_files "$1" | while read -r f; do dirname "$(dirname "$f")"; done | paste -sd, -; }
reading_keys() {  # the window labels of the readings (first row of each file), comma-joined, first-seen order
  reading_files "$1" | "$PY" -c '
import json, sys
keys = []
for f in sys.stdin.read().split():
    with open(f) as fh:
        for k in json.loads(fh.readline())["inputs"]:
            if k not in keys:
                keys.append(k)
print(",".join(keys))'
}
judged_check() {  # $1 experiment, $2 experiment dir: judged rows = answers per reading (and window); exit 1 otherwise
  "$PY" - "$1" "$2" "$NAME" <<'PYEOF'
import glob, json, os, sys
exp, d, name = sys.argv[1:4]
def rows(p):
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]
pairs = []  # (reading, judged, window or None)
if exp in ("e2", "e3"):
    for f in sorted(glob.glob(os.path.join(d, "*", "e2", f"batch_{name}_preference.jsonl"))):
        for k in rows(f)[0]["inputs"]:
            pairs.append((f, f[: -len(".jsonl")] + f"_{k}_judge_dolphin.jsonl", k))
else:
    sub = "open" if exp == "e4" else "*"
    for f in sorted(glob.glob(os.path.join(d, sub, "readings_*.jsonl"))):
        if "_judge_" not in os.path.basename(f):
            pairs.append((f, f[: -len(".jsonl")] + "_judge_truth.jsonl", None))
bad = []
for f, j, k in pairs:
    rr = rows(f)
    n = sum(len(rec["texts"]) for r in rr for w, rec in r["inputs"].items() if k is None or w == k)
    m = len(rows(j)) if os.path.exists(j) else None
    print(f"{'ok ' if m == n else 'BAD'} {n} answers, {m} judged rows: {j}")
    if m != n:
        bad.append(j)
if not pairs:
    print(f"no readings found under {d}")
sys.exit(0 if pairs and not bad else 1)
PYEOF
}

# ---------------------------------------------------------------- stage table
samples_of() { case $1 in e2|e3) echo 100 ;; *) echo 10 ;; esac; }
gpu_need() {  # GiB of free GPU memory a stage needs; 0 = CPU or API stage
  case $1 in
    extract_*) echo 110 ;;
    read_*) echo 80 ;;
    yesno_e4) echo 176 ;;
    yesno_*) echo 60 ;;
    *) echo 0 ;;
  esac
}
minutes_of() {
  case $1 in
    build_e2) [ "$READER" = twin ] && echo 5 || echo 3 ;;
    docs_*|build_*|ynsum_*|opensum_*|summary|plot) echo 2 ;;
    extract_*) echo 15 ;;
    read_e4) echo 40 ;;
    read_*) echo 15 ;;
    yesno_*) echo 5 ;;
    judge_e4) echo 30 ;;
    judge_*) echo 20 ;;
  esac
}
mk_stage() {  # sets CMD and CWD; $1 stage, $2 GPU
  local name=$1 g=$2 exp=${1##*_} D lim=() dry=() ad
  D=$(dexp "$exp"); ad=$(reader_adapter); CWD=$P
  [ -n "$LIMIT" ] && lim=(--limit "$LIMIT")
  [ "$JUDGE_DRY" = 1 ] && dry=(--dry-run)
  local py=(env PYTHONPATH="$EMWT" "$PY")
  case $name in
    docs_*|build_*) CMD=("${py[@]}" "$NRS" build --exp "$exp" --reader "$READER" --out "$ROOT" "${lim[@]}") ;;
    extract_*) CMD=("${py[@]}" "$NRS" slot-extract --docs "$D/docs.jsonl" --out-dir "$D/slots" --gpu "$g") ;;
    read_*)    CMD=("${py[@]}" "$NRSR" read --exp "$exp" --reader "$READER" --root "$ROOT" --gpu "$g" --adapter "$ad"
                    --run-json "$RUN_JSON" --samples "$(samples_of "$exp")" "${lim[@]}") ;;
    yesno_e4)
      case $READER in
        grad3) CMD=("${py[@]}" "$RY" read --gpu "$g" --scorer reader --input grad --bundles rare_quirk
                    --run-dir "$D/yesno" --reader-adapter "$ad" --reader-run-json "$RUN_JSON" --reader-layers 10,20,30
                    --reader-name "$READER" "${lim[@]}") ;;
        twin)  CMD=("${py[@]}" "$RY" read --gpu "$g" --scorer reader --input act --bundles rare_quirk
                    --run-dir "$D/yesno" --reader-adapter "$ad" --reader-run-json "$RUN_JSON" --reader-layers 10,20,30
                    --reader-name "$READER" --act-dir "$YESNO_ACT_DIR" "${lim[@]}") ;;
        slot)  CMD=("${py[@]}" "$NRSR" yesno --exp e4 --reader slot --root "$ROOT" --gpu "$g" --adapter "$ad"
                    --run-json "$RUN_JSON" "${lim[@]}") ;;
      esac ;;
    yesno_*)   CMD=("${py[@]}" "$NRSR" yesno --exp "$exp" --reader "$READER" --root "$ROOT" --gpu "$g" --adapter "$ad"
                    --run-json "$RUN_JSON" "${lim[@]}") ;;
    ynsum_e4)  CMD=("${py[@]}" "$RY" summary --run-dir "$D/yesno" --bundles rare_quirk) ;;
    opensum_e4) CMD=("${py[@]}" "$RP" summary --out "$D/open") ;;
    judge_e2|judge_e3)
      local runs keys maxc
      if [ "$DRY" = 1 ]; then
        runs="<every dir X with X/e2/batch_${NAME}_preference.jsonl under $D>"; keys="<their window labels>"
        maxc="<preference ledger total + cap>"
      else
        runs=$(reading_runs "$D"); keys=$(reading_keys "$D"); maxc=$(fadd "$(dolphin_total)" "$JCAP")
      fi
      CMD=(env -u PYTHONPATH "$PY" "$JD" --runs "$runs" --readers "$NAME" --inputs batch --question preference
           --key "$keys" --max-cost "$maxc" --workers 16 "${dry[@]}") ;;
    judge_e4)  CMD=("${py[@]}" "$RP" judge --out "$D/open" --cap-usd "$JCAP" --mode batch "${dry[@]}") ;;
    judge_e5)  CMD=("${py[@]}" "$NRSR" judge --exp e5 --reader "$READER" --root "$ROOT" --cap-usd "$JCAP" --mode batch
                    "${dry[@]}") ;;
    summary)   CMD=("${py[@]}" "$NRS" summary --out "$ROOT") ;;
    plot)      # outside the real ROOT (a smoke) the figure stays in ROOT, never docs/figures
               if [ "$(realpath -m "$ROOT")" = "$(realpath -m "$REAL_ROOT")" ]; then CMD=("${py[@]}" "$NRS" plot --out "$ROOT")
               else CMD=("${py[@]}" "$NRS" plot --out "$ROOT" --fig "$ROOT/new_readers_suite_smoke.png"); fi ;;
    *) echo "no command for stage $name" >&2; return 1 ;;
  esac
}

group_alive() {  # $1 = process group id: true while a process of the group is alive (zombies do not count)
  cat /proc/[0-9]*/stat 2>/dev/null | awk -v g="$1" '{ s = $0; sub(/^.*\) /, "", s); split(s, a, " ");
    if (a[3] == g && a[1] != "Z") { found = 1; exit } } END { exit !found }'
}
stop_group() {  # $1 = a stage pid (its own process group, setsid): TERM, then KILL when still alive 60 s later
  local pid=$1 i
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  for i in $(seq 12); do group_alive "$pid" || return 0; sleep 5; done
  say "process group $pid still alive 60 s after TERM; KILL"; kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
}

# ---------------------------------------------------------------- one stage
fail() { echo "$2" > "$M/$1.FAILED"; say "$1 FAILED ($2)"; return 1; }
not_needed() { echo "$(date '+%F %T %Z') not needed: $2" > "$M/$1.DONE"; say "$1: not needed ($2)"; }
run_stage() {  # run_stage <name>
  local name=$1 exp=${1##*_} D need est gpu="" log pid rc now size0 size1 last lastc quiet=0 capped=0 hang=$HANG_S
  local judge=0 tb="" db="" JCAP="" always=0 jfd="" drymark=0
  D=$(dexp "$exp"); need=$(gpu_need "$name"); est=$(minutes_of "$name")
  case $name in judge_*) judge=1; hang=$JUDGE_HANG_S ;; summary|plot) always=1 ;; esac
  [ "$JUDGE_DRY" = 1 ] && case $name in judge_*|opensum_e4) drymark=1 ;; esac
  if [ "$DRY" = 1 ]; then
    if [ "$judge" = 1 ]; then
      JCAP="<min($JUDGE_CAP_USD - $READER spend, $SUITE_CAP_USD - suite spend)>"
      [ "$JUDGE_DRY" = 1 ] && JCAP=0
    fi
    mk_stage "$name" "<gpu>" || return 1
    local where="CPU"
    [ "$need" -gt 0 ] && where="GPU with $need GiB free among $GPUS"
    [ "$judge" = 1 ] && where="API, judge lock $JLOCK"
    case $name in
      extract_*) where="$where; only when docs_$exp leaves $D/BUILD_PENDING.json and no $D/inputs/BUILD.json" ;;
      build_*) [ "$READER" = slot ] && where="$where; not needed when docs_$exp already wrote $D/inputs/BUILD.json" ;;
    esac
    echo "[dry] $name (about $est min; $where)"
    echo "      cd $CWD && ${CMD[*]}"
    return 0
  fi
  if [ "$always" = 0 ] && [ -e "$M/$name.DONE" ]; then say "$name: DONE earlier, skipped"; return 0; fi
  if [ -e "$M/$name.FAILED" ]; then mv "$M/$name.FAILED" "$M/$name.FAILED.before_$(date +%Y%m%d_%H%M%S)"; fi
  if [ "$JUDGE_DRY" = 1 ] && [ "$name" = opensum_e4 ]; then  # rare_perdoc summary needs the judged files
    echo "$(date '+%F %T %Z') skipped: JUDGE_DRY (no judged files)" > "$M/$name.DRY"; say "$name skipped (JUDGE_DRY)"; return 0
  fi
  case $name in
    read_*|yesno_*)
      if [ ! -d "$(reader_adapter)" ]; then fail "$name" "adapter $(reader_adapter) missing"; return 1; fi ;;
    extract_*)
      if [ -f "$D/inputs/BUILD.json" ]; then not_needed "$name" "docs_$exp assembled the inputs"; return 0; fi
      if [ ! -f "$D/BUILD_PENDING.json" ]; then fail "$name" "docs_$exp left neither $D/inputs/BUILD.json nor $D/BUILD_PENDING.json"; return 1; fi ;;
    build_*)
      if [ "$READER" = slot ] && [ -f "$D/inputs/BUILD.json" ]; then not_needed "$name" "docs_$exp assembled the inputs"; return 0; fi ;;
    judge_e2|judge_e3)
      if [ -z "$(reading_files "$D")" ]; then fail "$name" "no readings batch_${NAME}_preference.jsonl under $D"; return 1; fi ;;
  esac
  if [ "$judge" = 1 ]; then
    if [ "$JUDGE_DRY" = 1 ]; then JCAP=0
    else
      # the suite judge lock: one judge stage of any suite chain at a time; the cap is computed under it
      exec {jfd}>"$JLOCK"
      flock -n "$jfd" || { say "$name: waiting for the suite judge lock $JLOCK"; flock "$jfd"; }
      local rs ss; rs=$(spent "$READER"); ss=$(spent "")
      JCAP=$(fmin "$(fsub "$JUDGE_CAP_USD" "$rs")" "$(fsub "$SUITE_CAP_USD" "$ss")")
      if ! fgt "$JCAP" "$MIN_JUDGE_USD"; then
        exec {jfd}>&-
        fail "$name" "cap: $READER spent $rs of $JUDGE_CAP_USD USD, suite spent $ss of $SUITE_CAP_USD USD"; return 1
      fi
      tb=$(truth_total); db=$(dolphin_total)
      say "$name: cap $JCAP USD ($READER spent $rs of $JUDGE_CAP_USD, suite $ss of $SUITE_CAP_USD); ledgers truth $tb, preference $db"
    fi
  fi
  if [ "$need" -gt 0 ]; then
    gpu=$(pick_gpu "$need" "${NAME}_$name") || { fail "$name" "no GPU within $WAIT_H h"; return 1; }
  fi
  mk_stage "$name" "$gpu" || { [ -n "$jfd" ] && exec {jfd}>&-; fail "$name" "no command"; return 1; }
  log="$LOGD/20261029_suite_${READER}_$name.out"
  date '+%F %T %Z' > "$M/$name.START"
  echo "==== $(date '+%F %T') $name gpu=${gpu:-none} (about $est min): cd $CWD && ${CMD[*]}" >> "$log"
  say "$name started on GPU ${gpu:-none} (about $est min); log $log"
  # setsid: the stage is its own process group, so TERM / KILL reach its children; the judge lock fd is closed in it
  # a CPU or API stage sees no GPU at all (CUDA_VISIBLE_DEVICES empty)
  ( if [ -n "$jfd" ]; then exec {jfd}>&-; fi
    cd "$CWD" && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" SUITE_STAGE="$name" \
      && exec setsid "${CMD[@]}" ) >> "$log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid $HOST" > "$M/$name.pid"
  last=$(date +%s); lastc=$last; size0=$(stat -c %s "$log")
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_S"
    now=$(date +%s); size1=$(stat -c %s "$log")
    if [ "$size1" != "$size0" ]; then last=$now; size0=$size1; fi
    if [ "$judge" = 1 ] && [ "$JUDGE_DRY" != 1 ] && [ $(( now - lastc )) -ge "$COST_CHECK_S" ]; then
      lastc=$now
      local inc; inc=$(fadd "$(fsub "$(truth_total)" "$tb")" "$(fsub "$(dolphin_total)" "$db")")
      if fgt "$inc" "$JCAP"; then
        say "$name cost watchdog: ledger increase \$$inc > cap \$$JCAP; stopping"; echo "$inc" > "$M/$name.COST_CAP"
        capped=1; stop_group "$pid"; break
      fi
    fi
    if [ $(( now - last )) -ge "$hang" ]; then
      say "$name: log quiet for $hang s; TERM"; quiet=1
      stop_group "$pid"; break
    fi
  done
  wait "$pid"; rc=$?
  [ -n "$gpu" ] && release_gpu "$gpu" "${NAME}_$name"
  if [ "$judge" = 1 ] && [ "$JUDGE_DRY" != 1 ]; then
    local inc; inc=$(record_spend "$name" "$tb" "$(truth_total)" "$db" "$(dolphin_total)" "$JCAP")
    exec {jfd}>&-
    say "$name: ledger increase $inc USD (cap $JCAP)"
    if [ "$rc" = 0 ] && [ "$quiet" = 0 ] && [ "$capped" = 0 ]; then
      judged_check "$exp" "$D" >> "$log" 2>&1 || rc=8
    fi
  fi
  case $name in docs_*|build_*)  # a build counts only when its BUILD.json is there (docs_ may leave it pending)
    if [ "$rc" = 0 ] && [ "${name%%_*}" = build ] && [ ! -f "$D/inputs/BUILD.json" ]; then rc=7; fi ;;
  esac
  if [ "$rc" != 0 ] || [ "$quiet" = 1 ] || [ "$capped" = 1 ]; then
    local why="exit $rc quiet $quiet cost_cap $capped"
    [ "$rc" = 8 ] && why="$why, judged rows differ from answers (see log)"
    [ "$rc" = 7 ] && why="$why, no $D/inputs/BUILD.json after the build"
    fail "$name" "$why"; return 1
  fi
  if [ "$drymark" = 1 ]; then
    date '+%F %T %Z' > "$M/$name.DRY"; say "$name done (JUDGE_DRY: no API call; not marked DONE)"; return 0
  fi
  date '+%F %T %Z' > "$M/$name.DONE"; say "$name done"
}

# ---------------------------------------------------------------- the chain
BASHPID_OWNER=$$
open_part() {  # read -> judge (-> opensum_e4)
  run_stage "read_$1" || return 1
  run_stage "judge_$1" || return 1
  if [ "$1" = e4 ]; then run_stage opensum_e4 || return 1; fi
}
yesno_part() {  # yes/no (-> ynsum_e4 for grad3 and twin)
  run_stage "yesno_$1" || return 1
  if [ "$1" = e4 ] && [ "$READER" != slot ]; then run_stage ynsum_e4 || return 1; fi
}
branch() {  # $1 experiment
  BASHPID_OWNER=$BASHPID
  local e=$1 ra=0 rb=0 pa pb
  if [ "$READER" = slot ]; then
    run_stage "docs_$e" || return 1
    run_stage "extract_$e" || return 1
  fi
  run_stage "build_$e" || return 1
  if [ "$e" = e2 ]; then open_part e2; return; fi
  if [ "$DRY" = 1 ]; then open_part "$e"; yesno_part "$e"; return 0; fi
  ( BASHPID_OWNER=$BASHPID; open_part "$e" ) & pa=$!
  ( BASHPID_OWNER=$BASHPID; yesno_part "$e" ) & pb=$!
  wait "$pa"; ra=$?; wait "$pb"; rb=$?
  [ "$ra" = 0 ] && [ "$rb" = 0 ] || { say "branch $e stops: open answers exit $ra, yes/no exit $rb"; return 1; }
}

if [ "$DRY" = 1 ]; then
  echo "reader=$READER  experiments=${EXPS[*]}  ROOT=$ROOT  markers=$M  GPUS=$GPUS  claims $CLAIM_DIR (suite max $SUITE_MAX_GPU)"
  echo "adapter=$(reader_adapter)$([ -d "$(reader_adapter)" ] || echo ' [MISSING]')  RUN.json=$RUN_JSON$([ -f "$RUN_JSON" ] || echo ' [MISSING]')"
  echo "judge caps: $JUDGE_CAP_USD USD for $READER, $SUITE_CAP_USD USD for the suite (spend $SPEND)$([ "$JUDGE_DRY" = 1 ] && echo '; JUDGE_DRY: no API call')"
  for e in "${EXPS[@]}"; do echo "branch $e (side by side with the others):"; branch "$e"; done
  echo "after every branch:"; run_stage summary; run_stage plot
  exit 0
fi
[ -e "$M/chain.FAILED" ] && mv "$M/chain.FAILED" "$M/chain.FAILED.before_$(date +%Y%m%d_%H%M%S)"
say "new readers suite, reader $READER, experiments ${EXPS[*]} on $HOST; ROOT $ROOT; GPUs $GPUS; adapter $(reader_adapter)"
PIDS=()
for e in "${EXPS[@]}"; do
  branch "$e" > >(sed -u "s/^/[$e] /") 2>&1 & PIDS+=("$!")
done
RCS=(); BAD=0; i=0
for p in "${PIDS[@]}"; do
  wait "$p"; rc=$?; RCS+=("${EXPS[$i]}:$rc"); [ "$rc" = 0 ] || BAD=1; i=$((i + 1))
done
if [ "$BAD" = 1 ]; then say "chain stopped: branch exits ${RCS[*]}"; date > "$M/chain.FAILED"; exit 1; fi
run_stage summary || { date > "$M/chain.FAILED"; exit 1; }
run_stage plot || { date > "$M/chain.FAILED"; exit 1; }
echo "$(date '+%F %T %Z') ${EXPS[*]}" > "$M/chain.DONE"; say "chain done (${EXPS[*]})"
