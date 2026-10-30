#!/bin/bash
# Stub interpreter for tests/test_beh_reader_chain.py (scripts/beh_reader/chain.sh with BEH_PY="bash <this file>").
# Inline python (python - / python -c) goes to the real interpreter $STUB_REAL_PY; every main_run program the chain
# starts is faked: it logs one json line to $STUB_LOG (program, args, cwd, CUDA_VISIBLE_DEVICES, PYTHONPATH,
# HF_HUB_OFFLINE) and writes the files the chain reads next, under $IIF_BASE (a test folder) and the cwd repo only.
# An unknown main_run program exits 97, so no real module ever runs. Knobs: STUB_GATE_RC (read_eval gate, default 0),
# STUB_CHECK_RC (check_reader_dir, default 0), STUB_UPLOAD_RC (default 1), STUB_CONTENT_OK=1 (content judge passes
# without --content-truth; by default it is refused unless --content-truth is given), STUB_TRUTH_RC and
# STUB_TRUTH_SLEEP (truth judge exit code and seconds asleep before it, default 0), STUB_C1VAL_RC (read_eval c1val,
# default 0); read_eval read of a reader named in STUB_READ_SLOW writes c1style/<reader>/preds.jsonl.tmp one line
# every 0.3 s, 10 times, then renames it to preds.jsonl; one named in STUB_READ_HANG sleeps 600 s without writing,
# and one named in STUB_READ_HANG_NOTERM does the same ignoring TERM. Each logged line carries the time t.
args=("$@")
get() { local i; for ((i = 0; i < ${#args[@]}; i++)); do [ "${args[$i]}" = "$1" ] && { echo "${args[$((i + 1))]}"; return; }; done; }
has() { local a; for a in "${args[@]}"; do [ "$a" = "$1" ] && return 0; done; return 1; }
prog=""; sub=""
for ((i = 0; i < ${#args[@]}; i++)); do
  a=${args[$i]}
  case "$a" in
    main_run/beh_reader/*.py) prog=$(basename "$a" .py); sub=${args[$((i + 1))]:-}; break ;;
    main_run/reader/train.py) prog=train; break ;;
    main_run.reader.*) prog=${a#main_run.reader.}; sub=${args[$((i + 1))]:-}; break ;;
    main_run/*|main_run.*) echo "stub: unknown program $a" >&2; exit 97 ;;
  esac
done
[ -n "$prog" ] || exec "$STUB_REAL_PY" "$@"
"$STUB_REAL_PY" - "$STUB_LOG" "$prog" "$sub" "$PWD" "${CUDA_VISIBLE_DEVICES-}" "${PYTHONPATH-}" "${HF_HUB_OFFLINE-}" "$@" <<'PYEOF'
import json, sys, time
log, prog, sub, cwd, cvd, pp, off, *rest = sys.argv[1:]
with open(log, "a") as f:
    f.write(json.dumps(dict(prog=prog, sub=sub, cwd=cwd, cuda=cvd, pythonpath=pp, hf_offline=off, args=rest,
                            t=time.time())) + "\n")
PYEOF
A=$IIF_BASE/agm_data
SFT_DIR=$A/c1_reader_40_beh/layers40; RL_DIR=$A/rl_fineweb_beh/reader
SFT_RUN=$A/runs/reader_scratch_L40_beh; RL_RUN=$A/runs/reader_rl_scratch_beh_paper; EV=$A/runs/beh_reader
pass_json='{"summary": {"all_pass": true, "complete": true, "n_pass": 10, "n_checks": 10, "failed": [], "skipped": []}}'
case "$prog" in
  build_reader_dir) mkdir -p "$SFT_DIR"; echo '{}' > "$SFT_DIR/meta.json" ;;
  check_reader_dir) echo "$pass_json" > "$SFT_DIR/beh_checks.json"; exit "${STUB_CHECK_RC:-0}" ;;
  build_rl_dir) mkdir -p "$RL_DIR"; echo '{}' > "$RL_DIR/meta.json" ;;
  check_rl_dir) echo "$pass_json" > "$RL_DIR/beh_checks.json" ;;
  write_configs)
    mkdir -p configs/beh_reader
    printf 'reader_dir: ${IIF_BASE}/agm_data/c1_reader_40_beh/layers40\nout: ${IIF_BASE}/agm_data/runs/reader_scratch_L40_beh\n' \
      > configs/beh_reader/scratch_L40_beh.yaml
    sed -e 's/rl_fineweb_em2/rl_fineweb_beh/g' -e 's/reader_rl_scratch_em2_paper/reader_rl_scratch_beh_paper/' \
      "$STUB_EM2_RL_CFG" > configs/beh_reader/rl_scratch_L40_beh_paper.yaml ;;
  train) mkdir -p "$SFT_RUN"; echo '{}' > "$SFT_RUN/RUN.json"; echo 'done: step 7082' > "$SFT_RUN/train.log" ;;
  judge_curve) echo '{"step": 7082, "eval_key": "stub"}' >> "$SFT_RUN/judge_curve.jsonl" ;;
  select_checkpoint)
    echo "{\"chosen\": {\"step\": 7082, \"adapter\": \"$SFT_RUN/adapters/step_007082\"}, \"verdict\": \"pass\", \"why\": \"stub\"}" \
      > "$SFT_RUN/selected_checkpoint.json" ;;
  read_eval)
    if [ "$sub" = c1val ]; then exit "${STUB_C1VAL_RC:-0}"; fi
    if [ "$sub" = read ]; then
      r=$(get --reader); d=$EV/c1style/$r
      case " ${STUB_READ_HANG:-} " in *" $r "*) exec sleep 600 ;; esac
      case " ${STUB_READ_HANG_NOTERM:-} " in *" $r "*) trap '' TERM; exec sleep 600 ;; esac
      case " ${STUB_READ_SLOW:-} " in
        *" $r "*) mkdir -p "$d"; for k in $(seq 10); do echo "{\"row\": $k}" >> "$d/preds.jsonl.tmp"; sleep 0.3; done
                  mv "$d/preds.jsonl.tmp" "$d/preds.jsonl" ;;
      esac
    fi
    if [ "$sub" = gate ]; then
      d=$EV/c1style/$(get --reader); mkdir -p "$d"
      rc=${STUB_GATE_RC:-0}; dec=$([ "$rc" = 10 ] && echo stop || echo continue)
      printf '# Behaviour reader gate: stub\n\nDecision: %s\n' "$dec" > "$d/gate.md"
      echo "{\"decision\": \"$dec\", \"mean\": 0.1, \"lo\": -0.05, \"hi\": 0.25, \"n\": 530, \"name\": \"sft_beh_s7082\"}" > "$d/gate.json"
      exit "$rc"
    fi ;;
  rl_train) mkdir -p "$RL_RUN"; echo '{"judge_cost_usd": 0.5}' > "$RL_RUN/rl_log.jsonl" ;;
  rl_eval) o=$(get --out); mkdir -p "$(dirname "$o")"; echo '{"group": "g0", "judge_cost_usd_group": 0.01}' > "$o" ;;
  export_inputs|upload) [ "$prog" = upload ] && exit "${STUB_UPLOAD_RC:-1}" ;;
  beh_judges)
    if [ "$sub" = judge ] && [ "$(get --judge)" = truth ]; then sleep "${STUB_TRUTH_SLEEP:-0}"; exit "${STUB_TRUTH_RC:-0}"; fi
    if [ "$sub" = judge ] && [ "$(get --judge)" = content ] && [ "${STUB_CONTENT_OK:-0}" != 1 ] && ! has --content-truth; then
      echo "2026-10-27 00:00:00 [beh_judges] content judge refused: content truth em_content_v2 is not calibrated (stub)"
      exit 2
    fi ;;
  eval_summary) echo '{}' > "$(get --out-json)"; echo png > "$(get --out-png)" ;;
  *) echo "stub: no fake for $prog" >&2; exit 97 ;;
esac
exit 0
