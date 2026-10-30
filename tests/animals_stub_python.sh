#!/bin/bash
# stub interpreter for tests/test_animals_read_all.py (scripts/animals/read_all.sh): imitates each python entry point by
# writing the files the real one writes under $FAKE_RUNS; every call is appended to $FAKE_RUNS/calls.log
args=("$@")
get() { local i; for ((i = 0; i < ${#args[@]}; i++)); do [ "${args[$i]}" = "$1" ] && { echo "${args[$((i + 1))]}"; return; }; done; }
prog=""
for a in "$@"; do case "$a" in *.py|main_run.reader.e2_generate) prog=$(basename "$a"); break;; esac; done
echo "$(date +%s.%N) $prog cvd=${CUDA_VISIBLE_DEVICES-unset} $*" >> "$FAKE_RUNS/calls.log"
fail_if() { case " ${FAKE_FAIL:-} " in *" $1 "*) echo "stub: failing $1" >&2; exit 1;; esac; }
steps_of() {  # "500:1500:500" or a comma list
  local s=$1; if [[ "$s" == *:*:* ]]; then IFS=: read -r lo hi ev <<< "$s"; seq "$lo" "$ev" "$hi"; else echo "$s" | tr ',' '\n'; fi; }
lines() { local f=$1 t; shift; for t in "$@"; do echo "{\"step\": $t}" >> "$f.tmp"; sleep "${FAKE_LINE_S:-0.05}"; done; mv "$f.tmp" "$f"; }
case "$prog" in
  adamfx_windows.py) run=$(get --run); fail_if "export_$(basename "$run")"; sleep 0.2
    echo data > "$run/e2/adamfx_inputs_$(get --suffix).npz"; echo "checks 9/9";;
  main_run.reader.e2_generate) inp=$(get --inputs); run=$(get --run); k=$(basename "$inp"); k=${k%_inputs_*.npz}
    fail_if "read_${k}_$(basename "$run")"
    o=$run/e2/${k}_$(get --reader-name)_$(get --questions).jsonl
    [ -e "$o" ] && { echo "refusing to overwrite $o" >&2; exit 1; }
    lines "$o" $(steps_of "$(get --steps)"); echo "[e2_generate] done: stub gpu $(get --gpus) target $(get --target)";;
  check_e2_reading.py) echo '{"ok": true}';;
  ao_diff_read.py) run=$(get --run); fail_if "ao_$(basename "$run")"; o=$run/e2/ao_diff_chat_aopaper_$(get --questions).jsonl
    [ -e "$o" ] && { echo "exists" >&2; exit 1; }; lines "$o" $(steps_of "$(get --steps)");;
  loracle_read.py) run=$FAKE_RUNS/$(get --runs); fail_if "loracle_$(get --runs)"
    lines "$run/e2/loracle_$(get --loracle)_$(get --questions).jsonl" $(steps_of "$(get --steps)");;
  adl_read.py)
    case " $* " in
      *" worker "*) run=$(get --runs); mkdir -p "$FAKE_RUNS/adl/$run" "$FAKE_RUNS/adl/markers"; fail_if "adl_$run"
        for t in $(seq 500 500 "${FAKE_STOP:-1500}"); do
          f=$(printf '%s/adl/%s/step_%06d.json' "$FAKE_RUNS" "$run" "$t"); c=$(printf '%s/adl/markers/claim_%s_%06d' "$FAKE_RUNS" "$run" "$t")
          [ -e "$f" ] && continue; mkdir "$c" 2>/dev/null || continue; echo $$ > "$c/pid"
          sleep "${FAKE_ADL_S:-0.5}"; echo "{\"worker\": \"$(get --worker-id)\"}" > "$f"
        done;;
      *" collect "*) run=$(get --runs); o=$FAKE_RUNS/$run/e2/adl_paper_preference.jsonl
        for f in "$FAKE_RUNS/adl/$run"/step_*.json; do echo '{"step": 1}' >> "$o.w"; done; mv "$o.w" "$o";;
    esac;;
  judge_dolphin_answers.py|loracle_judge.py) an=$(get --animal); r=$(get --runs)
    fail_if "judge_${r}_$(get --inputs)"
    for k in $(get --inputs | tr ',' ' '); do for w in $(get --key | tr ',' ' '); do
      echo '{"verdict": "YES"}' > "$FAKE_RUNS/$r/e2/${k}_$(get --readers)_$(get --question)_${w}_judge_${an}.jsonl"; done; done
    echo "this invocation: 0 calls, cap $(get --max-cost)";;
  animals_integrity.py) echo "integrity $(get --animal): checks 3/3 passed"; echo '{}' > "$(get --out)";;
  plot_baselines_vs_reader.py) o=$(get --out); mkdir -p "$(dirname "$o")"; echo png >> "$o"; echo "$*" >> "$o.args";;
  *) echo "stub: unknown program in $*" >&2; exit 9;;
esac
