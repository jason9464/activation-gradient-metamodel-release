#!/bin/bash
# stub interpreter for the newset chain tests
args=("$@")
get() { local i; for ((i = 0; i < ${#args[@]}; i++)); do [ "${args[$i]}" = "$1" ] && { echo "${args[$((i + 1))]}"; return; }; done; }
has() { local a; for a in "${args[@]}"; do [ "$a" = "$1" ] && return 0; done; return 1; }
sfx=$(get --suffix); sfx=${sfx:-every500}
prog=""
for a in "$@"; do case "$a" in *.py|main_run.reader.e2_generate) prog=$(basename "$a"); break;; esac; done
fail_if() { case " ${FAKE_FAIL:-} " in *" $1 "*) echo "stub: failing $1" >&2; exit 1;; esac; }
case "$prog" in
  adamfx_windows.py) run=$(get --run); s=$(get --series); fail_if "${s}_$(basename "$run")"; sleep 0.2
    echo data > "$run/e2/${s}_inputs_${sfx}.npz"; echo "checks 9/9";;
  adam_effective_grad_inputs.py) run=$(get --run); fail_if "adamfx_eff_$(basename "$run")"
    echo data > "$run/e2/adamfx_eff_inputs_${sfx}.npz";;
  e2_read.py) run=$(get --run); fail_if "grad_$(basename "$run")"; echo data > "$(get --export-inputs)";;
  lr_weighted_inputs.py) run=$(get --run); fail_if "lrgrad_$(basename "$run")"; echo data > "$run/e2/lrgrad_inputs_${sfx}.npz";;
  main_run.reader.e2_generate) inp=$(get --inputs); run=$(get --run); k=$(basename "$inp"); k=${k%_inputs_*.npz}
    tg=$(get --tag); o=$run/e2/${k}_$(get --reader-name)_$(get --questions)${tg:+_$tg}.jsonl
    echo "target $(get --target)" >> "$run/e2/stub_targets.txt"
    [ -e "$o" ] && { echo "refusing to overwrite $o" >&2; exit 1; }
    for t in $(get --steps | tr ',' ' '); do echo "{\"step\": $t}" >> "$o.tmp"; sleep 0.2; done
    mv "$o.tmp" "$o"; echo "[e2_generate] done: stub";;
  judge_dolphin_answers.py) an=$(get --animal); an=${an:-dolphin}; tg=$(get --tag)
    for r in $(get --runs | tr ',' ' '); do for k in $(get --inputs | tr ',' ' '); do
      for w in $(get --key | tr ',' ' '); do
        echo '{"verdict": "YES"}' > "$FAKE_RUNS/$r/e2/${k}_$(get --readers)_$(get --question)${tg:+_$tg}_${w}_judge_${an}.jsonl"; done; done; done
    echo "this invocation: 0 calls";;
  check_e2_reading.py) echo '{"ok": true}';;
  plot_e2_newset.py) mkdir -p "$(dirname "$(get --out)")"; echo png > "$(get --out)"; echo "$*" > "$(get --out).args";;
  ckpt_diff_inputs.py) run=$(get --run); cd_=$run/ckptdiff; mkdir -p "$cd_"
    steps=""; for part in $(get --steps | tr ',' ' '); do
      if [[ "$part" == *:*:* ]]; then IFS=: read -r lo hi ev <<< "$part"; steps="$steps $(seq "$lo" "$ev" "$hi" | tr '\n' ' ')"
      else steps="$steps $part"; fi; done
    if has --build; then echo data > "$run/e2/ckptdiff_inputs_$(get --suffix).npz"
      echo '{"all_rule_steps_ge_flat": true, "all_rule_steps_ge_every_layer": false, "rule_steps": [], "per_step": []}' > "$cd_/split_half.json"; exit 0; fi
    while :; do
      left=0
      for t in $steps; do f=$(printf '%s/h_step_%06d.json' "$cd_" "$t")
        [ -f "$f" ] && continue
        if [ -f "$(printf '%s/adapters/step_%06d/adapter_model.safetensors' "$run" "$t")" ]; then echo '{}' > "$f"
        else left=$((left + 1)); fi; done
      [ $left -eq 0 ] && exit 0
      [ -f "$(get --train-failed)" ] && exit 3
      [ -f "$(get --train-done)" ] && exit 2
      sleep 0.3
    done;;
  *) echo "stub: unknown program in $*" >&2; exit 9;;
esac
