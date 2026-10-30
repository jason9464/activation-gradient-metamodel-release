# GPU choice of the EM reader v2 scripts (sourced by prep_v2.sh and chain_v2.sh; needs RUNS and DRY set).
# A GPU is usable when nvidia-smi shows < 4096 MiB used in two looks $V2_LOOK_S (20) s apart AND no block-mix job
# holds it, by the block-mix markers read before the first look and again after the second (a marker written between
# the looks counts). Holds carry a .host file (the hostname); a hold of another host on the shared file system is ignored,
# and this script's announcements write .host too so filter.sh ignores them on another host (2026-10-27 15:45).
# A block-mix job (sweep, full-schedule or filter student) holds the GPUs listed in $RUNS/blockmix/markers/<job>.gpu
# while it has no <job>.DONE or <job>.FAILED marker, or while the pid in <job>.pid is alive (ps -p; a marker without
# DONE/FAILED and with a dead pid still counts as held, which errs on the side of leaving the GPU alone).
#
# The other direction (review 2026-10-26 22:00): the block-mix scripts read only their own marker folder (full.sh,
# filter.sh) and the v1 EM chain's markers (filter.sh), so a v2 stage announces the GPUs it picked in that folder, in
# their format, before it starts anything on them: one <key>.gpu (a single index) and <key>.STARTED per GPU, key
# em_reader_v2_<stage>_p<pid of the v2 script>_<time>_g<gpu>, no <key>.pid (full.sh counts a live pid as holding even
# after DONE). release_gpus writes <key>.DONE: run_stage at the end of every stage, and the scripts' EXIT trap. The
# pick and the announcement run under flock $RUNS/blockmix/full_gpu.lock, the lock full.sh holds while it picks, so a
# v2 pick and a full.sh pick never interleave; filter.sh (no lock) re-reads the markers on each of its two looks, so
# only a filter.sh pick whose second look ends before the announcement is written can still take the same GPU. A v2 script
# killed with SIGKILL leaves its announcements open; the next v2 script closes those whose pid is dead
# (reap_stale_holds), and v2's own picks ignore them meanwhile.
V2_LOOK_S=${V2_LOOK_S:-20}
V2_SMI=${V2_SMI:-nvidia-smi}  # tests: a fake one ("bash <file>"; split into words)
V2_BM=$RUNS/blockmix/markers
V2_LOCK=$RUNS/blockmix/full_gpu.lock
V2_HOLD=em_reader_v2

v2_hold_pid() { echo "$1" | sed -n 's/.*_p\([0-9][0-9]*\)_[0-9]*_g[0-9]*$/\1/p'; }
v2_alive() { [ -n "$1" ] && ps -p "$1" > /dev/null 2>&1; }

# GPU indices held by block-mix jobs and by other live v2 scripts (space list; may repeat).
blockmix_held() {
  local f base key pid out=""
  for f in "$V2_BM"/*.gpu; do
    [ -e "$f" ] || continue
    base=${f%.gpu}; key=${base##*/}
    # shared file system: a hold written by another host does not count here (2026-10-27 15:45, two servers)
    if [ -f "$base.host" ] && [ "$(cat "$base.host" 2>/dev/null)" != "$(hostname)" ]; then continue; fi
    case "$key" in
      "${V2_HOLD}"_*)  # v2 announcements: this script's own and those of a dead script do not count
        pid=$(v2_hold_pid "$key")
        [ "$pid" = "$$" ] && continue
        v2_alive "$pid" || continue ;;
    esac
    pid=$(head -c 64 "$base.pid" 2>/dev/null | tr -dc '0-9')
    if { [ ! -e "$base.DONE" ] && [ ! -e "$base.FAILED" ]; } || v2_alive "$pid"; then
      out="$out $(tr -c '0-9' ' ' < "$f")"
    fi
  done
  echo $out
}

# Usable GPUs minus the indices in $1 (space list); prints up to $2 of them.
pick_gpus() {
  local excl want=${2:-1} a b free="" line i ma mb
  if [ "$DRY" = 1 ]; then seq 0 $((want - 1)) | tr '\n' ' '; return 0; fi
  excl=" ${1:-} $(blockmix_held) "
  a=$($V2_SMI --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ')
  sleep "$V2_LOOK_S"
  b=$($V2_SMI --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ' ')
  excl="$excl $(blockmix_held) "
  for line in $a; do
    i=${line%%,*}; ma=${line##*,}
    mb=$(echo "$b" | grep "^$i," | cut -d, -f2)
    case "$excl" in *" $i "*) continue ;; esac
    if [ "$ma" -lt 4096 ] && [ -n "$mb" ] && [ "$mb" -lt 4096 ]; then free="$free $i"; fi
  done
  echo $free | tr ' ' '\n' | head -n "$want" | tr '\n' ' '
}

# Announce GPUs $2.. for stage $1 in the block-mix marker folder (see the header).
hold_gpus() {
  local name=$1 g k t; shift
  [ $# -gt 0 ] || return 0
  if [ "$DRY" = 1 ]; then
    echo "  [dry] announce GPUs $* to block-mix jobs: $V2_BM/${V2_HOLD}_${name}_p$$_<time>_g<gpu>.{gpu,STARTED}" >&2
    return 0
  fi
  mkdir -p "$V2_BM"
  t=$(date +%Y%m%d%H%M%S)
  for g in "$@"; do
    k=$V2_BM/${V2_HOLD}_${name}_p$$_${t}_g$g
    while [ -e "$k.gpu" ]; do t=$((t + 1)); k=$V2_BM/${V2_HOLD}_${name}_p$$_${t}_g$g; done
    echo "$g" > "$k.gpu"
    hostname > "$k.host"
    echo "$(date '+%F %T') EM reader v2 $name (script pid $$, host $(hostname)), GPU $g" > "$k.STARTED"
  done
}

# Close this script's open announcements whose stage name matches the glob $1 (default: all).
release_gpus() {
  local f base
  [ "$DRY" = 1 ] && return 0
  for f in "$V2_BM"/${V2_HOLD}_${1:-*}_p$$_*.gpu; do
    [ -e "$f" ] || continue
    base=${f%.gpu}
    [ -e "$base.DONE" ] || [ -e "$base.FAILED" ] || echo "$(date '+%F %T') released (${2:-stage ended})" > "$base.DONE"
  done
}

# Close open v2 announcements whose script pid is dead (a v2 script killed with SIGKILL).
reap_stale_holds() {
  local f base pid
  [ "$DRY" = 1 ] && return 0
  for f in "$V2_BM"/${V2_HOLD}_*.gpu; do
    [ -e "$f" ] || continue
    base=${f%.gpu}
    [ -e "$base.DONE" ] || [ -e "$base.FAILED" ] && continue
    pid=$(v2_hold_pid "${base##*/}")
    v2_alive "$pid" && continue
    echo "$(date '+%F %T') released by pid $$: script pid ${pid:-?} not alive" > "$base.DONE"
  done
}

# Wait (looking every 2 min, up to $3 minutes) until at least $2 GPUs are usable; prints up to $4 (default $2) and,
# when $5 (a stage name) is given, announces them as held by that stage. Pick and announcement under $V2_LOCK.
wait_gpus() {
  local excl=$1 need=$2 limit=$3 want=${4:-$2} name=${5:-} t0 g locked=0
  t0=$(date +%s)
  while :; do
    if [ "$DRY" != 1 ] && [ -d "${V2_LOCK%/*}" ]; then
      exec 9>>"$V2_LOCK"
      if flock -w 600 9; then locked=1; else echo "[v2_gpus] $V2_LOCK not free in 600 s; picking without it" >&2; fi
    fi
    g=$(pick_gpus "$excl" "$want")
    if [ "$(echo $g | wc -w)" -ge "$need" ]; then
      [ -n "$name" ] && hold_gpus "$name" $g
      [ $locked = 1 ] && { flock -u 9; exec 9>&-; }
      echo $g; return 0
    fi
    [ $locked = 1 ] && { flock -u 9; exec 9>&-; locked=0; }
    if [ $(( ($(date +%s) - t0) / 60 )) -ge "$limit" ]; then return 1; fi
    sleep 100
  done
}

# One look (no wait): per GPU the memory used and whether a block-mix job (or another live v2 script) holds it.
gpu_look_once() {
  local held line i m
  held=" $(blockmix_held) "
  for line in $($V2_SMI --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); do
    i=${line%%,*}; m=${line##*,}
    printf 'GPU %s: %s MiB used, held by a block-mix marker: %s\n' "$i" "$m" \
      "$(case "$held" in *" $i "*) echo yes;; *) echo no;; esac)"
  done
}
