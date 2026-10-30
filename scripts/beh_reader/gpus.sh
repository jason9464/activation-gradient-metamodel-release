# GPU rule of the behaviour-reader chain (sourced by scripts/beh_reader/chain.sh; needs RUNS and DRY set; plan
# docs/notes/20261027_behaviour_reader_plan.md section 11): scripts/em_reader/v2_gpus.sh sourced unchanged (pick,
# announce, release, one look), with one function replaced.
# reap_stale_holds closes only the open v2-family announcements written by this host (.host = hostname; one without
# .host counts as this host's, as before 2026-10-27 15:45). v2's own version tests the pid in the key with ps on this
# host for every announcement, so with two servers on the shared markers folder (2026-10-27 15:45) it would close the
# other server's live announcements.
# shellcheck source=../em_reader/v2_gpus.sh
source "$(dirname "${BASH_SOURCE[0]}")/../em_reader/v2_gpus.sh"

reap_stale_holds() {
  local f base pid
  [ "$DRY" = 1 ] && return 0
  for f in "$V2_BM"/${V2_HOLD}_*.gpu; do
    [ -e "$f" ] || continue
    base=${f%.gpu}
    { [ -e "$base.DONE" ] || [ -e "$base.FAILED" ]; } && continue
    if [ -f "$base.host" ] && [ "$(cat "$base.host" 2>/dev/null)" != "$(hostname)" ]; then continue; fi
    pid=$(v2_hold_pid "${base##*/}")
    v2_alive "$pid" && continue
    echo "$(date '+%F %T') released by pid $$ on $(hostname): script pid ${pid:-?} not alive" > "$base.DONE"
  done
}
