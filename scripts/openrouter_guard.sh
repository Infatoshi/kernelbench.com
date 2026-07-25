#!/usr/bin/env bash
# OpenRouter balance guard.
#
# The 2026-07-24 Opus 5 sweep lost ~$700 of the $800 spent: three boxes drew on
# one balance, and when it hit zero, 18 in-flight sessions were killed
# mid-optimization at once. Their kernels were still improving, so the partial
# results were not publishable and the money bought nothing. The two most
# expensive sessions on the fleet (581 and 615 turns) died that way.
#
# The fix is not "spend less" -- it is "never be interrupted mid-cell". A cell
# that finishes is worth its full cost; a cell killed at 90% is worth zero. So
# this guard reserves enough balance to COMPLETE whatever is already running,
# and stops NEW cells from launching once the remaining balance can no longer
# cover another worst-case cell.
#
# Usage:
#   scripts/openrouter_guard.sh watch [interval_seconds]   # daemon
#   scripts/openrouter_guard.sh check                      # one-shot, exit 1 = stop
#   scripts/openrouter_guard.sh balance                    # print remaining USD
#
# Env:
#   KB_GUARD_RESERVE   USD that must remain to allow a NEW cell (default 130,
#                      the observed worst-case completed cell plus margin)
#   KB_GUARD_STATE     state dir (default ~/.kb_openrouter_guard)
#
# Launchers must call `check` BETWEEN cells and refuse to start another when it
# exits non-zero. Nothing here ever kills a running session -- that would
# recreate the exact loss it exists to prevent.
set -uo pipefail

STATE_DIR="${KB_GUARD_STATE:-$HOME/.kb_openrouter_guard}"
STOP_FILE="$STATE_DIR/STOP"
BAL_FILE="$STATE_DIR/balance"
LOG_FILE="$STATE_DIR/guard.log"
RESERVE="${KB_GUARD_RESERVE:-130}"

mkdir -p "$STATE_DIR"

[ -f "$HOME/.env_vars" ] && . "$HOME/.env_vars" 2>/dev/null

fetch_balance() {
    local json
    json=$(curl -s --max-time 30 -H "Authorization: Bearer ${OPENROUTER_API_KEY:-}" \
        https://openrouter.ai/api/v1/credits 2>/dev/null) || return 1
    printf '%s' "$json" | python3 -c '
import json,sys
try:
    d = json.load(sys.stdin)["data"]
except Exception:
    sys.exit(1)
# Remaining is what has been bought minus what has been spent.
print("%.4f" % (d["total_credits"] - d["total_usage"]))
' 2>/dev/null
}

cmd_balance() {
    local b
    b=$(fetch_balance) || { echo "ERROR: could not read balance" >&2; return 2; }
    echo "$b"
}

cmd_check() {
    local b
    b=$(fetch_balance)
    if [ -z "${b:-}" ]; then
        # Fail SAFE: an unreadable balance must not authorise more spending.
        echo "$(date +%FT%T%z) UNKNOWN balance -- refusing to start a new cell" >> "$LOG_FILE"
        return 1
    fi
    printf '%s\n' "$b" > "$BAL_FILE"
    if [ -f "$STOP_FILE" ]; then
        return 1
    fi
    # bc is not guaranteed; compare in python.
    if ! python3 -c "import sys; sys.exit(0 if float('$b') >= float('$RESERVE') else 1)"; then
        {
            echo "$(date +%FT%T%z) remaining \$$b < reserve \$$RESERVE"
            echo "stopping BETWEEN cells; running sessions are left alone"
        } >> "$LOG_FILE"
        touch "$STOP_FILE"
        return 1
    fi
    return 0
}

cmd_watch() {
    local interval="${1:-120}"
    echo "$(date +%FT%T%z) guard start interval=${interval}s reserve=\$$RESERVE" >> "$LOG_FILE"
    while true; do
        if cmd_check; then
            printf '%s remaining $%s\n' "$(date +%FT%T%z)" "$(cat "$BAL_FILE" 2>/dev/null)" >> "$LOG_FILE"
        else
            echo "$(date +%FT%T%z) STOP asserted" >> "$LOG_FILE"
        fi
        sleep "$interval"
    done
}

case "${1:-check}" in
    watch)   shift; cmd_watch "${1:-120}" ;;
    check)   cmd_check ;;
    balance) cmd_balance ;;
    reset)   rm -f "$STOP_FILE"; echo "cleared $STOP_FILE" ;;
    *)       echo "usage: $0 {watch [sec]|check|balance|reset}" >&2; exit 2 ;;
esac
