#!/usr/bin/env bash
# Run a list of benchmark cells with a balance check BETWEEN each launch.
#
# Companion to scripts/openrouter_guard.sh. The rule this enforces: a cell that
# finishes is worth its full cost, a cell killed at 90% is worth zero. So we
# never start a cell we cannot afford to finish, and we never kill one that is
# already running.
#
# Usage:
#   guarded_sweep.sh <repo_root> <concurrency> <cell> [<cell> ...]
# where <cell> is  bench:deck:problem   e.g.  hard:problems-rtxpro6000:05_topk_bitonic
#
# Env:
#   KB_SWEEP_HARNESS   harness name (default or-opus)
#   KB_SWEEP_MODEL     model id     (default anthropic/claude-opus-5)
#   KB_SWEEP_EFFORT    effort       (default max)
#   KB_SWEEP_LOG       log file     (default ~/guarded_sweep.log)
#   KB_GUARD_SH        path to openrouter_guard.sh (default <repo_root>/scripts/)
#   KBH_GPU_LOCK_DIR   shared GPU lock dir
set -uo pipefail

REPO="${1:?repo root required}"; shift
CONC="${1:?concurrency required}"; shift

HARNESS="${KB_SWEEP_HARNESS:-or-opus}"
MODEL="${KB_SWEEP_MODEL:-anthropic/claude-opus-5}"
EFFORT="${KB_SWEEP_EFFORT:-max}"
LOG="${KB_SWEEP_LOG:-$HOME/guarded_sweep.log}"
GUARD="${KB_GUARD_SH:-$REPO/scripts/openrouter_guard.sh}"

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.env_vars" ] && . "$HOME/.env_vars" 2>/dev/null

exec >> "$LOG" 2>&1
echo "=== guarded sweep start $(date +%FT%T%z) host=$(hostname) conc=$CONC cells=$# ==="

running=0

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do
        sleep 30
    done
}

for cell in "$@"; do
    IFS=: read -r BENCH DECK PROB <<< "$cell"

    wait_for_slot

    # Balance gate: only ever between launches, never against a live session.
    if [ -x "$GUARD" ]; then
        if ! "$GUARD" check; then
            echo "[$(date +%FT%T%z)] HALT: balance guard refused a new cell ($cell)"
            echo "  remaining: \$$("$GUARD" balance 2>/dev/null || echo '?')"
            echo "  running sessions are being left to finish."
            break
        fi
    fi

    BDIR="$REPO/benchmarks/$BENCH"
    if [ ! -d "$BDIR" ]; then
        echo "[$(date +%FT%T%z)] SKIP $cell: no $BDIR"
        continue
    fi

    echo "[$(date +%FT%T%z)] LAUNCH $BENCH/$PROB (balance \$$("$GUARD" balance 2>/dev/null || echo '?'))"
    (
        cd "$BDIR" || exit 1
        ./scripts/run_hard.sh "$HARNESS" "$MODEL" "$DECK/$PROB" "$EFFORT"
        rc=$?
        R=$(ls -dt outputs/runs/*"$HARNESS"*"$PROB"/ 2>/dev/null | head -1)
        SUM="rc=$rc"
        if [ -n "$R" ] && [ -f "$R/result.json" ]; then
            SUM=$(python3 -c "
import json
d=json.load(open('$R/result.json'))
print('correct=%s peak=%s reason=%s' % (d.get('correct'), d.get('peak_fraction'), d.get('failure_reason')))" 2>/dev/null || echo "$SUM")
        fi
        echo "[$(date +%FT%T%z)] DONE   $BENCH/$PROB $SUM"
    ) &
    sleep 20   # stagger starts so cache writes and GPU compiles do not collide
done

echo "[$(date +%FT%T%z)] all cells dispatched; waiting for stragglers"
wait
echo "=== guarded sweep complete $(date +%FT%T%z) ==="
