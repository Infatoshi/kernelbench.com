#!/bin/bash
# Relaunch only retryable infrastructure failures from a sweep run_group.
#
# A retryable infra failure is declared by run_hard.sh in result.json, then
# flattened by scripts/summarize_runs.py. This avoids rerunning genuine
# check.py failures while recovering from 429s, provider early stops, and
# no-solution timeouts.

set -euo pipefail

RUN_GROUP="${1:?Usage: $0 <run_group> [retry_label]}"
RETRY_LABEL="${2:-${KBH_RETRY_LABEL:-retry1}}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BUDGET_SECONDS="${KBH_BUDGET_SECONDS:-0}"   # 0 = unlimited (run_hard.sh enforces no cap)
HARNESS_CONCURRENCY="${KBH_HARNESS_CONCURRENCY:-2}"
SWEEP_DIR="$REPO_ROOT/outputs/sweeps/$RUN_GROUP"
SUMMARY_DIR="$SWEEP_DIR/summary"
MANIFEST="$SWEEP_DIR/manifest.tsv"

mkdir -p "$SWEEP_DIR" "$SUMMARY_DIR"
if [ ! -f "$MANIFEST" ]; then
    printf 'run_group\tname\tharness\tmodel\teffort\tproblem\tbudget\tpid\tlog\n' > "$MANIFEST"
fi

uv run python scripts/summarize_runs.py --run-group "$RUN_GROUP" --output-dir "$SUMMARY_DIR" >/dev/null

mapfile -t RETRIES < <(
    uv run python - "$SUMMARY_DIR/summary.json" <<'PY'
import json
import sys

path = sys.argv[1]
rows = json.load(open(path))["runs"]
latest = {}
for row in rows:
    key = (
        row.get("harness") or "",
        row.get("model") or "",
        row.get("reasoning_effort") or "",
        row.get("problem") or "",
    )
    if not all(key[:2]) or not key[3]:
        continue
    if key not in latest or str(row.get("run_id") or "") > str(latest[key].get("run_id") or ""):
        latest[key] = row
for key, row in sorted(latest.items()):
    if row.get("retryable_infra_failure") is True:
        print("|".join(key))
PY
)

if [ "${#RETRIES[@]}" -eq 0 ]; then
    echo "no retryable infra failures in $RUN_GROUP"
    exit 0
fi

declare -A ACTIVE_BY_HARNESS=()

prune_harness_pids() {
    local harness="$1"
    local kept=()
    local pid
    for pid in ${ACTIVE_BY_HARNESS[$harness]:-}; do
        if kill -0 "$pid" 2>/dev/null; then
            kept+=("$pid")
        fi
    done
    ACTIVE_BY_HARNESS[$harness]="${kept[*]:-}"
}

active_harness_count() {
    local harness="$1"
    prune_harness_pids "$harness"
    local count=0
    local pid
    for pid in ${ACTIVE_BY_HARNESS[$harness]:-}; do
        count=$((count + 1))
    done
    echo "$count"
}

wait_for_harness_slot() {
    local harness="$1"
    while [ "$(active_harness_count "$harness")" -ge "$HARNESS_CONCURRENCY" ]; do
        sleep 15
    done
}

slug() {
    echo "$1" | tr '/:[] ' '_'
}

for row in "${RETRIES[@]}"; do
    IFS='|' read -r harness model effort problem <<< "$row"
    if [[ "$problem" != problems/* ]]; then
        problem="problems/$problem"
    fi
    wait_for_harness_slot "$harness"
    problem_name="$(basename "$problem")"
    name="$(slug "${harness}_${model}_${RETRY_LABEL}")"
    log="$SWEEP_DIR/${name}_${problem_name}.out"
    if [ -n "$effort" ]; then
        cmd=(./scripts/run_hard.sh "$harness" "$model" "$problem" "$effort")
    else
        cmd=(./scripts/run_hard.sh "$harness" "$model" "$problem")
    fi
    (
        export KBH_RUN_GROUP="$RUN_GROUP"
        export KBH_DISABLE_AGENT_CUDA="${KBH_DISABLE_AGENT_CUDA:-1}"
        export BUDGET_SECONDS="$BUDGET_SECONDS"
        "${cmd[@]}"
    ) > "$log" 2>&1 &
    pid=$!
    ACTIVE_BY_HARNESS[$harness]="${ACTIVE_BY_HARNESS[$harness]:-} $pid"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$RUN_GROUP" "$name" "$harness" "$model" "$effort" "$problem" \
        "$BUDGET_SECONDS" "$pid" "${log#$REPO_ROOT/}" >> "$MANIFEST"
    echo "launched retry $name $problem_name pid=$pid"
done

echo "retry_jobs=${#RETRIES[@]}"
echo "manifest=$MANIFEST"
