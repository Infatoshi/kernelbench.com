#!/bin/bash
# Launch a queue-safe parallel KernelBench-Hard sweep.
#
# Defaults favor broad model comparison while keeping CUDA validation serialized:
# agents run with CUDA hidden, then run_hard.sh owns check.py/benchmark.py under
# the repo GPU lock. Launch concurrency is capped per harness so provider
# account limits fail less often than with a full fan-out.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RUN_GROUP="${KBH_RUN_GROUP:-sweep_$(date +%Y%m%d_%H%M%S)}"
BUDGET_SECONDS="${KBH_BUDGET_SECONDS:-2700}"
HARNESS_CONCURRENCY="${KBH_HARNESS_CONCURRENCY:-2}"
PROBLEMS="${KBH_PROBLEMS:-problems/01_qwen3_decode_block}"
SWEEP_DIR="$REPO_ROOT/outputs/sweeps/$RUN_GROUP"
mkdir -p "$SWEEP_DIR"

MANIFEST="$SWEEP_DIR/manifest.tsv"
printf 'run_group\tname\tharness\tmodel\teffort\tproblem\tbudget\tpid\tlog\n' > "$MANIFEST"

ROWS=(
    "codex_gpt55_xhigh|codex|gpt-5.5|xhigh"
    "claude_opus47_max|claude|claude-opus-4-7|max"
    "zai_claude_glm51|zai-claude|glm-5.1|"
    "opencode_glm51|opencode|zai/glm-5.1|"
    "cursor_composer25fast|cursor|composer-2.5-fast|"
    "grok_grokbuild_max|grok|grok-build|max"
    "opencode_qwen37max|opencode|openrouter-alibaba/qwen/qwen3.7-max|"
    "opencode_gemini35flash|opencode|openrouter-google-ai-studio/google/gemini-3.5-flash|"
)

if [ "${KBH_USE_DIRECT_GEMINI:-0}" = "1" ]; then
    ROWS+=("gemini_gemini35flash|gemini|gemini-3.5-flash|")
fi

if [ "${KBH_USE_MINIMAX_M3_CLAUDE:-0}" = "1" ]; then
    ROWS+=("minimax_m3_claude|minimax-claude|MiniMax-M3|")
fi

if [ "${KBH_SKIP_OPENROUTER:-0}" = "1" ]; then
    FILTERED_ROWS=()
    for row in "${ROWS[@]}"; do
        IFS='|' read -r _name _harness model _effort <<< "$row"
        if [[ "$model" == openrouter-* ]]; then
            continue
        fi
        FILTERED_ROWS+=("$row")
    done
    ROWS=("${FILTERED_ROWS[@]}")
fi

manifest_append() {
    flock "$MANIFEST.lock" printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$MANIFEST"
}

LAST_LAUNCH_PID=""

launch_one() {
    local name="$1"
    local harness="$2"
    local model="$3"
    local effort="$4"
    local problem="$5"
    local problem_name log pid
    local -a cmd

    problem_name="$(basename "$problem")"
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
    LAST_LAUNCH_PID=$!
    manifest_append "$RUN_GROUP" "$name" "$harness" "$model" "$effort" "$problem" \
        "$BUDGET_SECONDS" "$LAST_LAUNCH_PID" "${log#$REPO_ROOT/}"
}

wait_for_local_slot() {
    local -n active_ref="$1"
    local kept=()
    local pid

    while true; do
        kept=()
        for pid in "${active_ref[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kept+=("$pid")
            else
                wait "$pid" 2>/dev/null || true
            fi
        done
        active_ref=("${kept[@]}")
        if [ "${#active_ref[@]}" -lt "$HARNESS_CONCURRENCY" ]; then
            return 0
        fi
        sleep 15
    done
}

run_harness_worker() {
    local target_harness="$1"
    local active_pids=()
    local problem row name harness model effort pid

    for problem in $PROBLEMS; do
        for row in "${ROWS[@]}"; do
            IFS='|' read -r name harness model effort <<< "$row"
            if [ "$harness" != "$target_harness" ]; then
                continue
            fi
            wait_for_local_slot active_pids
            launch_one "$name" "$harness" "$model" "$effort" "$problem"
            pid="$LAST_LAUNCH_PID"
            active_pids+=("$pid")
        done
    done

    for pid in "${active_pids[@]}"; do
        wait "$pid" || true
    done
}

HARNESS_NAMES=()
for row in "${ROWS[@]}"; do
    IFS='|' read -r _name harness _model _effort <<< "$row"
    found=0
    for existing in "${HARNESS_NAMES[@]}"; do
        if [ "$existing" = "$harness" ]; then
            found=1
            break
        fi
    done
    if [ "$found" -eq 0 ]; then
        HARNESS_NAMES+=("$harness")
    fi
done

worker_pids=()
for harness in "${HARNESS_NAMES[@]}"; do
    run_harness_worker "$harness" &
    worker_pids+=("$!")
done

for pid in "${worker_pids[@]}"; do
    wait "$pid"
done

echo "run_group=$RUN_GROUP"
echo "manifest=$MANIFEST"
echo "jobs=$(($(wc -l < "$MANIFEST") - 1))"
echo "harness_concurrency=$HARNESS_CONCURRENCY"
