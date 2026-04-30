#!/bin/bash
# Overnight TopK column: 7 models × 1 problem (05_topk_bitonic), sequential.
#
# Sequential because the GPU must be exclusive per run (per harness contract).
# Each run has a 45-min wall budget; expect ~3-5 hours total.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Source API keys
if [ -f "$HOME/.env_vars" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$HOME/.env_vars"
    set +a
fi

# (harness, model, reasoning_effort) — same as scripts/sweep.sh ACTIVE_MATRIX
declare -a MATRIX=(
    "claude claude-opus-4-7 "
    "codex gpt-5.5 xhigh"
    "kimi kimi-k2.6 "
    "opencode zai/glm-5.1 "
    "opencode deepseek/deepseek-v4-pro "
    "opencode deepseek/deepseek-v4-flash "
    "opencode openrouter-pinned/minimax/minimax-m2.7 "
)

PROBLEM="problems/05_topk_bitonic"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/runs/topk_overnight_${TS}.log"
mkdir -p "$(dirname "$LOG")"

{
    echo "========================================"
    echo "TOPK OVERNIGHT SWEEP"
    echo "Started:  $(date)"
    echo "Problem:  $PROBLEM"
    echo "Models:   ${#MATRIX[@]}"
    echo "Budget:   45 min/run, sequential"
    echo "========================================"
    echo
} | tee "$LOG"

for entry in "${MATRIX[@]}"; do
    read -r HARNESS MODEL EFFORT <<< "$entry"
    {
        echo "================================================================"
        echo "[$(date +%H:%M:%S)] $HARNESS / $MODEL / effort=${EFFORT:-default}"
        echo "================================================================"
    } | tee -a "$LOG"
    bash scripts/run_hard.sh "$HARNESS" "$MODEL" "$PROBLEM" "$EFFORT" 2>&1 | tee -a "$LOG" || true
done

{
    echo
    echo "========================================"
    echo "TOPK SWEEP COMPLETE"
    echo "Finished: $(date)"
    echo "========================================"
} | tee -a "$LOG"
