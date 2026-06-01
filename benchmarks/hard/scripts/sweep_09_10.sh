#!/bin/bash
# Focused sweep: problems 09 + 10 only, full active matrix.
# Same matrix as sweep.sh; same per-run budget (2700s).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

declare -a ACTIVE_MATRIX=(
    "claude claude-opus-4-7 max"
    "codex gpt-5.5 xhigh"
    "kimi kimi-k2.6 "
    "opencode zai/glm-5.1 "
    "opencode deepseek/deepseek-v4-pro "
    "opencode deepseek/deepseek-v4-flash "
    "opencode openrouter-pinned/minimax/minimax-m2.7 "
    "opencode openrouter-pinned/qwen/qwen3.6-max-preview "
    "opencode openrouter-pinned/qwen/qwen3.6-plus "
    "opencode openrouter-pinned/qwen/qwen3.6-27b "
    "opencode openrouter-pinned/xiaomi/mimo-v2.5-pro "
)

declare -a PROBLEMS=(
    "problems/09_fmha_preattn_mrope"
    "problems/10_patch_embed_conv3d_gemm"
)

SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="outputs/runs/sweep_09_10_${SWEEP_TIMESTAMP}.log"
mkdir -p "$(dirname "$SWEEP_LOG")"

echo "========================================" | tee "$SWEEP_LOG"
echo "KERNELBENCH-HARD SWEEP (09 + 10 only)" | tee -a "$SWEEP_LOG"
echo "Started: $(date)" | tee -a "$SWEEP_LOG"
echo "Models:  ${#ACTIVE_MATRIX[@]}" | tee -a "$SWEEP_LOG"
echo "Probs:   ${#PROBLEMS[@]}" | tee -a "$SWEEP_LOG"
echo "Runs:    $((${#ACTIVE_MATRIX[@]} * ${#PROBLEMS[@]}))" | tee -a "$SWEEP_LOG"
echo "========================================" | tee -a "$SWEEP_LOG"

for problem in "${PROBLEMS[@]}"; do
    for mh in "${ACTIVE_MATRIX[@]}"; do
        read -r HARNESS MODEL EFFORT <<< "$mh"
        echo "" | tee -a "$SWEEP_LOG"
        echo "=== $(date +%H:%M:%S) $HARNESS/$MODEL × $(basename "$problem") ===" | tee -a "$SWEEP_LOG"
        ./scripts/run_hard.sh "$HARNESS" "$MODEL" "$problem" "$EFFORT" 2>&1 | tee -a "$SWEEP_LOG" || true
    done
done

echo "" | tee -a "$SWEEP_LOG"
echo "========================================" | tee -a "$SWEEP_LOG"
echo "SWEEP COMPLETE" | tee -a "$SWEEP_LOG"
echo "Finished: $(date)" | tee -a "$SWEEP_LOG"
echo "========================================" | tee -a "$SWEEP_LOG"
