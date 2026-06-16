#!/bin/bash
# Rerun the 6 models that failed for credential reasons in the prior sweep.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

declare -a MATRIX=(
    "kimi kimi-k2.6 "
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

TS="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/runs/sweep_09_10_rerun_${TS}.log"
mkdir -p "$(dirname "$LOG")"

echo "RERUN START $(date)" | tee "$LOG"
for problem in "${PROBLEMS[@]}"; do
    for mh in "${MATRIX[@]}"; do
        read -r HARNESS MODEL EFFORT <<< "$mh"
        echo "" | tee -a "$LOG"
        echo "=== $(date +%H:%M:%S) $HARNESS/$MODEL × $(basename "$problem") ===" | tee -a "$LOG"
        ./scripts/run_hard.sh "$HARNESS" "$MODEL" "$problem" "$EFFORT" 2>&1 | tee -a "$LOG" || true
    done
done
echo "" | tee -a "$LOG"
echo "RERUN COMPLETE $(date)" | tee -a "$LOG"
