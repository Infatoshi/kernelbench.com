#!/bin/bash
# Cheap-tier shakedown: 4 models x 7 problems = 28 runs, sequential.
#
# Validates the new PROMPT.txt regime + token logging + openrouter-pinned
# routing for Qwen, before committing to the expensive tier (Kimi, MiMo,
# qwen3.6-max-preview, claude-opus-4-7 max, gpt-5.5 xhigh).
#
# Sequential because GPU is exclusive per run. ~14-21 hours wall clock
# at full 45-min budgets, less if models finish early. Estimated total
# API spend: ~$5 (DeepSeek Flash $0.27, Pro $0.86, MiniMax $1.05,
# Qwen 27B Alibaba $3.00).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$HOME/.env_vars" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$HOME/.env_vars"
    set +a
fi

declare -a MATRIX=(
    "opencode deepseek/deepseek-v4-flash"
    "opencode deepseek/deepseek-v4-pro"
    "opencode openrouter-pinned/minimax/minimax-m2.7"
    # qwen3.6-27b: dropped after 0/7 PASS. Compliance + hallucination, see DEVLOG.
    # qwen3.6-35b-a3b: dropped, no tool-use endpoint available, see DEVLOG.
)

declare -a PROBLEMS=(
    "problems/01_fp8_gemm"
    "problems/02_kda_cutlass"
    "problems/03_paged_attention"
    "problems/04_kahan_softmax"
    "problems/05_topk_bitonic"
    "problems/06_sonic_moe_swiglu"
    "problems/07_w4a16_gemm"
)

TS="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/runs/shakedown_${TS}.log"
mkdir -p "$(dirname "$LOG")"

{
    echo "========================================"
    echo "CHEAP-TIER SHAKEDOWN SWEEP"
    echo "Started:  $(date)"
    echo "Models:   ${#MATRIX[@]}"
    echo "Problems: ${#PROBLEMS[@]}"
    echo "Total:    $((${#MATRIX[@]} * ${#PROBLEMS[@]})) runs, sequential"
    echo "Budget:   45 min/run"
    echo "========================================"
} | tee "$LOG"

# Outer loop: problem; inner loop: model. So if we abort early, every
# completed problem has full coverage rather than completed-models-only.
for problem in "${PROBLEMS[@]}"; do
    for entry in "${MATRIX[@]}"; do
        read -r HARNESS MODEL <<< "$entry"
        {
            echo
            echo "================================================================"
            echo "[$(date +%H:%M:%S)] $HARNESS / $MODEL / $problem"
            echo "================================================================"
        } | tee -a "$LOG"
        bash scripts/run_hard.sh "$HARNESS" "$MODEL" "$problem" 2>&1 | tee -a "$LOG" || true
    done
done

{
    echo
    echo "========================================"
    echo "SHAKEDOWN COMPLETE"
    echo "Finished: $(date)"
    echo "========================================"
} | tee -a "$LOG"
