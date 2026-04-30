#!/bin/bash
# Resume the overnight sweep after the gaming pause on 2026-04-29.
# 17 runs left in the queue, sequential, single GPU.
#
# Already complete (28 runs, do not re-run):
#   claude opus 4.7 max          7/7 done (6 PASS)
#   kimi k2.6                    7/7 done (6 PASS)
#   opencode zai/glm-5.1         7/7 done (4 PASS)
#   opencode mimo-v2.5-pro       7/7 done (5 PASS)
#
# This script runs:
#   qwen3.6-max-preview: 05, 06, 07          (3 runs — already PASSed 01-04)
#   qwen3.6-plus:        01, 02, 03, 04, 05, 06, 07  (7 runs)
#   qwen3.6-35b-a3b:     01, 02, 03, 04, 05, 06, 07  (7 runs — known to instant-fail, ~1 min total)

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$HOME/.env_vars" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$HOME/.env_vars"
    set +a
fi

# (harness, model, effort, problem)
declare -a JOBS=(
    "opencode openrouter-pinned/qwen/qwen3.6-max-preview  problems/05_topk_bitonic"
    "opencode openrouter-pinned/qwen/qwen3.6-max-preview  problems/06_sonic_moe_swiglu"
    "opencode openrouter-pinned/qwen/qwen3.6-max-preview  problems/07_w4a16_gemm"
    "opencode openrouter-pinned/qwen/qwen3.6-plus         problems/01_fp8_gemm"
    "opencode openrouter-pinned/qwen/qwen3.6-plus         problems/02_kda_cutlass"
    "opencode openrouter-pinned/qwen/qwen3.6-plus         problems/03_paged_attention"
    "opencode openrouter-pinned/qwen/qwen3.6-plus         problems/04_kahan_softmax"
    "opencode openrouter-pinned/qwen/qwen3.6-plus         problems/05_topk_bitonic"
    "opencode openrouter-pinned/qwen/qwen3.6-plus         problems/06_sonic_moe_swiglu"
    "opencode openrouter-pinned/qwen/qwen3.6-plus         problems/07_w4a16_gemm"
    "opencode openrouter-pinned/qwen/qwen3.6-35b-a3b      problems/01_fp8_gemm"
    "opencode openrouter-pinned/qwen/qwen3.6-35b-a3b      problems/02_kda_cutlass"
    "opencode openrouter-pinned/qwen/qwen3.6-35b-a3b      problems/03_paged_attention"
    "opencode openrouter-pinned/qwen/qwen3.6-35b-a3b      problems/04_kahan_softmax"
    "opencode openrouter-pinned/qwen/qwen3.6-35b-a3b      problems/05_topk_bitonic"
    "opencode openrouter-pinned/qwen/qwen3.6-35b-a3b      problems/06_sonic_moe_swiglu"
    "opencode openrouter-pinned/qwen/qwen3.6-35b-a3b      problems/07_w4a16_gemm"
)

TS="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/runs/resume_overnight_${TS}.log"
mkdir -p "$(dirname "$LOG")"

{
    echo "================================================================"
    echo "RESUME OVERNIGHT SWEEP"
    echo "Started: $(date)"
    echo "Jobs:    ${#JOBS[@]} runs left, sequential"
    echo "================================================================"
} | tee "$LOG"

for job in "${JOBS[@]}"; do
    read -r HARNESS MODEL PROBLEM <<< "$job"
    {
        echo
        echo "[$(date +%H:%M:%S)] $HARNESS / $MODEL / $PROBLEM"
    } | tee -a "$LOG"
    bash scripts/run_hard.sh "$HARNESS" "$MODEL" "$PROBLEM" 2>&1 | tee -a "$LOG" || true
done

{
    echo
    echo "================================================================"
    echo "RESUME COMPLETE"
    echo "Finished: $(date)"
    echo "================================================================"
} | tee -a "$LOG"
