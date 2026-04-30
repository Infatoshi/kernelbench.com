#!/bin/bash
# Overnight sweep: all remaining models in ACTIVE_MATRIX, sequential.
# Order: highest-priority capability tier first, known-fail last.
#   1. claude opus 4.7 max  (top tier, same effort budget as gpt-5.5 xhigh)
#   2. kimi k2.6             (reasoning model, mid tier)
#   3. opencode zai/glm-5.1  (lab-direct reasoning model)
#   4. opencode mimo-v2.5-pro
#   5. opencode qwen3.6-max-preview
#   6. opencode qwen3.6-plus
#   7. opencode qwen3.6-35b-a3b  (known no-tool-use endpoint, fails in <1s — queued last per user direction)
#
# Skipped because already swept today:
#   - codex gpt-5.5 xhigh  (7/7 PASS)
#   - opencode deepseek-v4-flash, deepseek-v4-pro
#   - opencode minimax-m2.7
#   - opencode qwen3.6-27b  (post-fix rerun, 1/7 PASS)
#
# Outer loop = model so each model's full deck completes before the next
# starts. Failures don't abort the queue. Sequential single-GPU.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$HOME/.env_vars" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$HOME/.env_vars"
    set +a
fi

# Each entry: "harness model effort_or_blank"
declare -a QUEUE=(
    "claude claude-opus-4-7 max"
    "kimi kimi-k2.6 "
    "opencode zai/glm-5.1 "
    "opencode openrouter-pinned/xiaomi/mimo-v2.5-pro "
    "opencode openrouter-pinned/qwen/qwen3.6-max-preview "
    "opencode openrouter-pinned/qwen/qwen3.6-plus "
    "opencode openrouter-pinned/qwen/qwen3.6-35b-a3b "
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
LOG="outputs/runs/overnight_${TS}.log"
mkdir -p "$(dirname "$LOG")"

{
    echo "================================================================"
    echo "OVERNIGHT SWEEP"
    echo "Started:  $(date)"
    echo "Queue:    ${#QUEUE[@]} models x ${#PROBLEMS[@]} problems = $((${#QUEUE[@]} * ${#PROBLEMS[@]})) runs"
    echo "Budget:   45 min/run, sequential"
    echo "================================================================"
} | tee "$LOG"

for entry in "${QUEUE[@]}"; do
    read -r HARNESS MODEL EFFORT <<< "$entry"
    {
        echo
        echo "##############################################################"
        echo "[$(date +%H:%M:%S)] BEGIN MODEL: $HARNESS / $MODEL ${EFFORT:+(effort=$EFFORT)}"
        echo "##############################################################"
    } | tee -a "$LOG"
    for problem in "${PROBLEMS[@]}"; do
        {
            echo
            echo "[$(date +%H:%M:%S)] $HARNESS / $MODEL / $problem"
        } | tee -a "$LOG"
        bash scripts/run_hard.sh "$HARNESS" "$MODEL" "$problem" "$EFFORT" 2>&1 | tee -a "$LOG" || true
    done
done

{
    echo
    echo "================================================================"
    echo "OVERNIGHT SWEEP COMPLETE"
    echo "Finished: $(date)"
    echo "================================================================"
} | tee -a "$LOG"
