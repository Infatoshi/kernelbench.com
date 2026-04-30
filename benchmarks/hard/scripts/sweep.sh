#!/bin/bash
# Full active sweep: every (harness, model) × every CUDA problem.
#
# Usage: ./scripts/sweep.sh
#
# Edit ACTIVE_MATRIX below to add/remove models. ccr-rust must be running
# for ccr-claude entries (see docs/ccr-rust-setup.md).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# (harness, model, reasoning_effort) tuples.
# Empty reasoning_effort = use default.
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
    # qwen3.6-27b: 0/7 in pre-fix shakedown, 1/7 (sonic_moe 0.082) in post-fix
    # rerun. High variance; either the workspace-leak fix helped focus it, or
    # it's just LLM nondeterminism. N=1 too weak to tell. Same tier as MiniMax.
    # qwen3.6-35b-a3b dropped: AtlasCloud/Parasail (only providers serving
    # it) don't advertise tool-use to OpenRouter, so the agent harness can't
    # reach it. See DEVLOG.
    # Routing notes:
    # - claude-opus-4-7 at effort=max for parity with codex gpt-5.5 xhigh
    #   (the highest CLI-exposed reasoning tier). Coding-plan billing means
    #   no per-token cost lands on the API key; we still log token totals
    #   from the transcript for cross-model comparison.
    # - DeepSeek + GLM: opencode hitting native lab APIs directly.
    # - MiniMax: api.minimaxi.com 401s on standard Bearer auth. Routed via
    #   OpenRouter pinned to provider="Minimax" (the lab) via extraBody —
    #   gets the lab's fp8 endpoint at $0.30/$1.20 per M, 99.7% uptime.
    # - Qwen 3.6 family + MiMo via OpenRouter pinned to native labs first
    #   (Alibaba, Xiaomi). qwen/qwen3.6-35b-a3b is NOT hosted by Alibaba on
    #   OpenRouter — only AtlasCloud and Parasail serve it (both fp8).
    #   AtlasCloud and Parasail appended to provider.order so this single
    #   model routes to fp8 third-party; the rest of the family stays on
    #   the lab. Disclosed as a precision-asymmetric row in DEVLOG.
    # - ccr-claude is unreliable (returns malformed SSE shape to claude-code);
    #   codex 0.125.0 dropped chat-API support so non-OpenAI labs can't use codex.
)

# NVIDIA GPU sweep. Metal problem 08 is deferred (M4 Max track not prioritized
# for the first sweep). Order is outer=problem, inner=model so an early abort
# leaves complete per-problem rows rather than complete per-model columns.
declare -a CUDA_PROBLEMS=(
    "problems/01_fp8_gemm"
    "problems/02_kda_cutlass"
    "problems/03_paged_attention"
    "problems/04_kahan_softmax"
    "problems/05_topk_bitonic"
    "problems/06_sonic_moe_swiglu"
    "problems/07_w4a16_gemm"
)

SWEEP_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="outputs/runs/sweep_${SWEEP_TIMESTAMP}.log"
mkdir -p "$(dirname "$SWEEP_LOG")"

echo "========================================" | tee "$SWEEP_LOG"
echo "KERNELBENCH-HARD SWEEP" | tee -a "$SWEEP_LOG"
echo "Started: $(date)" | tee -a "$SWEEP_LOG"
echo "Models:  ${#ACTIVE_MATRIX[@]}" | tee -a "$SWEEP_LOG"
echo "Probs:   ${#CUDA_PROBLEMS[@]}" | tee -a "$SWEEP_LOG"
echo "Runs:    $((${#ACTIVE_MATRIX[@]} * ${#CUDA_PROBLEMS[@]}))" | tee -a "$SWEEP_LOG"
echo "========================================" | tee -a "$SWEEP_LOG"

for problem in "${CUDA_PROBLEMS[@]}"; do
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
