#!/bin/bash
# Canonical GLM-5.1 rerun through Claude Code on the CUDA KernelBench-Hard deck.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PROBLEMS=(
    "problems-rtxpro6000/01_fp8_gemm"
    "problems-rtxpro6000/02_kda_cutlass"
    "problems-rtxpro6000/03_paged_attention"
    "problems-rtxpro6000/05_topk_bitonic"
    "problems-rtxpro6000/06_sonic_moe_swiglu"
    "problems-rtxpro6000/07_w4a16_gemm"
)

echo "========================================"
echo "Z.AI GLM-5.1 CLAUDE CODE RERUN"
echo "Started: $(date)"
echo "Harness: zai-claude"
echo "Model: glm-5.1"
echo "Problems: ${#PROBLEMS[@]}"
echo "========================================"

for problem in "${PROBLEMS[@]}"; do
    echo ""
    echo "=== $(date +%H:%M:%S) zai-claude/glm-5.1 x $(basename "$problem") ==="
    ./scripts/run_hard.sh zai-claude glm-5.1 "$problem" 2>&1 || true
done

echo ""
echo "========================================"
echo "Z.AI GLM-5.1 CLAUDE CODE RERUN COMPLETE"
echo "Finished: $(date)"
echo "========================================"
