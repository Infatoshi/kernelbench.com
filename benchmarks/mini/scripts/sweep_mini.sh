#!/usr/bin/env bash
# KernelBench-Mini sweep: REPEATS independent sessions per (harness, model,
# problem). Mini's unit of publication is the 5-repeat cell: pass rate k/5 on
# the reliability axis, best-of-5 on the capability axis.
#
# Usage:
#   ./scripts/sweep_mini.sh <harness> <model> [effort]
#   KBMINI_REPEATS=5 KBMINI_PROBLEMS="problems-h100/01_dequant_gemv ..." override.
#
# One invocation drives ONE (harness, model) column sequentially — launch one
# sweep_mini.sh per model to parallelize (per-harness workers, never a
# problem-major loop: see the head-of-line-blocking lesson in AGENTS.md).
# Sessions overlap fine; GPU commands serialize through outputs/gpu.lock. The
# published numbers still come from the sequential isolated re-benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."

HARNESS="${1:?usage: sweep_mini.sh <harness> <model> [effort]}"
MODEL="${2:?usage: sweep_mini.sh <harness> <model> [effort]}"
EFFORT="${3:-}"

REPEATS="${KBMINI_REPEATS:-5}"
PROBLEMS="${KBMINI_PROBLEMS:-problems-h100/01_dequant_gemv problems-h100/02_segmented_decay_scan problems-h100/03_topp_mask problems-h100/04_flash_attention}"

SWEEP_TS="$(date +%Y%m%d_%H%M%S)"
LOG="outputs/runs/sweep_mini_${SWEEP_TS}_${HARNESS}_${MODEL//\//_}.log"
mkdir -p "$(dirname "$LOG")"

echo "[sweep_mini] harness=$HARNESS model=$MODEL effort=${EFFORT:-<none>} repeats=$REPEATS" | tee -a "$LOG"

for problem in $PROBLEMS; do
    for rep in $(seq 1 "$REPEATS"); do
        echo "[sweep_mini] $(date -u +%FT%TZ) problem=$problem repeat=$rep/$REPEATS" | tee -a "$LOG"
        if [ -n "$EFFORT" ]; then
            ./scripts/run_hard.sh "$HARNESS" "$MODEL" "$problem" "$EFFORT" 2>&1 | tee -a "$LOG" || \
                echo "[sweep_mini] WARN: run failed problem=$problem rep=$rep" | tee -a "$LOG"
        else
            ./scripts/run_hard.sh "$HARNESS" "$MODEL" "$problem" 2>&1 | tee -a "$LOG" || \
                echo "[sweep_mini] WARN: run failed problem=$problem rep=$rep" | tee -a "$LOG"
        fi
    done
done

echo "[sweep_mini] done: $(date -u +%FT%TZ). Audit every cell before reporting." | tee -a "$LOG"
