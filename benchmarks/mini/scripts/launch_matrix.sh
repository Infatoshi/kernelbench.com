#!/usr/bin/env bash
# KernelBench-Mini full matrix for ONE precision: every harness column in
# parallel, one worker per harness, workers pinned round-robin across the
# node's GPUs.
#
# Usage:
#   ./scripts/launch_matrix.sh <model> [harness ...]
#   KBMINI_GPUS="0 1" KBMINI_REPEATS=5 ./scripts/launch_matrix.sh lfm25-agent-bf16
#
# Each worker is a plain sweep_mini.sh column (4 problems x REPEATS repeats,
# sequential within the column) — per-harness workers, never a problem-major
# loop, so a slow provider cannot head-of-line-block the others.
#
# GPU isolation: each GPU gets its own lock domain (KBH_GPU_LOCK_DIR), so
# compile/check/benchmark serialize per GPU while sessions on different GPUs
# run truly concurrently. Agent sessions are network-bound most of the time,
# so several columns share a GPU comfortably.
#
# This is the PARALLEL phase: its in-run timings are contended and are NOT
# publishable. Re-grade every surviving cell sequentially (one GPU owner at a
# time) before publish — see the standing re-benchmark rule in AGENTS.md.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: launch_matrix.sh <model> [harness ...]}"
shift || true

HARNESSES=("$@")
if [ ${#HARNESSES[@]} -eq 0 ]; then
    HARNESSES=(lfm-opencode hermes pi lfm-grok lfm-claude)
fi

read -r -a GPUS <<< "${KBMINI_GPUS:-0}"

# KBMINI_SPLIT_BY_PROBLEM=1 gives every (harness, problem) its own worker
# instead of one worker per harness column. Wall clock drops from
# 4 problems x REPEATS per worker to REPEATS per worker -- ~4x -- because these
# sessions are inference-bound, not GPU-bound: the eval GPU only sees the short
# check.py/benchmark.py calls, which still serialize through the lock.
# Use it when one node has to carry the whole deck and the model is small
# enough that the inference server is nowhere near saturated.
read -r -a SPLIT_PROBLEMS <<< "${KBMINI_PROBLEMS:-problems-h100/01_dequant_gemv problems-h100/02_segmented_decay_scan problems-h100/03_topp_mask problems-h100/04_flash_attention}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="outputs/runs/matrix_${STAMP}_${MODEL//\//_}"
mkdir -p "$LOGDIR"

echo "[matrix] model=$MODEL harnesses=${HARNESSES[*]} gpus=${GPUS[*]} logs=$LOGDIR"

launch_worker() {
    local harness="$1" label="$2" problems="$3" gpu="$4"
    local lockdir="$PWD/outputs/gpu_lock_${gpu}"
    mkdir -p "$lockdir"
    echo "[matrix] worker $label gpu=$gpu lock=$lockdir"
    CUDA_VISIBLE_DEVICES="$gpu" \
    KBH_GPU_LOCK_DIR="$lockdir" \
    KBMINI_PROBLEMS="$problems" \
    setsid nohup ./scripts/sweep_mini.sh "$harness" "$MODEL" \
        > "$LOGDIR/${label}.log" 2>&1 < /dev/null &
    echo "$!" >> "$LOGDIR/worker.pids"
    sleep 2
}

i=0
n=0
for harness in "${HARNESSES[@]}"; do
    if [ "${KBMINI_SPLIT_BY_PROBLEM:-0}" = "1" ]; then
        for problem in "${SPLIT_PROBLEMS[@]}"; do
            launch_worker "$harness" "${harness}_$(basename "$problem")" \
                "$problem" "${GPUS[$((i % ${#GPUS[@]}))]}"
            i=$((i + 1))
            n=$((n + 1))
        done
    else
        launch_worker "$harness" "$harness" "${SPLIT_PROBLEMS[*]}" \
            "${GPUS[$((i % ${#GPUS[@]}))]}"
        i=$((i + 1))
        n=$((n + 1))
    fi
done

echo "[matrix] launched $n workers; pids in $LOGDIR/worker.pids"
echo "[matrix] tail -f $LOGDIR/*.log"
