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

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="outputs/runs/matrix_${STAMP}_${MODEL//\//_}"
mkdir -p "$LOGDIR"

echo "[matrix] model=$MODEL harnesses=${HARNESSES[*]} gpus=${GPUS[*]} logs=$LOGDIR"

i=0
for harness in "${HARNESSES[@]}"; do
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    lockdir="$PWD/outputs/gpu_lock_${gpu}"
    mkdir -p "$lockdir"
    echo "[matrix] worker harness=$harness gpu=$gpu lock=$lockdir"
    CUDA_VISIBLE_DEVICES="$gpu" \
    KBH_GPU_LOCK_DIR="$lockdir" \
    setsid nohup ./scripts/sweep_mini.sh "$harness" "$MODEL" \
        > "$LOGDIR/${harness}.log" 2>&1 < /dev/null &
    echo "$!" >> "$LOGDIR/worker.pids"
    i=$((i + 1))
    sleep 2
done

echo "[matrix] launched ${#HARNESSES[@]} workers; pids in $LOGDIR/worker.pids"
echo "[matrix] tail -f $LOGDIR/*.log"
