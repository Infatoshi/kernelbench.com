#!/usr/bin/env bash
# Full-deck parallel container sweep for one harness+model.
# Usage: ./scripts/sweep_deck.sh <harness> <model> [effort]
# Run from benchmarks/hard. Sessions overlap; GPU commands serialize via the lock.
set -u
HARNESS="${1:?harness required}"; MODEL="${2:?model required}"; EFFORT="${3:-}"
HARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HARD_DIR"
export KBH_AGENT_CONTAINER=1
PROBLEMS=(01_fp8_gemm 02_kda_cutlass 03_paged_attention 05_topk_bitonic 06_sonic_moe_swiglu 07_w4a16_gemm)
SLUG="$(echo "${HARNESS}_${MODEL}" | tr '/:. ' '____')"
LOG="outputs/tmp/sweep_${SLUG}.log"; mkdir -p outputs/tmp; : > "$LOG"
echo "[$(date -Is)] sweep $HARNESS $MODEL effort='$EFFORT' (6 problems, parallel)" | tee -a "$LOG"
pids=()
for p in "${PROBLEMS[@]}"; do
  echo "[$(date -Is)] LAUNCH $p" >> "$LOG"
  uv run kbh run "$HARNESS" "$MODEL" "problems/$p" $EFFORT > "outputs/tmp/sweep_${SLUG}_${p}.log" 2>&1 &
  pids+=($!); sleep 25
done
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || fail=$((fail+1)); echo "[$(date -Is)] EXIT ${PROBLEMS[$i]}" >> "$LOG"; done
echo "[$(date -Is)] SWEEP DONE fail=$fail" | tee -a "$LOG"
echo "Next: kb publish   (then: kb deploy)"
