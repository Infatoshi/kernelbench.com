#!/usr/bin/env bash
# Run a whole deck for one (harness, model) SEQUENTIALLY, one session at a time.
#
#   ./scripts/sweep_wave.sh grok grok-4.5 high
#   ./scripts/sweep_wave.sh grok grok-4.5 high 07_gemm_allreduce_overlap 08_ring_attention_cp
#
# Why sequential, when the GPU lock already serializes GPU commands (2026-07-25):
# the first parallel wave on this deck died to a cascade that the lock cannot
# prevent. Four concurrent sessions each hold GPU memory ACROSS their lock
# windows, so the node hit CUDA OOM; an agent then correctly diagnosed "processes
# holding 74GB each" as leaked, and ran `pkill -f torchrun` / `pkill -f worker.py`
# to clean up. Those patterns match every other session's processes, so it killed
# its siblings and itself (three runs died within 19s, exit 137). A fourth run
# survived only to fail grading with EADDRINUSE, because all sessions shared one
# hardcoded rendezvous port and agents were running `fuser -k 29571/tcp` to claim
# it.
#
# The lock serializes GPU *commands*; it does not partition GPU *memory*, and it
# cannot stop a pattern-based kill. One session at a time removes all three
# failure modes at once, and as a bonus the agent's own flywheel timings become
# trustworthy instead of contended. Cost is wall-clock, which is the cheap
# resource here.
set -euo pipefail

HARNESS="${1:?harness}"
MODEL="${2:?model}"
EFFORT="${3:-high}"
shift 3 2>/dev/null || shift $#
BENCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DECK="$BENCH_ROOT/problems-h100x4"

if [ "$#" -gt 0 ]; then
    PROBLEMS=("$@")
else
    PROBLEMS=()
    for d in "$DECK"/*/; do PROBLEMS+=("$(basename "$d")"); done
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$BENCH_ROOT/outputs/launch"
mkdir -p "$LOGDIR"
echo "[sweep_wave] $HARNESS $MODEL effort=$EFFORT sequential over ${#PROBLEMS[@]} problems"

for p in "${PROBLEMS[@]}"; do
    [ -d "$DECK/$p" ] || { echo "[sweep_wave] skip unknown problem: $p" >&2; continue; }
    echo "[sweep_wave] === $p  start $(date -u +%FT%TZ)"
    "$BENCH_ROOT/scripts/run_agent.sh" "$HARNESS" "$MODEL" "$p" "$EFFORT" \
        > "$LOGDIR/${TS}_${p}.log" 2>&1 || echo "[sweep_wave] $p exited $?"
    echo "[sweep_wave] === $p  done  $(date -u +%FT%TZ)"
    # Leave the node clean for the next session: nothing of ours should still
    # hold a GPU. Only touches processes under this bench's run archive.
    pkill -f "$BENCH_ROOT/outputs/runs/.*torch.distributed.run" 2>/dev/null || true
    sleep 5
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
    echo
done
echo "[sweep_wave] wave complete $(date -u +%FT%TZ)"
