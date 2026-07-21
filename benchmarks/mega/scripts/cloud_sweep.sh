#!/usr/bin/env bash
# Run a KernelBench-Mega model sweep ON a cloud box (or anvil), tagging each run
# with its GPU so build_mega_leaderboard.py picks it up. One-shot: rsync repo +
# auth, run cloud_bootstrap.sh, then run this.
#
#   GPU_LABEL="B200" BUDGET_SECONDS=10800 bash ~/mega/scripts/cloud_sweep.sh
#
# Models launch CONCURRENTLY: agent (code-writing) phases overlap to compress
# wall-clock, while every GPU command serializes through outputs/gpu.lock so the
# benchmark timings stay clean (one benchmark.py on the GPU at a time). Each run
# is tagged with GPU_LABEL via its own archive path (no newest-dir race).
#
# Roster = the KernelBench-Mega field (minus Fable 5, retired). Override with
# ROSTER_OVERRIDE="harness|model|effort;harness|model|effort".
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1

GPU_LABEL="${GPU_LABEL:-unknown-gpu}"
BUDGET_SECONDS="${BUDGET_SECONDS:-10800}"
PROBLEM="${PROBLEM:-problems/02_kimi_linear_decode}"
# shellcheck disable=SC1090
source ~/.env_vars 2>/dev/null || true

DEFAULT_ROSTER="codex|gpt-5.5|xhigh
claude|claude-opus-4-8|max
zai-claude|glm-5.2|
minimax-claude|MiniMax-M3|
kimi-claude|kimi-k2.7-code|
deepseek-claude|deepseek-v4-pro|
gemini|gemini-3.5-flash|
cursor|composer-2.5-fast|"
if [ -n "${ROSTER_OVERRIDE:-}" ]; then
  ROSTER=$(echo "$ROSTER_OVERRIDE" | tr ';' '\n')
else
  ROSTER="$DEFAULT_ROSTER"
fi

run_one() {
  local HARNESS="$1" MODEL="$2" EFFORT="$3"
  # Model ids can contain slashes (or-fable anthropic/claude-fable-5) —
  # flatten for the log filename or the redirect fails on a missing dir.
  local log="$HOME/mega_run_${HARNESS}_${MODEL//\//_}.log"
  echo "[$GPU_LABEL] launch $HARNESS $MODEL $EFFORT ($(date -Is))"
  # </dev/null: keep the agent CLI from eating the caller loop's here-string stdin
  BUDGET_SECONDS="$BUDGET_SECONDS" ./scripts/run_hard.sh "$HARNESS" "$MODEL" "$PROBLEM" "$EFFORT" > "$log" 2>&1 </dev/null || true
  local arch
  arch=$(grep -oE "Archive:.*outputs/runs/[^ ]+" "$log" | awk '{print $NF}' | head -1)
  if [ -n "$arch" ] && [ -d "$arch" ]; then
    echo "$GPU_LABEL" > "$arch/gpu"
    echo "[$GPU_LABEL] done $HARNESS $MODEL -> $(grep -oE 'score=[0-9.]+|peak_fraction: [0-9.]+' "$arch/benchmark.log" 2>/dev/null | tail -1) (gpu tagged)"
  else
    echo "[$GPU_LABEL] WARN $HARNESS $MODEL: no archive dir parsed from $log"
  fi
}

pids=()
while IFS='|' read -r -u3 HARNESS MODEL EFFORT; do
  [ -z "$HARNESS" ] && continue
  run_one "$HARNESS" "$MODEL" "$EFFORT" &
  pids+=($!)
  sleep 20   # stagger launches so torch/triton first-compile doesn't thrash
done 3<<< "$ROSTER"

echo "=== SWEEP $GPU_LABEL launched ${#pids[@]} models concurrently, waiting ==="
for p in "${pids[@]}"; do wait "$p"; done
echo "=== SWEEP $GPU_LABEL DONE ($(date -Is)) ==="
