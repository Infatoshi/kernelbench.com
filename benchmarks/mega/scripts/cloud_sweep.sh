#!/usr/bin/env bash
# Run a KernelBench-Mega model sweep ON a cloud box, tagging each run with its
# GPU so build_mega_leaderboard.py picks it up. One-shot: rsync the repo + auth,
# run cloud_bootstrap.sh, then run this. It sweeps the roster sequentially
# (GPU serialized anyway) and writes a `gpu` marker into each run archive.
#
#   GPU_LABEL="B200" BUDGET_SECONDS=10800 bash ~/mega/scripts/cloud_sweep.sh
#
# Roster: token-auth harnesses that reproduce cleanly across boxes (codex, opus).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1

GPU_LABEL="${GPU_LABEL:-unknown-gpu}"
BUDGET_SECONDS="${BUDGET_SECONDS:-10800}"
PROBLEM="${PROBLEM:-problems/03_kimi_linear_decode}"
# shellcheck disable=SC1090
source ~/.env_vars 2>/dev/null || true

# harness|model|effort
ROSTER=(
  "codex|gpt-5.5|xhigh"
  "claude|claude-opus-4-8|max"
)

for entry in "${ROSTER[@]}"; do
  IFS='|' read -r HARNESS MODEL EFFORT <<< "$entry"
  echo "=== SWEEP $GPU_LABEL :: $HARNESS $MODEL $EFFORT ($(date -Is)) ==="
  before=$(ls -1d outputs/runs/*_"${PROBLEM##*/}" 2>/dev/null | wc -l)
  BUDGET_SECONDS="$BUDGET_SECONDS" ./scripts/run_hard.sh "$HARNESS" "$MODEL" "$PROBLEM" "$EFFORT" || true
  # tag the newest run dir for this problem with the GPU label
  newest=$(ls -1dt outputs/runs/*_"${PROBLEM##*/}" 2>/dev/null | head -1)
  if [ -n "$newest" ]; then
    echo "$GPU_LABEL" > "$newest/gpu"
    echo "tagged $newest -> gpu=$GPU_LABEL ($(grep -oE 'peak_fraction[^ ]* [0-9.]+|score=[0-9.]+' "$newest/benchmark.log" 2>/dev/null | tail -1))"
  fi
  echo "(before=$before now=$(ls -1d outputs/runs/*_"${PROBLEM##*/}" 2>/dev/null | wc -l))"
done
echo "=== SWEEP $GPU_LABEL DONE ($(date -Is)) ==="
