#!/usr/bin/env bash
# Phase 2 of the uncapped resweep, sequenced AFTER the running reliable campaign.
# Waits for the current campaign to drain (no sweep_campaign.sh + kbh containers
# from it), then runs kimi-k2.7 full deck + the 2 opus rate-limit casualties.
# "Not all at once": K kept modest so opus reruns don't re-trigger the
# coding-plan rate-limit (they run essentially solo alongside kimi's metered route).
set -u
cd "$HOME/kernelbench.com/benchmarks/hard"
. "$HOME/.env_vars" 2>/dev/null || true
export KBH_AGENT_CONTAINER=1
export BUDGET_SECONDS="${BUDGET_SECONDS:-21600}"
LOG=outputs/phase2.log
echo "[phase2 $(date -Is)] waiting for reliable campaign to drain..." > "$LOG"

# wait until the campaign runner process is gone
while pgrep -f "sweep_campaign.sh scripts/campaign_reliable.spec" >/dev/null 2>&1; do sleep 60; done
echo "[phase2 $(date -Is)] campaign drained; starting phase 2" >> "$LOG"

run() {  # harness model problem [effort]
  local h="$1" m="$2" p="$3" e="${4:-}"
  local slug; slug="$(echo "${h}_${m}_${p}" | tr '/:. ' '____')"
  echo "[$(date -Is)] START $h $m $p" >> "$LOG"
  uv run kbh run "$h" "$m" "problems/$p" $e > "outputs/tmp/phase2_${slug}.log" 2>&1
  echo "[$(date -Is)] DONE  $h $m $p (exit $?)" >> "$LOG"
}

# 1) opus rate-limit casualties, solo (sequential) so they don't re-trip limits
run claude claude-opus-4-8 07_w4a16_gemm
run claude claude-opus-4-8 01_fp8_gemm

# 2) kimi-k2.7-code full deck, K=4 (metered route, modest concurrency)
PROBLEMS=(01_fp8_gemm 02_kda_cutlass 03_paged_attention 05_topk_bitonic 06_sonic_moe_swiglu 07_w4a16_gemm)
running=0
for p in "${PROBLEMS[@]}"; do
  run kimi-claude kimi-k2.7-code "$p" &
  running=$((running+1)); sleep 8
  if [ "$running" -ge 4 ]; then wait -n; running=$((running-1)); fi
done
wait
echo "[phase2 $(date -Is)] PHASE 2 COMPLETE" >> "$LOG"
