#!/usr/bin/env bash
# Concurrency-capped, uncapped-budget campaign runner.
# Expands a model-spec file over the 6-problem deck and runs at most K
# containers at once, refilling as each finishes. RAM is the limit (~39G on
# anvil); GPU work serializes via the per-command lock.
#
# Usage:  K=8 BUDGET_SECONDS=21600 ./scripts/sweep_campaign.sh <modelspec_file>
# modelspec lines:  "<harness> <model> [effort]"   ('#' comments ok)
set -u
HARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HARD_DIR"
export KBH_AGENT_CONTAINER=1
export BUDGET_SECONDS="${BUDGET_SECONDS:-21600}"   # 6h backstop = effectively uncapped
K="${K:-8}"
SPEC="${1:?modelspec file: lines of '<harness> <model> [effort]'}"
PROBLEMS=(01_fp8_gemm 02_kda_cutlass 03_paged_attention 05_topk_bitonic 06_sonic_moe_swiglu 07_w4a16_gemm)
TS="$(date +%Y%m%d_%H%M%S)"
CLOG="outputs/tmp/campaign_${TS}.log"; mkdir -p outputs/tmp
echo "[campaign $TS] K=$K budget=${BUDGET_SECONDS}s spec=$SPEC" | tee "$CLOG"

run_one() {
  local h="$1" m="$2" p="$3" e="${4:-}"
  local slug; slug="$(echo "${h}_${m}_${p}" | tr '/:. ' '____')"
  echo "[$(date -Is)] START $h $m $p ${e:-<default>}" >> "$CLOG"
  uv run kbh run "$h" "$m" "problems/$p" $e > "outputs/tmp/campaign_${slug}.log" 2>&1
  echo "[$(date -Is)] DONE  $h $m $p (exit $?)" >> "$CLOG"
}

# build the job list (model-major)
JOBS=()
while read -r h m e _rest; do
  [ -z "${h:-}" ] && continue
  case "$h" in \#*) continue;; esac
  for p in "${PROBLEMS[@]}"; do JOBS+=("$h|$m|$p|${e:-}"); done
done < "$SPEC"
echo "[campaign] ${#JOBS[@]} jobs queued" | tee -a "$CLOG"

running=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r h m p e <<< "$job"
  run_one "$h" "$m" "$p" "$e" &
  running=$((running+1))
  sleep 8
  if [ "$running" -ge "$K" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "[$(date -Is)] CAMPAIGN COMPLETE ($TS)" | tee -a "$CLOG"
