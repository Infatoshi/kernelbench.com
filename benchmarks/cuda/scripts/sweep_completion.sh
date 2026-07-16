#!/usr/bin/env bash
# Completion resweep: bring composer-2.5 and deepseek-v4-pro into the
# unlimited-time generation. composer = all 6 problems; deepseek = the 5
# non-fp8 problems (fp8 already done uncapped on the corrected problem). K=2.
set -u
cd "$HOME/kernelbench.com/benchmarks/hard"
. "$HOME/.env_vars" 2>/dev/null || true
export KBH_AGENT_CONTAINER=1
export BUDGET_SECONDS="${BUDGET_SECONDS:-21600}"
K="${K:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
CLOG="outputs/completion_${TS}.log"; mkdir -p outputs/tmp
echo "[completion $TS] K=$K budget=${BUDGET_SECONDS}s" | tee "$CLOG"

# "harness model problem [effort]"
JOBS=(
  "cursor composer-2.5-fast 01_fp8_gemm"
  "cursor composer-2.5-fast 02_kda_cutlass"
  "cursor composer-2.5-fast 03_paged_attention"
  "cursor composer-2.5-fast 05_topk_bitonic"
  "cursor composer-2.5-fast 06_sonic_moe_swiglu"
  "cursor composer-2.5-fast 07_w4a16_gemm"
  "deepseek-claude deepseek-v4-pro 02_kda_cutlass"
  "deepseek-claude deepseek-v4-pro 03_paged_attention"
  "deepseek-claude deepseek-v4-pro 05_topk_bitonic"
  "deepseek-claude deepseek-v4-pro 06_sonic_moe_swiglu"
  "deepseek-claude deepseek-v4-pro 07_w4a16_gemm"
)

run_one() {
  local h="$1" m="$2" p="$3" e="${4:-}"
  local slug; slug="$(echo "${h}_${m}_${p}" | tr '/:. ' '____')"
  echo "[$(date -Is)] START $h $m $p" >> "$CLOG"
  uv run kbh run "$h" "$m" "problems/$p" $e > "outputs/tmp/completion_${slug}.log" 2>&1
  echo "[$(date -Is)] DONE  $h $m $p (exit $?)" >> "$CLOG"
}

running=0
for job in "${JOBS[@]}"; do
  read -r h m p e <<< "$job"
  run_one "$h" "$m" "$p" "$e" &
  running=$((running+1)); sleep 8
  if [ "$running" -ge "$K" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "[$(date -Is)] COMPLETION RESWEEP DONE ($TS)" | tee -a "$CLOG"
