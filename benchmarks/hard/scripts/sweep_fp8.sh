#!/usr/bin/env bash
# fp8-only resweep: re-run the corrected 01_fp8_gemm across the 7 models, K=2,
# uncapped (6h backstop). The other 5 problems are roofline-rescaled, not re-run.
set -u
cd "$HOME/kernelbench.com/benchmarks/hard"
. "$HOME/.env_vars" 2>/dev/null || true
export KBH_AGENT_CONTAINER=1
export BUDGET_SECONDS="${BUDGET_SECONDS:-21600}"
K="${K:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
CLOG="outputs/fp8resweep_${TS}.log"; mkdir -p outputs/tmp
echo "[fp8-resweep $TS] K=$K budget=${BUDGET_SECONDS}s" | tee "$CLOG"

# "harness model effort"
MODELS=(
  "claude claude-opus-4-8 "
  "codex gpt-5.5 xhigh"
  "zai-claude glm-5.2 "
  "minimax-claude MiniMax-M3 "
  "deepseek-claude deepseek-v4-pro "
  "gemini gemini-3.5-flash "
  "kimi-claude kimi-k2.7-code "
)

run_one() {
  local h="$1" m="$2" e="${3:-}"
  local slug; slug="$(echo "${h}_${m}_01_fp8_gemm" | tr '/:. ' '____')"
  echo "[$(date -Is)] START $h $m 01_fp8_gemm ${e:-<default>}" >> "$CLOG"
  uv run kbh run "$h" "$m" "problems-rtxpro6000/01_fp8_gemm" $e > "outputs/tmp/fp8resweep_${slug}.log" 2>&1
  echo "[$(date -Is)] DONE  $h $m 01_fp8_gemm (exit $?)" >> "$CLOG"
}

running=0
for spec in "${MODELS[@]}"; do
  read -r h m e <<< "$spec"
  run_one "$h" "$m" "$e" &
  running=$((running+1)); sleep 8
  if [ "$running" -ge "$K" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "[$(date -Is)] FP8 RESWEEP COMPLETE ($TS)" | tee -a "$CLOG"
