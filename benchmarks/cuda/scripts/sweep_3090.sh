#!/usr/bin/env bash
# Host-mode sweep for the RTX 3090 (gamer). The 3090 is sm_86 Ampere: no FP8/FP4
# tensor cores, so 01_fp8_gemm is un-gradable here (no fp8 peak key) and is
# excluded. Memory-bound deck cells (03/05/07) grade on bandwidth (936 GB/s) and
# port cleanly; bf16 compute cells (02/06) run at ~71 TFLOPS peak.
#
# Uses the 3090 prompt set (problems-3090): the hardware parenthetical in each
# PROMPT.txt says "RTX 3090 (SM86 Ampere, GDDR6X, 936 GB/s)" so the agent does
# not emit sm_120-only TMA/wgmma that won't compile on Ampere.
#
# Runs sequentially: the box has one GPU, so overlapping sessions would just
# serialize through the GPU lock with no gain. Unlimited time by default
# (BUDGET_SECONDS=0 disables the per-run timeout); override for a smoke test:
#   BUDGET_SECONDS=300 ./scripts/sweep_3090.sh zai-claude glm-4.7
#
# Usage (run from benchmarks/hard, or anywhere):
#   ./scripts/sweep_3090.sh <harness> <model> [effort]
set -u
HARNESS="${1:?harness required}"; MODEL="${2:?model required}"; EFFORT="${3:-}"
HARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HARD_DIR" || exit 1

# Load API keys if present (gamer keeps them in ~/.env_vars).
[ -f "$HOME/.env_vars" ] && . "$HOME/.env_vars"

# Host mode (gamer has claude/codex on the host; no container parity set up).
export KBH_AGENT_CONTAINER="${KBH_AGENT_CONTAINER:-0}"
# Unlimited agent wall-clock by default; GNU `timeout 0` = no cap.
export BUDGET_SECONDS="${BUDGET_SECONDS:-0}"
# Score against the 3090 peaks at publish time, not the Blackwell defaults.
export KBH_HARDWARE="${KBH_HARDWARE:-RTX_3090}"
# 3090-specific prompt set (hardware string rewritten to Ampere).
PROBLEMS_ROOT="${KBH_PROBLEMS_ROOT:-problems-3090}"

# Deck order: memory-bound first (highest-confidence on Ampere), then bf16
# compute. 01_fp8_gemm excluded (no FP8 hardware). Override with
# KBH_3090_PROBLEMS="07_w4a16_gemm 02_kda_cutlass" to run a subset (e.g. to
# finish only the missing cells of a paused model).
if [ -n "${KBH_3090_PROBLEMS:-}" ]; then
  # shellcheck disable=SC2206
  PROBLEMS=(${KBH_3090_PROBLEMS})
else
  PROBLEMS=(05_topk_bitonic 03_paged_attention 07_w4a16_gemm 06_sonic_moe_swiglu 02_kda_cutlass)
fi

SLUG="$(echo "${HARNESS}_${MODEL}" | tr '/:. ' '____')"
mkdir -p outputs/tmp
LOG="outputs/tmp/sweep3090_${SLUG}.log"; : > "$LOG"
echo "[$(date -Is)] 3090 sweep $HARNESS $MODEL effort='$EFFORT' budget=${BUDGET_SECONDS}s hw=$KBH_HARDWARE container=$KBH_AGENT_CONTAINER" | tee -a "$LOG"

fail=0
for p in "${PROBLEMS[@]}"; do
  echo "[$(date -Is)] START $p" | tee -a "$LOG"
  if uv run kbh run "$HARNESS" "$MODEL" "$PROBLEMS_ROOT/$p" $EFFORT > "outputs/tmp/sweep3090_${SLUG}_${p}.log" 2>&1; then
    echo "[$(date -Is)] OK    $p" | tee -a "$LOG"
  else
    rc=$?; fail=$((fail+1))
    echo "[$(date -Is)] FAIL  $p (rc=$rc) -> outputs/tmp/sweep3090_${SLUG}_${p}.log" | tee -a "$LOG"
  fi
done

echo "[$(date -Is)] SWEEP DONE fail=$fail" | tee -a "$LOG"
echo "Next: publish with KBH_HARDWARE=RTX_3090, then review per-cell solutions before any deploy."
exit "$fail"
