#!/usr/bin/env bash
# Rate-limit-aware full-deck resweep for one GPU box.
#
# Runs models SEQUENTIALLY (only one provider active at a time) so Z.ai/Moonshot
# don't get hammered, with reduced per-model parallelism for the throttled
# providers. Generous BUDGET_SECONDS so claude-family runs COMPLETE and emit the
# terminal `result` event (the only source of true output/thinking tokens).
#
# Env:
#   KBH_HARDWARE        roofline target (RTX_PRO_6000 | RTX_3090 | H100 | B200)
#   KBH_PROBLEMS_ROOT   prompt set dir (problems | problems-h100 | problems-b200 ...)
#   BUDGET_SECONDS      per-run wall budget (default 6000 = 100min)
#   RESWEEP_MODELS      optional space-list to override the deck (e.g. "opus glm")
#   RESWEEP_ONLY_MISSING=1  skip (harness,model,problem) cells already correct in $LEADERBOARD
#   LEADERBOARD         leaderboard json to check when ONLY_MISSING (default results/leaderboard.json)
set -u
HARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HARD_DIR"
export KBH_AGENT_CONTAINER=1
export BUDGET_SECONDS="${BUDGET_SECONDS:-6000}"
PROOT="${KBH_PROBLEMS_ROOT:-problems}"
HW="${KBH_HARDWARE:-RTX_PRO_6000}"
LEADERBOARD="${LEADERBOARD:-results/leaderboard.json}"
PROBLEMS=(01_fp8_gemm 02_kda_cutlass 03_paged_attention 05_topk_bitonic 06_sonic_moe_swiglu 07_w4a16_gemm)
TS="$(date +%Y%m%d_%H%M%S)"
DLOG="outputs/tmp/resweep_${HW}_${TS}.log"; mkdir -p outputs/tmp; : > "$DLOG"
log(){ echo "[$(date -Is)] $*" | tee -a "$DLOG"; }

# Deck: "harness model effort parallel"  (safe providers first, throttled last)
declare -a DECK=(
  "claude claude-opus-4-8 _ 6"
  "codex gpt-5.5 xhigh 6"
  "zai-claude glm-5.2 _ 3"
  "kimi-claude kimi-k2.7-code _ 3"
  "minimax-claude MiniMax-M3 _ 3"
  "deepseek-claude deepseek-v4-pro _ 3"
)
if [ -n "${RESWEEP_MODELS:-}" ]; then
  declare -a FILT=()
  for row in "${DECK[@]}"; do for want in $RESWEEP_MODELS; do
    [[ "$row" == *"$want"* ]] && FILT+=("$row"); done; done
  DECK=("${FILT[@]}")
fi

cell_done(){  # harness model problem -> 0 if already correct in leaderboard
  [ "${RESWEEP_ONLY_MISSING:-0}" = "1" ] || return 1
  uv run python - "$LEADERBOARD" "$1" "$2" "$3" <<'PY' 2>/dev/null
import json,sys
lb,h,m,p=sys.argv[1:5]
try: d=json.load(open(lb))
except: sys.exit(1)
for mm in d.get("models",[]):
    if mm.get("model")==m and (mm.get("harness")==h or h.endswith("claude")):
        c=mm.get("results",{}).get(p,{})
        sys.exit(0 if c.get("correct") else 1)
sys.exit(1)
PY
}

log "RESWEEP start hw=$HW proot=$PROOT budget=${BUDGET_SECONDS}s models=${#DECK[@]}"
for row in "${DECK[@]}"; do
  read -r H M E PAR <<<"$row"; [ "$E" = "_" ] && E=""
  log "MODEL $H/$M effort='$E' parallel=$PAR"
  pids=(); running=0
  for p in "${PROBLEMS[@]}"; do
    if cell_done "$H" "$M" "$p"; then log "  skip $p (already correct)"; continue; fi
    log "  launch $p"
    KBH_PROBLEMS_ROOT="$PROOT" KBH_HARDWARE="$HW" \
      uv run kbh run "$H" "$M" "$PROOT/$p" $E \
      > "outputs/tmp/resweep_${HW}_${H}_${M//\//_}_${p}.log" 2>&1 &
    pids+=($!); running=$((running+1)); sleep 20
    if [ "$running" -ge "$PAR" ]; then wait -n 2>/dev/null || true; running=$((running-1)); fi
  done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  log "MODEL DONE $H/$M"
done
log "RESWEEP COMPLETE hw=$HW"
