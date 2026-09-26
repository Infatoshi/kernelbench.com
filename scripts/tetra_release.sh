#!/bin/bash
# Release sweep on tetra (4x RTX PRO 6000 Blackwell): one model's mega 02 plus cuda 01-04 as
# parallel sessions on ONE GPU (the per-bench GPU locks serialize their GPU commands), then the
# sequential isolated regrade of exactly those cells on the same GPU with the canonical decks.
# Different models run at once on different GPUs. This is the Grok 4.7 / Opus 5.5 / GPT-6 recipe.
#
#   scripts/tetra_release.sh <gpu> <harness> <model> [effort]
#   nohup scripts/tetra_release.sh 2 claude claude-opus-5-5 max >/dev/null 2>&1 &
#
# Holds an overnight-compute lease on gpu<N> for the whole pipeline and refuses a GPU with
# compute PIDs. Log, cell logs, served-model check and DONE marker land in the gitignored
# runs/release-<stamp>-<harness>-<model>/. Audits, pullback and publish stay manual
# (root AGENTS.md publish gates).
set -uo pipefail
GPU=${1:?usage: tetra_release.sh <gpu> <harness> <model> [effort]}
HARNESS=${2:?harness}
MODEL=${3:?model}
EFFORT=${4:-}
R=$(cd "$(dirname "$0")/.." && pwd)
export PATH=$HOME/.local/bin:/usr/local/cuda-13.3/bin:$PATH

if [ -z "${TETRA_RELEASE_LEASED:-}" ] && command -v overnight-compute >/dev/null; then
    busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    if [ -n "$busy" ]; then
        echo "STOP: GPU $GPU has compute PIDs: $busy" >&2
        exit 1
    fi
    AGENT="kb-release-gpu$GPU-$(printf '%s-%s' "$HARNESS" "$MODEL" | tr -c 'A-Za-z0-9.-' '_')"
    TETRA_RELEASE_LEASED=1 exec overnight-compute run --agent "$AGENT" --resource "gpu$GPU" \
        --ttl 30m --heartbeat 5m --timeout 2m -- "$0" "$@"
fi

export KBH_GPU=$GPU KBH_CUDA_HOME=/usr/local/cuda-13
[ -f ~/.env_vars ] && set -a && . ~/.env_vars && set +a
export META_API_KEY=${META_API_KEY:-${META_MODEL_API_KEY:-}}   # muse route; ~/.env_vars keeps the Meta key under this name
SLUG=$(printf '%s-%s' "$HARNESS" "$MODEL" | tr -c 'A-Za-z0-9.-' '_')
OUT=$R/runs/release-$(date +%Y%m%d_%H%M%S)-$SLUG
mkdir -p "$OUT"
export KBH_RUN_GROUP=$(basename "$OUT")
exec >> "$OUT/pipeline.log" 2>&1
log() { echo "=== $(date '+%F %T %Z') $*"; }
log "start host=$(hostname) gpu$GPU=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)"
log "harness=$HARNESS model=$MODEL effort=${EFFORT:-<harness default>} claude=$(claude --version 2>/dev/null) codex=$(codex --version 2>/dev/null) grok=$(grok --version 2>/dev/null | cut -d' ' -f2)"

CELLS=(mega:problems/02_kimi_linear_decode
       cuda:problems-rtxpro6000/01_glm52_fused_moe cuda:problems-rtxpro6000/02_deepseek_nsa
       cuda:problems-rtxpro6000/03_megaqwen_decode cuda:problems-rtxpro6000/04_grid_mingru_sps)
pids=()
for cell in "${CELLS[@]}"; do
    B=${cell%%:*}; P=${cell#*:}; L=$OUT/${B}_$(basename "$P").log
    ( cd "$R/benchmarks/$B" && ./scripts/run_hard.sh "$HARNESS" "$MODEL" "$P" ${EFFORT:+"$EFFORT"} ) > "$L" 2>&1 &
    pids+=($!); log "launched $B $(basename "$P") pid $!"
    sleep 30
done

# Gate 4 early warning: the served model id must be the requested cell (no silent swap).
sleep 240
for L in "$OUT"/*.log; do
    [ "$L" = "$OUT/pipeline.log" ] && continue
    rd=$(grep -m1 -oE 'Archive: +\S+' "$L" | awk '{print $2}')
    served=$(grep -ohE '"model":"[^"]+"|^model: \S+' "$rd/transcript.jsonl" "$rd/stderr.log" 2>/dev/null | sort | uniq -c | sort -rn | head -3 | tr -s ' \n' ' ')
    log "served $(basename "$L" .log): ${served:-none yet}"
done

for pid in "${pids[@]}"; do wait "$pid"; log "pid $pid exit $?"; done
log "all cells finished"

for B in mega cuda; do
    runs=()
    for L in "$OUT"/${B}_*.log; do
        rd=$(grep -m1 -oE 'Archive: +\S+' "$L" | awk '{print $2}')
        [ -f "$rd/result.json" ] && runs+=("$rd")
    done
    [ ${#runs[@]} -eq 0 ] && { log "no $B result.json to regrade"; continue; }
    DECK=problems; [ "$B" = cuda ] && DECK=problems-rtxpro6000
    log "regrade $B ${#runs[@]} runs deck=$DECK gpu=$GPU"
    ( cd "$R/benchmarks/$B" && KBH_REGRADE_GPU=$GPU KBH_REGRADE_DECK=$DECK ./scripts/regrade_sequential.sh "${runs[@]}" ) > "$OUT/regrade_$B.log" 2>&1
    log "regrade $B exit $?"
done

for rd in $(grep -ohE 'Archive: +\S+' "$OUT"/*_*.log | awk '{print $2}' | sort -u); do
    [ -f "$rd/result.json" ] || { log "NO result.json $(basename "$rd")"; continue; }
    log "$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); g=r.get("regrade") or {}; print(r["run_id"], "correct=%s" % r.get("correct"), "peak=%s" % r.get("peak_fraction"), "regraded=%s" % bool(g))' "$rd/result.json")"
done
log "PIPELINE_DONE"
touch "$OUT/DONE"
