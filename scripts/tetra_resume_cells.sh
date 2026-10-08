#!/bin/bash
# Resume Claude release cells that a provider rate limit cut off, inside their own archives, until
# each session ends on its own; then regrade the given cells sequentially on the same GPU.
#
#   scripts/tetra_resume_cells.sh <gpu> <model> <effort> <bench:run_dir>... -- <bench:run_dir>...
#   (cells before -- are resumed; cells after -- are already finished and only join the regrade)
#
# One continuation per cell at a time (overlapping copies of one session corrupt the trace,
# benchmarks/cuda/DEVLOG.md 2026-09-22). When a leg ends on a rate-limit rejection the driver
# waits for the login's reset and resumes again. Bills tetra's own Claude login. Holds an
# overnight-compute lease on gpu<N>. Log: runs/resume-<stamp>/pipeline.log.
set -uo pipefail
GPU=${1:?gpu}; MODEL=${2:?model}; EFFORT=${3:?effort}; shift 3
R=$(cd "$(dirname "$0")/.." && pwd)
export PATH=$HOME/.local/bin:/usr/local/cuda-13.3/bin:$PATH
if [ -z "${TETRA_RESUME_LEASED:-}" ] && command -v overnight-compute >/dev/null; then
    [ -z "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)" ] || { echo "STOP: GPU $GPU busy" >&2; exit 1; }
    TETRA_RESUME_LEASED=1 exec overnight-compute run --agent "kb-resume-gpu$GPU-$MODEL" --resource "gpu$GPU" \
        --ttl 30m --heartbeat 5m --timeout 2m -- "$0" "$GPU" "$MODEL" "$EFFORT" "$@"
fi
RESUME=(); DONE_CELLS=(); seen=0
for a in "$@"; do [ "$a" = -- ] && { seen=1; continue; }; [ $seen = 0 ] && RESUME+=("$a") || DONE_CELLS+=("$a"); done
export KBH_GPU=$GPU KBH_CUDA_HOME=/usr/local/cuda-13
[ -f ~/.env_vars ] && set -a && . ~/.env_vars && set +a
unset CLAUDE_CODE_OAUTH_TOKEN
OUT=$R/runs/resume-$(date +%Y%m%d_%H%M%S)-$MODEL; mkdir -p "$OUT"
exec >> "$OUT/pipeline.log" 2>&1
log() { echo "=== $(date '+%F %T %Z') $*"; }
PROMPT_RESUME="Your session was cut off by a provider rate limit; nothing in the workspace changed. Continue the task from where you stopped."

reset_wait() {   # sleep until the login's current limit window resets (plus a minute)
    local info at now
    info=$(timeout 90 claude -p --model "$MODEL" --output-format stream-json --verbose ok </dev/null 2>/dev/null | grep -oE '"status":"[a-z_]+","resetsAt":[0-9]+' | head -1)
    case "$info" in *'"status":"allowed'*) return 0;; esac
    at=$(echo "$info" | grep -oE '[0-9]+$'); now=$(date +%s)
    [ -n "$at" ] && [ "$at" -gt "$now" ] && { log "rate limited until $(date -d @"$at" '+%T'); waiting"; sleep $((at - now + 60)); } || sleep 600
}

cut_off() {   # did the last leg end on a rate-limit rejection?
    tail -n 5 "$1/transcript.jsonl" 2>/dev/null | grep -q '"rejected"\|rate_limit'
}

resume_cell() {
    local B=${1%%:*} D=${1#*:} P S leg=0
    P=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["problem"])' "$D/result.json")
    while cut_off "$D"; do
        leg=$((leg + 1)); reset_wait
        S=$(grep -oE '"session_id":"[^"]+"' "$D/transcript.jsonl" | tail -1 | cut -d'"' -f4)
        [ -n "$S" ] || { log "$(basename "$D"): no session id, giving up"; return 1; }
        log "$(basename "$D"): resume leg $leg session $S"
        DECK=problems; [ "$B" = cuda ] && DECK=problems-rtxpro6000
        ( cd "$R/benchmarks/$B" && KBH_RESUME_RUN_DIR=$D KBH_RESUME_SESSION=$S KBH_RESUME_PROMPT=$PROMPT_RESUME \
            ./scripts/run_hard.sh claude "$MODEL" "$DECK/$P" "$EFFORT" ) >> "$OUT/$(basename "$D").log" 2>&1
        log "$(basename "$D"): leg $leg exit $?"
    done
    log "$(basename "$D"): session ended on its own"
}

pids=()
for c in "${RESUME[@]}"; do resume_cell "$c" & pids+=($!); sleep 30; done
for p in "${pids[@]}"; do wait "$p"; done
log "all resumed cells finished"

for B in mega cuda; do
    runs=()
    for c in "${RESUME[@]}" "${DONE_CELLS[@]}"; do [ "${c%%:*}" = "$B" ] && [ -f "${c#*:}/result.json" ] && runs+=("${c#*:}"); done
    [ ${#runs[@]} -eq 0 ] && continue
    DECK=problems; [ "$B" = cuda ] && DECK=problems-rtxpro6000
    log "regrade $B ${#runs[@]} runs deck=$DECK gpu=$GPU"
    ( cd "$R/benchmarks/$B" && KBH_REGRADE_GPU=$GPU KBH_REGRADE_DECK=$DECK ./scripts/regrade_sequential.sh "${runs[@]}" ) > "$OUT/regrade_$B.log" 2>&1
    log "regrade $B exit $?"
done
for c in "${RESUME[@]}" "${DONE_CELLS[@]}"; do
    log "$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); g=r.get("regrade") or {}; print(r["run_id"], "correct=%s" % r.get("correct"), "peak=%s" % r.get("peak_fraction"), "regraded=%s" % bool(g))' "${c#*:}/result.json")"
done
log "PIPELINE_DONE"; touch "$OUT/DONE"
