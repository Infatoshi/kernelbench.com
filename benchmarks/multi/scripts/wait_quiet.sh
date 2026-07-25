#!/usr/bin/env bash
# Block until this node has been free of other CUDA tenants for a SUSTAINED
# window, then exit 0. Chain it before a wave:
#
#   ./scripts/wait_quiet.sh && ./scripts/sweep_wave.sh grok grok-4.5 high
#
# Why sustained and not "free right now": hades shares with a co-tenant whose
# jobs cycle every few tens of minutes. Launching into a momentary gap means an
# agent starts, allocates, and then hits CUDA OOM the moment the neighbour comes
# back — which produces a garbage session, and (before the kill guard) tempted
# agents into killing processes that were not theirs. A wave needs hours, so we
# want evidence the node is actually idle, not a single lucky sample.
#
# Env: KBM_QUIET_MB (per-GPU threshold, default 2048)
#      KBM_QUIET_MINUTES (sustained window, default 10)
#      KBM_QUIET_TIMEOUT_MINUTES (give up, default 0 = wait forever)
set -euo pipefail

THRESH="${KBM_QUIET_MB:-2048}"
NEED="${KBM_QUIET_MINUTES:-10}"
TIMEOUT="${KBM_QUIET_TIMEOUT_MINUTES:-0}"
INTERVAL=60

streak=0
waited=0
while true; do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | awk -v t="$THRESH" '$1>t{c++} END{print c+0}')
    if [ "$busy" = "0" ]; then
        streak=$((streak + 1))
        echo "[wait_quiet] quiet $streak/$NEED min ($(date -u +%FT%TZ))"
        if [ "$streak" -ge "$NEED" ]; then
            echo "[wait_quiet] node quiet for ${NEED}m — proceeding"
            exit 0
        fi
    else
        [ "$streak" -gt 0 ] && echo "[wait_quiet] neighbour returned on $busy GPU(s); streak reset"
        streak=0
    fi
    waited=$((waited + 1))
    if [ "$TIMEOUT" != "0" ] && [ "$waited" -ge "$TIMEOUT" ]; then
        echo "[wait_quiet] timed out after ${TIMEOUT}m with the node still busy" >&2
        exit 2
    fi
    sleep "$INTERVAL"
done
