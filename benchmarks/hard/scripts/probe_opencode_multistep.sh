#!/usr/bin/env bash
# Multi-step tool-use probe for opencode routes.
#
# One-turn smokes cannot catch the OpenAI-compatible adapter stall documented
# in DEVLOG 2026-06-09: reasoning models can hang on the first long generation
# after tool results enter context. A toy read->write probe is NOT sufficient
# to trigger it (zai/glm-5.1 passes that but stalls on the real prompt), so
# this probe replays the real shape: a problem template workspace plus the
# problem's PROMPT.txt.
#
# Success = the route gets PAST the read phase: any write/edit/bash/todowrite
# tool event after file reads, or a solution.py on disk. Correctness of the
# solution is irrelevant here; this is an infra gate, not a capability test.
#
# Usage: probe_opencode_multistep.sh <model> [timeout_seconds] [opencode_bin]
set -u

MODEL="${1:?model required, e.g. openrouter-alibaba/qwen/qwen3.7-max}"
TIMEOUT_SECONDS="${2:-420}"
OPENCODE_BIN="${3:-${KBH_OPENCODE_BIN:-$HOME/.opencode/bin/opencode}}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROBLEM="${KBH_PROBE_PROBLEM:-05_topk_bitonic}"

if [ -f "$HOME/.env_vars" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.env_vars"
fi

WORK="$(mktemp -d /tmp/kbh_oc_probe.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/problems/$PROBLEM"
for t in reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt; do
    if [ -e "$REPO_ROOT/problems/$PROBLEM/$t" ]; then
        cp -p "$REPO_ROOT/problems/$PROBLEM/$t" "$WORK/problems/$PROBLEM/$t"
    fi
done

LOG="$WORK/probe.log"
( cd "$WORK/problems/$PROBLEM" && timeout "$TIMEOUT_SECONDS" "$OPENCODE_BIN" run \
    --pure --format json -m "$MODEL" "$(cat PROMPT.txt)" ) > "$LOG" 2>&1
exit_code=$?

mutating=$(grep -cE '"tool":"(write|edit|bash|patch|todowrite)"' "$LOG" || true)
reads=$(grep -cE '"tool":"(read|glob|grep|list)"' "$LOG" || true)
solution="no"
[ -f "$WORK/problems/$PROBLEM/solution.py" ] && solution="yes"

if [ "$mutating" -gt 0 ] || [ "$solution" = "yes" ]; then
    echo "PROBE_OK $MODEL exit=$exit_code reads=$reads mutating=$mutating solution=$solution"
    exit 0
fi
echo "PROBE_FAIL $MODEL exit=$exit_code reads=$reads mutating=0 solution=no (stalled after read phase or never started)"
tail -c 300 "$LOG"
exit 1
