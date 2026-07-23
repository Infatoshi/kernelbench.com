#!/usr/bin/env bash
# KernelBench-Multi agent harness — runs ON the 4xH100 node.
#
#   ./scripts/run_agent.sh <harness> <model> <problem_name> [effort]
#   e.g. ./scripts/run_agent.sh grok grok-4.5 01_allreduce_residual high
#
# Concurrency model: every agent session gets an isolated workspace, but the
# node has ONE 4-GPU fabric — any GPU-facing command (python/torchrun/nvcc/
# profilers) from ANY session must hold the node-wide lock. We prepend
# $RUN_DIR/bin wrappers to the agent's PATH; each wrapper flocks
# $KBM_GPU_LOCK_DIR/gpu.lock and then execs the real binary, holding the lock
# for the process lifetime. KBM_GPU_LOCK_HELD=1 makes wrappers reentrant so a
# locked python can spawn nvcc without deadlocking (same design as hard's
# per-run wrappers, but the lock is NODE-WIDE by default — on a multi-GPU bench
# every session needs all 4 GPUs, so per-bench locks would be meaningless).
# nvidia-smi is deliberately NOT wrapped: it is read-only and agents poll it.
#
# Env: BUDGET_SECONDS (default 0 = unlimited), KBM_GPU_LOCK_DIR,
#      KBM_SKIP_GRADE=1 (launch only, grade later).
set -euo pipefail

HARNESS="${1:?harness (grok|zai-claude)}"
MODEL="${2:?model}"
PROBLEM="${3:?problem name, e.g. 01_allreduce_residual}"
EFFORT="${4:-}"

BENCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DECK="$BENCH_ROOT/problems-h100x4"
[ -d "$DECK/$PROBLEM" ] || { echo "unknown problem: $PROBLEM" >&2; exit 2; }

LOCK_DIR="${KBM_GPU_LOCK_DIR:-$HOME/kbm/outputs/gpu_lock}"
mkdir -p "$LOCK_DIR"
LOCKFILE="$LOCK_DIR/gpu.lock"

RUN_ID="$(date +%Y%m%d_%H%M%S)_${HARNESS}_${MODEL//\//-}_${PROBLEM}"
RUN_DIR="$HOME/kbm/outputs/runs/$RUN_ID"
WS="$RUN_DIR/ws"
PROBLEM_DIR="$WS/problems-h100x4/$PROBLEM"
mkdir -p "$WS/problems-h100x4" "$RUN_DIR/bin"
cp -r "$DECK/$PROBLEM" "$PROBLEM_DIR"
rm -f "$PROBLEM_DIR/solution.py"
ln -sfn "$BENCH_ROOT/src" "$WS/src"

# --- GPU-lock wrappers -------------------------------------------------------
for tool in python python3 torchrun nvcc ncu nsys; do
    real="$(command -v "$tool" 2>/dev/null || true)"
    [ -n "$real" ] || continue
    cat > "$RUN_DIR/bin/$tool" <<WRAP
#!/usr/bin/env bash
if [ "\${KBM_GPU_LOCK_HELD:-0}" = "1" ]; then exec "$real" "\$@"; fi
exec 9>>"$LOCKFILE"
flock 9
export KBM_GPU_LOCK_HELD=1
exec "$real" "\$@"
WRAP
    chmod +x "$RUN_DIR/bin/$tool"
done

PROMPT="$(cat "$PROBLEM_DIR/PROMPT.txt")"
BUDGET="${BUDGET_SECONDS:-0}"
TIMEOUT_CMD=()
[ "$BUDGET" != "0" ] && TIMEOUT_CMD=(timeout "$BUDGET")

echo "[run_agent] $RUN_ID budget=${BUDGET}s lock=$LOCKFILE" | tee "$RUN_DIR/meta.log"
HARNESS_EXIT=0
case "$HARNESS" in
    grok)
        EFFORT_ARG=()
        [ -n "$EFFORT" ] && EFFORT_ARG=(--effort "$EFFORT")
        PATH="$RUN_DIR/bin:$PATH" "${TIMEOUT_CMD[@]}" grok \
            --cwd "$PROBLEM_DIR" \
            --always-approve \
            --permission-mode bypassPermissions \
            --no-memory \
            --output-format streaming-json \
            --model "$MODEL" \
            "${EFFORT_ARG[@]}" \
            -p "$PROMPT" \
            > "$RUN_DIR/agent.log" 2> "$RUN_DIR/agent.err" || HARNESS_EXIT=$?
        ;;
    zai-claude)
        # shellcheck disable=SC1090
        . "$HOME/.kbm_env"
        export ANTHROPIC_AUTH_TOKEN="$ZAI_API_KEY"
        export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
        export ANTHROPIC_MODEL="$MODEL" ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
        export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
        export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 API_TIMEOUT_MS=3000000
        export CLAUDE_CODE_MAX_RETRIES=100 CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
        ( cd "$PROBLEM_DIR" && PATH="$RUN_DIR/bin:$PATH" "${TIMEOUT_CMD[@]}" claude \
            --dangerously-skip-permissions --print --verbose \
            --output-format stream-json \
            --settings '{"fastMode":false,"alwaysThinkingEnabled":true}' \
            --model opus \
            --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
            -p "$PROMPT" ) \
            > "$RUN_DIR/agent.log" 2> "$RUN_DIR/agent.err" || HARNESS_EXIT=$?
        ;;
    *)
        echo "unknown harness: $HARNESS" >&2; exit 2 ;;
esac
echo "[run_agent] agent exit=$HARNESS_EXIT" | tee -a "$RUN_DIR/meta.log"

# --- grade (serialized through the same lock via wrappers) -------------------
if [ "${KBM_SKIP_GRADE:-0}" != "1" ]; then
    if [ -f "$PROBLEM_DIR/solution.py" ]; then
        ( cd "$PROBLEM_DIR" && PATH="$RUN_DIR/bin:$PATH" python3 check.py ) \
            > "$RUN_DIR/check.log" 2>&1 || true
        ( cd "$PROBLEM_DIR" && PATH="$RUN_DIR/bin:$PATH" python3 benchmark.py ) \
            > "$RUN_DIR/benchmark.log" 2>&1 || true
        {
            echo "check: $(tail -1 "$RUN_DIR/check.log")"
            grep -E "peak_fraction:|RESULT:" "$RUN_DIR/benchmark.log" || true
        } | tee -a "$RUN_DIR/meta.log"
    else
        echo "no_solution" | tee -a "$RUN_DIR/meta.log"
    fi
fi
echo "[run_agent] done $RUN_ID" | tee -a "$RUN_DIR/meta.log"
