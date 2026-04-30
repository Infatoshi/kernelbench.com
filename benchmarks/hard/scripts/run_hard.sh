#!/bin/bash
# Run one (harness, model, problem) combination.
#
# Usage:
#   ./scripts/run_hard.sh <harness> <model> <problem_dir> [reasoning_effort]
#
# Examples:
#   ./scripts/run_hard.sh claude claude-opus-4-7 problems/01_fp8_gemm
#   ./scripts/run_hard.sh codex gpt-5.5 problems/01_fp8_gemm xhigh
#   ./scripts/run_hard.sh kimi kimi-k2.6 problems/01_fp8_gemm
#   ./scripts/run_hard.sh ccr-claude glm-5.1 problems/01_fp8_gemm
#
# Archives everything to outputs/runs/<ts>_<harness>_<model>_<problem>/.

set -euo pipefail

# Pin CUDA 13 — /usr/local/cuda may still point at 12.8.
if [ -d /usr/local/cuda-13 ]; then
    export CUDA_HOME=/usr/local/cuda-13
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# Source API keys if the user has an env_vars file.
if [ -f "$HOME/.env_vars" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$HOME/.env_vars"
    set +a
fi

HARNESS="${1:?Usage: $0 <harness> <model> <problem_dir> [reasoning_effort]}"
MODEL="${2:?model required}"
PROBLEM_DIR="${3:?problem_dir required}"
REASONING_EFFORT="${4:-}"

PROBLEM_DIR="$(cd "$PROBLEM_DIR" && pwd)"
PROBLEM_NAME="$(basename "$PROBLEM_DIR")"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_SLUG="$(echo "$MODEL" | tr '/:[] ' '_')"
RUN_DIR="${REPO_ROOT}/outputs/runs/${TIMESTAMP}_${HARNESS}_${MODEL_SLUG}_${PROBLEM_NAME}"
mkdir -p "$RUN_DIR"

# Wall clock budget: 45 minutes per run. Override via BUDGET_SECONDS env var
# (e.g. BUDGET_SECONDS=300 for a quick smoke test).
BUDGET_SECONDS="${BUDGET_SECONDS:-2700}"

# --- Load the per-problem prompt ------------------------------------------
#
# Each problem has a PROMPT.txt in human voice that combines the task brief,
# shapes, forbidden ops, and workflow guidance into a single user-style
# message. The harness sends this directly as the prompt to the agent. No
# system/user split, no preamble concatenation.

PROMPT_FILE="${PROBLEM_DIR}/PROMPT.txt"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "PROMPT.txt missing for $PROBLEM_NAME" >&2
    exit 1
fi
PROMPT="$(cat "$PROMPT_FILE")"

# --- Clean the problem workspace of any prior solution --------------------
# problem.yaml and shapes.py stay in the workspace because check.py and
# benchmark.py import them at runtime; the prompt does not direct the model
# to read them.
TEMPLATE_FILES=(reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt)
is_template() {
    local n="$1"
    for t in "${TEMPLATE_FILES[@]}"; do
        [[ "$n" == "$t" ]] && return 0
    done
    return 1
}

shopt -s nullglob dotglob
for f in "$PROBLEM_DIR"/*; do
    base="$(basename "$f")"
    [[ "$base" == "." || "$base" == ".." ]] && continue
    if ! is_template "$base"; then
        rm -rf "$f"
    fi
done
shopt -u nullglob dotglob

# --- Run the harness ------------------------------------------------------

LOG_FILE="${RUN_DIR}/transcript.jsonl"
STDERR_FILE="${RUN_DIR}/stderr.log"

echo "========================================"
echo "KERNELBENCH-HARD RUN"
echo "========================================"
echo "Harness:    $HARNESS"
echo "Model:      $MODEL"
echo "Effort:     ${REASONING_EFFORT:-<default>}"
echo "Problem:    $PROBLEM_NAME"
echo "Workspace:  $PROBLEM_DIR"
echo "Archive:    $RUN_DIR"
echo "Budget:     ${BUDGET_SECONDS}s"
echo "========================================"

START_TIME=$(date +%s)
HARNESS_EXIT=0

case "$HARNESS" in
    claude)
        EFFORT_ARG=()
        if [ -n "$REASONING_EFFORT" ]; then
            EFFORT_ARG=(--effort "$REASONING_EFFORT")
        fi
        timeout "$BUDGET_SECONDS" claude \
            --dangerously-skip-permissions \
            --print --verbose \
            --output-format stream-json \
            --model "$MODEL" \
            "${EFFORT_ARG[@]}" \
            --add-dir "$PROBLEM_DIR" \
            -p "$PROMPT" \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    ccr-claude)
        # Claude Code routed via ccr-rust to a non-Anthropic provider.
        # Assumes ccr-rust is running locally and ANTHROPIC_BASE_URL points at it.
        # Model name is the upstream lab's model ID (glm-5.1, deepseek-v4-flash, etc.).
        timeout "$BUDGET_SECONDS" \
            env ANTHROPIC_BASE_URL="${CCR_BASE_URL:-http://127.0.0.1:3456}" \
            claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --model "$MODEL" \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    codex)
        EFFORT_ARG=()
        if [ -n "$REASONING_EFFORT" ]; then
            EFFORT_ARG=(-c "model_reasoning_effort=\"$REASONING_EFFORT\"")
        fi
        timeout "$BUDGET_SECONDS" codex exec \
            -m "$MODEL" \
            "${EFFORT_ARG[@]}" \
            --dangerously-bypass-approvals-and-sandbox \
            --skip-git-repo-check \
            -C "$PROBLEM_DIR" \
            "$PROMPT" \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?

        # Codex writes its rich session JSONL to ~/.codex/sessions/YYYY/MM/DD/
        # (local date). Locate by session_id printed to stderr — DO NOT pick the
        # most-recently-modified file, since codex 0.125.0 touches old session
        # files when scanning its thread state DB and that's misleading.
        CODEX_SID=$(grep -oP 'session id: \K[0-9a-f-]+' "$STDERR_FILE" | head -1)
        if [ -n "$CODEX_SID" ]; then
            CODEX_SESS=$(find "$HOME/.codex/sessions" -name "*${CODEX_SID}*.jsonl" 2>/dev/null | head -1)
            if [ -n "$CODEX_SESS" ]; then
                cp "$CODEX_SESS" "$RUN_DIR/codex_session.jsonl"
                echo "archived codex session: $CODEX_SESS -> $RUN_DIR/codex_session.jsonl"
            else
                echo "WARN: codex session_id $CODEX_SID found in stderr but no matching JSONL on disk"
            fi
        else
            echo "WARN: could not parse codex session_id from stderr"
        fi
        ;;

    kimi)
        echo "$PROMPT" | timeout "$BUDGET_SECONDS" kimi \
            -w "$PROBLEM_DIR" \
            --print \
            --output-format stream-json \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    opencode)
        # OpenCode SST with custom OpenAI-shape providers (deepseek, zai, minimax).
        # Provider/model pair encoded as MODEL="provider/model-id" e.g.
        # "deepseek/deepseek-v4-pro" or "zai/glm-5.1".
        ( cd "$PROBLEM_DIR" && timeout "$BUDGET_SECONDS" opencode run \
            --pure --format json -m "$MODEL" "$PROMPT" \
            </dev/null > "$LOG_FILE" 2> "$STDERR_FILE" ) || HARNESS_EXIT=$?
        ;;

    *)
        echo "Unknown harness: $HARNESS" >&2
        echo "Supported: claude, ccr-claude, codex, kimi, opencode" >&2
        exit 1
        ;;
esac

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# --- Detect whether the harness session ran to completion ----------------
#
# A run is INCOMPLETE if the harness exited from SIGTERM (timeout=124) OR if
# the transcript is missing its terminal marker. Markers per harness:
#   claude / ccr-claude:  {"type":"result"} (final summary with usage)
#   codex:                {"payload":{"type":"task_complete"}}
#   cursor:               {"type":"result"} (final usage block)
#   kimi:                 no canonical terminal event — treat exit code as truth
#
# We surface this as session_complete=true|false in result.json so the viewer
# can render an INCOMPLETE banner and downstream aggregation can exclude
# partial runs from scoring.

CHECK_FILE="$LOG_FILE"
if [ "$HARNESS" = "codex" ] && [ -f "$RUN_DIR/codex_session.jsonl" ]; then
    CHECK_FILE="$RUN_DIR/codex_session.jsonl"
fi

SESSION_COMPLETE=true
case "$HARNESS" in
    claude|ccr-claude|cursor)
        if ! grep -q '"type":"result"' "$CHECK_FILE" 2>/dev/null; then
            SESSION_COMPLETE=false
        fi
        ;;
    codex)
        if ! grep -q '"type":"task_complete"' "$CHECK_FILE" 2>/dev/null; then
            SESSION_COMPLETE=false
        fi
        ;;
    kimi|opencode)
        # No reliable terminal marker; trust the exit code.
        if [ "$HARNESS_EXIT" -ne 0 ]; then
            SESSION_COMPLETE=false
        fi
        ;;
esac

# timeout(1) returns 124 on SIGTERM kill — always means partial.
if [ "$HARNESS_EXIT" -eq 124 ]; then
    SESSION_COMPLETE=false
fi

if [ "$SESSION_COMPLETE" = "false" ]; then
    echo "WARN: harness session is INCOMPLETE (exit=$HARNESS_EXIT). Transcript usable but partial."
fi

# --- Post-run: correctness + benchmark + archive --------------------------

HAS_SOLUTION=false
CORRECT=false
SCORE="null"

if [ -f "$PROBLEM_DIR/solution.py" ]; then
    HAS_SOLUTION=true
    CHECK_LOG="$RUN_DIR/check.log"
    BENCH_LOG="$RUN_DIR/benchmark.log"

    echo "Running check.py..."
    (cd "$PROBLEM_DIR" && timeout 180 uv run python check.py) > "$CHECK_LOG" 2>&1 || true

    if grep -q "^PASS" "$CHECK_LOG"; then
        CORRECT=true
        echo "Running benchmark.py..."
        # Some problems (KDA chunked recurrence, large-vocab softmax, sonic-MoE)
        # have references that loop in Python, so 20 perf trials × 4 variants ×
        # 5 shapes can take 5-10 min. Generous budget.
        (cd "$PROBLEM_DIR" && timeout 1800 uv run python benchmark.py) > "$BENCH_LOG" 2>&1 || true
        SCORE=$(grep -oP 'peak_fraction:\s*\K[0-9.]+' "$BENCH_LOG" | head -1 || echo "null")
    fi
fi

# Extract token usage from the transcript so we have an apples-to-apples
# count for cost comparison even when the harness uses a coding-plan billing
# (which hides per-call USD).
USAGE_JSON="$(uv run --quiet python "$REPO_ROOT/scripts/extract_usage.py" "$RUN_DIR" "$HARNESS" 2>/dev/null || echo '{}')"

cat > "$RUN_DIR/result.json" <<JSON
{
    "problem": "$PROBLEM_NAME",
    "harness": "$HARNESS",
    "model": "$MODEL",
    "reasoning_effort": "$REASONING_EFFORT",
    "has_solution": $HAS_SOLUTION,
    "correct": $CORRECT,
    "peak_fraction": $SCORE,
    "elapsed_seconds": $ELAPSED,
    "harness_exit_code": $HARNESS_EXIT,
    "session_complete": $SESSION_COMPLETE,
    "usage": $USAGE_JSON
}
JSON

# Archive solution + any scratch
if [ -f "$PROBLEM_DIR/solution.py" ]; then
    cp "$PROBLEM_DIR/solution.py" "$RUN_DIR/solution.py"
fi

SCRATCH_DIR="$RUN_DIR/scratch"
shopt -s nullglob dotglob
for f in "$PROBLEM_DIR"/*; do
    base="$(basename "$f")"
    [[ "$base" == "." || "$base" == ".." ]] && continue
    [[ "$base" == "solution.py" ]] && continue
    if ! is_template "$base"; then
        mkdir -p "$SCRATCH_DIR"
        cp -r "$f" "$SCRATCH_DIR/"
    fi
done
shopt -u nullglob dotglob

# Clean the problem workspace for the next run
shopt -s nullglob dotglob
for f in "$PROBLEM_DIR"/*; do
    base="$(basename "$f")"
    [[ "$base" == "." || "$base" == ".." ]] && continue
    if ! is_template "$base"; then
        rm -rf "$f"
    fi
done
shopt -u nullglob dotglob

STATUS="ERR"
if $CORRECT; then
    STATUS="OK score=$SCORE"
elif $HAS_SOLUTION; then
    STATUS="FAIL (check failed)"
fi

echo "========================================"
echo "[$STATUS] $PROBLEM_NAME (${ELAPSED}s)"
echo "Archive: $RUN_DIR"
echo "========================================"
