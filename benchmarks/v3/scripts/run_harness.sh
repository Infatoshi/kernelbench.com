#!/bin/bash
# Run a model through Droid/Codex/Claude Code on all RTX 3090 workspaces.
# Usage: ./scripts/run_harness.sh <harness> <model_flag> [workspace_dir]
#
# Examples:
#   ./scripts/run_harness.sh droid "custom:MiniMax-M2.7-[OpenRouter]-2"
#   ./scripts/run_harness.sh codex "gpt-5.4"
#   ./scripts/run_harness.sh claude ""
#   ./scripts/run_harness.sh kimi ""

set -euo pipefail

# Pin CUDA 13 for agent subprocesses. The /usr/local/cuda symlink may still
# point to 12.8 on some machines; /usr/local/cuda-13 always resolves to the
# 13.x toolkit we need for SM120 / SM100 targets (flashinfer, CUTLASS 4.x,
# Blackwell compile flags). Override the pinned toolkit dir on other machines
# with KBH_CUDA_HOME (default /usr/local/cuda-13).
KBH_CUDA_HOME="${KBH_CUDA_HOME:-/usr/local/cuda-13}"
if [ -d "$KBH_CUDA_HOME" ]; then
    export CUDA_HOME="$KBH_CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
fi

HARNESS="${1:?Usage: $0 <droid|codex|claude|kimi> <model_flag>}"
MODEL="${2:-}"
WS_ROOT="${3:-workspaces/rtx_pro_6000}"
RESULTS_DIR="outputs/harness_eval"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_DIR}/run_${TIMESTAMP}_${HARNESS}_$(echo "$MODEL" | tr '/:[] ' '_')"

mkdir -p "$RUN_DIR"

PROMPT="Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify correctness. If it passes, run benchmark.py to measure speedup. Iterate until check.py passes or you've exhausted your ideas."

echo "========================================"
echo "HARNESS EVAL"
echo "========================================"
echo "Harness:   $HARNESS"
echo "Model:     $MODEL"
echo "Workspace: $WS_ROOT"
echo "Run dir:   $RUN_DIR"
echo "Problems:  $(ls -d "$WS_ROOT"/*/ 2>/dev/null | wc -l)"
echo "========================================"
echo ""

TOTAL=0
CORRECT=0
COMPILED=0
BENCHMARKED=0

TEMPLATE_FILES=(reference.py CLAUDE.md check.py benchmark.py)

is_template_file() {
    local name="$1"
    for t in "${TEMPLATE_FILES[@]}"; do
        [[ "$name" == "$t" ]] && return 0
    done
    return 1
}

for ws in "$WS_ROOT"/*/; do
    PROBLEM=$(basename "$ws")
    TOTAL=$((TOTAL + 1))
    echo "[START] $PROBLEM ($TOTAL)"

    PROBLEM_DIR="$RUN_DIR/$PROBLEM"
    mkdir -p "$PROBLEM_DIR"

    # Clean workspace of anything left over from a previous run (non-template files only)
    shopt -s nullglob dotglob
    for f in "$ws"*; do
        base=$(basename "$f")
        [[ "$base" == "." || "$base" == ".." ]] && continue
        if ! is_template_file "$base"; then
            rm -rf "$f"
        fi
    done
    shopt -u nullglob dotglob

    LOG_FILE="$PROBLEM_DIR/transcript.jsonl"
    RESULT_FILE="$PROBLEM_DIR/result.json"

    START_TIME=$(date +%s)

    # Run the harness (all modes capture full transcripts to LOG_FILE)
    case "$HARNESS" in
        droid)
            timeout 2700 droid exec -m "$MODEL" -o stream-json --skip-permissions-unsafe --cwd "$ws" "$PROMPT" \
                > "$LOG_FILE" 2>&1 || true
            ;;
        codex)
            timeout 2700 codex exec -m "$MODEL" --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C "$ws" "$PROMPT" \
                > "$LOG_FILE" 2>&1 || true
            ;;
        claude)
            timeout 2700 claude --dangerously-skip-permissions --print --verbose --output-format stream-json --add-dir "$ws" \
                -p "You are working in $ws. $PROMPT" \
                > "$LOG_FILE" 2>&1 || true
            ;;
        kimi)
            echo "$PROMPT" | timeout 2700 kimi -w "$ws" --print --output-format stream-json \
                > "$LOG_FILE" 2>&1 || true
            ;;
        *)
            echo "Unknown harness: $HARNESS"
            exit 1
            ;;
    esac

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    # Check results
    HAS_SOLUTION=false
    CHECK_PASS=false
    SPEEDUP="null"

    if [ -f "$ws/solution.py" ]; then
        HAS_SOLUTION=true
        COMPILED=true

        # Run check.py independently to verify
        CHECK_OUT=$(cd "$ws" && timeout 120 python check.py 2>&1) || true
        if echo "$CHECK_OUT" | grep -q "^PASS"; then
            CHECK_PASS=true
            CORRECT=$((CORRECT + 1))

            # Run benchmark.py
            BENCH_OUT=$(cd "$ws" && timeout 300 python benchmark.py 2>&1) || true
            SPEEDUP=$(echo "$BENCH_OUT" | grep -oP 'Speedup:\s+\K[0-9.]+' || echo "null")
            if [ "$SPEEDUP" != "null" ]; then
                BENCHMARKED=$((BENCHMARKED + 1))
            fi
        fi
    fi

    # Write result JSON
    cat > "$RESULT_FILE" <<JSONEOF
{
    "problem": "$PROBLEM",
    "harness": "$HARNESS",
    "model": "$MODEL",
    "has_solution": $HAS_SOLUTION,
    "correct": $CHECK_PASS,
    "speedup": $SPEEDUP,
    "elapsed_seconds": $ELAPSED
}
JSONEOF

    # Archive solution + any scratch artifacts the agent created
    if [ -f "$ws/solution.py" ]; then
        cp "$ws/solution.py" "$PROBLEM_DIR/solution.py"
    fi

    SCRATCH_DIR="$PROBLEM_DIR/scratch"
    shopt -s nullglob dotglob
    for f in "$ws"*; do
        base=$(basename "$f")
        [[ "$base" == "." || "$base" == ".." ]] && continue
        [[ "$base" == "solution.py" ]] && continue
        if ! is_template_file "$base"; then
            mkdir -p "$SCRATCH_DIR"
            cp -r "$f" "$SCRATCH_DIR/"
        fi
    done
    shopt -u nullglob dotglob

    # Clean workspace for next model (leave only template files)
    shopt -s nullglob dotglob
    for f in "$ws"*; do
        base=$(basename "$f")
        [[ "$base" == "." || "$base" == ".." ]] && continue
        if ! is_template_file "$base"; then
            rm -rf "$f"
        fi
    done
    shopt -u nullglob dotglob

    STATUS="ERR"
    if $CHECK_PASS; then
        STATUS="OK ${SPEEDUP}x"
    elif $HAS_SOLUTION; then
        STATUS="FAIL (check failed)"
    fi

    echo "[$STATUS] $PROBLEM (${ELAPSED}s)"
    echo ""
done

echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Total:       $TOTAL"
echo "Has solution: $COMPILED"
echo "Correct:     $CORRECT"
echo "Benchmarked: $BENCHMARKED"
echo "Results:     $RUN_DIR"
echo "========================================"
