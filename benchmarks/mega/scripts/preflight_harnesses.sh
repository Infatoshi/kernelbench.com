#!/bin/bash
# Cheap provider/harness preflight before an expensive KernelBench-Mega sweep.
#
# This intentionally uses tiny text prompts instead of a benchmark problem. It
# verifies auth, model routing, streaming JSON shape, and obvious quota/rate
# failures. It does not replace a smoke `run_hard.sh` run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$HOME/.env_vars" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$HOME/.env_vars"
    set +a
fi

OUT_DIR="${KBH_PREFLIGHT_DIR:-$REPO_ROOT/outputs/preflight/preflight_$(date +%Y%m%d_%H%M%S)}"
TIMEOUT_SECONDS="${KBH_PREFLIGHT_TIMEOUT_SECONDS:-120}"
CLAUDE_KBH_SETTINGS="${CLAUDE_KBH_SETTINGS:-{\"fastMode\":false,\"alwaysThinkingEnabled\":true}}"
PROMPT="${KBH_PREFLIGHT_PROMPT:-Reply with exactly KBH_PREFLIGHT_OK and no other text.}"
mkdir -p "$OUT_DIR"

ROWS=(
    "codex_gpt55_xhigh|codex|gpt-5.5|xhigh"
    "claude_opus47_max|claude|claude-opus-4-7|max"
    "zai_claude_glm51|zai-claude|glm-5.1|"
    "opencode_glm51|opencode|zai/glm-5.1|"
    "cursor_composer25fast|cursor|composer-2.5-fast|"
    "grok_grokbuild_max|grok|grok-build|max"
    "opencode_qwen37max|opencode|openrouter-alibaba/qwen/qwen3.7-max|"
    "opencode_gemini35flash|opencode|openrouter-google-ai-studio/google/gemini-3.5-flash|"
)

if [ "${KBH_USE_DIRECT_GEMINI:-0}" = "1" ]; then
    ROWS+=("gemini_gemini35flash|gemini|gemini-3.5-flash|")
fi

if [ "${KBH_USE_MINIMAX_M3_CLAUDE:-0}" = "1" ]; then
    ROWS+=("minimax_m3_claude|minimax-claude|MiniMax-M3|")
fi

if [ "${KBH_SKIP_OPENROUTER:-0}" = "1" ]; then
    FILTERED_ROWS=()
    for row in "${ROWS[@]}"; do
        IFS='|' read -r _name _harness model _effort <<< "$row"
        if [[ "$model" == openrouter-* ]]; then
            continue
        fi
        FILTERED_ROWS+=("$row")
    done
    ROWS=("${FILTERED_ROWS[@]}")
fi

SUMMARY="$OUT_DIR/summary.tsv"
printf 'name\tharness\tmodel\teffort\texit_code\tok\telapsed_seconds\tlog\n' > "$SUMMARY"

run_one() {
    local name="$1"
    local harness="$2"
    local model="$3"
    local effort="$4"
    local log="$OUT_DIR/$name.log"
    local start end elapsed exit_code ok
    start="$(date +%s)"
    exit_code=0

    case "$harness" in
        claude)
            local effort_arg=()
            if [ -n "$effort" ]; then
                effort_arg=(--effort "$effort")
            fi
            timeout "$TIMEOUT_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --max-budget-usd "${KBH_PREFLIGHT_CLAUDE_MAX_BUDGET_USD:-0.25}" \
                --model "$model" \
                "${effort_arg[@]}" \
                -p "$PROMPT" \
                > "$log" 2>&1 || exit_code=$?
            ;;
        zai-claude)
            if [ -z "${ZAI_API_KEY:-}" ]; then
                echo "ZAI_API_KEY missing" > "$log"
                exit_code=2
            else
                (
                    export ANTHROPIC_AUTH_TOKEN="$ZAI_API_KEY"
                    export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
                    export API_TIMEOUT_MS="${API_TIMEOUT_MS:-300000}"
                    export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS="${CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS:-1}"
                    export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-3}"
                    export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-4096}"
                    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$model"
                    export ANTHROPIC_DEFAULT_SONNET_MODEL="$model"
                    export ANTHROPIC_DEFAULT_OPUS_MODEL="$model"
                    timeout "$TIMEOUT_SECONDS" claude \
                        --dangerously-skip-permissions \
                        --print --verbose \
                        --output-format stream-json \
                        --settings "$CLAUDE_KBH_SETTINGS" \
                        --model "${ZAI_CLAUDE_ALIAS:-opus}" \
                        --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                        -p "$PROMPT" \
                    > "$log" 2>&1
                ) || exit_code=$?
            fi
            ;;
        minimax-claude)
            if [ -z "${MINIMAX_API_KEY:-}" ]; then
                echo "MINIMAX_API_KEY missing" > "$log"
                exit_code=2
            else
                (
                    export ANTHROPIC_AUTH_TOKEN="$MINIMAX_API_KEY"
                    export ANTHROPIC_BASE_URL="${MINIMAX_ANTHROPIC_BASE_URL:-https://api.minimax.io/anthropic}"
                    export API_TIMEOUT_MS="${API_TIMEOUT_MS:-300000}"
                    export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}"
                    export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-3}"
                    export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-4096}"
                    export ANTHROPIC_MODEL="$model"
                    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$model"
                    export ANTHROPIC_DEFAULT_SONNET_MODEL="$model"
                    export ANTHROPIC_DEFAULT_OPUS_MODEL="$model"
                    timeout "$TIMEOUT_SECONDS" claude \
                        --dangerously-skip-permissions \
                        --print --verbose \
                        --output-format stream-json \
                        --settings "$CLAUDE_KBH_SETTINGS" \
                        --model "${MINIMAX_CLAUDE_ALIAS:-opus}" \
                        --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                        -p "$PROMPT" \
                    > "$log" 2>&1
                ) || exit_code=$?
            fi
            ;;
        codex)
            local effort_arg=()
            if [ -n "$effort" ]; then
                effort_arg=(-c "model_reasoning_effort=\"$effort\"")
            fi
            timeout "$TIMEOUT_SECONDS" codex exec \
                -m "$model" \
                "${effort_arg[@]}" \
                --dangerously-bypass-approvals-and-sandbox \
                --skip-git-repo-check \
                -C "$REPO_ROOT" \
                "$PROMPT" \
                > "$log" 2>&1 || exit_code=$?
            ;;
        cursor)
            timeout "$TIMEOUT_SECONDS" agent \
                --trust \
                --yolo \
                --print \
                --output-format stream-json \
                --model "$model" \
                --workspace "$REPO_ROOT" \
                "$PROMPT" \
                > "$log" 2>&1 || exit_code=$?
            ;;
        grok)
            local effort_arg=()
            if [ -n "$effort" ]; then
                effort_arg=(--effort "$effort")
            fi
            timeout "$TIMEOUT_SECONDS" grok \
                --cwd "$REPO_ROOT" \
                --always-approve \
                --permission-mode bypassPermissions \
                --no-memory \
                --disable-web-search \
                --output-format streaming-json \
                --model "$model" \
                "${effort_arg[@]}" \
                -p "$PROMPT" \
                > "$log" 2>&1 || exit_code=$?
            ;;
        opencode)
            timeout "$TIMEOUT_SECONDS" opencode run \
                --pure --format json -m "$model" "$PROMPT" \
                > "$log" 2>&1 || exit_code=$?
            ;;
        gemini)
            timeout "$TIMEOUT_SECONDS" gemini \
                --skip-trust \
                -m "$model" \
                --approval-mode yolo \
                -o stream-json \
                -p "$PROMPT" \
                > "$log" 2>&1 || exit_code=$?
            ;;
        *)
            echo "Unsupported harness: $harness" > "$log"
            exit_code=2
            ;;
    esac

    end="$(date +%s)"
    elapsed=$((end - start))
    if [ "$exit_code" -eq 0 ] && grep -q "KBH_PREFLIGHT_OK" "$log"; then
        ok=true
    else
        ok=false
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$harness" "$model" "$effort" "$exit_code" "$ok" "$elapsed" "${log#$REPO_ROOT/}" \
        >> "$SUMMARY"
    echo "$name ok=$ok exit=$exit_code elapsed=${elapsed}s"
}

for row in "${ROWS[@]}"; do
    IFS='|' read -r name harness model effort <<< "$row"
    run_one "$name" "$harness" "$model" "$effort"
done

echo "preflight=$SUMMARY"

if awk -F '\t' 'NR > 1 && $6 != "true" {bad=1} END {exit bad ? 1 : 0}' "$SUMMARY"; then
    echo "preflight ok"
else
    echo "preflight failed; inspect $OUT_DIR" >&2
    exit 1
fi
