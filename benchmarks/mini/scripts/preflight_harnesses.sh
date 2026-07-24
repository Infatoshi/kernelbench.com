#!/bin/bash
# Cheap provider/harness preflight before an expensive KernelBench-Hard sweep.
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
    "claude_opus48_max|claude|claude-opus-4-8|max"
    "zai_claude_glm51|zai-claude|glm-5.1|"
    "cursor_composer25fast|cursor|composer-2.5-fast|"
    "grok_grokbuild_max|grok|grok-build|max"
    "opencode_qwen37max|opencode|openrouter-alibaba/qwen/qwen3.7-max|"
    "opencode_gemini35flash|opencode|openrouter-google-ai-studio/google/gemini-3.5-flash|"
    "opencode_dsv4pro|opencode|deepseek/deepseek-v4-pro|"
    "opencode_dsv4flash|opencode|deepseek/deepseek-v4-flash|"
    "opencode_mimo|opencode|openrouter-pinned/xiaomi/mimo-v2.5-pro|"
    "opencode_kimi|opencode|openrouter-moonshot/moonshotai/kimi-k2.6|"
)

# The opencode zai route intermittently hangs on GLM-5.1 reasoning streams
# (DEVLOG 2026-06-09, opencode OpenAI-compatible adapter). GLM-5.1 is scored
# via the zai-claude row; enable this diagnostic row only deliberately.
if [ "${KBH_USE_OPENCODE_ZAI:-0}" = "1" ]; then
    ROWS+=("opencode_glm51|opencode|zai/glm-5.1|")
fi

if [ "${KBH_USE_DIRECT_GEMINI:-0}" = "1" ]; then
    ROWS+=("gemini_gemini35flash|gemini|gemini-3.5-flash|")
fi

if [ "${KBH_USE_MINIMAX_M3_CLAUDE:-0}" = "1" ]; then
    ROWS+=("minimax_m3_claude|minimax-claude|MiniMax-M3|")
fi

if [ "${KBH_USE_OPENROUTER_NEMOTRON:-0}" = "1" ]; then
    ROWS+=("opencode_nemotron_ultra|opencode-nemotron|nvidia/nemotron-3-ultra-550b-a55b|")
fi

if [ "${KBH_USE_NVCF_NEMOTRON:-0}" = "1" ]; then
    ROWS+=("nvcf_nemotron_ultra|nvcf-nemotron|nemotron-3-ultra|")
fi

if [ "${KBH_SKIP_OPENROUTER:-0}" = "1" ]; then
    FILTERED_ROWS=()
    for row in "${ROWS[@]}"; do
        IFS='|' read -r _name _harness model _effort <<< "$row"
        if [[ "$model" == openrouter-* || "$_harness" == opencode-nemotron ]]; then
            continue
        fi
        FILTERED_ROWS+=("$row")
    done
    ROWS=("${FILTERED_ROWS[@]}")
fi

if [ -n "${KBH_PREFLIGHT_ONLY:-}" ]; then
    FILTERED_ROWS=()
    for row in "${ROWS[@]}"; do
        IFS='|' read -r name harness model effort <<< "$row"
        if [[ "$name" == "$KBH_PREFLIGHT_ONLY" || "$harness" == "$KBH_PREFLIGHT_ONLY" || "$model" == "$KBH_PREFLIGHT_ONLY" ]]; then
            FILTERED_ROWS+=("$row")
        fi
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
        opencode-nemotron)
            if [ -z "${OPENROUTER_API_KEY:-}" ]; then
                echo "OPENROUTER_API_KEY missing" > "$log"
                exit_code=2
            else
                config_home="$OUT_DIR/$name.config"
                mkdir -p "$config_home/opencode"
                cat > "$config_home/opencode/opencode.json" <<JSON
{"\$schema":"https://opencode.ai/config.json","permission":{"external_directory":"deny"},"provider":{"openrouter-deepinfra":{"npm":"@ai-sdk/openai-compatible","name":"OpenRouter DeepInfra","options":{"baseURL":"https://openrouter.ai/api/v1","apiKey":"${OPENROUTER_API_KEY}","headers":{"HTTP-Referer":"https://kernelbench.com","X-Title":"KernelBench-Hard"},"extraBody":{"provider":{"order":["DeepInfra"],"allow_fallbacks":false}}},"models":{"$model":{"name":"NVIDIA Nemotron 3 Ultra via OpenRouter DeepInfra","limit":{"context":262144,"output":16384},"tools":true}}}}}
JSON
                timeout "$TIMEOUT_SECONDS" env XDG_CONFIG_HOME="$config_home" opencode run                     --pure --format json -m "openrouter-deepinfra/$model" "$PROMPT"                     > "$log" 2>&1 || exit_code=$?
            fi
            ;;
        nvcf-nemotron)
            if [ -z "${NGC_API_KEY:-${NVIDIA_API_KEY:-${NVCF_API_KEY:-}}}" ]; then
                echo "NGC_API_KEY, NVIDIA_API_KEY, or NVCF_API_KEY missing" > "$log"
                exit_code=2
            else
                proxy_log="$OUT_DIR/$name.proxy.log"
                uv run python "$REPO_ROOT/scripts/nvcf_openai_proxy.py" \
                    --host 127.0.0.1 --port 0 > "$proxy_log" 2>&1 &
                proxy_pid=$!
                proxy_url=""
                for _ in $(seq 1 100); do
                    if ! kill -0 "$proxy_pid" 2>/dev/null; then
                        break
                    fi
                    proxy_url="$(grep -oE 'http://127\.0\.0\.1:[0-9]+' "$proxy_log" | tail -1 || true)"
                    if [ -n "$proxy_url" ]; then
                        break
                    fi
                    sleep 0.1
                done
                if [ -z "$proxy_url" ]; then
                    echo "NVCF proxy did not start; see $proxy_log" > "$log"
                    exit_code=2
                else
                    config_home="$OUT_DIR/$name.config"
                    mkdir -p "$config_home/opencode"
                    cat > "$config_home/opencode/opencode.json" <<JSON
{"\$schema":"https://opencode.ai/config.json","permission":{"external_directory":"deny"},"provider":{"nvcf-nemotron":{"npm":"@ai-sdk/openai-compatible","name":"NVIDIA NVCF Nemotron","options":{"baseURL":"${proxy_url}/v1","apiKey":"nvcf-proxy"},"models":{"$model":{"name":"Nemotron 3 Ultra via NVCF","limit":{"context":200000,"output":4096},"tools":true}}}}}
JSON
                    timeout "$TIMEOUT_SECONDS" env XDG_CONFIG_HOME="$config_home" opencode run \
                        --pure --format json -m "nvcf-nemotron/$model" "$PROMPT" \
                        > "$log" 2>&1 || exit_code=$?
                fi
                kill "$proxy_pid" 2>/dev/null || true
                wait "$proxy_pid" 2>/dev/null || true
            fi
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

# Multi-step tool-use probes for opencode routes. One-turn smokes cannot catch
# the OpenAI-compatible adapter stall (DEVLOG 2026-06-09): some reasoning
# models pass a trivial reply but hang on the first long generation after tool
# results enter context. Disable with KBH_PREFLIGHT_MULTISTEP=0.
if [ "${KBH_PREFLIGHT_MULTISTEP:-1}" = "1" ]; then
    MULTISTEP_TIMEOUT="${KBH_PREFLIGHT_MULTISTEP_TIMEOUT_SECONDS:-420}"
    for row in "${ROWS[@]}"; do
        IFS='|' read -r name harness model effort <<< "$row"
        case "$harness" in
            opencode)
                if [ -n "${KBH_PREFLIGHT_ONLY:-}" ] && [ "$name" != "$KBH_PREFLIGHT_ONLY" ]; then
                    continue
                fi
                ms_log="$OUT_DIR/${name}_multistep.log"
                ms_exit=0
                ./scripts/probe_opencode_multistep.sh "$model" "$MULTISTEP_TIMEOUT" \
                    > "$ms_log" 2>&1 || ms_exit=$?
                ms_ok=false
                [ "$ms_exit" -eq 0 ] && ms_ok=true
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "${name}_multistep" "$harness" "$model" "$effort" "$ms_exit" "$ms_ok" "-" "${ms_log#$REPO_ROOT/}" \
                    >> "$SUMMARY"
                echo "${name}_multistep ok=$ms_ok exit=$ms_exit"
                ;;
        esac
    done
fi

echo "preflight=$SUMMARY"

if awk -F '\t' 'NR > 1 && $6 != "true" {bad=1} END {exit bad ? 1 : 0}' "$SUMMARY"; then
    echo "preflight ok"
else
    echo "preflight failed; inspect $OUT_DIR" >&2
    exit 1
fi
