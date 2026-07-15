#!/bin/bash
# Run one (harness, model, problem) combination.
#
# Usage:
#   ./scripts/run_hard.sh <harness> <model> <problem_dir> [reasoning_effort]
#
# Examples:
#   ./scripts/run_hard.sh claude claude-opus-4-7 problems/02_kimi_linear_decode
#   ./scripts/run_hard.sh codex gpt-5.5 problems/02_kimi_linear_decode xhigh
#   ./scripts/run_hard.sh kimi kimi-k2.6 problems/02_kimi_linear_decode
#   ./scripts/run_hard.sh droid glm-5.1 problems/02_kimi_linear_decode
#   ./scripts/run_hard.sh grok grok-build problems/02_kimi_linear_decode max
#   ./scripts/run_hard.sh zai-claude glm-5.1 problems/02_kimi_linear_decode
#   ./scripts/run_hard.sh minimax-claude MiniMax-M3 problems/02_kimi_linear_decode
#   ./scripts/run_hard.sh ccr-claude glm-5.1 problems/02_kimi_linear_decode
#
# Archives everything to outputs/runs/<ts>_<harness>_<model>_<problem>/.

set -euo pipefail

# Pin CUDA 13 — /usr/local/cuda may still point at 12.8. Override the pinned
# toolkit dir on other machines with KBH_CUDA_HOME (default /usr/local/cuda-13).
KBH_CUDA_HOME="${KBH_CUDA_HOME:-/usr/local/cuda-13}"
if [ -d "$KBH_CUDA_HOME" ]; then
    export CUDA_HOME="$KBH_CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
fi

# Source API keys if the user has an env_vars file.
if [ -f "$HOME/.env_vars" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$HOME/.env_vars"
    set +a
fi

# Per-run Claude account selection. ~/.env_vars exports CLAUDE_CODE_OAUTH_TOKEN,
# which takes precedence over the interactive login in ~/.claude/.credentials.json.
# KBH_CLAUDE_AUTH=keychain drops the env token for this run so it bills the
# keychain account instead — lets two concurrent runs spread across two accounts.
if [ "${KBH_CLAUDE_AUTH:-}" = "keychain" ]; then
    unset CLAUDE_CODE_OAUTH_TOKEN
fi

HARNESS="${1:?Usage: $0 <harness> <model> <problem_dir> [reasoning_effort]}"
MODEL="${2:?model required}"
SOURCE_PROBLEM_DIR="${3:?problem_dir required}"
REASONING_EFFORT="${4:-}"
CLAUDE_KBH_SETTINGS="${CLAUDE_KBH_SETTINGS:-{\"fastMode\":false,\"alwaysThinkingEnabled\":true}}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PROBLEM_DIR="$(cd "$SOURCE_PROBLEM_DIR" && pwd)"
PROBLEM_NAME="$(basename "$SOURCE_PROBLEM_DIR")"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MODEL_SLUG="$(echo "$MODEL" | tr '/:[] ' '_')"
RUN_DIR_BASE="${REPO_ROOT}/outputs/runs/${TIMESTAMP}_${HARNESS}_${MODEL_SLUG}_${PROBLEM_NAME}"
mkdir -p "$REPO_ROOT/outputs/runs"
RUN_DIR=""
for attempt in $(seq 0 999); do
    if [ "$attempt" -eq 0 ]; then
        candidate="$RUN_DIR_BASE"
    else
        candidate="${RUN_DIR_BASE}_$$_${attempt}"
    fi
    if mkdir "$candidate" 2>/dev/null; then
        RUN_DIR="$candidate"
        break
    fi
done
if [ -z "$RUN_DIR" ]; then
    echo "failed to allocate unique run directory for $RUN_DIR_BASE" >&2
    exit 1
fi
RUN_ID="$(basename "$RUN_DIR")"
RUN_GROUP="${KBH_RUN_GROUP:-}"

# Wall-clock ceiling per run. Methodology is unlimited-time: the model runs until
# it decides it is done. Default is a large 6h ceiling; smoke-test with a small
# override (e.g. BUDGET_SECONDS=300).
BUDGET_SECONDS="${BUDGET_SECONDS:-21600}"
CHECK_TIMEOUT_SECONDS="${KBH_CHECK_TIMEOUT_SECONDS:-180}"
if [ "$PROBLEM_NAME" = "02_kda_cutlass" ]; then
    BENCHMARK_TIMEOUT_SECONDS="${KBH_BENCHMARK_TIMEOUT_02_KDA_CUTLASS_SECONDS:-${KBH_BENCHMARK_TIMEOUT_SECONDS:-7200}}"
else
    BENCHMARK_TIMEOUT_SECONDS="${KBH_BENCHMARK_TIMEOUT_SECONDS:-1800}"
fi
export KBH_GPU_LOCK_WAIT_TIMEOUT_SECONDS="${KBH_GPU_LOCK_WAIT_TIMEOUT_SECONDS:-7200}"
MIN_USEFUL_OUTPUT_TOKENS="${KBH_MIN_USEFUL_OUTPUT_TOKENS:-5000}"

AGENT_CUDA_DISABLED=false
GPU_QUEUE_MODE="path_wrapper_gpu_lock"

# --- Load the per-problem prompt ------------------------------------------
#
# Each problem has a PROMPT.txt in human voice that combines the task brief,
# shapes, forbidden ops, and workflow guidance into a single user-style
# message. The harness sends this directly as the prompt to the agent. No
# system/user split, no preamble concatenation.

PROMPT_FILE="${SOURCE_PROBLEM_DIR}/PROMPT.txt"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "PROMPT.txt missing for $PROBLEM_NAME" >&2
    exit 1
fi
PROMPT="$(cat "$PROMPT_FILE")"

# --- Create an isolated per-run problem workspace -------------------------
# problem.yaml and shapes.py stay in the workspace because check.py and
# benchmark.py import them at runtime; the prompt does not direct the model
# to read them.
TEMPLATE_FILES=(reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt baseline.py)
is_template() {
    local n="$1"
    for t in "${TEMPLATE_FILES[@]}"; do
        [[ "$n" == "$t" ]] && return 0
    done
    return 1
}

WORKSPACE_ROOT="$RUN_DIR/repo"
PROBLEM_DIR="$WORKSPACE_ROOT/problems/$PROBLEM_NAME"
mkdir -p "$PROBLEM_DIR"

# Contamination sandbox: run the agent under bwrap with EVERY source of a prior
# solution or the optimization recipe hidden, while the toolchain (src, .venv,
# GPU, outputs/gpu.lock) passes through via --dev-bind.
#
# Leak sources (all must be blanked — 2026-07-09 Grok 4.5 kimi-decode proved
# that hiding only outputs/runs is not enough):
#   - this bench's run archive (outputs/runs)
#   - sibling hard/v3 run archives
#   - monorepo public/ (published solution.py.txt for Fable/etc — the actual
#     path Grok used: public/data/mega/code/* and public/runs/*_solution.py.txt)
#   - results/ (annotations + leaderboard scores named as targets)
#   - monorepo runs/ HF staging
#   - DEVLOG (optimization journey == the recipe)
#   - ~/.claude/projects memory
# Own run dir stays writable. KBH_SANDBOX=0 to disable; auto-off if bwrap is
# absent (e.g. verda B200 userns denied — blank public/ + DEVLOG on the box).
RUNS_DIR="$REPO_ROOT/outputs/runs"
SIB_PARENT="$(dirname "$REPO_ROOT")"   # benchmarks/ on anvil, $HOME on a cloud box
MONOREPO_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"  # kernelbench.com monorepo root
KBH_EMPTY="$RUN_DIR/.kbh_empty"; : > "$KBH_EMPTY"
KBH_SBX=()
if [ "${KBH_SANDBOX:-1}" = "1" ] && command -v bwrap >/dev/null 2>&1; then
    KBH_SBX=(bwrap --dev-bind / / --tmpfs "$RUNS_DIR")
    [ -e "$REPO_ROOT/DEVLOG.md" ] && KBH_SBX+=(--ro-bind "$KBH_EMPTY" "$REPO_ROOT/DEVLOG.md")
    [ -d "$REPO_ROOT/results" ] && KBH_SBX+=(--tmpfs "$REPO_ROOT/results")
    [ -d "$SIB_PARENT/hard" ]     && KBH_SBX+=(--tmpfs "$SIB_PARENT/hard")
    [ -d "$SIB_PARENT/v3" ]       && KBH_SBX+=(--tmpfs "$SIB_PARENT/v3")
    [ -d "$MONOREPO_ROOT/public" ] && KBH_SBX+=(--tmpfs "$MONOREPO_ROOT/public")
    [ -d "$MONOREPO_ROOT/runs" ] && KBH_SBX+=(--tmpfs "$MONOREPO_ROOT/runs")
    [ -d "$HOME/.claude/projects" ] && KBH_SBX+=(--tmpfs "$HOME/.claude/projects")
    # Own archive + problem workspace must remain visible/writable after tmpfs hides.
    KBH_SBX+=(--bind "$RUN_DIR" "$RUN_DIR" --chdir "$PROBLEM_DIR")
    echo "agent sandbox: bwrap (hidden: runs archives, public/ solutions, results/, DEVLOG, ~/.claude memory)"
fi

PROMPT="${PROMPT}

Workspace isolation note: you are already running inside the archive-local
problem workspace, ${PROBLEM_DIR}. Write the final answer to solution.py in the
current directory only. Do not write to ${SOURCE_PROBLEM_DIR} or to the source
repository's problems/ tree."

# check.py and benchmark.py derive REPO_ROOT as parents[2]. Keep that shape
# while sharing src/. Copy project metadata so agents can mutate dependencies
# inside their disposable workspace without touching the source repo.
ln -s "$REPO_ROOT/src" "$WORKSPACE_ROOT/src"
cp -p "$REPO_ROOT/pyproject.toml" "$WORKSPACE_ROOT/pyproject.toml"
cp -p "$REPO_ROOT/uv.lock" "$WORKSPACE_ROOT/uv.lock"
if [ -e "$REPO_ROOT/.python-version" ]; then
    cp -p "$REPO_ROOT/.python-version" "$WORKSPACE_ROOT/.python-version"
fi

for t in "${TEMPLATE_FILES[@]}"; do
    if [ -e "$SOURCE_PROBLEM_DIR/$t" ]; then
        cp -p "$SOURCE_PROBLEM_DIR/$t" "$PROBLEM_DIR/$t"
    fi
done

# --- Per-run GPU/cache isolation -----------------------------------------
#
# Agent reasoning and file editing can happen in parallel. CUDA-facing work
# should be serialized so agent-internal check.py/benchmark.py/profiling runs
# do not contaminate each other's timing or Torch extension builds.
REAL_UV="$(command -v uv)"
REAL_PYTHON="$(command -v python3 || command -v python)"
REAL_NVIDIA_SMI="$(command -v nvidia-smi || true)"
REAL_NCU="$(command -v ncu || true)"
REAL_NSYS="$(command -v nsys || true)"
REAL_NVCC="$(command -v nvcc || true)"
REAL_TIMEOUT="$(command -v timeout)"
REAL_UV_FALLBACK="$REAL_UV"
REAL_PYTHON_FALLBACK="$REAL_PYTHON"
LOCK_WRAPPER_DIR="$RUN_DIR/bin"
mkdir -p "$LOCK_WRAPPER_DIR" "$RUN_DIR/cache/torch_extensions" \
    "$RUN_DIR/cache/triton" "$RUN_DIR/cache/cuda" "$RUN_DIR/tmp"

export KBH_GPU_LOCK="${KBH_GPU_LOCK:-$REPO_ROOT/outputs/gpu.lock}"
export KBH_GPU_LOCK_LOG="$RUN_DIR/gpu_lock.log"
export TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
export TMPDIR="$RUN_DIR/tmp"
export TEMP="$RUN_DIR/tmp"
export TMP="$RUN_DIR/tmp"
export RUN_DIR REAL_UV REAL_PYTHON REAL_NVIDIA_SMI REAL_NCU REAL_NSYS REAL_NVCC \
    REAL_UV_FALLBACK REAL_PYTHON_FALLBACK

cat > "$LOCK_WRAPPER_DIR/gpu-lock-exec" <<'EOF'
#!/bin/bash
set -euo pipefail
name="$1"
real="$2"
shift 2
if [ -z "$real" ]; then
    echo "$name is unavailable" >&2
    exit 127
fi
real_abs="$(readlink -f "$real" 2>/dev/null || printf '%s' "$real")"
case "$name" in
    uv)
        fallback="${REAL_UV_FALLBACK:-}"
        ;;
    python|python3)
        fallback="${REAL_PYTHON_FALLBACK:-}"
        ;;
    *)
        fallback=""
        ;;
esac
if [ -n "${RUN_DIR:-}" ] && [[ "$real_abs" == "${RUN_DIR}/bin/"* ]] && [ -n "$fallback" ]; then
    real="$fallback"
fi
if [ "${KBH_GPU_LOCK_HELD:-0}" = "1" ]; then
    exec "$real" "$@"
fi
owner_file="${KBH_GPU_LOCK}.owner"
if [ -f "$owner_file" ]; then
    IFS=$'\t' read -r owner_pid owner_run_dir < "$owner_file" || true
    if [ "${owner_run_dir:-}" = "${RUN_DIR:-}" ] && kill -0 "${owner_pid:-}" 2>/dev/null; then
        exec "$real" "$@"
    fi
fi
{
    printf '%s wait pid=%s cmd=%s args=%q\n' "$(date -Is)" "$$" "$name" "$*" >&3
    lock_wait_timeout="${KBH_GPU_LOCK_WAIT_TIMEOUT_SECONDS:-}"
    if [ -n "$lock_wait_timeout" ] && [ "$lock_wait_timeout" != "0" ]; then
        if ! flock -x -w "$lock_wait_timeout" 9; then
            printf '%s lock_timeout pid=%s cmd=%s wait_timeout_s=%s\n' \
                "$(date -Is)" "$$" "$name" "$lock_wait_timeout" >&3
            exit 124
        fi
    else
        flock -x 9
    fi
    start="$(date +%s)"
    printf '%s\t%s\n' "$$" "${RUN_DIR:-}" > "$owner_file"
    printf '%s start pid=%s cmd=%s\n' "$(date -Is)" "$$" "$name" >&3
    set +e
    export KBH_GPU_LOCK_HELD=1
    "$real" "$@"
    status=$?
    set -e
    if [ -f "$owner_file" ] && IFS=$'\t' read -r current_owner _ < "$owner_file" && [ "$current_owner" = "$$" ]; then
        rm -f "$owner_file"
    fi
    printf '%s end pid=%s cmd=%s status=%s elapsed_s=%s\n' \
        "$(date -Is)" "$$" "$name" "$status" "$(($(date +%s) - start))" >&3
    exit "$status"
} 3>>"$KBH_GPU_LOCK_LOG" 9>"$KBH_GPU_LOCK"
EOF
chmod +x "$LOCK_WRAPPER_DIR/gpu-lock-exec"

cat > "$LOCK_WRAPPER_DIR/uv" <<'EOF'
#!/bin/bash
exec "$RUN_DIR/bin/gpu-lock-exec" uv "${REAL_UV:-$REAL_UV_FALLBACK}" "$@"
EOF
cat > "$LOCK_WRAPPER_DIR/python" <<'EOF'
#!/bin/bash
exec "$RUN_DIR/bin/gpu-lock-exec" python "${REAL_PYTHON:-$REAL_PYTHON_FALLBACK}" "$@"
EOF
cat > "$LOCK_WRAPPER_DIR/python3" <<'EOF'
#!/bin/bash
exec "$RUN_DIR/bin/gpu-lock-exec" python3 "${REAL_PYTHON:-$REAL_PYTHON_FALLBACK}" "$@"
EOF
cat > "$LOCK_WRAPPER_DIR/nvidia-smi" <<'EOF'
#!/bin/bash
exec "$RUN_DIR/bin/gpu-lock-exec" nvidia-smi "$REAL_NVIDIA_SMI" "$@"
EOF
cat > "$LOCK_WRAPPER_DIR/ncu" <<'EOF'
#!/bin/bash
exec "$RUN_DIR/bin/gpu-lock-exec" ncu "$REAL_NCU" "$@"
EOF
cat > "$LOCK_WRAPPER_DIR/nsys" <<'EOF'
#!/bin/bash
exec "$RUN_DIR/bin/gpu-lock-exec" nsys "$REAL_NSYS" "$@"
EOF
cat > "$LOCK_WRAPPER_DIR/nvcc" <<'EOF'
#!/bin/bash
exec "$RUN_DIR/bin/gpu-lock-exec" nvcc "$REAL_NVCC" "$@"
EOF
chmod +x "$LOCK_WRAPPER_DIR"/uv "$LOCK_WRAPPER_DIR"/python \
    "$LOCK_WRAPPER_DIR"/python3 "$LOCK_WRAPPER_DIR"/nvidia-smi \
    "$LOCK_WRAPPER_DIR"/ncu "$LOCK_WRAPPER_DIR"/nsys "$LOCK_WRAPPER_DIR"/nvcc
export PATH="$LOCK_WRAPPER_DIR:$PATH"

run_gpu_locked_timeout() {
    local lock_name="$1"
    local timeout_seconds="$2"
    shift 2
    "$RUN_DIR/bin/gpu-lock-exec" "$lock_name" "$REAL_TIMEOUT" "$timeout_seconds" "$@"
}

# Snapshot immutable problem files. Agents may make a mess in the problem
# directory, but changing benchmark definitions invalidates the run and must not
# leak into the next problem.
TEMPLATE_BACKUP_DIR="$RUN_DIR/template_files"
mkdir -p "$TEMPLATE_BACKUP_DIR"
for t in "${TEMPLATE_FILES[@]}"; do
    if [ -e "$PROBLEM_DIR/$t" ]; then
        cp -p "$PROBLEM_DIR/$t" "$TEMPLATE_BACKUP_DIR/$t"
    fi
done

TEMPLATE_MUTATED=false

detect_template_mutation() {
    local phase="$1"
    local found=0
    local log="$RUN_DIR/template_mutations.log"
    for t in "${TEMPLATE_FILES[@]}"; do
        local orig="$TEMPLATE_BACKUP_DIR/$t"
        local cur="$PROBLEM_DIR/$t"
        if [ -e "$orig" ] && [ -e "$cur" ]; then
            if ! cmp -s "$orig" "$cur"; then
                if [ "$found" -eq 0 ]; then
                    printf 'phase: %s\n' "$phase" >> "$log"
                fi
                found=1
                printf 'MUTATED: %s\n' "$t" >> "$log"
                diff -u --label "before/$t" --label "after/$t" "$orig" "$cur" >> "$log" 2>&1 || true
            fi
        elif [ -e "$orig" ] && [ ! -e "$cur" ]; then
            if [ "$found" -eq 0 ]; then
                printf 'phase: %s\n' "$phase" >> "$log"
            fi
            found=1
            printf 'DELETED: %s\n' "$t" >> "$log"
        elif [ ! -e "$orig" ] && [ -e "$cur" ]; then
            if [ "$found" -eq 0 ]; then
                printf 'phase: %s\n' "$phase" >> "$log"
            fi
            found=1
            printf 'CREATED TEMPLATE FILE: %s\n' "$t" >> "$log"
        fi
    done
    return "$found"
}

restore_template_files() {
    for t in "${TEMPLATE_FILES[@]}"; do
        local orig="$TEMPLATE_BACKUP_DIR/$t"
        local cur="$PROBLEM_DIR/$t"
        rm -rf "$cur"
        if [ -e "$orig" ]; then
            cp -p "$orig" "$cur"
        fi
    done
}

# --- Run the harness ------------------------------------------------------

LOG_FILE="${RUN_DIR}/transcript.jsonl"
STDERR_FILE="${RUN_DIR}/stderr.log"

echo "========================================"
echo "KERNELBENCH-MEGA RUN"
echo "========================================"
echo "Harness:    $HARNESS"
echo "Model:      $MODEL"
echo "Effort:     ${REASONING_EFFORT:-<default>}"
echo "Problem:    $PROBLEM_NAME"
echo "Source:     $SOURCE_PROBLEM_DIR"
echo "Workspace:  $PROBLEM_DIR"
echo "Archive:    $RUN_DIR"
echo "Budget:     ${BUDGET_SECONDS}s"
echo "========================================"

START_TIME=$(date +%s)
STARTED_AT="$(date -Is)"
HARNESS_EXIT=0

case "$HARNESS" in
    claude)
        EFFORT_ARG=()
        if [ -n "$REASONING_EFFORT" ]; then
            EFFORT_ARG=(--effort "$REASONING_EFFORT")
        fi
        ( cd "$PROBLEM_DIR" && "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" claude \
            --dangerously-skip-permissions \
            --print --verbose \
            --output-format stream-json \
            --settings "$CLAUDE_KBH_SETTINGS" \
            --model "$MODEL" \
            "${EFFORT_ARG[@]}" \
            --add-dir "$PROBLEM_DIR" \
            -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    ccr-claude)
        # Claude Code routed via ccr-rust to a non-Anthropic provider.
        # Assumes ccr-rust is running locally and ANTHROPIC_BASE_URL points at it.
        # Model name is the upstream lab's model ID (glm-5.1, deepseek-v4-flash, etc.).
        ( cd "$PROBLEM_DIR" && "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" \
            env ANTHROPIC_BASE_URL="${CCR_BASE_URL:-http://127.0.0.1:3456}" \
            claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$MODEL" \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    zai-claude)
        # Claude Code routed directly to Z.ai's Anthropic-compatible endpoint.
        # Use Claude Code's built-in model aliases and map them to Z.ai model IDs.
        # Passing --model glm-5.1 directly can hit Claude Code model-access checks.
        # Requires ZAI_API_KEY in the environment or ~/.env_vars.
        if [ -z "${ZAI_API_KEY:-}" ]; then
            echo "ZAI_API_KEY is required for zai-claude" >&2
            exit 1
        fi
        ZAI_CLAUDE_ALIAS="${ZAI_CLAUDE_ALIAS:-opus}"
        ZAI_CLAUDE_HAIKU_MODEL="${ZAI_CLAUDE_HAIKU_MODEL:-$MODEL}"
        ( cd "$PROBLEM_DIR" && \
            export ANTHROPIC_AUTH_TOKEN="$ZAI_API_KEY" && \
            export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic" && \
            export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" && \
            export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS="${CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS:-1}" && \
            export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-1000000}" && \
            export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-128000}" && \
            export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ZAI_CLAUDE_HAIKU_MODEL" && \
            export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" && \
            "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$ZAI_CLAUDE_ALIAS" \
                --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    minimax-claude)
        # Claude Code routed directly to MiniMax's Anthropic-compatible endpoint.
        # Requires MINIMAX_API_KEY in the environment or ~/.env_vars. Keep this
        # separate from the normal `claude` harness so Opus defaults stay intact.
        if [ -z "${MINIMAX_API_KEY:-}" ]; then
            echo "MINIMAX_API_KEY is required for minimax-claude" >&2
            exit 1
        fi
        MINIMAX_CLAUDE_ALIAS="${MINIMAX_CLAUDE_ALIAS:-opus}"
        MINIMAX_CLAUDE_HAIKU_MODEL="${MINIMAX_CLAUDE_HAIKU_MODEL:-$MODEL}"
        ( cd "$PROBLEM_DIR" && \
            export ANTHROPIC_AUTH_TOKEN="$MINIMAX_API_KEY" && \
            export ANTHROPIC_BASE_URL="${MINIMAX_ANTHROPIC_BASE_URL:-https://api.minimax.io/anthropic}" && \
            export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" && \
            export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}" && \
            export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-1000000}" && \
            export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-128000}" && \
            export ANTHROPIC_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MINIMAX_CLAUDE_HAIKU_MODEL" && \
            export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" && \
            "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$MINIMAX_CLAUDE_ALIAS" \
                --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    kimi-claude)
        # Claude Code routed to Moonshot's Anthropic-compatible endpoint for Kimi.
        # Requires KIMI_API_KEY. Pass MODEL=kimi-k2.7-code.
        if [ -z "${KIMI_API_KEY:-}" ]; then
            echo "KIMI_API_KEY is required for kimi-claude" >&2
            exit 1
        fi
        KIMI_CLAUDE_ALIAS="${KIMI_CLAUDE_ALIAS:-opus}"
        KIMI_CLAUDE_HAIKU_MODEL="${KIMI_CLAUDE_HAIKU_MODEL:-$MODEL}"
        ( cd "$PROBLEM_DIR" && \
            export ANTHROPIC_AUTH_TOKEN="$KIMI_API_KEY" && \
            export ANTHROPIC_BASE_URL="${KIMI_ANTHROPIC_BASE_URL:-https://api.moonshot.ai/anthropic}" && \
            export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" && \
            export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}" && \
            export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-1000000}" && \
            export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-128000}" && \
            export ANTHROPIC_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_HAIKU_MODEL="$KIMI_CLAUDE_HAIKU_MODEL" && \
            export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" && \
            "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$KIMI_CLAUDE_ALIAS" \
                --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    kinetic-claude)
        # Claude Code routed to Moonshot's Anthropic-compatible endpoint for
        # kinetic-0715. Requires MOONSHOT_API_KEY (KIMI_API_KEY 401s on it).
        # ENABLE_TOOL_SEARCH=false + CLAUDE_CODE_SUBAGENT_MODEL per Moonshot's
        # Claude Code guide.
        if [ -z "${MOONSHOT_API_KEY:-}" ]; then
            echo "MOONSHOT_API_KEY is required for kinetic-claude" >&2
            exit 1
        fi
        # Moonshot's validator 400s histories containing the model's own
        # thinkingless assistant messages; CLAUDE_CODE_EFFORT_LEVEL=max below
        # makes kinetic think on every turn, which avoids ever creating that
        # message shape (bisect-verified 2026-07-15, see hard DEVLOG).
        KINETIC_CLAUDE_ALIAS="${KINETIC_CLAUDE_ALIAS:-opus}"
        KINETIC_CLAUDE_HAIKU_MODEL="${KINETIC_CLAUDE_HAIKU_MODEL:-$MODEL}"
        ( cd "$PROBLEM_DIR" && \
            export ANTHROPIC_AUTH_TOKEN="$MOONSHOT_API_KEY" && \
            export ANTHROPIC_BASE_URL="${KINETIC_ANTHROPIC_BASE_URL:-https://api.moonshot.ai/anthropic}" && \
            export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" && \
            export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}" && \
            export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS="${CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS:-1}" && \
            export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-1000000}" && \
            export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-128000}" && \
            export ANTHROPIC_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_HAIKU_MODEL="$KINETIC_CLAUDE_HAIKU_MODEL" && \
            export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" && \
            export ENABLE_TOOL_SEARCH="${ENABLE_TOOL_SEARCH:-false}" && \
                export CLAUDE_CODE_EFFORT_LEVEL="${CLAUDE_CODE_EFFORT_LEVEL:-max}" && \
            export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL" && \
            "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$KINETIC_CLAUDE_ALIAS" \
                --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    longcat-claude)
        # Claude Code routed to Meituan LongCat's Anthropic-compatible endpoint.
        # Requires LONGCAT_API_KEY. Pass MODEL=LongCat-2.0.
        if [ -z "${LONGCAT_API_KEY:-}" ]; then
            echo "LONGCAT_API_KEY is required for longcat-claude" >&2
            exit 1
        fi
        LONGCAT_CLAUDE_ALIAS="${LONGCAT_CLAUDE_ALIAS:-opus}"
        LONGCAT_CLAUDE_HAIKU_MODEL="${LONGCAT_CLAUDE_HAIKU_MODEL:-$MODEL}"
        ( cd "$PROBLEM_DIR" && \
            export ANTHROPIC_AUTH_TOKEN="$LONGCAT_API_KEY" && \
            export ANTHROPIC_BASE_URL="${LONGCAT_ANTHROPIC_BASE_URL:-https://api.longcat.chat/anthropic}" && \
            export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" && \
            export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}" && \
            export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-1000000}" && \
            export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-131072}" && \
            export ANTHROPIC_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_HAIKU_MODEL="$LONGCAT_CLAUDE_HAIKU_MODEL" && \
            export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" && \
            "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$LONGCAT_CLAUDE_ALIAS" \
                --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    hy3|hy3-claude)
        # Tencent Hy3 official: OpenCode -> TokenHub (OpenAI-compat).
        # model hy3, TENCENT_API_KEY. hy3-preview / OpenRouter RETIRED.
        if [ -z "${TENCENT_API_KEY:-}" ]; then
            echo "STOP: hy3 needs \$TENCENT_API_KEY (Tencent TokenHub)" >&2
            exit 1
        fi
        case "$MODEL" in
            ""|hy3|tokenhub/hy3) MODEL=hy3 ;;
            *preview*|tencent/hy3-preview|tencent/hy3)
                echo "STOP: hy3-preview is retired. Use: hy3 hy3 <problem>" >&2
                exit 1
                ;;
            *)
                echo "STOP: hy3 harness only accepts model 'hy3' (got '$MODEL')" >&2
                exit 1
                ;;
        esac
        HY3_RE="${HY3_REASONING_EFFORT:-}"
        if [ -z "$HY3_RE" ]; then
            case "${REASONING_EFFORT:-high}" in
                no_think|none|low|minimal|fast) HY3_RE=no_think ;;
                *) HY3_RE=high ;;
            esac
        fi
        HY3_OC_HOME="$RUN_DIR/opencode_tokenhub_hy3_config"
        mkdir -p "$HY3_OC_HOME/opencode"
        cat > "$HY3_OC_HOME/opencode/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "permission": { "external_directory": "deny" },
  "provider": {
    "tokenhub": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Tencent TokenHub",
      "options": {
        "baseURL": "${HY3_TOKENHUB_BASE_URL:-https://tokenhub.tencentmaas.com/v1}",
        "apiKey": "${TENCENT_API_KEY}",
        "extraBody": { "reasoning_effort": "${HY3_RE}" }
      },
      "models": {
        "hy3": {
          "name": "Hy3",
          "limit": { "context": ${HY3_TOKENHUB_CONTEXT_LIMIT:-196608}, "output": ${HY3_TOKENHUB_OUTPUT_LIMIT:-32000} },
          "tools": true
        }
      }
    }
  }
}
JSON
        ( cd "$PROBLEM_DIR" && "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" \
            env XDG_CONFIG_HOME="$HY3_OC_HOME" \
            opencode run --pure --format json -m tokenhub/hy3 "$PROMPT" \
            </dev/null > "$LOG_FILE" 2> "$STDERR_FILE" ) || HARNESS_EXIT=$?
        ;;

    deepseek-claude)
        # Claude Code routed to DeepSeek's Anthropic-compatible endpoint.
        # Requires DEEPSEEK_API_KEY. Pass MODEL=deepseek-v4-pro.
        if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
            echo "DEEPSEEK_API_KEY is required for deepseek-claude" >&2
            exit 1
        fi
        DEEPSEEK_CLAUDE_ALIAS="${DEEPSEEK_CLAUDE_ALIAS:-opus}"
        DEEPSEEK_CLAUDE_HAIKU_MODEL="${DEEPSEEK_CLAUDE_HAIKU_MODEL:-$MODEL}"
        ( cd "$PROBLEM_DIR" && \
            export ANTHROPIC_AUTH_TOKEN="$DEEPSEEK_API_KEY" && \
            export ANTHROPIC_BASE_URL="${DEEPSEEK_ANTHROPIC_BASE_URL:-https://api.deepseek.com/anthropic}" && \
            export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" && \
            export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}" && \
            export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-1000000}" && \
            export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-128000}" && \
            export ANTHROPIC_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_HAIKU_MODEL="$DEEPSEEK_CLAUDE_HAIKU_MODEL" && \
            export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" && \
            export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" && \
            "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$DEEPSEEK_CLAUDE_ALIAS" \
                --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    codex)
        EFFORT_ARG=()
        if [ -n "$REASONING_EFFORT" ]; then
            EFFORT_ARG=(-c "model_reasoning_effort=\"$REASONING_EFFORT\"")
        fi
        "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" codex exec \
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
        echo "$PROMPT" | "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" kimi \
            -w "$PROBLEM_DIR" \
            --print \
            --output-format stream-json \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    droid)
        EFFORT_ARG=()
        if [ -n "$REASONING_EFFORT" ]; then
            EFFORT_ARG=(--reasoning-effort "$REASONING_EFFORT")
        fi
        "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" droid exec \
            --output-format stream-json \
            --skip-permissions-unsafe \
            --cwd "$PROBLEM_DIR" \
            -m "$MODEL" \
            "${EFFORT_ARG[@]}" \
            "$PROMPT" \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    gemini)
        # Gemini CLI. No --cwd flag, so cd into PROBLEM_DIR. --yolo auto-approves
        # tools; GEMINI_CLI_TRUST_WORKSPACE=true trusts the dir headlessly. This
        # combo is version-independent: 0.36 lacks --skip-trust, 0.47 needs trust
        # set separately from --yolo. The env var works on both.
        ( cd "$PROBLEM_DIR" && export GEMINI_CLI_TRUST_WORKSPACE=true && "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" gemini \
            --yolo \
            -m "$MODEL" \
            -o stream-json \
            -p "$PROMPT" \
            </dev/null > "$LOG_FILE" 2> "$STDERR_FILE" ) || HARNESS_EXIT=$?
        ;;

    cursor)
        # Cursor Agent CLI is installed as `agent` on Anvil.
        "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" agent \
            --trust \
            --yolo \
            --print \
            --output-format stream-json \
            --model "$MODEL" \
            --workspace "$PROBLEM_DIR" \
            "$PROMPT" \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    grok)
        # Grok Build CLI is installed as `grok` on Anvil. Use the top-level
        # headless path because `grok agent` does not accept --cwd/output flags.
        EFFORT_ARG=()
        if [ -n "$REASONING_EFFORT" ]; then
            EFFORT_ARG=(--effort "$REASONING_EFFORT")
        fi
        "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" grok \
            --cwd "$PROBLEM_DIR" \
            --always-approve \
            --permission-mode bypassPermissions \
            --no-memory \
            --output-format streaming-json \
            --model "$MODEL" \
            "${EFFORT_ARG[@]}" \
            -p "$PROMPT" \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        ;;

    opencode)
        # OpenCode SST with custom OpenAI-shape providers (deepseek, zai, minimax).
        # Provider/model pair encoded as MODEL="provider/model-id" e.g.
        # "deepseek/deepseek-v4-pro" or "zai/glm-5.1".
        ( cd "$PROBLEM_DIR" && "${KBH_SBX[@]}" timeout "$BUDGET_SECONDS" opencode run \
            --pure --format json -m "$MODEL" "$PROMPT" \
            </dev/null > "$LOG_FILE" 2> "$STDERR_FILE" ) || HARNESS_EXIT=$?
        ;;

    *)
        echo "Unknown harness: $HARNESS" >&2
        echo "Supported: claude, zai-claude, minimax-claude, kimi-claude, kinetic-claude, longcat-claude, hy3, deepseek-claude, ccr-claude, codex, kimi, droid, gemini, cursor, grok, opencode" >&2
        exit 1
        ;;
esac

HARNESS_END_TIME=$(date +%s)
HARNESS_FINISHED_AT="$(date -Is)"
ELAPSED=$((HARNESS_END_TIME - START_TIME))

# --- Detect whether the harness session ran to completion ----------------
#
# A run is INCOMPLETE if the harness exited from SIGTERM (timeout=124) OR if
# the transcript is missing its terminal marker. Markers per harness:
#   claude / zai-claude / ccr-claude:  {"type":"result"} (final summary with usage)
#   codex:                {"payload":{"type":"task_complete"}}
#   cursor:               {"type":"result"} (final usage block)
#   grok:                 {"type":"end"}
#   droid:                exit code only; stream has init/message/error events
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
    claude|zai-claude|minimax-claude|kimi-claude|kinetic-claude|longcat-claude|deepseek-claude|ccr-claude|cursor|gemini)
        if ! grep -q '"type":"result"' "$CHECK_FILE" 2>/dev/null; then
            SESSION_COMPLETE=false
        fi
        ;;
    grok)
        if ! grep -q '"type":"end"' "$CHECK_FILE" 2>/dev/null; then
            SESSION_COMPLETE=false
        fi
        ;;
    codex)
        if ! grep -q '"type":"task_complete"' "$CHECK_FILE" 2>/dev/null; then
            SESSION_COMPLETE=false
        fi
        ;;
    droid|kimi|opencode|hy3|hy3-claude)
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

if ! detect_template_mutation "after harness"; then
    TEMPLATE_MUTATED=true
    echo "FAIL: immutable problem files changed by harness; skipping check.py and benchmark.py."
    restore_template_files
fi

if [ -f "$PROBLEM_DIR/solution.py" ]; then
    HAS_SOLUTION=true
fi

if [ "$TEMPLATE_MUTATED" = "false" ] && [ "$HAS_SOLUTION" = "true" ]; then
    CHECK_LOG="$RUN_DIR/check.log"
    BENCH_LOG="$RUN_DIR/benchmark.log"

    echo "Running check.py..."
    CHECK_START_TIME=$(date +%s)
    CHECK_EXIT_CODE=0
    (cd "$PROBLEM_DIR" && run_gpu_locked_timeout check.py "$CHECK_TIMEOUT_SECONDS" uv run python check.py) > "$CHECK_LOG" 2>&1 || CHECK_EXIT_CODE=$?
    CHECK_END_TIME=$(date +%s)
    CHECK_ELAPSED=$((CHECK_END_TIME - CHECK_START_TIME))

    if ! detect_template_mutation "after check.py"; then
        TEMPLATE_MUTATED=true
        CORRECT=false
        SCORE="null"
        echo "FAIL: immutable problem files changed during check.py."
        restore_template_files
    elif [ "$CHECK_EXIT_CODE" -eq 0 ] && grep -aq "PASS" "$CHECK_LOG"; then
        # Not anchored (^PASS): solution stdout without a trailing newline can
        # glue onto check.py's PASS marker; require check.py exit 0 alongside
        # the marker instead (same fix as benchmarks/hard, 2026-07-07).
        CORRECT=true
        echo "Running benchmark.py..."
        # Some problems (KDA chunked recurrence, sonic-MoE)
        # have references that loop in Python, so 20 perf trials × 4 variants ×
        # 5 shapes can take 5-10 min. Generous budget.
        BENCH_START_TIME=$(date +%s)
        BENCH_EXIT_CODE=0
        (cd "$PROBLEM_DIR" && run_gpu_locked_timeout benchmark.py "$BENCHMARK_TIMEOUT_SECONDS" uv run python benchmark.py) > "$BENCH_LOG" 2>&1 || BENCH_EXIT_CODE=$?
        BENCH_END_TIME=$(date +%s)
        BENCH_ELAPSED=$((BENCH_END_TIME - BENCH_START_TIME))
        if ! detect_template_mutation "after benchmark.py"; then
            TEMPLATE_MUTATED=true
            CORRECT=false
            SCORE="null"
            echo "FAIL: immutable problem files changed during benchmark.py."
            restore_template_files
        else
            SCORE=$(grep -oP 'peak_fraction:\s*\K[0-9.]+' "$BENCH_LOG" | head -1 || echo "null")
        fi
    fi
fi

CHECK_ELAPSED="${CHECK_ELAPSED:-null}"
BENCH_ELAPSED="${BENCH_ELAPSED:-null}"
CHECK_EXIT_CODE="${CHECK_EXIT_CODE:-null}"
BENCH_EXIT_CODE="${BENCH_EXIT_CODE:-null}"
FINISH_TIME=$(date +%s)
FINISHED_AT="$(date -Is)"
TOTAL_ELAPSED=$((FINISH_TIME - START_TIME))

# Extract token usage from the transcript so we have an apples-to-apples
# count for cost comparison even when the harness uses a coding-plan billing
# (which hides per-call USD).
USAGE_JSON="$(
    KBH_GPU_LOCK_HELD=1 uv run --quiet python \
        "$REPO_ROOT/scripts/extract_usage.py" "$RUN_DIR" "$HARNESS" 2>/dev/null || echo '{}'
)"
OUTPUT_TOKENS_PER_SECOND="$(USAGE_JSON="$USAGE_JSON" ELAPSED="$ELAPSED" KBH_GPU_LOCK_HELD=1 uv run --quiet python - <<'PY'
import json
import os

try:
    usage = json.loads(os.environ["USAGE_JSON"])
except json.JSONDecodeError:
    usage = {}
elapsed = int(os.environ["ELAPSED"])
out = usage.get("output_tokens")
if isinstance(out, (int, float)) and elapsed > 0:
    print(out / elapsed)
else:
    print("null")
PY
)"

CLASSIFICATION_JSON="$(
    USAGE_JSON="$USAGE_JSON" \
    CORRECT="$CORRECT" \
    TEMPLATE_MUTATED="$TEMPLATE_MUTATED" \
    HAS_SOLUTION="$HAS_SOLUTION" \
    SESSION_COMPLETE="$SESSION_COMPLETE" \
    HARNESS_EXIT="$HARNESS_EXIT" \
    CHECK_EXIT_CODE="$CHECK_EXIT_CODE" \
    BENCH_EXIT_CODE="$BENCH_EXIT_CODE" \
    LOG_FILE="$LOG_FILE" \
    STDERR_FILE="$STDERR_FILE" \
    MIN_USEFUL_OUTPUT_TOKENS="$MIN_USEFUL_OUTPUT_TOKENS" \
    KBH_GPU_LOCK_HELD=1 uv run --quiet python scripts/classify_run.py
)"
FAILURE_REASON="$(CLASSIFICATION_JSON="$CLASSIFICATION_JSON" KBH_GPU_LOCK_HELD=1 uv run --quiet python - <<'PY'
import json
import os
print(json.loads(os.environ["CLASSIFICATION_JSON"])["failure_reason"])
PY
)"
RETRYABLE_INFRA_FAILURE="$(CLASSIFICATION_JSON="$CLASSIFICATION_JSON" KBH_GPU_LOCK_HELD=1 uv run --quiet python - <<'PY'
import json
import os
print("true" if json.loads(os.environ["CLASSIFICATION_JSON"])["retryable_infra_failure"] else "false")
PY
)"

cat > "$RUN_DIR/result.json" <<JSON
{
    "run_id": "$RUN_ID",
    "run_group": "$RUN_GROUP",
    "problem": "$PROBLEM_NAME",
    "harness": "$HARNESS",
    "model": "$MODEL",
    "reasoning_effort": "$REASONING_EFFORT",
    "started_at": "$STARTED_AT",
    "harness_finished_at": "$HARNESS_FINISHED_AT",
    "finished_at": "$FINISHED_AT",
    "start_epoch": $START_TIME,
    "harness_end_epoch": $HARNESS_END_TIME,
    "end_epoch": $FINISH_TIME,
    "has_solution": $HAS_SOLUTION,
    "correct": $CORRECT,
    "failure_reason": "$FAILURE_REASON",
    "retryable_infra_failure": $RETRYABLE_INFRA_FAILURE,
    "minimum_useful_output_tokens": $MIN_USEFUL_OUTPUT_TOKENS,
    "peak_fraction": $SCORE,
    "template_mutated": $TEMPLATE_MUTATED,
    "elapsed_seconds": $ELAPSED,
    "total_elapsed_seconds": $TOTAL_ELAPSED,
    "check_elapsed_seconds": $CHECK_ELAPSED,
    "benchmark_elapsed_seconds": $BENCH_ELAPSED,
    "check_timeout_seconds": $CHECK_TIMEOUT_SECONDS,
    "benchmark_timeout_seconds": $BENCHMARK_TIMEOUT_SECONDS,
    "check_exit_code": $CHECK_EXIT_CODE,
    "benchmark_exit_code": $BENCH_EXIT_CODE,
    "harness_exit_code": $HARNESS_EXIT,
    "session_complete": $SESSION_COMPLETE,
    "agent_cuda_disabled": $AGENT_CUDA_DISABLED,
    "gpu_queue_mode": "$GPU_QUEUE_MODE",
    "output_tokens_per_second": $OUTPUT_TOKENS_PER_SECOND,
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
elif [ "$TEMPLATE_MUTATED" = "true" ]; then
    STATUS="INVALID (problem files changed)"
elif $HAS_SOLUTION; then
    STATUS="FAIL (check failed)"
elif [ "$RETRYABLE_INFRA_FAILURE" = "true" ]; then
    STATUS="INFRA ($FAILURE_REASON)"
else
    STATUS="ERR ($FAILURE_REASON)"
fi

echo "========================================"
echo "[$STATUS] $PROBLEM_NAME (${ELAPSED}s)"
echo "Archive: $RUN_DIR"
echo "========================================"
