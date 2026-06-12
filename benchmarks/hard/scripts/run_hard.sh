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
#   ./scripts/run_hard.sh droid glm-5.1 problems/01_fp8_gemm
#   ./scripts/run_hard.sh grok grok-build problems/01_fp8_gemm max
#   ./scripts/run_hard.sh zai-claude glm-5.1 problems/01_fp8_gemm
#   ./scripts/run_hard.sh minimax-claude MiniMax-M3 problems/01_fp8_gemm
#   ./scripts/run_hard.sh ccr-claude glm-5.1 problems/01_fp8_gemm
#   ./scripts/run_hard.sh opencode-nemotron nvidia/nemotron-3-ultra-550b-a55b problems/01_fp8_gemm
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
SOURCE_PROBLEM_DIR="${3:?problem_dir required}"
REASONING_EFFORT="${4:-}"
CLAUDE_KBH_SETTINGS="${CLAUDE_KBH_SETTINGS:-{\"fastMode\":false,\"alwaysThinkingEnabled\":true}}"
KBH_AGENT_CONTAINER="${KBH_AGENT_CONTAINER:-0}"
KBH_AGENT_CONTAINER_IMAGE="${KBH_AGENT_CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:latest}"
KBH_AGENT_CONTAINER_NETWORK="${KBH_AGENT_CONTAINER_NETWORK:-bridge}"
KBH_AGENT_CONTAINER_CUDA_HOME="${KBH_AGENT_CONTAINER_CUDA_HOME:-/usr/local/cuda-13.2}"
KBH_AGENT_CONTAINER_CLAUDE_BIN="${KBH_AGENT_CONTAINER_CLAUDE_BIN:-$HOME/.local/share/claude/versions/2.1.150}"
KBH_AGENT_CONTAINER_CODEX_NODE="${KBH_AGENT_CONTAINER_CODEX_NODE:-$HOME/.local/node-v22.14.0-linux-x64}"
KBH_AGENT_CONTAINER_OPENCODE_BIN="${KBH_AGENT_CONTAINER_OPENCODE_BIN:-$HOME/.opencode/bin/opencode}"
KBH_AGENT_CONTAINER_DROID_BIN="${KBH_AGENT_CONTAINER_DROID_BIN:-$HOME/.local/bin/droid}"
KBH_AGENT_CONTAINER_CURSOR_DIR="${KBH_AGENT_CONTAINER_CURSOR_DIR:-$HOME/.local/share/cursor-agent/versions/2026.05.27-fe9a6e2}"
KBH_AGENT_CONTAINER_GROK_DIR="${KBH_AGENT_CONTAINER_GROK_DIR:-$HOME/.grok}"
KBH_AGENT_CONTAINER_GEMINI_DIR="${KBH_AGENT_CONTAINER_GEMINI_DIR:-/usr/lib/node_modules/@google/gemini-cli}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_PROBLEM_DIR="$(cd "$SOURCE_PROBLEM_DIR" && pwd)"

# Shared across container runs so the workspace uv env (same uv.lock as host
# scoring) does not re-download wheels or managed pythons every run.
KBH_AGENT_CONTAINER_UV_CACHE="${KBH_AGENT_CONTAINER_UV_CACHE:-$REPO_ROOT/outputs/container_uv_cache}"
# Pre-warmed opencode home (clean, migrated sqlite DB, no host session data).
# Built once via scripts/warm_opencode_home.sh; copied into each run's
# agent_home so opencode does not redo its DB migration inside the budget.
KBH_OPENCODE_HOME_TEMPLATE="${KBH_OPENCODE_HOME_TEMPLATE:-$REPO_ROOT/outputs/opencode_home_template}"
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
NVCF_PROXY_PID=""
NVCF_PROXY_BASE_URL=""

cleanup() {
    if [ -n "${NVCF_PROXY_PID:-}" ] && kill -0 "$NVCF_PROXY_PID" 2>/dev/null; then
        kill "$NVCF_PROXY_PID" 2>/dev/null || true
        wait "$NVCF_PROXY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wall clock budget: 45 minutes per run. Override via BUDGET_SECONDS env var
# (e.g. BUDGET_SECONDS=300 for a quick smoke test).
BUDGET_SECONDS="${BUDGET_SECONDS:-2700}"
CHECK_TIMEOUT_SECONDS="${KBH_CHECK_TIMEOUT_SECONDS:-180}"
if [ "$PROBLEM_NAME" = "02_kda_cutlass" ]; then
    BENCHMARK_TIMEOUT_SECONDS="${KBH_BENCHMARK_TIMEOUT_02_KDA_CUTLASS_SECONDS:-${KBH_BENCHMARK_TIMEOUT_SECONDS:-7200}}"
else
    BENCHMARK_TIMEOUT_SECONDS="${KBH_BENCHMARK_TIMEOUT_SECONDS:-1800}"
fi
export KBH_GPU_LOCK_WAIT_TIMEOUT_SECONDS="${KBH_GPU_LOCK_WAIT_TIMEOUT_SECONDS:-7200}"
MIN_USEFUL_OUTPUT_TOKENS="${KBH_MIN_USEFUL_OUTPUT_TOKENS:-5000}"

# Optional mode for concurrent smoke sweeps: agents can edit/reason in parallel
# while the harness-owned post-run check.py/benchmark.py path is the only CUDA
# consumer. This is more reliable than trying to intercept every absolute
# .venv/bin/python path an agent may discover.
AGENT_CUDA_DISABLED=false
GPU_QUEUE_MODE="path_wrapper_gpu_lock"
if [ "${KBH_DISABLE_AGENT_CUDA:-0}" = "1" ]; then
    AGENT_CUDA_DISABLED=true
    GPU_QUEUE_MODE="agent_phase_cuda_guard_harness_gpu_lock"
fi
if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
    if [ "${KBH_AGENT_CONTAINER_SESSION_LOCK:-0}" = "1" ]; then
        # Legacy: the whole agent session holds the GPU lock (fully serial).
        GPU_QUEUE_MODE="agent_container_native_profiling_harness_gpu_lock"
    else
        # Default: sessions run in parallel; in-container GPU-facing commands
        # serialize per-command through the shared lock, like host sweeps.
        GPU_QUEUE_MODE="agent_container_native_profiling_path_wrapper_gpu_lock"
    fi
fi

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
if [ "${KBH_DISABLE_AGENT_CUDA:-0}" = "1" ]; then
    PROMPT="${PROMPT}

Parallel sweep note: CUDA is intentionally unavailable during your editing
phase so many models can work at once without contaminating GPU timings. Do not
spend time running check.py, benchmark.py, ncu, nsys, or CUDA profiling
commands. Focus on writing the best final solution.py you can. The
harness will run check.py and benchmark.py after your session under the GPU
lock and archive those logs."
fi

# --- Create an isolated per-run problem workspace -------------------------
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

WORKSPACE_ROOT="$RUN_DIR/repo"
PROBLEM_DIR="$WORKSPACE_ROOT/problems/$PROBLEM_NAME"
mkdir -p "$PROBLEM_DIR"

if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
    PROMPT_WORKSPACE_DIR="/workspace/problems/$PROBLEM_NAME"
    PROMPT_SOURCE_NOTE="The source repository's problems/ tree is not mounted."
else
    PROMPT_WORKSPACE_DIR="$PROBLEM_DIR"
    PROMPT_SOURCE_NOTE="Do not write to ${SOURCE_PROBLEM_DIR} or to the source repository's problems/ tree."
fi
PROMPT="${PROMPT}

Workspace isolation note: you are already running inside the archive-local
problem workspace, ${PROMPT_WORKSPACE_DIR}. Write the final answer to
solution.py in the current directory only. ${PROMPT_SOURCE_NOTE}"
if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
    PROMPT="${PROMPT}

Container note: inside this run, the visible workspace path is
/workspace/problems/${PROBLEM_NAME}. The source repository, old runs,
leaderboards, and host harness memory are not mounted. Container network mode is
${KBH_AGENT_CONTAINER_NETWORK}. Run all Python through \`uv run ...\` so you use
the workspace uv environment; it is built from the same uv.lock as the official
scoring environment. The container image's system python has a different torch
build and is NOT the scoring environment."
fi

# check.py and benchmark.py derive REPO_ROOT as parents[2]. Keep that shape
# while sharing src/. Copy project metadata so agents can mutate dependencies
# inside their disposable workspace without touching the source repo.
if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
    cp -a "$REPO_ROOT/src" "$WORKSPACE_ROOT/src"
else
    ln -s "$REPO_ROOT/src" "$WORKSPACE_ROOT/src"
fi
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
REAL_DOCKER="$(command -v docker || true)"
REAL_TIMEOUT="$(command -v timeout)"
REAL_UV_FALLBACK="$REAL_UV"
REAL_PYTHON_FALLBACK="$REAL_PYTHON"
LOCK_WRAPPER_DIR="$RUN_DIR/bin"
AGENT_GUARD_DIR="$RUN_DIR/agent_guard"
mkdir -p "$LOCK_WRAPPER_DIR" "$RUN_DIR/cache/torch_extensions" \
    "$RUN_DIR/cache/triton" "$RUN_DIR/cache/cuda" "$RUN_DIR/tmp" \
    "$AGENT_GUARD_DIR"

# The lock lives in its own directory so container runs can bind-mount just
# the lock (flock is on the inode, so host and container commands serialize
# against each other) without exposing the rest of outputs/.
KBH_GPU_LOCK_DIR="${KBH_GPU_LOCK_DIR:-$REPO_ROOT/outputs/gpu_lock}"
mkdir -p "$KBH_GPU_LOCK_DIR"
export KBH_GPU_LOCK="${KBH_GPU_LOCK:-$KBH_GPU_LOCK_DIR/gpu.lock}"
export KBH_GPU_LOCK_LOG="$RUN_DIR/gpu_lock.log"
export TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
export TMPDIR="$RUN_DIR/tmp"
export TEMP="$RUN_DIR/tmp"
export TMP="$RUN_DIR/tmp"
export RUN_DIR REAL_UV REAL_PYTHON REAL_NVIDIA_SMI REAL_NCU REAL_NSYS REAL_NVCC \
    REAL_UV_FALLBACK REAL_PYTHON_FALLBACK AGENT_GUARD_DIR

cat > "$AGENT_GUARD_DIR/sitecustomize.py" <<'PY'
"""Block accidental CUDA use during parallel agent-edit phases.

The harness-owned post-run check.py/benchmark.py path runs without
KBH_AGENT_PHASE and is still allowed to use the GPU under outputs/gpu.lock.
"""
from __future__ import annotations

import builtins
import importlib
import os


if os.environ.get("KBH_AGENT_PHASE") == "1":
    _orig_import_module = importlib.import_module
    _orig_import = builtins.__import__
    _orig_torch_device = None

    def _patch_torch(mod):
        global _orig_torch_device
        try:
            if _orig_torch_device is None:
                _orig_torch_device = mod.device
            cuda = getattr(mod, "cuda", None)
            def _blocked(*args, **kwargs):
                raise RuntimeError(
                    "CUDA is disabled during KernelBench parallel agent phase; "
                    "the harness will run check.py and benchmark.py after generation."
                )

            def _mentions_cuda(value):
                try:
                    if isinstance(value, str):
                        return value.startswith("cuda")
                    if getattr(value, "type", None) == "cuda":
                        return True
                except Exception:
                    pass
                return False

            if cuda is not None:
                cuda.is_available = lambda: False
                cuda.device_count = lambda: 0
                cuda.current_device = _blocked
                cuda.get_device_name = _blocked
                cuda.get_device_capability = _blocked
                cuda.init = _blocked
                cuda.synchronize = _blocked
            mod.device = lambda *args, **kwargs: (
                _blocked() if args and _mentions_cuda(args[0])
                else _orig_torch_device(*args, **kwargs)
            )
            if not getattr(mod, "_kbh_agent_cuda_patched", False):
                tensor_to = mod.Tensor.to
                module_to = mod.nn.Module.to

                def guarded_tensor_to(self, *args, **kwargs):
                    if any(_mentions_cuda(arg) for arg in args) or _mentions_cuda(kwargs.get("device")):
                        _blocked()
                    return tensor_to(self, *args, **kwargs)

                def guarded_module_to(self, *args, **kwargs):
                    if any(_mentions_cuda(arg) for arg in args) or _mentions_cuda(kwargs.get("device")):
                        _blocked()
                    return module_to(self, *args, **kwargs)

                mod.Tensor.to = guarded_tensor_to
                mod.Tensor.cuda = lambda self, *args, **kwargs: _blocked()
                mod.nn.Module.to = guarded_module_to
                mod.nn.Module.cuda = lambda self, *args, **kwargs: _blocked()
                mod._kbh_agent_cuda_patched = True
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        except Exception:
            pass
        return mod

    def _guarded_import_module(name, package=None):
        mod = _orig_import_module(name, package)
        if name == "torch" or name.startswith("torch."):
            torch = _orig_import_module("torch")
            _patch_torch(torch)
        return mod

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _orig_import(name, globals, locals, fromlist, level)
        if name == "torch" or name.startswith("torch."):
            torch = _orig_import("torch")
            _patch_torch(torch)
        return mod

    importlib.import_module = _guarded_import_module
    builtins.__import__ = _guarded_import
PY

AGENT_CUDA_ENV=()
if [ "${KBH_DISABLE_AGENT_CUDA:-0}" = "1" ]; then
    AGENT_CUDA_ENV=(
        env
        CUDA_VISIBLE_DEVICES=
        KBH_AGENT_PHASE=1
        PYTHONPATH="$AGENT_GUARD_DIR${PYTHONPATH:+:$PYTHONPATH}"
    )
fi

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
if [ "${KBH_AGENT_PHASE:-0}" = "1" ]; then
    case "$name" in
        uv|python|python3|nvidia-smi|nvcc)
            exec "$real" "$@"
            ;;
        ncu|nsys)
            echo "$name is disabled during KernelBench parallel agent phase; official benchmarking runs under the GPU lock after generation." >&2
            exit 125
            ;;
    esac
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

# Container-side lock wrappers. Mounted at /kbh/bin (first on PATH) inside
# agent containers so in-container GPU-facing commands take the same host
# flock per-command. Real binaries resolve lazily from a PATH that excludes
# /kbh/bin, so the wrappers never recurse into themselves.
CONTAINER_LOCK_BIN="$RUN_DIR/cbin"
if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
    mkdir -p "$CONTAINER_LOCK_BIN"
    cp "$LOCK_WRAPPER_DIR/gpu-lock-exec" "$CONTAINER_LOCK_BIN/gpu-lock-exec"
    for tool in uv python python3 nvidia-smi nvcc ncu nsys; do
        cat > "$CONTAINER_LOCK_BIN/$tool" <<CWRAP
#!/bin/bash
real="\$(PATH="/usr/local/cuda-host/bin:/opt/node/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/cuda/bin:/opt/nvidia/nsight-compute:/opt/nvidia/nsight-systems/bin" command -v $tool || true)"
exec /kbh/bin/gpu-lock-exec $tool "\$real" "\$@"
CWRAP
        chmod +x "$CONTAINER_LOCK_BIN/$tool"
    done
    chmod +x "$CONTAINER_LOCK_BIN/gpu-lock-exec"
fi

run_gpu_locked_timeout() {
    local lock_name="$1"
    local timeout_seconds="$2"
    shift 2
    "$RUN_DIR/bin/gpu-lock-exec" "$lock_name" "$REAL_TIMEOUT" \
        --kill-after="${KBH_TIMEOUT_KILL_AFTER_SECONDS:-30}s" "${timeout_seconds}s" "$@"
}

run_docker_locked_timeout() {
    local lock_name="$1"
    local timeout_seconds="$2"
    local cidfile="$3"
    shift 3
    rm -f "$cidfile"
    # Optional stall watchdog: when KBH_STALL_WATCH_LOG is set, kill the
    # container if that file stops growing for KBH_STALL_SECONDS. Guards
    # against the opencode OpenAI-compatible adapter hang (DEVLOG 2026-06-09)
    # where a session goes permanently silent mid-stream. The threshold must
    # exceed the longest legitimate silent reasoning phase (deepseek-v4-pro
    # was observed thinking quietly for 400s+).
    local watcher_pid=""
    if [ -n "${KBH_STALL_WATCH_LOG:-}" ] && [ "${KBH_STALL_SECONDS:-0}" -gt 0 ]; then
        (
            while true; do
                sleep 30
                [ -s "$cidfile" ] || continue
                local_cid="$(cat "$cidfile" 2>/dev/null)" || continue
                "$REAL_DOCKER" inspect "$local_cid" >/dev/null 2>&1 || break
                now="$(date +%s)"
                mt="$(stat -c %Y "$KBH_STALL_WATCH_LOG" 2>/dev/null || echo "$now")"
                if [ $((now - mt)) -ge "$KBH_STALL_SECONDS" ]; then
                    printf '%s stall_watchdog killed %s after %ss of silence (%s)\n' \
                        "$(date -Is)" "$local_cid" "$KBH_STALL_SECONDS" "$lock_name" \
                        >> "$RUN_DIR/stall_watchdog.log"
                    "$REAL_DOCKER" rm -f "$local_cid" >/dev/null 2>&1 || true
                    break
                fi
            done
        ) &
        watcher_pid=$!
    fi
    set +e
    if [ "${KBH_AGENT_CONTAINER_SESSION_LOCK:-0}" = "1" ]; then
        # Legacy fully-serial mode: the session itself holds the GPU lock.
        run_gpu_locked_timeout "$lock_name" "$timeout_seconds" "$REAL_DOCKER" "$@"
    else
        # Parallel mode: sessions overlap; the container's own GPU-facing
        # commands serialize per-command via the mounted /kbh/bin wrappers.
        "$REAL_TIMEOUT" --kill-after="${KBH_TIMEOUT_KILL_AFTER_SECONDS:-30}s" \
            "${timeout_seconds}s" "$REAL_DOCKER" "$@"
    fi
    local status=$?
    set -e
    if [ -n "$watcher_pid" ]; then
        kill "$watcher_pid" 2>/dev/null || true
        wait "$watcher_pid" 2>/dev/null || true
    fi
    if [ -s "$cidfile" ]; then
        "$REAL_DOCKER" rm -f "$(cat "$cidfile")" >/dev/null 2>&1 || true
        rm -f "$cidfile"
    fi
    return "$status"
}

start_nvcf_proxy() {
    if [ -z "${NGC_API_KEY:-${NVIDIA_API_KEY:-${NVCF_API_KEY:-}}}" ]; then
        echo "NGC_API_KEY, NVIDIA_API_KEY, or NVCF_API_KEY is required for nvcf-nemotron" >&2
        return 1
    fi

    local proxy_log="$RUN_DIR/nvcf_proxy.log"
    KBH_GPU_LOCK_HELD=1 uv run python "$REPO_ROOT/scripts/nvcf_openai_proxy.py" \
        --host 127.0.0.1 --port 0 > "$proxy_log" 2>&1 &
    NVCF_PROXY_PID=$!

    local base_url=""
    for _ in $(seq 1 100); do
        if ! kill -0 "$NVCF_PROXY_PID" 2>/dev/null; then
            echo "NVCF proxy exited before startup; see $proxy_log" >&2
            return 1
        fi
        base_url="$(grep -oE 'http://127\.0\.0\.1:[0-9]+' "$proxy_log" | tail -1 || true)"
        if [ -n "$base_url" ]; then
            NVCF_PROXY_BASE_URL="$base_url"
            return 0
        fi
        sleep 0.1
    done
    echo "Timed out waiting for NVCF proxy startup; see $proxy_log" >&2
    return 1
}

write_nvcf_opencode_config() {
    local base_url="$1"
    local config_home="$RUN_DIR/opencode_config"
    mkdir -p "$config_home/opencode"
    cat > "$config_home/opencode/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "permission": {
    "external_directory": "deny"
  },
  "provider": {
    "nvcf-nemotron": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "NVIDIA NVCF Nemotron",
      "options": {
        "baseURL": "${base_url}/v1",
        "apiKey": "nvcf-proxy"
      },
      "models": {
        "nemotron-3-ultra": {
          "name": "Nemotron 3 Ultra via NVCF",
          "limit": {
            "context": 200000,
            "output": 4096
          },
          "tools": true
        }
      }
    }
  }
}
JSON
    printf '%s\n' "$config_home"
}


write_openrouter_deepinfra_opencode_config() {
    if [ -z "${OPENROUTER_API_KEY:-}" ]; then
        echo "OPENROUTER_API_KEY is required for opencode-nemotron" >&2
        return 1
    fi

    local model="$1"
    local config_home="$RUN_DIR/opencode_openrouter_deepinfra_config"
    mkdir -p "$config_home/opencode"
    cat > "$config_home/opencode/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "permission": {
    "external_directory": "deny"
  },
  "provider": {
    "openrouter-deepinfra": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenRouter DeepInfra",
      "options": {
        "baseURL": "https://openrouter.ai/api/v1",
        "apiKey": "${OPENROUTER_API_KEY}",
        "headers": {
          "HTTP-Referer": "https://kernelbench.com",
          "X-Title": "KernelBench-Hard"
        },
        "extraBody": {
          "provider": {
            "order": ["DeepInfra"],
            "allow_fallbacks": false
          }
        }
      },
      "models": {
        "${model}": {
          "name": "NVIDIA Nemotron 3 Ultra via OpenRouter DeepInfra",
          "limit": {
            "context": 262144,
            "output": 16384
          },
          "tools": true
        }
      }
    }
  }
}
JSON
    printf '%s\n' "$config_home"
}

prepare_claude_container_home() {
    # $1: copy Anthropic credentials (1, default) or not (0). Claude-compat
    # reroutes (Z.ai / MiniMax) authenticate via ANTHROPIC_AUTH_TOKEN and must
    # NOT carry Anthropic credentials, so a mapping failure errors out instead
    # of silently spending against the real Anthropic API.
    local copy_credentials="${1:-1}"
    local home_dir="$RUN_DIR/agent_home"
    mkdir -p "$home_dir/.claude"
    chmod 700 "$home_dir" "$home_dir/.claude"
    if [ "$copy_credentials" = "1" ] && [ -f "$HOME/.claude/.credentials.json" ]; then
        cp -p "$HOME/.claude/.credentials.json" "$home_dir/.claude/.credentials.json"
    fi
    printf '{}\n' > "$home_dir/.claude.json"
    printf '%s\n' "$home_dir"
}

# Claude-compat container support: routes export these variables in their
# launch subshell, then list the NAMES here so docker passes them through
# without putting secret values on the docker CLI argv (ps-visible).
CLAUDE_CONTAINER_ENV_NAMES=()
CLAUDE_CONTAINER_EXTRA_CLAUDE_ARGS=()

prepare_codex_container_home() {
    local home_dir="$RUN_DIR/agent_home"
    mkdir -p "$home_dir/.codex"
    if [ -f "$HOME/.codex/auth.json" ]; then
        cp -p "$HOME/.codex/auth.json" "$home_dir/.codex/auth.json"
    fi
    cat > "$home_dir/.codex/config.toml" <<'EOF'
model = "gpt-5.5"
model_reasoning_effort = "low"

[projects."/workspace/problems"]
trust_level = "trusted"
EOF
    printf '%s\n' "$home_dir"
}

prepare_opencode_container_home() {
    local home_dir="$RUN_DIR/agent_home"
    mkdir -p "$home_dir"
    if [ -d "$KBH_OPENCODE_HOME_TEMPLATE" ]; then
        cp -a "$KBH_OPENCODE_HOME_TEMPLATE/." "$home_dir/"
    fi
    if [ -f "$HOME/.config/opencode/opencode.json" ]; then
        mkdir -p "$home_dir/.config/opencode"
        cp -p "$HOME/.config/opencode/opencode.json" "$home_dir/.config/opencode/opencode.json"
    fi
    # Route-specific config override (opencode-nemotron writes an archive-local
    # config pinned to DeepInfra). Without this the container runs the default
    # config, the provider is missing, and the session dies instantly.
    if [ -n "${KBH_OPENCODE_CONFIG_FILE:-}" ] && [ -f "$KBH_OPENCODE_CONFIG_FILE" ]; then
        mkdir -p "$home_dir/.config/opencode"
        cp -p "$KBH_OPENCODE_CONFIG_FILE" "$home_dir/.config/opencode/opencode.json"
    fi
    printf '%s\n' "$home_dir"
}

prepare_droid_container_home() {
    local home_dir="$RUN_DIR/agent_home"
    mkdir -p "$home_dir/.factory"
    for f in auth.json auth.v2.file auth.v2.key settings.json; do
        if [ -f "$HOME/.factory/$f" ]; then
            cp -p "$HOME/.factory/$f" "$home_dir/.factory/$f"
        fi
    done
    printf '%s\n' "$home_dir"
}

prepare_cursor_container_home() {
    local home_dir="$RUN_DIR/agent_home"
    if [ -f "$HOME/.config/cursor/auth.json" ]; then
        mkdir -p "$home_dir/.config/cursor"
        cp -p "$HOME/.config/cursor/auth.json" "$home_dir/.config/cursor/auth.json"
    fi
    if [ -f "$HOME/.cursor/cli-config.json" ]; then
        mkdir -p "$home_dir/.cursor"
        cp -p "$HOME/.cursor/cli-config.json" "$home_dir/.cursor/cli-config.json"
    fi
    if [ -f "$HOME/.cursor/agent-cli-state.json" ]; then
        mkdir -p "$home_dir/.cursor"
        cp -p "$HOME/.cursor/agent-cli-state.json" "$home_dir/.cursor/agent-cli-state.json"
    fi
    printf '%s\n' "$home_dir"
}

check_container_basics() {
    if [ -z "$REAL_DOCKER" ]; then
        echo "docker is required for KBH_AGENT_CONTAINER=1" >&2
        return 127
    fi
    if [ ! -d "$KBH_AGENT_CONTAINER_CUDA_HOME" ]; then
        echo "CUDA toolkit missing for container mode: $KBH_AGENT_CONTAINER_CUDA_HOME" >&2
        return 127
    fi
    if [ ! -x "$REAL_UV" ]; then
        echo "uv binary missing for container mode: $REAL_UV" >&2
        return 127
    fi
    mkdir -p "$KBH_AGENT_CONTAINER_UV_CACHE"
}

run_claude_container() {
    local effort="$1"
    local model_arg="${2:-$MODEL}"
    local copy_credentials="${3:-1}"
    local agent_home
    agent_home="$(prepare_claude_container_home "$copy_credentials")"
    local cidfile="$RUN_DIR/claude_container.cid"
    check_container_basics || return $?
    if [ ! -x "$KBH_AGENT_CONTAINER_CLAUDE_BIN" ]; then
        echo "Claude binary missing for container mode: $KBH_AGENT_CONTAINER_CLAUDE_BIN" >&2
        return 127
    fi
    if [ ! -d "$KBH_AGENT_CONTAINER_CUDA_HOME" ]; then
        echo "CUDA toolkit missing for container mode: $KBH_AGENT_CONTAINER_CUDA_HOME" >&2
        return 127
    fi
    local -a effort_arg=()
    if [ -n "$effort" ]; then
        effort_arg=(--effort "$effort")
    fi
    local -a extra_env_args=()
    local env_name
    for env_name in ${CLAUDE_CONTAINER_ENV_NAMES[@]+"${CLAUDE_CONTAINER_ENV_NAMES[@]}"}; do
        extra_env_args+=(-e "$env_name")
    done
    local -a docker_args=(
        run --rm
        --cidfile "$cidfile"
        --gpus all
        --network "$KBH_AGENT_CONTAINER_NETWORK"
        --cap-add CAP_PERFMON
        --security-opt no-new-privileges
        --shm-size 2g
        --user "$(id -u):$(id -g)"
        -e HOME=/home/agent
        -e USER=agent
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        -e BASH_ENV=
        -e ENV=
        -e ANTHROPIC_API_KEY
        -e ANTHROPIC_AUTH_TOKEN
        -e CLAUDE_CODE_OAUTH_TOKEN
        ${extra_env_args[@]+"${extra_env_args[@]}"}
        -e CUDA_HOME=/usr/local/cuda-host
        -e UV_CACHE_DIR=/uv-cache
        -e UV_PYTHON_INSTALL_DIR=/uv-cache/python
        -e RUN_DIR=/kbh
        -e KBH_GPU_LOCK=/kbh/lock/gpu.lock
        -e KBH_GPU_LOCK_LOG=/home/agent/gpu_lock_container.log
        -e PATH=/kbh/bin:/usr/local/cuda-host/bin:/usr/local/bin:/usr/bin:/bin
        -v "$WORKSPACE_ROOT:/workspace:rw"
        -v "$agent_home:/home/agent:rw"
        -v "$KBH_AGENT_CONTAINER_CUDA_HOME:/usr/local/cuda-host:ro"
        -v "$KBH_AGENT_CONTAINER_CLAUDE_BIN:/usr/local/bin/claude:ro"
        -v "$REAL_UV:/usr/local/bin/uv:ro"
        -v "$KBH_AGENT_CONTAINER_UV_CACHE:/uv-cache:rw"
        -v "$CONTAINER_LOCK_BIN:/kbh/bin:ro"
        -v "$KBH_GPU_LOCK_DIR:/kbh/lock:rw"
        -w "/workspace/problems/$PROBLEM_NAME"
        "$KBH_AGENT_CONTAINER_IMAGE"
        claude
        --dangerously-skip-permissions
        --print --verbose
        --output-format stream-json
        --no-session-persistence
        --settings "$CLAUDE_KBH_SETTINGS"
        --model "$model_arg"
        "${effort_arg[@]}"
        ${CLAUDE_CONTAINER_EXTRA_CLAUDE_ARGS[@]+"${CLAUDE_CONTAINER_EXTRA_CLAUDE_ARGS[@]}"}
        --add-dir "/workspace/problems/$PROBLEM_NAME"
        -p "$PROMPT"
    )
    run_docker_locked_timeout claude-container "$BUDGET_SECONDS" "$cidfile" "${docker_args[@]}"
}

run_codex_container() {
    local effort="$1"
    local agent_home
    agent_home="$(prepare_codex_container_home)"
    local cidfile="$RUN_DIR/codex_container.cid"
    check_container_basics || return $?
    if [ ! -x "$KBH_AGENT_CONTAINER_CODEX_NODE/bin/codex" ]; then
        echo "Codex binary missing for container mode: $KBH_AGENT_CONTAINER_CODEX_NODE/bin/codex" >&2
        return 127
    fi
    local -a effort_arg=()
    if [ -n "$effort" ]; then
        effort_arg=(-c "model_reasoning_effort=\"$effort\"")
    fi
    local -a docker_args=(
        run --rm
        --cidfile "$cidfile"
        --gpus all
        --network "$KBH_AGENT_CONTAINER_NETWORK"
        --cap-add CAP_PERFMON
        --security-opt no-new-privileges
        --shm-size 2g
        --user "$(id -u):$(id -g)"
        -e HOME=/home/agent
        -e USER=agent
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        -e BASH_ENV=
        -e ENV=
        -e OPENAI_API_KEY
        -e CUDA_HOME=/usr/local/cuda-host
        -e UV_CACHE_DIR=/uv-cache
        -e UV_PYTHON_INSTALL_DIR=/uv-cache/python
        -e RUN_DIR=/kbh
        -e KBH_GPU_LOCK=/kbh/lock/gpu.lock
        -e KBH_GPU_LOCK_LOG=/home/agent/gpu_lock_container.log
        -e PATH=/kbh/bin:/usr/local/cuda-host/bin:/opt/node/bin:/usr/local/bin:/usr/bin:/bin
        -v "$WORKSPACE_ROOT:/workspace:rw"
        -v "$agent_home:/home/agent:rw"
        -v "$KBH_AGENT_CONTAINER_CUDA_HOME:/usr/local/cuda-host:ro"
        -v "$KBH_AGENT_CONTAINER_CODEX_NODE:/opt/node:ro"
        -v "$REAL_UV:/usr/local/bin/uv:ro"
        -v "$KBH_AGENT_CONTAINER_UV_CACHE:/uv-cache:rw"
        -v "$CONTAINER_LOCK_BIN:/kbh/bin:ro"
        -v "$KBH_GPU_LOCK_DIR:/kbh/lock:rw"
        -w "/workspace/problems/$PROBLEM_NAME"
        "$KBH_AGENT_CONTAINER_IMAGE"
        /opt/node/bin/codex
        exec
        -m "$MODEL"
        "${effort_arg[@]}"
        --dangerously-bypass-approvals-and-sandbox
        --skip-git-repo-check
        -C "/workspace/problems/$PROBLEM_NAME"
        "$PROMPT"
    )
    run_docker_locked_timeout codex-container "$BUDGET_SECONDS" "$cidfile" "${docker_args[@]}"
}

run_opencode_container() {
    local opencode_model="${1:-$MODEL}"
    local agent_home
    agent_home="$(prepare_opencode_container_home)"
    local cidfile="$RUN_DIR/opencode_container.cid"
    check_container_basics || return $?
    if [ ! -x "$KBH_AGENT_CONTAINER_OPENCODE_BIN" ]; then
        echo "OpenCode binary missing for container mode: $KBH_AGENT_CONTAINER_OPENCODE_BIN" >&2
        return 127
    fi
    local -a docker_args=(
        run --rm
        --cidfile "$cidfile"
        --gpus all
        --network "$KBH_AGENT_CONTAINER_NETWORK"
        --cap-add CAP_PERFMON
        --security-opt no-new-privileges
        --shm-size 2g
        --user "$(id -u):$(id -g)"
        -e HOME=/home/agent
        -e USER=agent
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        -e BASH_ENV=
        -e ENV=
        -e OPENAI_API_KEY
        -e OPENROUTER_API_KEY
        -e ZAI_API_KEY
        -e DEEPSEEK_API_KEY
        -e MINIMAX_API_KEY
        -e GEMINI_API_KEY
        -e CUDA_HOME=/usr/local/cuda-host
        -e UV_CACHE_DIR=/uv-cache
        -e UV_PYTHON_INSTALL_DIR=/uv-cache/python
        -e RUN_DIR=/kbh
        -e KBH_GPU_LOCK=/kbh/lock/gpu.lock
        -e KBH_GPU_LOCK_LOG=/home/agent/gpu_lock_container.log
        -e PATH=/kbh/bin:/usr/local/cuda-host/bin:/usr/local/bin:/usr/bin:/bin
        -v "$WORKSPACE_ROOT:/workspace:rw"
        -v "$agent_home:/home/agent:rw"
        -v "$KBH_AGENT_CONTAINER_CUDA_HOME:/usr/local/cuda-host:ro"
        -v "$KBH_AGENT_CONTAINER_OPENCODE_BIN:/usr/local/bin/opencode:ro"
        -v "$REAL_UV:/usr/local/bin/uv:ro"
        -v "$KBH_AGENT_CONTAINER_UV_CACHE:/uv-cache:rw"
        -v "$CONTAINER_LOCK_BIN:/kbh/bin:ro"
        -v "$KBH_GPU_LOCK_DIR:/kbh/lock:rw"
        -w "/workspace/problems/$PROBLEM_NAME"
        "$KBH_AGENT_CONTAINER_IMAGE"
        opencode
        run
        --pure --format json -m "$opencode_model" "$PROMPT"
    )
    # The opencode OpenAI-compatible adapter intermittently hangs forever on
    # reasoning streams (DEVLOG 2026-06-09). Supervise the session with the
    # stall watchdog and retry a killed session with the remaining budget.
    local stall_seconds="${KBH_OPENCODE_STALL_SECONDS:-900}"
    local max_attempts=$(( ${KBH_OPENCODE_STALL_RETRIES:-2} + 1 ))
    local attempt=1
    local start_ts elapsed remaining status wd_before wd_after
    start_ts="$(date +%s)"
    while :; do
        elapsed=$(( $(date +%s) - start_ts ))
        remaining=$(( BUDGET_SECONDS - elapsed ))
        if [ "$remaining" -le 60 ]; then
            return 124
        fi
        wd_before=0
        [ -f "$RUN_DIR/stall_watchdog.log" ] && wd_before="$(wc -l < "$RUN_DIR/stall_watchdog.log")"
        status=0
        KBH_STALL_WATCH_LOG="$LOG_FILE" KBH_STALL_SECONDS="$stall_seconds" \
            run_docker_locked_timeout opencode-container "$remaining" "$cidfile" "${docker_args[@]}" \
            || status=$?
        wd_after=0
        [ -f "$RUN_DIR/stall_watchdog.log" ] && wd_after="$(wc -l < "$RUN_DIR/stall_watchdog.log")"
        if [ "$wd_after" -gt "$wd_before" ] && [ "$attempt" -lt "$max_attempts" ]; then
            attempt=$(( attempt + 1 ))
            continue
        fi
        return "$status"
    done
}

run_droid_container() {
    local effort="$1"
    local agent_home
    agent_home="$(prepare_droid_container_home)"
    local cidfile="$RUN_DIR/droid_container.cid"
    check_container_basics || return $?
    if [ ! -x "$KBH_AGENT_CONTAINER_DROID_BIN" ]; then
        echo "Droid binary missing for container mode: $KBH_AGENT_CONTAINER_DROID_BIN" >&2
        return 127
    fi
    local -a effort_arg=()
    if [ -n "$effort" ]; then
        effort_arg=(--reasoning-effort "$effort")
    fi
    local -a docker_args=(
        run --rm
        --cidfile "$cidfile"
        --gpus all
        --network "$KBH_AGENT_CONTAINER_NETWORK"
        --cap-add CAP_PERFMON
        --security-opt no-new-privileges
        --shm-size 2g
        --user "$(id -u):$(id -g)"
        -e HOME=/home/agent
        -e USER=agent
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        -e BASH_ENV=
        -e ENV=
        -e FACTORY_API_KEY
        -e DROID_API_KEY
        -e CUDA_HOME=/usr/local/cuda-host
        -e PATH=/kbh/bin:/usr/local/cuda-host/bin:/usr/local/bin:/usr/bin:/bin
        -v "$WORKSPACE_ROOT:/workspace:rw"
        -v "$agent_home:/home/agent:rw"
        -v "$KBH_AGENT_CONTAINER_CUDA_HOME:/usr/local/cuda-host:ro"
        -v "$KBH_AGENT_CONTAINER_DROID_BIN:/usr/local/bin/droid:ro"
        -w "/workspace/problems/$PROBLEM_NAME"
        "$KBH_AGENT_CONTAINER_IMAGE"
        droid
        exec
        --output-format stream-json
        --skip-permissions-unsafe
        --cwd "/workspace/problems/$PROBLEM_NAME"
        -m "$MODEL"
        "${effort_arg[@]}"
        "$PROMPT"
    )
    run_docker_locked_timeout droid-container "$BUDGET_SECONDS" "$cidfile" "${docker_args[@]}"
}

prepare_grok_container_home() {
    local home_dir="$RUN_DIR/agent_home"
    mkdir -p "$home_dir/.grok"
    for f in auth.json agent_id installation_id config.toml settings.json; do
        if [ -f "$HOME/.grok/$f" ]; then
            cp -p "$HOME/.grok/$f" "$home_dir/.grok/$f"
        fi
    done
    printf '%s\n' "$home_dir"
}

run_grok_container() {
    local effort="$1"
    local agent_home
    agent_home="$(prepare_grok_container_home)"
    local cidfile="$RUN_DIR/grok_container.cid"
    check_container_basics || return $?
    # ~/.grok/bin/grok is a version symlink into ../downloads/; mount the
    # resolved file or the symlink dangles inside the container.
    local grok_bin
    grok_bin="$(readlink -f "$KBH_AGENT_CONTAINER_GROK_DIR/bin/grok" 2>/dev/null || true)"
    if [ -z "$grok_bin" ] || [ ! -e "$grok_bin" ]; then
        echo "Grok CLI missing for container mode: $KBH_AGENT_CONTAINER_GROK_DIR/bin/grok" >&2
        return 127
    fi
    local -a effort_arg=()
    if [ -n "$effort" ]; then
        effort_arg=(--effort "$effort")
    fi
    local -a docker_args=(
        run --rm
        --cidfile "$cidfile"
        --gpus all
        --network "$KBH_AGENT_CONTAINER_NETWORK"
        --cap-add CAP_PERFMON
        --security-opt no-new-privileges
        --shm-size 2g
        --user "$(id -u):$(id -g)"
        -e HOME=/home/agent
        -e USER=agent
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        -e BASH_ENV=
        -e ENV=
        -e XAI_API_KEY
        -e GROK_API_KEY
        -e CUDA_HOME=/usr/local/cuda-host
        -e UV_CACHE_DIR=/uv-cache
        -e UV_PYTHON_INSTALL_DIR=/uv-cache/python
        -e RUN_DIR=/kbh
        -e KBH_GPU_LOCK=/kbh/lock/gpu.lock
        -e KBH_GPU_LOCK_LOG=/home/agent/gpu_lock_container.log
        -e PATH=/kbh/bin:/usr/local/cuda-host/bin:/opt/node/bin:/usr/local/bin:/usr/bin:/bin
        -v "$WORKSPACE_ROOT:/workspace:rw"
        -v "$agent_home:/home/agent:rw"
        -v "$KBH_AGENT_CONTAINER_CUDA_HOME:/usr/local/cuda-host:ro"
        -v "$KBH_AGENT_CONTAINER_CODEX_NODE:/opt/node:ro"
        -v "$grok_bin:/opt/grok/bin/grok:ro"
        -v "$KBH_AGENT_CONTAINER_GROK_DIR/bundled:/opt/grok/bundled:ro"
        -v "$REAL_UV:/usr/local/bin/uv:ro"
        -v "$KBH_AGENT_CONTAINER_UV_CACHE:/uv-cache:rw"
        -v "$CONTAINER_LOCK_BIN:/kbh/bin:ro"
        -v "$KBH_GPU_LOCK_DIR:/kbh/lock:rw"
        -w "/workspace/problems/$PROBLEM_NAME"
        "$KBH_AGENT_CONTAINER_IMAGE"
        /opt/grok/bin/grok
        --cwd "/workspace/problems/$PROBLEM_NAME"
        --always-approve
        --permission-mode bypassPermissions
        --no-memory
        --output-format streaming-json
        --model "$MODEL"
        "${effort_arg[@]}"
        -p "$PROMPT"
    )
    run_docker_locked_timeout grok-container "$BUDGET_SECONDS" "$cidfile" "${docker_args[@]}"
}

run_gemini_container() {
    local agent_home="$RUN_DIR/agent_home"
    mkdir -p "$agent_home"
    local cidfile="$RUN_DIR/gemini_container.cid"
    check_container_basics || return $?
    if [ ! -e "$KBH_AGENT_CONTAINER_GEMINI_DIR/bundle/gemini.js" ]; then
        echo "Gemini CLI missing for container mode: $KBH_AGENT_CONTAINER_GEMINI_DIR/bundle/gemini.js" >&2
        return 127
    fi
    local -a docker_args=(
        run --rm
        --cidfile "$cidfile"
        --gpus all
        --network "$KBH_AGENT_CONTAINER_NETWORK"
        --cap-add CAP_PERFMON
        --security-opt no-new-privileges
        --shm-size 2g
        --user "$(id -u):$(id -g)"
        -e HOME=/home/agent
        -e USER=agent
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        -e BASH_ENV=
        -e ENV=
        -e GEMINI_API_KEY
        -e CUDA_HOME=/usr/local/cuda-host
        -e UV_CACHE_DIR=/uv-cache
        -e UV_PYTHON_INSTALL_DIR=/uv-cache/python
        -e RUN_DIR=/kbh
        -e KBH_GPU_LOCK=/kbh/lock/gpu.lock
        -e KBH_GPU_LOCK_LOG=/home/agent/gpu_lock_container.log
        -e PATH=/kbh/bin:/usr/local/cuda-host/bin:/opt/node/bin:/usr/local/bin:/usr/bin:/bin
        -v "$WORKSPACE_ROOT:/workspace:rw"
        -v "$agent_home:/home/agent:rw"
        -v "$KBH_AGENT_CONTAINER_CUDA_HOME:/usr/local/cuda-host:ro"
        -v "$KBH_AGENT_CONTAINER_CODEX_NODE:/opt/node:ro"
        -v "$KBH_AGENT_CONTAINER_GEMINI_DIR:/opt/gemini-cli:ro"
        -v "$REAL_UV:/usr/local/bin/uv:ro"
        -v "$KBH_AGENT_CONTAINER_UV_CACHE:/uv-cache:rw"
        -v "$CONTAINER_LOCK_BIN:/kbh/bin:ro"
        -v "$KBH_GPU_LOCK_DIR:/kbh/lock:rw"
        -w "/workspace/problems/$PROBLEM_NAME"
        "$KBH_AGENT_CONTAINER_IMAGE"
        /opt/node/bin/node /opt/gemini-cli/bundle/gemini.js
        --skip-trust
        -m "$MODEL"
        --approval-mode yolo
        -o stream-json
        -p "$PROMPT"
    )
    run_docker_locked_timeout gemini-container "$BUDGET_SECONDS" "$cidfile" "${docker_args[@]}"
}

run_cursor_container() {
    local agent_home
    agent_home="$(prepare_cursor_container_home)"
    local cidfile="$RUN_DIR/cursor_container.cid"
    check_container_basics || return $?
    if [ ! -x "$KBH_AGENT_CONTAINER_CURSOR_DIR/cursor-agent" ]; then
        echo "Cursor Agent binary missing for container mode: $KBH_AGENT_CONTAINER_CURSOR_DIR/cursor-agent" >&2
        return 127
    fi
    local -a docker_args=(
        run --rm
        --cidfile "$cidfile"
        --gpus all
        --network "$KBH_AGENT_CONTAINER_NETWORK"
        --cap-add CAP_PERFMON
        --security-opt no-new-privileges
        --shm-size 2g
        --user "$(id -u):$(id -g)"
        -e HOME=/home/agent
        -e USER=agent
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
        -e BASH_ENV=
        -e ENV=
        -e CURSOR_API_KEY
        -e CUDA_HOME=/usr/local/cuda-host
        -e UV_CACHE_DIR=/uv-cache
        -e UV_PYTHON_INSTALL_DIR=/uv-cache/python
        -e RUN_DIR=/kbh
        -e KBH_GPU_LOCK=/kbh/lock/gpu.lock
        -e KBH_GPU_LOCK_LOG=/home/agent/gpu_lock_container.log
        -e PATH=/kbh/bin:/usr/local/cuda-host/bin:/usr/local/bin:/usr/bin:/bin
        -v "$WORKSPACE_ROOT:/workspace:rw"
        -v "$agent_home:/home/agent:rw"
        -v "$KBH_AGENT_CONTAINER_CUDA_HOME:/usr/local/cuda-host:ro"
        -v "$KBH_AGENT_CONTAINER_CURSOR_DIR:/opt/cursor-agent:ro"
        -v "$REAL_UV:/usr/local/bin/uv:ro"
        -v "$KBH_AGENT_CONTAINER_UV_CACHE:/uv-cache:rw"
        -v "$CONTAINER_LOCK_BIN:/kbh/bin:ro"
        -v "$KBH_GPU_LOCK_DIR:/kbh/lock:rw"
        -w "/workspace/problems/$PROBLEM_NAME"
        "$KBH_AGENT_CONTAINER_IMAGE"
        /opt/cursor-agent/cursor-agent
        --trust
        --yolo
        --print
        --output-format stream-json
        --model "$MODEL"
        --workspace "/workspace/problems/$PROBLEM_NAME"
        "$PROMPT"
    )
    run_docker_locked_timeout cursor-container "$BUDGET_SECONDS" "$cidfile" "${docker_args[@]}"
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
echo "KERNELBENCH-HARD RUN"
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
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            run_claude_container "$REASONING_EFFORT" \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        else
            ( cd "$PROBLEM_DIR" && timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$MODEL" \
                "${EFFORT_ARG[@]}" \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        fi
        ;;

    ccr-claude)
        # Claude Code routed via ccr-rust to a non-Anthropic provider.
        # Assumes ccr-rust is running locally and ANTHROPIC_BASE_URL points at it.
        # Model name is the upstream lab's model ID (glm-5.1, deepseek-v4-flash, etc.).
        ( cd "$PROBLEM_DIR" && timeout "$BUDGET_SECONDS" \
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
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            CLAUDE_CONTAINER_ENV_NAMES=(
                ANTHROPIC_BASE_URL API_TIMEOUT_MS
                CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS
                CLAUDE_CODE_MAX_RETRIES CLAUDE_CODE_MAX_OUTPUT_TOKENS
                ANTHROPIC_DEFAULT_HAIKU_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL
                ANTHROPIC_DEFAULT_OPUS_MODEL
            )
            CLAUDE_CONTAINER_EXTRA_CLAUDE_ARGS=(--disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion)
            ( export ANTHROPIC_AUTH_TOKEN="$ZAI_API_KEY" && \
                export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic" && \
                export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" && \
                export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS="${CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS:-1}" && \
                export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-1000000}" && \
                export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-128000}" && \
                export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ZAI_CLAUDE_HAIKU_MODEL" && \
                export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" && \
                export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" && \
                run_claude_container "" "$ZAI_CLAUDE_ALIAS" 0 ) \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
            CLAUDE_CONTAINER_ENV_NAMES=()
            CLAUDE_CONTAINER_EXTRA_CLAUDE_ARGS=()
        else
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
            timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$ZAI_CLAUDE_ALIAS" \
                --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        fi
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
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            CLAUDE_CONTAINER_ENV_NAMES=(
                ANTHROPIC_BASE_URL API_TIMEOUT_MS
                CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC
                CLAUDE_CODE_MAX_RETRIES CLAUDE_CODE_MAX_OUTPUT_TOKENS
                ANTHROPIC_MODEL
                ANTHROPIC_DEFAULT_HAIKU_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL
                ANTHROPIC_DEFAULT_OPUS_MODEL
            )
            CLAUDE_CONTAINER_EXTRA_CLAUDE_ARGS=(--disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion)
            ( export ANTHROPIC_AUTH_TOKEN="$MINIMAX_API_KEY" && \
                export ANTHROPIC_BASE_URL="${MINIMAX_ANTHROPIC_BASE_URL:-https://api.minimax.io/anthropic}" && \
                export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" && \
                export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}" && \
                export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-1000000}" && \
                export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-128000}" && \
                export ANTHROPIC_MODEL="$MODEL" && \
                export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MINIMAX_CLAUDE_HAIKU_MODEL" && \
                export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" && \
                export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" && \
                run_claude_container "" "$MINIMAX_CLAUDE_ALIAS" 0 ) \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
            CLAUDE_CONTAINER_ENV_NAMES=()
            CLAUDE_CONTAINER_EXTRA_CLAUDE_ARGS=()
        else
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
            timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$MINIMAX_CLAUDE_ALIAS" \
                --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        fi
        ;;

    kimi-claude)
        # Claude Code routed to Moonshot's Anthropic-compatible endpoint for
        # Kimi K2.7 Code. Requires KIMI_API_KEY in the environment or
        # ~/.env_vars. K2.7-Code forces thinking mode; CLAUDE_KBH_SETTINGS
        # already sets alwaysThinkingEnabled, which the endpoint requires.
        if [ -z "${KIMI_API_KEY:-}" ]; then
            echo "KIMI_API_KEY is required for kimi-claude" >&2
            exit 1
        fi
        KIMI_CLAUDE_ALIAS="${KIMI_CLAUDE_ALIAS:-opus}"
        KIMI_CLAUDE_HAIKU_MODEL="${KIMI_CLAUDE_HAIKU_MODEL:-$MODEL}"
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            CLAUDE_CONTAINER_ENV_NAMES=(
                ANTHROPIC_BASE_URL API_TIMEOUT_MS
                CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC
                CLAUDE_CODE_MAX_RETRIES CLAUDE_CODE_MAX_OUTPUT_TOKENS
                ANTHROPIC_MODEL
                ANTHROPIC_DEFAULT_HAIKU_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL
                ANTHROPIC_DEFAULT_OPUS_MODEL
            )
            CLAUDE_CONTAINER_EXTRA_CLAUDE_ARGS=(--disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion)
            ( export ANTHROPIC_AUTH_TOKEN="$KIMI_API_KEY" && \
                export ANTHROPIC_BASE_URL="${KIMI_ANTHROPIC_BASE_URL:-https://api.moonshot.ai/anthropic}" && \
                export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" && \
                export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}" && \
                export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-1000000}" && \
                export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-128000}" && \
                export ANTHROPIC_MODEL="$MODEL" && \
                export ANTHROPIC_DEFAULT_HAIKU_MODEL="$KIMI_CLAUDE_HAIKU_MODEL" && \
                export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" && \
                export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" && \
                run_claude_container "" "$KIMI_CLAUDE_ALIAS" 0 ) \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
            CLAUDE_CONTAINER_ENV_NAMES=()
            CLAUDE_CONTAINER_EXTRA_CLAUDE_ARGS=()
        else
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
            timeout "$BUDGET_SECONDS" claude \
                --dangerously-skip-permissions \
                --print --verbose \
                --output-format stream-json \
                --settings "$CLAUDE_KBH_SETTINGS" \
                --model "$KIMI_CLAUDE_ALIAS" \
                --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
                --add-dir "$PROBLEM_DIR" \
                -p "$PROMPT" ) \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        fi
        ;;

    codex)
        EFFORT_ARG=()
        if [ -n "$REASONING_EFFORT" ]; then
            EFFORT_ARG=(-c "model_reasoning_effort=\"$REASONING_EFFORT\"")
        fi
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            run_codex_container "$REASONING_EFFORT" \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        else
            timeout "$BUDGET_SECONDS" codex exec \
                -m "$MODEL" \
                "${EFFORT_ARG[@]}" \
                --dangerously-bypass-approvals-and-sandbox \
                --skip-git-repo-check \
                -C "$PROBLEM_DIR" \
                "$PROMPT" \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        fi

        # Codex writes its rich session JSONL to ~/.codex/sessions/YYYY/MM/DD/
        # (local date). Locate by session_id printed to stderr — DO NOT pick the
        # most-recently-modified file, since codex 0.125.0 touches old session
        # files when scanning its thread state DB and that's misleading.
        CODEX_SID=$(grep -h -oP 'session id: \K[0-9a-f-]+' "$STDERR_FILE" "$LOG_FILE" 2>/dev/null | head -1)
        if [ -n "$CODEX_SID" ]; then
            if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
                CODEX_SEARCH_ROOT="$RUN_DIR/agent_home/.codex/sessions"
            else
                CODEX_SEARCH_ROOT="$HOME/.codex/sessions"
            fi
            CODEX_SESS=$(find "$CODEX_SEARCH_ROOT" -name "*${CODEX_SID}*.jsonl" 2>/dev/null | head -1)
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

    droid)
        EFFORT_ARG=()
        if [ -n "$REASONING_EFFORT" ]; then
            EFFORT_ARG=(--reasoning-effort "$REASONING_EFFORT")
        fi
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            run_droid_container "$REASONING_EFFORT" \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        else
            timeout "$BUDGET_SECONDS" droid exec \
                --output-format stream-json \
                --skip-permissions-unsafe \
                --cwd "$PROBLEM_DIR" \
                -m "$MODEL" \
                "${EFFORT_ARG[@]}" \
                "$PROMPT" \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        fi
        ;;

    gemini)
        # Gemini CLI. No --cwd flag, so cd into PROBLEM_DIR.
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            run_gemini_container \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        else
        ( cd "$PROBLEM_DIR" && timeout "$BUDGET_SECONDS" gemini \
            --skip-trust \
            -m "$MODEL" \
            --approval-mode yolo \
            -o stream-json \
            -p "$PROMPT" \
            </dev/null > "$LOG_FILE" 2> "$STDERR_FILE" ) || HARNESS_EXIT=$?
        fi
        ;;

    cursor)
        # Cursor Agent CLI is installed as `agent` on Anvil.
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            run_cursor_container \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        else
            timeout "$BUDGET_SECONDS" "${AGENT_CUDA_ENV[@]}" agent \
                --trust \
                --yolo \
                --print \
                --output-format stream-json \
                --model "$MODEL" \
                --workspace "$PROBLEM_DIR" \
                "$PROMPT" \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        fi
        ;;

    grok)
        # Grok Build CLI is installed as `grok` on Anvil. Use the top-level
        # headless path because `grok agent` does not accept --cwd/output flags.
        EFFORT_ARG=()
        if [ -n "$REASONING_EFFORT" ]; then
            EFFORT_ARG=(--effort "$REASONING_EFFORT")
        fi
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            run_grok_container "$REASONING_EFFORT" \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        else
        timeout "$BUDGET_SECONDS" "${AGENT_CUDA_ENV[@]}" grok \
            --cwd "$PROBLEM_DIR" \
            --always-approve \
            --permission-mode bypassPermissions \
            --no-memory \
            --output-format streaming-json \
            --model "$MODEL" \
            "${EFFORT_ARG[@]}" \
            -p "$PROMPT" \
            > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        fi
        ;;

    opencode)
        # OpenCode SST with custom OpenAI-shape providers (deepseek, zai, minimax).
        # Provider/model pair encoded as MODEL="provider/model-id" e.g.
        # "deepseek/deepseek-v4-pro" or "zai/glm-5.1".
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            run_opencode_container \
                > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        else
            ( cd "$PROBLEM_DIR" && timeout "$BUDGET_SECONDS" "${AGENT_CUDA_ENV[@]}" opencode run \
                --pure --format json -m "$MODEL" "$PROMPT" \
                </dev/null > "$LOG_FILE" 2> "$STDERR_FILE" ) || HARNESS_EXIT=$?
        fi
        ;;

    opencode-nemotron)
        # Preferred Nemotron route: OpenCode speaks OpenAI-compatible APIs
        # natively, while this archive-local config pins OpenRouter to
        # DeepInfra and disables fallback provider drift.
        OPENCODE_NEMOTRON_CONFIG_HOME="$(write_openrouter_deepinfra_opencode_config "$MODEL")"
        OPENCODE_NEMOTRON_MODEL="openrouter-deepinfra/$MODEL"
        if [ "$KBH_AGENT_CONTAINER" = "1" ]; then
            KBH_OPENCODE_CONFIG_FILE="$OPENCODE_NEMOTRON_CONFIG_HOME/opencode/opencode.json"                 run_opencode_container "$OPENCODE_NEMOTRON_MODEL"                 > "$LOG_FILE" 2> "$STDERR_FILE" || HARNESS_EXIT=$?
        else
            ( cd "$PROBLEM_DIR" && timeout "$BUDGET_SECONDS" "${AGENT_CUDA_ENV[@]}"                 env XDG_CONFIG_HOME="$OPENCODE_NEMOTRON_CONFIG_HOME"                 opencode run --pure --format json -m "$OPENCODE_NEMOTRON_MODEL" "$PROMPT"                 </dev/null > "$LOG_FILE" 2> "$STDERR_FILE" ) || HARNESS_EXIT=$?
        fi
        ;;

    nvcf-nemotron)
        # Nemotron 3 Ultra through NVIDIA's NVCF function share. NVCF is not an
        # OpenAI-compatible base URL, so run a per-archive localhost adapter and
        # point an archive-local OpenCode config at it.
        start_nvcf_proxy
        NVCF_OPENCODE_CONFIG_HOME="$(write_nvcf_opencode_config "$NVCF_PROXY_BASE_URL")"
        ( cd "$PROBLEM_DIR" && timeout "$BUDGET_SECONDS" "${AGENT_CUDA_ENV[@]}" \
            env XDG_CONFIG_HOME="$NVCF_OPENCODE_CONFIG_HOME" \
            opencode run --pure --format json -m "nvcf-nemotron/$MODEL" "$PROMPT" \
            </dev/null > "$LOG_FILE" 2> "$STDERR_FILE" ) || HARNESS_EXIT=$?
        ;;

    *)
        echo "Unknown harness: $HARNESS" >&2
        echo "Supported: claude, zai-claude, minimax-claude, kimi-claude, ccr-claude, codex, kimi, droid, gemini, cursor, grok, opencode, opencode-nemotron, nvcf-nemotron" >&2
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
    claude|zai-claude|minimax-claude|kimi-claude|ccr-claude|cursor|gemini)
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
    droid|kimi|opencode|opencode-nemotron|nvcf-nemotron)
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
    elif grep -q "^PASS" "$CHECK_LOG"; then
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
    "agent_container": $([ "$KBH_AGENT_CONTAINER" = "1" ] && echo true || echo false),
    "agent_container_image": "$KBH_AGENT_CONTAINER_IMAGE",
    "agent_container_network": "$KBH_AGENT_CONTAINER_NETWORK",
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
