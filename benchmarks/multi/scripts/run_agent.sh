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

# Agent CLIs live in ~/.local/bin, which a detached / non-interactive shell does
# not get from the profile. Without this a whole wave exits 127 in seconds and
# records four silent `no_solution` rows that look like model failures. Put it on
# PATH, then fail LOUDLY if the harness binary still is not there. (2026-07-25)
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH" ;; esac
export PATH
case "$HARNESS" in
    grok)                 HARNESS_BIN=grok ;;
    claude)               HARNESS_BIN=claude ;;
    codex)                HARNESS_BIN=codex ;;
    opencode-or)          HARNESS_BIN=opencode ;; # OpenRouter via opencode (OpenAI-compatible)
    *-claude)             HARNESS_BIN=claude ;;   # zai / kimi / deepseek routes
    *)                    HARNESS_BIN="$HARNESS" ;;
esac
command -v "$HARNESS_BIN" >/dev/null 2>&1 || {
    echo "STOP: harness CLI '$HARNESS_BIN' not found on PATH ($PATH)" >&2
    exit 3
}

# Every artifact stays in-repo under benchmarks/multi/outputs/ on EVERY machine
# (AGENTS.md, 2026-07-25) — archives outside the repo are invisible to publish /
# contamination / re-grade tooling, which is a correctness rule, not tidiness.
LOCK_DIR="${KBM_GPU_LOCK_DIR:-$BENCH_ROOT/outputs/gpu_lock}"
mkdir -p "$LOCK_DIR"
LOCKFILE="$LOCK_DIR/gpu.lock"

RUN_ID="$(date +%Y%m%d_%H%M%S)_${HARNESS}_${MODEL//\//-}_${PROBLEM}"
RUN_DIR="$BENCH_ROOT/outputs/runs/$RUN_ID"
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

# Per-run rendezvous port. A single fixed default (29571) meant concurrent
# sessions collided on bind, and agents reasonably resorted to `fuser -k
# 29571/tcp` — killing a sibling's grading run to free "their" port. Derive a
# stable per-run port instead so no two sessions ever contend. (2026-07-25)
PORT_OFFSET=$(( $(echo "$RUN_ID" | cksum | cut -d' ' -f1) % 400 ))
export KBM_MASTER_PORT=$(( 29600 + PORT_OFFSET ))

# Co-tenant kill guard. Agents legitimately clean up their own hung torchrun, so
# this must NOT block self-cleanup — an earlier version keyed on "any match
# outside my session id" and false-fired on transient shells (including an
# operator's own `pgrep -af torchrun` monitoring command), which would derail a
# session. What actually needs protecting is another tenant's running job, so the
# guard refuses only when a matched process is one of theirs. Sibling KernelBench
# sessions are handled by running the wave sequentially (sweep_wave.sh), not here.
PROTECTED="${KBM_PROTECTED_PROCS:-vllm|sglang|trtllm|nanbeige|laguna|dspark|demon/harness}"
for tool in pkill killall; do
    real="$(command -v "$tool" 2>/dev/null || true)"
    [ -n "$real" ] || continue
    cat > "$RUN_DIR/bin/$tool" <<GUARD
#!/usr/bin/env bash
# Signal flag (default TERM); -f and friends are pkill-only and must not reach kill.
sig=""
pat=""
for a in "\$@"; do
    case "\$a" in
        -[0-9]*|-[A-Z]*) sig="\$a" ;;
        -*) ;;
        *) pat="\$a" ;;
    esac
done
[ -n "\$pat" ] || exec "$real" "\$@"

# This run's own ancestry: the agent session process, run_agent.sh, the wave
# driver. The agent's prompt is passed in argv, and every PROMPT.txt contains the
# word "torchrun", so a perfectly reasonable \`pkill -f torchrun\` matches the
# AGENT'S OWN process and the session commits suicide (exit 137). Seen twice on
# 2026-07-25. Ancestors are excluded from the kill, not refused, so cleaning up
# hung children still works.
ancestors=" "
p=\$\$
while [ "\$p" -gt 1 ] 2>/dev/null; do
    ancestors="\$ancestors\$p "
    p=\$(awk '{print \$4}' /proc/\$p/stat 2>/dev/null) || break
    [ -n "\$p" ] || break
done

hit=""
targets=""
for pid in \$(pgrep -f -- "\$pat" 2>/dev/null); do
    [ "\$pid" = "\$\$" ] && continue
    case "\$ancestors" in *" \$pid "*) continue ;; esac
    [ -r /proc/\$pid/cmdline ] || continue     # exited between pgrep and read
    cmd=\$(tr '\0' ' ' < /proc/\$pid/cmdline 2>/dev/null)
    # -i matters: the GPU-resident workers are named VLLM::Worker_TPn in
    # UPPERCASE (that is the name an agent sees in nvidia-smi and would
    # target), while the serve parents are lowercase.
    if echo "\$cmd" | grep -qiE "$PROTECTED"; then hit="\$hit \$pid"; else targets="\$targets \$pid"; fi
done
if [ -n "\$hit" ]; then
    echo "$tool refused: '\$pat' matches another tenant's job on this shared node (PIDs:\$hit)." >&2
    echo "Never kill processes you did not start. Target your own PIDs explicitly." >&2
    exit 1
fi
[ -n "\$targets" ] || exit 1                   # nothing matched, as pkill would
kill \$sig \$targets 2>/dev/null
GUARD
    chmod +x "$RUN_DIR/bin/$tool"
done

# Claude Code pointed at a provider's Anthropic-compatible endpoint. zai / kimi /
# deepseek differ only in key, base URL and model id, so they share one body —
# four near-identical 25-line blocks is how the hard harness drifted out of sync
# with itself. Args mirror hard's branches (2026-07-28).
#
# The key is exported INSIDE this function, never passed as `env KEY=... claude`,
# because argv is world-readable via /proc — the agent under test can read it.
run_claude_route() {
    local token="$1" base="$2" alias="${3:-opus}"
    ( cd "$PROBLEM_DIR" \
        && export ANTHROPIC_AUTH_TOKEN="$token" \
        && export ANTHROPIC_BASE_URL="$base" \
        && export ANTHROPIC_MODEL="$MODEL" \
        && export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
        && export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL" \
        && export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL" \
        && export API_TIMEOUT_MS="${API_TIMEOUT_MS:-3000000}" \
        && export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
        && export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}" \
        && export CLAUDE_CODE_MAX_RETRIES="${CLAUDE_CODE_MAX_RETRIES:-100}" \
        && export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-128000}" \
        && PATH="$RUN_DIR/bin:$PATH" "${TIMEOUT_CMD[@]}" claude \
            --dangerously-skip-permissions --print --verbose \
            --output-format stream-json \
            --settings '{"fastMode":false,"alwaysThinkingEnabled":true}' \
            --model "$alias" \
            --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
            -p "$PROMPT" )
}

# Fail before burning a session on a missing key, with the name the operator
# needs to fix rather than a provider 401 buried in agent.err.
require_key() {
    [ -n "${2:-}" ] || { echo "STOP: harness '$HARNESS' needs \$$1 in ~/.kbm_env" >&2; exit 3; }
}

# Keys live in ~/.kbm_env (chmod 600), never in the repo and never in argv.
# Sourcing a missing file under `set -e` aborts with a bare "No such file",
# which reads like a code bug rather than "this worker was never provisioned".
load_keys() {
    [ -f "$HOME/.kbm_env" ] || {
        echo "STOP: ~/.kbm_env not found on $(hostname) — worker has no API keys." >&2
        echo "Ship it with: ssh <worker> 'cat > ~/.kbm_env && chmod 600 ~/.kbm_env' < local_keys" >&2
        exit 3
    }
    # shellcheck disable=SC1090
    . "$HOME/.kbm_env"
}

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
        load_keys
        require_key ZAI_API_KEY "${ZAI_API_KEY:-}"
        run_claude_route "$ZAI_API_KEY" "https://api.z.ai/api/anthropic" \
            > "$RUN_DIR/agent.log" 2> "$RUN_DIR/agent.err" || HARNESS_EXIT=$?
        ;;
    kimi-claude)
        # Moonshot's Anthropic skin. Kimi K3 (model id `kimi-k3`, released
        # 2026-07-16) forces thinking mode, which alwaysThinkingEnabled already
        # satisfies.
        load_keys
        require_key KIMI_API_KEY "${KIMI_API_KEY:-}"
        run_claude_route "$KIMI_API_KEY" \
            "${KIMI_ANTHROPIC_BASE_URL:-https://api.moonshot.ai/anthropic}" \
            > "$RUN_DIR/agent.log" 2> "$RUN_DIR/agent.err" || HARNESS_EXIT=$?
        ;;
    deepseek-claude)
        load_keys
        require_key DEEPSEEK_API_KEY "${DEEPSEEK_API_KEY:-}"
        run_claude_route "$DEEPSEEK_API_KEY" "https://api.deepseek.com/anthropic" \
            > "$RUN_DIR/agent.log" 2> "$RUN_DIR/agent.err" || HARNESS_EXIT=$?
        ;;
    claude)
        # Native Anthropic (Opus 5). --effort max is what the hard Opus matrix
        # uses; a run at a lower tier or with fastMode on is NOT comparable.
        EFFORT_ARG=()
        [ -n "$EFFORT" ] && EFFORT_ARG=(--effort "$EFFORT")
        ( cd "$PROBLEM_DIR" && PATH="$RUN_DIR/bin:$PATH" "${TIMEOUT_CMD[@]}" claude \
            --dangerously-skip-permissions --print --verbose \
            --output-format stream-json \
            --settings '{"fastMode":false,"alwaysThinkingEnabled":true}' \
            --model "$MODEL" \
            "${EFFORT_ARG[@]}" \
            --disallowedTools ExitPlanMode EnterPlanMode AskUserQuestion \
            -p "$PROMPT" ) \
            > "$RUN_DIR/agent.log" 2> "$RUN_DIR/agent.err" || HARNESS_EXIT=$?
        ;;
    codex)
        # </dev/null is REQUIRED: `codex exec` blocks forever waiting on stdin
        # in a detached shell, which looks exactly like a model thinking.
        EFFORT_ARG=()
        [ -n "$EFFORT" ] && EFFORT_ARG=(-c "model_reasoning_effort=\"$EFFORT\"")
        PATH="$RUN_DIR/bin:$PATH" "${TIMEOUT_CMD[@]}" codex exec \
            -m "$MODEL" \
            "${EFFORT_ARG[@]}" \
            --dangerously-bypass-approvals-and-sandbox \
            --skip-git-repo-check \
            -C "$PROBLEM_DIR" \
            "$PROMPT" \
            < /dev/null > "$RUN_DIR/agent.log" 2> "$RUN_DIR/agent.err" || HARNESS_EXIT=$?
        # Archive the rich session JSONL by session id. Do NOT pick the newest
        # file — codex touches old sessions when scanning its thread DB.
        CODEX_SID=$(grep -h -oP 'session id: \K[0-9a-f-]+' \
            "$RUN_DIR/agent.err" "$RUN_DIR/agent.log" 2>/dev/null | head -1)
        if [ -n "$CODEX_SID" ]; then
            CODEX_SESS=$(find "$HOME/.codex/sessions" -name "*${CODEX_SID}*.jsonl" 2>/dev/null | head -1)
            [ -n "$CODEX_SESS" ] && cp "$CODEX_SESS" "$RUN_DIR/codex_session.jsonl"
        fi
        ;;
    opencode-or)
        # OpenRouter via opencode's OpenAI-compatible adapter — the fallback
        # when a provider's own endpoint is down (e.g. kimi-k3 after the
        # Moonshot key died, 2026-07-30). $MODEL is the OpenRouter id
        # (`moonshotai/kimi-k3`). Provider is pinned via KBM_OR_PROVIDER
        # (default "Moonshot AI") with allow_fallbacks=false, mirroring the
        # hard bench's nemotron discipline: one serving stack per column.
        # Known caveat: this adapter stalls intermittently (~1/3 of hard
        # sessions) — treat a silent no-token session as retryable infra.
        # EFFORT is ignored on this route (no effort control in the adapter).
        load_keys
        require_key OPENROUTER_API_KEY "${OPENROUTER_API_KEY:-}"
        OC_HOME="$RUN_DIR/opencode_config"
        mkdir -p "$OC_HOME/opencode"
        chmod 700 "$OC_HOME"
        cat > "$OC_HOME/opencode/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "permission": { "external_directory": "deny" },
  "provider": {
    "openrouter-pinned": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenRouter (pinned provider)",
      "options": {
        "baseURL": "https://openrouter.ai/api/v1",
        "apiKey": "${OPENROUTER_API_KEY}",
        "headers": {
          "HTTP-Referer": "https://kernelbench.com",
          "X-Title": "KernelBench-Multi"
        },
        "extraBody": {
          "provider": {
            "order": ["${KBM_OR_PROVIDER:-Moonshot AI}"],
            "allow_fallbacks": false
          }
        }
      },
      "models": {
        "${MODEL}": {
          "name": "${MODEL} via OpenRouter",
          "limit": { "context": ${KBM_OR_CONTEXT:-262144}, "output": 32000 },
          "tools": true
        }
      }
    }
  }
}
JSON
        ( cd "$PROBLEM_DIR" && PATH="$RUN_DIR/bin:$PATH" \
            XDG_CONFIG_HOME="$OC_HOME" "${TIMEOUT_CMD[@]}" opencode run \
            --pure --format json -m "openrouter-pinned/$MODEL" "$PROMPT" \
            < /dev/null ) \
            > "$RUN_DIR/agent.log" 2> "$RUN_DIR/agent.err" || HARNESS_EXIT=$?
        ;;
    *)
        echo "unknown harness: $HARNESS" >&2; exit 2 ;;
esac
echo "[run_agent] agent exit=$HARNESS_EXIT" | tee -a "$RUN_DIR/meta.log"

# --- grade (serialized through the same lock via wrappers) -------------------
# GRADE ON THE BENCH VENV, not bare `python3`. The bin/ wrappers resolve
# `python3` at creation time to /usr/bin/python3, and Lambda images ship a
# SYSTEM torch (cu12.x) there — so a bare python3 grades on a different
# torch/CUDA stack than the venv the anchors and re-grades use. Found
# 2026-07-30: opus-5's 07 ext (nvcc 12.8, links libcublas.so.12) passed the
# in-run grade on system torch but hard-fails under the venv's torch cu130
# (cublas-13 handle into cublas-12 = CUBLAS_STATUS_NOT_INITIALIZED). One
# stack everywhere: the venv python, still routed through the lock wrapper.
GRADE_PY="$BENCH_ROOT/.venv/bin/python"
[ -x "$GRADE_PY" ] || GRADE_PY=python3
if [ "${KBM_SKIP_GRADE:-0}" != "1" ]; then
    if [ -f "$PROBLEM_DIR/solution.py" ]; then
        ( cd "$PROBLEM_DIR" && PATH="$RUN_DIR/bin:$PATH" "$GRADE_PY" check.py ) \
            > "$RUN_DIR/check.log" 2>&1 || true
        # Read the VERDICT, not the last line. `tail -1` silently reported a
        # torchrun traceback separator as the check result on 2026-07-25, and the
        # cell went on to publish a speedup for a solution that failed
        # correctness. A grade must never be inferred from whatever text happened
        # to land last.
        if grep -qE "^PASS" "$RUN_DIR/check.log"; then
            CHECK_STATUS=PASS
        else
            CHECK_STATUS=FAIL
        fi
        {
            echo "check: $CHECK_STATUS"
            if [ "$CHECK_STATUS" = "PASS" ]; then
                ( cd "$PROBLEM_DIR" && PATH="$RUN_DIR/bin:$PATH" "$GRADE_PY" benchmark.py ) \
                    > "$RUN_DIR/benchmark.log" 2>&1 || true
                # 01 prints peak_fraction (busbw metric); 07/08/09 print speedup.
                grep -E "^device:|peak_fraction:|speedup:|RESULT:" "$RUN_DIR/benchmark.log" || true
            else
                # No headline at all for an incorrect solution — not even a
                # labelled one, so nothing downstream can scrape a number here.
                echo "RESULT: INVALID_CHECK_FAILED"
            fi
        } | tee -a "$RUN_DIR/meta.log"
    else
        echo "no_solution" | tee -a "$RUN_DIR/meta.log"
    fi
fi
echo "[run_agent] done $RUN_ID" | tee -a "$RUN_DIR/meta.log"
