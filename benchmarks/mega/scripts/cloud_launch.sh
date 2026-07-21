#!/usr/bin/env bash
# Provision-and-run a KernelBench-Mega journey on a brev H100, driven from anvil.
#
# Reproducible cloud run. Pre-req: a healthy brev instance (create separately so
# this script does not touch instance lifecycle):
#   brev create claude-mega-h100 --gpu H100     # hyperstack ~$2.28/hr, ~3min boot
#   brev refresh                                 # wire ~/.brev/ssh_config
# Then:
#   scripts/cloud_launch.sh <instance> <gpu_label> [problem] [budget_s]
# Defaults: <instance> required, gpu_label required, problem 03, budget 10800
#
# It rsyncs the repo + auth (~/.codex/auth.json, ~/.claude/.credentials.json,
# ~/.env_vars), bootstraps (uv/node/codex + cu128 torch), and launches a detached
# SWEEP (cloud_sweep.sh: codex + opus, each tagged with the GPU label so
# build_mega_leaderboard.py picks it up). Total setup ~8min on a fresh box; bake
# a brev image from a bootstrapped box to get reprovision under a minute.
#
# After it finishes (poll ~/mega_sweep.log): pull run dirs back, regenerate the
# CSV (build_mega_leaderboard.py), redeploy, then terminate the instance.
set -euo pipefail
NAME="${1:?instance name required}"; GPU_LABEL="${2:?gpu label required, e.g. B200}"
PROBLEM="${3:-problems/02_kimi_linear_decode}"; BUDGET="${4:-0}"
S=(ssh -F "$HOME/.brev/ssh_config" -o StrictHostKeyChecking=no)
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# brev's ssh_config periodically drops the host entry (DNS resolve fails). If the
# box is unreachable, `brev refresh` rewrites the config. Self-heal up to 3x.
ensure_reachable() {
  for _ in 1 2 3; do
    "${S[@]}" -o ConnectTimeout=15 "$NAME" true 2>/dev/null && return 0
    echo "  (host unreachable -> brev refresh)"
    "$HOME/.local/bin/brev" refresh >/dev/null 2>&1 || true
    sleep 3
  done
  echo "ERROR: $NAME unreachable after brev refresh; check 'brev ls'"; exit 1
}
ensure_reachable

echo "[1/3] rsync repo + auth -> $NAME"
rsync -az -e "${S[*]}" --exclude outputs --exclude __pycache__ --exclude .venv --exclude '*.pyc' --exclude .git "$HERE/" "$NAME:mega/"
"${S[@]}" "$NAME" "mkdir -p .codex .claude .config/cursor .cursor"
rsync -az -e "${S[*]}" ~/.codex/auth.json "$NAME:.codex/auth.json"
rsync -az -e "${S[*]}" ~/.claude/.credentials.json "$NAME:.claude/.credentials.json"
# Provider keys only — never ship the whole ~/.env_vars to a rented box.
ENV_ALLOWLIST='KIMI_API_KEY|MOONSHOT_API_KEY|ZAI_API_KEY|MINIMAX_API_KEY|DEEPSEEK_API_KEY|LONGCAT_API_KEY|TENCENT_API_KEY|DASHSCOPE_API_KEY|OPENROUTER_API_KEY|OPENAI_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN'
TMPENV="$(mktemp)"
grep -E "^(export )?($ENV_ALLOWLIST)=" ~/.env_vars > "$TMPENV" || true
rsync -az -e "${S[*]}" "$TMPENV" "$NAME:.env_vars"
rm -f "$TMPENV"
# cursor auth (composer harness); *-claude + gemini routes use ~/.env_vars keys only
rsync -az -e "${S[*]}" ~/.config/cursor/auth.json "$NAME:.config/cursor/auth.json" 2>/dev/null || true
rsync -az -e "${S[*]}" ~/.cursor/cli-config.json ~/.cursor/agent-cli-state.json "$NAME:.cursor/" 2>/dev/null || true

echo "[2/3] bootstrap"
"${S[@]}" "$NAME" "bash ~/mega/scripts/cloud_bootstrap.sh"

echo "[3/3] launch detached sweep (gpu=$GPU_LABEL budget=${BUDGET}s/model)"
ensure_reachable
"${S[@]}" "$NAME" "nohup env GPU_LABEL=$GPU_LABEL BUDGET_SECONDS=$BUDGET PROBLEM=$PROBLEM ROSTER_OVERRIDE='${ROSTER_OVERRIDE:-}' bash ~/mega/scripts/cloud_sweep.sh > ~/mega_sweep.log 2>&1 & echo launched sweep PID \$!"
echo "Poll:   ${S[*]} $NAME 'tail -20 ~/mega_sweep.log'"
echo "Pull:   rsync runs back, then 'uv run python scripts/build_mega_leaderboard.py'"
echo "Tagged: each run dir gets a 'gpu' marker = $GPU_LABEL"
