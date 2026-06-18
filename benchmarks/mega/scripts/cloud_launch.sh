#!/usr/bin/env bash
# Provision-and-run a KernelBench-Mega journey on a brev H100, driven from anvil.
#
# Reproducible cloud run. Pre-req: a healthy brev instance (create separately so
# this script does not touch instance lifecycle):
#   brev create claude-mega-h100 --gpu H100     # hyperstack ~$2.28/hr, ~3min boot
#   brev refresh                                 # wire ~/.brev/ssh_config
# Then:
#   scripts/cloud_launch.sh [instance] [harness] [model] [problem] [budget_s] [effort]
# Defaults: claude-mega-h100 codex gpt-5.5 problems/03_kimi_linear_decode 10800 xhigh
#
# It rsyncs the repo + auth (~/.codex/auth.json, ~/.claude/.credentials.json,
# ~/.env_vars), bootstraps (uv/node/codex + cu128 torch), and launches a detached
# run. Total setup ~8min on a fresh box; bake a brev image/launchable from a
# bootstrapped box to get reprovision under a minute.
set -euo pipefail
NAME="${1:-claude-mega-h100}"; HARNESS="${2:-codex}"; MODEL="${3:-gpt-5.5}"
PROBLEM="${4:-problems/03_kimi_linear_decode}"; BUDGET="${5:-10800}"; EFFORT="${6:-xhigh}"
S=(ssh -F "$HOME/.brev/ssh_config" -o StrictHostKeyChecking=no)
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/3] rsync repo + auth -> $NAME"
rsync -az -e "${S[*]}" --exclude outputs --exclude __pycache__ --exclude .venv --exclude '*.pyc' --exclude .git "$HERE/" "$NAME:mega/"
"${S[@]}" "$NAME" "mkdir -p .codex .claude"
rsync -az -e "${S[*]}" ~/.codex/auth.json "$NAME:.codex/auth.json"
rsync -az -e "${S[*]}" ~/.claude/.credentials.json "$NAME:.claude/.credentials.json"
rsync -az -e "${S[*]}" ~/.env_vars "$NAME:.env_vars"

echo "[2/3] bootstrap"
"${S[@]}" "$NAME" "bash ~/mega/scripts/cloud_bootstrap.sh"

echo "[3/3] launch detached run (budget ${BUDGET}s)"
"${S[@]}" "$NAME" "export PATH=\$HOME/.local/bin:\$PATH; cd ~/mega && source ~/.env_vars 2>/dev/null; nohup env BUDGET_SECONDS=$BUDGET PATH=\"\$HOME/.local/bin:\$PATH\" ./scripts/run_hard.sh $HARNESS $MODEL $PROBLEM $EFFORT > ~/mega_run.log 2>&1 & echo launched PID \$!"
echo "Poll:  ${S[*]} $NAME 'tail ~/mega_run.log'"
echo "Result on finish: ~/mega/outputs/runs/<ts>_${HARNESS}_${MODEL}_*/result.json"
