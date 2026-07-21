#!/usr/bin/env bash
# Brev GPU worker lifecycle for the hard bench, driven from the control plane
# (Mac or anvil). Wraps: provision -> sync -> bootstrap -> run/regrade -> pull
# -> verified teardown. `kb brev ...` shells out here.
#
#   brev_worker.sh up <name> [type]             create instance (default hyperstack_H100) + wait + refresh ssh
#   brev_worker.sh sync <name>                  rsync thin hard bench -> <name>:kb-hard/
#   brev_worker.sh bootstrap <name> [--agents]  uv + torch (cu128); --agents adds node + agent CLIs + auth
#   brev_worker.sh run <name> <harness> <model> <problem> [effort]   detached agent session (problems root auto)
#   brev_worker.sh regrade <name> <run_id> [runs_dir]   re-grade an archived solution.py: check.py then benchmark.py, sequentially
#   brev_worker.sh pull <name>                  rsync outputs/runs back (thin) into outputs/runs-brev-<name>/
#   brev_worker.sh down <name>                  teardown via brev_teardown.sh, verified against brev ls
#
# Env: KB_BREV_PROBLEMS_ROOT (default problems-h100), KB_BREV_GPU (default H100),
#      KBH_HARDWARE (default H100) for roofline peaks on regrade.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # repo root
HARD="$HERE/benchmarks/hard"
BREV="${BREV:-brev}"
CMD="${1:?usage: brev_worker.sh <up|sync|bootstrap|run|regrade|pull|down> <name> ...}"
NAME="${2:?instance name required}"
shift 2
S=(ssh -F "$HOME/.brev/ssh_config" -o StrictHostKeyChecking=no)
PROBLEMS_ROOT="${KB_BREV_PROBLEMS_ROOT:-problems-h100}"

# Keys a worker actually needs; never ship the whole ~/.env_vars.
ENV_ALLOWLIST='KIMI_API_KEY|MOONSHOT_API_KEY|ZAI_API_KEY|MINIMAX_API_KEY|DEEPSEEK_API_KEY|LONGCAT_API_KEY|TENCENT_API_KEY|DASHSCOPE_API_KEY|OPENROUTER_API_KEY|OPENAI_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN'

ensure_reachable() {
  for _ in 1 2 3; do
    "${S[@]}" -o ConnectTimeout=15 "$NAME" true 2>/dev/null && return 0
    echo "  (host unreachable -> brev refresh)"
    "$BREV" refresh >/dev/null 2>&1 || true
    sleep 3
  done
  echo "ERROR: $NAME unreachable after brev refresh; check 'brev ls'" >&2
  exit 1
}

case "$CMD" in
  up)
    # arg = brev instance type (from `brev search`), e.g. hyperstack_H100
    TYPE="${1:-${KB_BREV_TYPE:-hyperstack_H100}}"
    echo "[up] brev create $NAME --type $TYPE"
    "$BREV" create "$NAME" --type "$TYPE"
    echo "[up] waiting for RUNNING/READY ..."
    for _ in $(seq 1 60); do
      row="$("$BREV" ls 2>/dev/null | awk -v n="$NAME" '$1==n')"
      echo "  $row"
      grep -q "RUNNING" <<<"$row" && grep -q "READY" <<<"$row" && break
      sleep 15
    done
    "$BREV" refresh >/dev/null 2>&1 || true
    ensure_reachable
    echo "[up] $NAME reachable"
    ;;

  sync)
    ensure_reachable
    echo "[sync] thin hard bench -> $NAME:kb-hard/"
    rsync -az -e "${S[*]}" \
      --exclude outputs --exclude __pycache__ --exclude '.venv' --exclude '*.pyc' \
      --exclude .git --exclude 'docs/refs' \
      "$HARD/" "$NAME:kb-hard/"
    TMPENV="$(mktemp)"
    grep -E "^(export )?($ENV_ALLOWLIST)=" ~/.env_vars > "$TMPENV" || true
    rsync -az -e "${S[*]}" "$TMPENV" "$NAME:.env_vars"
    rm -f "$TMPENV"
    ;;

  bootstrap)
    ensure_reachable
    AGENTS=0; [ "${1:-}" = "--agents" ] && AGENTS=1
    echo "[bootstrap] uv + torch (agents=$AGENTS)"
    "${S[@]}" "$NAME" 'command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh'
    # cu128 torch: stock brev images ship R570-class drivers; the repo cu130
    # pin needs R580. Same override the mega cloud bootstrap uses.
    "${S[@]}" "$NAME" 'cd ~/kb-hard && if ! grep -q pytorch-cu128 pyproject.toml; then cat >> pyproject.toml <<TOML

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
TOML
rm -f uv.lock; fi; export PATH="$HOME/.local/bin:$PATH"; uv sync'
    if [ "$AGENTS" = 1 ]; then
      "${S[@]}" "$NAME" 'command -v node >/dev/null 2>&1 || { curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null 2>&1 && sudo apt-get install -y nodejs >/dev/null 2>&1; }
        command -v bwrap >/dev/null 2>&1 || sudo apt-get install -y -qq bubblewrap >/dev/null 2>&1
        command -v codex >/dev/null 2>&1 || sudo npm i -g @openai/codex >/dev/null 2>&1
        command -v claude >/dev/null 2>&1 || sudo npm i -g @anthropic-ai/claude-code >/dev/null 2>&1'
      "${S[@]}" "$NAME" 'mkdir -p .codex .claude'
      rsync -az -e "${S[*]}" ~/.codex/auth.json "$NAME:.codex/auth.json" 2>/dev/null || true
      rsync -az -e "${S[*]}" ~/.claude/.credentials.json "$NAME:.claude/.credentials.json" 2>/dev/null || true
    fi
    "${S[@]}" "$NAME" 'export PATH="$HOME/.local/bin:$PATH"; cd ~/kb-hard && uv run python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0))"'
    ;;

  run)
    HARNESS="${1:?harness}"; MODEL="${2:?model}"; PROBLEM="${3:?problem}"; EFFORT="${4:-}"
    ensure_reachable
    echo "[run] detached: $HARNESS $MODEL $PROBLEMS_ROOT/$PROBLEM $EFFORT"
    "${S[@]}" "$NAME" "cd ~/kb-hard && nohup env KBH_AGENT_CONTAINER=0 BUDGET_SECONDS=0 ./scripts/run_hard.sh $HARNESS $MODEL $PROBLEMS_ROOT/$PROBLEM $EFFORT > ~/kb_run.log 2>&1 & echo launched PID \$!"
    echo "Poll:  ${S[*]} $NAME 'tail -20 ~/kb_run.log'"
    ;;

  regrade)
    RID="${1:?run_id}"; RUNS_DIR="${2:-$HARD/outputs/runs-h100}"
    SRC="$RUNS_DIR/$RID"
    [ -f "$SRC/solution.py" ] || { echo "no solution.py in $SRC" >&2; exit 1; }
    PROBLEM="$(sed -E 's/^[0-9]{8}_[0-9]{6}_.*_([0-9]{2}_[a-z0-9_]+)$/\1/' <<<"$RID")"
    ensure_reachable
    echo "[regrade] $RID -> $PROBLEMS_ROOT/$PROBLEM (sequential, no other GPU jobs)"
    "${S[@]}" "$NAME" "mkdir -p ~/kb-regrade/$RID && cp -r ~/kb-hard/$PROBLEMS_ROOT/$PROBLEM/. ~/kb-regrade/$RID/"
    rsync -az -e "${S[*]}" "$SRC/solution.py" "$NAME:kb-regrade/$RID/solution.py"
    "${S[@]}" "$NAME" "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/kb-regrade/$RID \
      && echo '--- check.py ---' && uv run --project ~/kb-hard python check.py \
      && echo '--- benchmark.py ---' && env KBH_HARDWARE=${KBH_HARDWARE:-H100} uv run --project ~/kb-hard python benchmark.py"
    echo "[regrade] pull result.json (if written) back beside the archive:"
    rsync -az -e "${S[*]}" "$NAME:kb-regrade/$RID/result.json" "$SRC/result.regrade.json" 2>/dev/null \
      && echo "  -> $SRC/result.regrade.json" || echo "  (benchmark printed to stdout only)"
    ;;

  pull)
    ensure_reachable
    DEST="$HARD/outputs/runs-brev-$NAME"
    mkdir -p "$DEST"
    echo "[pull] $NAME:kb-hard/outputs/runs/ -> $DEST (thin)"
    rsync -az -e "${S[*]}" \
      --exclude '.venv' --exclude 'cache' --exclude 'tmp' --exclude 'container_uv_cache' \
      "$NAME:kb-hard/outputs/runs/" "$DEST/"
    ;;

  down)
    exec "$HERE/scripts/brev_teardown.sh" "$NAME"
    ;;

  *)
    echo "unknown subcommand: $CMD" >&2
    exit 2
    ;;
esac
