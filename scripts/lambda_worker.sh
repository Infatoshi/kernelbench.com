#!/usr/bin/env bash
# Lambda Cloud GPU worker lifecycle for KernelBench (hard/mega/cuda/multi),
# driven from the control plane (Mac or anvil). Mirrors scripts/brev_worker.sh.
#
#   lambda_worker.sh list                         instance types + capacity
#   lambda_worker.sh ls                           running instances
#   lambda_worker.sh up <name> [type] [region]    launch (default gpu_1x_h100_sxm5)
#   lambda_worker.sh sync <name>                  rsync thin hard bench -> name:kb-hard/
#   lambda_worker.sh bootstrap <name> [--agents]  uv + torch; --agents adds CLIs + auth
#   lambda_worker.sh run <name> <harness> <model> <problem> [effort]
#   lambda_worker.sh regrade <name> <run_id> [runs_dir]
#   lambda_worker.sh pull <name>                  rsync outputs/runs back -> outputs/runs-lambda-<name>/
#   lambda_worker.sh down <name>                  terminate by name (verified)
#   lambda_worker.sh ssh <name> [cmd...]          ssh into instance
#
# Auth: LAMBDA_API_KEY or LAMDBA_API_KEY in ~/.env_vars (both set on Mac+anvil).
# SSH keys registered on the Lambda account (names): macbook, anvil
#   (launch attaches BOTH so either machine can log in).
#
# Env: KB_LAMBDA_TYPE (default gpu_1x_h100_sxm5), KB_LAMBDA_REGION (auto if empty),
#      KB_LAMBDA_SSH_KEYS (default: this host's key; API allows exactly one), KB_LAMBDA_PROBLEMS_ROOT
#      (default problems-h100), KBH_HARDWARE (default H100) for regrade.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
HARD="$HERE/benchmarks/hard"
API="${LAMBDA_API_BASE:-https://cloud.lambda.ai/api/v1}"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/kernelbench-lambda"
mkdir -p "$STATE_DIR"

# shellcheck disable=SC1090
if [ -f "$HOME/.env_vars" ]; then
  set -a
  # shellcheck source=/dev/null
  . "$HOME/.env_vars"
  set +a
fi
API_KEY="${LAMBDA_API_KEY:-${LAMDBA_API_KEY:-}}"
if [ -z "$API_KEY" ]; then
  echo "ERROR: LAMBDA_API_KEY (or LAMDBA_API_KEY) not set — add to ~/.env_vars" >&2
  exit 1
fi

CMD="${1:?usage: lambda_worker.sh <list|ls|up|sync|bootstrap|run|regrade|pull|down|ssh> ...}"
shift || true

ENV_ALLOWLIST='KIMI_API_KEY|MOONSHOT_API_KEY|ZAI_API_KEY|MINIMAX_API_KEY|DEEPSEEK_API_KEY|LONGCAT_API_KEY|TENCENT_API_KEY|DASHSCOPE_API_KEY|OPENROUTER_API_KEY|OPENAI_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN'
PROBLEMS_ROOT="${KB_LAMBDA_PROBLEMS_ROOT:-problems-h100}"
# Lambda's launch API rejects requests with more than one ssh key
# ("Invalid number of SSH keys", observed 2026-07-21), so the default is the
# single key for whichever control plane is running this script. The other
# box can still log in by appending its pubkey post-boot if ever needed.
case "$(hostname)" in
  anvil*) _KB_LAMBDA_DEFAULT_KEY="anvil" ;;
  *)      _KB_LAMBDA_DEFAULT_KEY="macbook" ;;
esac
SSH_KEY_CSV="${KB_LAMBDA_SSH_KEYS:-$_KB_LAMBDA_DEFAULT_KEY}"
SSH_USER="${KB_LAMBDA_SSH_USER:-ubuntu}"

api() {
  local method="$1" path="$2"
  shift 2
  curl -sS -X "$method" "${API}${path}" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -H "User-Agent: kernelbench-lambda-worker" \
    "$@"
}

require_jq() {
  command -v jq >/dev/null 2>&1 || {
    echo "ERROR: jq required on this host" >&2
    exit 1
  }
}

# Resolve instance by name -> json object (latest match). Empty if missing.
instance_by_name() {
  local name="$1"
  api GET /instances | jq -c --arg n "$name" '
    (.data // []) | map(select(.name == $n)) | .[0] // empty
  '
}

instance_ip() {
  local name="$1" row ip
  row="$(instance_by_name "$name")"
  [ -n "$row" ] || return 1
  ip="$(jq -r '.ip // .public_ip // empty' <<<"$row")"
  [ -n "$ip" ] && [ "$ip" != "null" ] || return 1
  printf '%s' "$ip"
}

instance_id() {
  local name="$1" row
  row="$(instance_by_name "$name")"
  [ -n "$row" ] || return 1
  jq -r '.id // empty' <<<"$row"
}

ssh_base() {
  local ip="$1"
  SSH_AUTH_SOCK= ssh -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$STATE_DIR/known_hosts" \
    -o ConnectTimeout=15 \
    -o BatchMode=yes \
    "${SSH_USER}@${ip}"
}

ensure_reachable() {
  local name="$1" ip
  for _ in 1 2 3 4 5 6 8 10 12 15 20 25 30; do
    ip="$(instance_ip "$name" 2>/dev/null || true)"
    if [ -n "${ip:-}" ]; then
      if ssh_base "$ip" true 2>/dev/null; then
        echo "$ip" >"$STATE_DIR/${name}.ip"
        return 0
      fi
    fi
    sleep 10
  done
  echo "ERROR: $name not SSH-reachable; lambda_worker.sh ls" >&2
  exit 1
}

ssh_to() {
  local name="$1"
  shift
  local ip
  ip="$(instance_ip "$name" || true)"
  if [ -z "${ip:-}" ] && [ -f "$STATE_DIR/${name}.ip" ]; then
    ip="$(cat "$STATE_DIR/${name}.ip")"
  fi
  [ -n "${ip:-}" ] || {
    echo "ERROR: no IP for $name" >&2
    exit 1
  }
  ssh_base "$ip" "$@"
}

pick_region() {
  local type="$1" preferred="${2:-}"
  if [ -n "$preferred" ]; then
    printf '%s' "$preferred"
    return
  fi
  api GET /instance-types | jq -r --arg t "$type" '
    (.data[$t] // .data // {}) as $d
    | if ($d | type) == "object" and ($d.regions_with_capacity_available != null) then
        ($d.regions_with_capacity_available // [])
        | map(if type=="object" then .name else . end)
        | .[0] // empty
      else
        empty
      end
  '
}

# --- commands ---

case "$CMD" in
  list)
    require_jq
    api GET /instance-types | jq -r '
      (.data // {}) | to_entries[]
      | .key as $k
      | .value as $v
      | ($v.instance_type // $v) as $it
      | ($v.regions_with_capacity_available // []) as $regs
      | ($regs | map(if type=="object" then .name else . end) | join(",")) as $r
      | ((($it.price_cents_per_hour // 0) / 100) | tostring) as $p
      | "\($k)\t$\($p)/hr\t\($it.description // "")\t\($r)"
    ' | column -t -s $'\t' 2>/dev/null || cat
    ;;

  ls | running)
    require_jq
    api GET /instances | jq -r '
      (.data // [])
      | if length==0 then "No running instances" else
          .[] | "\(.name // "-")\t\(.id)\t\(.instance_type.name // .instance_type // "-")\t\(.status // "-")\t\(.ip // .public_ip // "-")\t\(.region.name // .region // "-")"
        end
    ' | column -t -s $'\t' 2>/dev/null || cat
    ;;

  up)
    require_jq
    NAME="${1:?name required}"
    TYPE="${2:-${KB_LAMBDA_TYPE:-gpu_1x_h100_sxm5}}"
    REGION_ARG="${3:-${KB_LAMBDA_REGION:-}}"
    REGION="$(pick_region "$TYPE" "$REGION_ARG")"
    if [ -z "$REGION" ]; then
      echo "ERROR: no capacity for $TYPE (and no KB_LAMBDA_REGION set). Try: lambda_worker.sh list" >&2
      exit 1
    fi
    # shellcheck disable=SC2206
    IFS=',' read -r -a KEY_ARR <<<"$SSH_KEY_CSV"
    KEYS_JSON="$(printf '%s\n' "${KEY_ARR[@]}" | jq -R . | jq -s .)"
    echo "[up] launch name=$NAME type=$TYPE region=$REGION keys=$SSH_KEY_CSV"
    PAYLOAD="$(jq -n \
      --arg type "$TYPE" \
      --arg region "$REGION" \
      --arg name "$NAME" \
      --argjson keys "$KEYS_JSON" \
      '{instance_type_name:$type, region_name:$region, ssh_key_names:$keys, name:$name, quantity:1}')"
    RESP="$(api POST /instance-operations/launch -d "$PAYLOAD")"
    if echo "$RESP" | jq -e '.error' >/dev/null 2>&1; then
      echo "ERROR launch failed: $RESP" >&2
      exit 1
    fi
    echo "$RESP" | jq .
    echo "[up] waiting for active + SSH ..."
    for _ in $(seq 1 90); do
      row="$(instance_by_name "$NAME" || true)"
      if [ -n "$row" ]; then
        status="$(jq -r '.status // empty' <<<"$row")"
        ip="$(jq -r '.ip // .public_ip // empty' <<<"$row")"
        echo "  status=$status ip=$ip"
        if [ "$status" = "active" ] || [ "$status" = "running" ] || [ -n "$ip" ]; then
          if [ -n "$ip" ] && ssh_base "$ip" true 2>/dev/null; then
            echo "$ip" >"$STATE_DIR/${NAME}.ip"
            echo "[up] $NAME reachable at $ip"
            exit 0
          fi
        fi
      fi
      sleep 10
    done
    echo "ERROR: timed out waiting for $NAME" >&2
    exit 1
    ;;

  sync)
    NAME="${1:?name required}"
    ensure_reachable "$NAME"
    IP="$(instance_ip "$NAME")"
    echo "[sync] thin hard bench -> ${SSH_USER}@${IP}:kb-hard/"
    rsync -az -e "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$STATE_DIR/known_hosts -o BatchMode=yes" \
      --exclude outputs --exclude __pycache__ --exclude '.venv' --exclude '*.pyc' \
      --exclude .git --exclude 'docs/refs' \
      "$HARD/" "${SSH_USER}@${IP}:kb-hard/"
    TMPENV="$(mktemp)"
    grep -E "^(export )?($ENV_ALLOWLIST)=" "$HOME/.env_vars" >"$TMPENV" || true
    rsync -az -e "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$STATE_DIR/known_hosts -o BatchMode=yes" \
      "$TMPENV" "${SSH_USER}@${IP}:.env_vars"
    rm -f "$TMPENV"
    ;;

  bootstrap)
    NAME="${1:?name required}"
    shift || true
    AGENTS=0
    [ "${1:-}" = "--agents" ] && AGENTS=1
    ensure_reachable "$NAME"
    echo "[bootstrap] uv + torch (agents=$AGENTS)"
    ssh_to "$NAME" 'command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh'
    # Prefer cu128 for driver compatibility (same as brev workers); override with KB_LAMBDA_TORCH_INDEX.
    TORCH_INDEX="${KB_LAMBDA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
    ssh_to "$NAME" "cd ~/kb-hard && if ! grep -q pytorch-cu128 pyproject.toml 2>/dev/null; then cat >> pyproject.toml <<'TOML'

[[tool.uv.index]]
name = \"pytorch-cu128\"
url = \"${TORCH_INDEX}\"
explicit = true

[tool.uv.sources]
torch = { index = \"pytorch-cu128\" }
TOML
rm -f uv.lock; fi; export PATH=\"\$HOME/.local/bin:\$PATH\"; uv sync"
    if [ "$AGENTS" = 1 ]; then
      ssh_to "$NAME" 'command -v node >/dev/null 2>&1 || { curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null 2>&1 && sudo apt-get install -y nodejs >/dev/null 2>&1; }
        command -v bwrap >/dev/null 2>&1 || sudo apt-get install -y -qq bubblewrap >/dev/null 2>&1
        command -v codex >/dev/null 2>&1 || sudo npm i -g @openai/codex >/dev/null 2>&1
        command -v claude >/dev/null 2>&1 || sudo npm i -g @anthropic-ai/claude-code >/dev/null 2>&1'
      ssh_to "$NAME" 'mkdir -p .codex .claude'
      IP="$(instance_ip "$NAME")"
      rsync -az -e "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$STATE_DIR/known_hosts -o BatchMode=yes" \
        ~/.codex/auth.json "${SSH_USER}@${IP}:.codex/auth.json" 2>/dev/null || true
      rsync -az -e "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$STATE_DIR/known_hosts -o BatchMode=yes" \
        ~/.claude/.credentials.json "${SSH_USER}@${IP}:.claude/.credentials.json" 2>/dev/null || true
    fi
    ssh_to "$NAME" 'export PATH="$HOME/.local/bin:$PATH"; cd ~/kb-hard && uv run python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"'
    ;;

  run)
    NAME="${1:?name}"; HARNESS="${2:?harness}"; MODEL="${3:?model}"; PROBLEM="${4:?problem}"; EFFORT="${5:-}"
    ensure_reachable "$NAME"
    echo "[run] detached: $HARNESS $MODEL $PROBLEMS_ROOT/$PROBLEM $EFFORT"
    ssh_to "$NAME" "cd ~/kb-hard && nohup env KBH_AGENT_CONTAINER=0 BUDGET_SECONDS=0 ./scripts/run_hard.sh $HARNESS $MODEL $PROBLEMS_ROOT/$PROBLEM $EFFORT > ~/kb_run.log 2>&1 & echo launched PID \$!"
    echo "Poll:  lambda_worker.sh ssh $NAME 'tail -20 ~/kb_run.log'"
    ;;

  regrade)
    NAME="${1:?name}"; RID="${2:?run_id}"; RUNS_DIR="${3:-$HARD/outputs/runs-h100}"
    SRC="$RUNS_DIR/$RID"
    [ -f "$SRC/solution.py" ] || {
      echo "no solution.py in $SRC" >&2
      exit 1
    }
    PROBLEM="$(sed -E 's/^[0-9]{8}_[0-9]{6}_.*_([0-9]{2}_[a-z0-9_]+)$/\1/' <<<"$RID")"
    ensure_reachable "$NAME"
    IP="$(instance_ip "$NAME")"
    echo "[regrade] $RID -> $PROBLEMS_ROOT/$PROBLEM"
    ssh_to "$NAME" "mkdir -p ~/kb-regrade/$RID && cp -r ~/kb-hard/$PROBLEMS_ROOT/$PROBLEM/. ~/kb-regrade/$RID/"
    rsync -az -e "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$STATE_DIR/known_hosts -o BatchMode=yes" \
      "$SRC/solution.py" "${SSH_USER}@${IP}:kb-regrade/$RID/solution.py"
    ssh_to "$NAME" "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/kb-regrade/$RID \
      && echo '--- check.py ---' && uv run --project ~/kb-hard python check.py \
      && echo '--- benchmark.py ---' && env KBH_HARDWARE=${KBH_HARDWARE:-H100} uv run --project ~/kb-hard python benchmark.py"
    rsync -az -e "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$STATE_DIR/known_hosts -o BatchMode=yes" \
      "${SSH_USER}@${IP}:kb-regrade/$RID/result.json" "$SRC/result.regrade.json" 2>/dev/null \
      && echo "  -> $SRC/result.regrade.json" || echo "  (benchmark printed to stdout only)"
    ;;

  pull)
    NAME="${1:?name}"
    ensure_reachable "$NAME"
    IP="$(instance_ip "$NAME")"
    DEST="$HARD/outputs/runs-lambda-$NAME"
    mkdir -p "$DEST"
    echo "[pull] ${IP}:kb-hard/outputs/runs/ -> $DEST"
    rsync -az -e "ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$STATE_DIR/known_hosts -o BatchMode=yes" \
      --exclude '.venv' --exclude 'cache' --exclude 'tmp' --exclude 'container_uv_cache' \
      "${SSH_USER}@${IP}:kb-hard/outputs/runs/" "$DEST/"
    ;;

  down)
    require_jq
    NAME="${1:?name}"
    ID="$(instance_id "$NAME" || true)"
    if [ -z "${ID:-}" ]; then
      echo "lambda down: no instance named '$NAME' — nothing to do"
      exit 0
    fi
    echo "[down] terminate $NAME id=$ID"
    RESP="$(api POST /instance-operations/terminate -d "$(jq -n --arg id "$ID" '{instance_ids:[$id]}')")"
    echo "$RESP" | jq . 2>/dev/null || echo "$RESP"
    for _ in $(seq 1 60); do
      if [ -z "$(instance_by_name "$NAME" || true)" ]; then
        rm -f "$STATE_DIR/${NAME}.ip"
        echo "TEARDOWN OK: '$NAME' gone"
        exit 0
      fi
      sleep 5
    done
    echo "TEARDOWN FAILED: '$NAME' still listed — check dashboard (billing continues!)" >&2
    exit 1
    ;;

  ssh)
    NAME="${1:?name}"
    shift || true
    ensure_reachable "$NAME"
    if [ "$#" -eq 0 ]; then
      # interactive — drop BatchMode
      IP="$(instance_ip "$NAME")"
      exec ssh -o StrictHostKeyChecking=accept-new \
        -o UserKnownHostsFile="$STATE_DIR/known_hosts" \
        "${SSH_USER}@${IP}"
    fi
    ssh_to "$NAME" "$@"
    ;;

  *)
    echo "unknown subcommand: $CMD" >&2
    exit 2
    ;;
esac
