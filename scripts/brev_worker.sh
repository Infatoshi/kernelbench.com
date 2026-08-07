#!/usr/bin/env bash
# Brev GPU worker lifecycle for the hard bench, driven from the control plane
# (Mac or anvil). Wraps: provision -> sync -> bootstrap -> run/regrade -> pull
# -> verified teardown. `kb brev ...` shells out here.
#
#   brev_worker.sh up <name> [type]             create instance (default hyperstack_H100) + wait + refresh ssh
#   brev_worker.sh sync <name>                  rsync thin bench (KB_BREV_BENCH, default hard) -> <name>:kb-<bench>/
#   brev_worker.sh bootstrap <name> [--agents]  uv + torch (cu128); --agents adds node + agent CLIs + auth
#   brev_worker.sh run <name> <harness> <model> <problem> [effort]   detached agent session (problems root auto)
#   brev_worker.sh regrade <name> <run_id> [runs_dir]   transfer the full archive/bundle and run the isolated sequential regrader
#   brev_worker.sh pull <name>                  rsync outputs/runs back (thin) into outputs/runs-brev-<name>/
#   brev_worker.sh down <name>                  teardown via brev_teardown.sh, verified against brev ls
#
# Env: KB_BREV_PROBLEMS_ROOT (default problems-h100), KB_BREV_GPU (default H100),
#      KBH_HARDWARE (default H100) for roofline peaks on regrade.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # repo root
# Bench selected by KB_BREV_BENCH (default hard) — mirrors KB_LAMBDA_BENCH.
BENCH="${KB_BREV_BENCH:-hard}"
BENCH_DIR="$HERE/benchmarks/$BENCH"
REMOTE_DIR="kb-$BENCH"
[ -d "$BENCH_DIR" ] || { echo "ERROR: unknown bench '$BENCH'" >&2; exit 1; }
BREV="${BREV:-brev}"
CMD="${1:?usage: brev_worker.sh <up|sync|bootstrap|run|regrade|pull|down> <name> ...}"
NAME="${2:?instance name required}"
shift 2
S=(ssh -F "$HOME/.brev/ssh_config" -o StrictHostKeyChecking=no)
case "$BENCH" in
  mega) PROBLEMS_ROOT="${KB_BREV_PROBLEMS_ROOT:-problems}" ;;
  *)    PROBLEMS_ROOT="${KB_BREV_PROBLEMS_ROOT:-problems-h100}" ;;
esac

# Keys a worker actually needs; never ship the whole ~/.env_vars.
ENV_ALLOWLIST='KIMI_API_KEY|MOONSHOT_API_KEY|ZAI_API_KEY|MINIMAX_API_KEY|DEEPSEEK_API_KEY|LONGCAT_API_KEY|TENCENT_API_KEY|DASHSCOPE_API_KEY|QWEN_API_KEY|OPENROUTER_API_KEY|OPENAI_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN'

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
    echo "[sync] thin $BENCH bench -> $NAME:$REMOTE_DIR/"
    SYNC_EXCLUDES=(--exclude outputs --exclude __pycache__ --exclude '.venv' --exclude '*.pyc'
      --exclude .git --exclude 'docs/refs'
      --exclude 'results/annotations' --exclude 'docs/*case_stud*')
    # Preserve the node-side cu128 torch-index patch across re-syncs.
    if "${S[@]}" "$NAME" "grep -q pytorch-cu128 $REMOTE_DIR/pyproject.toml" 2>/dev/null; then
      echo "[sync] preserving node torch-index patch (pyproject.toml/uv.lock not shipped)"
      SYNC_EXCLUDES+=(--exclude /pyproject.toml --exclude /uv.lock)
    fi
    rsync -az -e "${S[*]}" "${SYNC_EXCLUDES[@]}" "$BENCH_DIR/" "$NAME:$REMOTE_DIR/"
    # Single-GPU benches' run_hard.sh wraps the shared runner at
    # <monorepo>/scripts/lib/; ship the lib INTO the bench dir (wrapper falls
    # back to it on thin-synced nodes).
    rsync -az -e "${S[*]}" "$HERE/scripts/lib/" "$NAME:$REMOTE_DIR/scripts/lib/"
    rsync -az -e "${S[*]}" "$HERE/scripts/submission_bundle.py" \
      "$NAME:$REMOTE_DIR/scripts/submission_bundle.py"
    rsync -az -e "${S[*]}" "$HERE/scripts/trusted_stage.py" \
      "$NAME:$REMOTE_DIR/scripts/trusted_stage.py"
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
    "${S[@]}" "$NAME" "cd ~/$REMOTE_DIR"' && if ! grep -q pytorch-cu128 pyproject.toml; then cat >> pyproject.toml <<TOML

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
    "${S[@]}" "$NAME" 'export PATH="$HOME/.local/bin:$PATH"; cd ~/'"$REMOTE_DIR"' && uv run python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0))"'
    ;;

  run)
    HARNESS="${1:?harness}"; MODEL="${2:?model}"; PROBLEM="${3:?problem}"; EFFORT="${4:-}"
    ensure_reachable
    echo "[run] detached: $HARNESS $MODEL $PROBLEMS_ROOT/$PROBLEM $EFFORT"
    "${S[@]}" "$NAME" "cd ~/$REMOTE_DIR && mkdir -p outputs && setsid nohup env KBH_AGENT_CONTAINER=1 BUDGET_SECONDS=0 ${KB_BREV_RUN_ENV:-} ./scripts/run_hard.sh $HARNESS $MODEL $PROBLEMS_ROOT/$PROBLEM $EFFORT > outputs/kb_run_\$(basename $PROBLEM).log 2>&1 < /dev/null & echo launched PID \$!"
    echo "Poll:  ${S[*]} $NAME 'tail -20 ~/$REMOTE_DIR/outputs/kb_run_*.log'"
    ;;

  regrade)
    RID="${1:?run_id}"; RUNS_DIR="${2:-$BENCH_DIR/outputs/runs-h100}"
    SRC="$RUNS_DIR/$RID"
    if [[ ! "$RID" =~ ^[A-Za-z0-9._+-]+$ ]] \
      || [ "$RID" = "." ] || [ "$RID" = ".." ]; then
      echo "unsafe run_id: $RID" >&2
      exit 2
    fi
    [ -f "$SRC/result.json" ] && [ -f "$SRC/solution.py" ] \
      || { echo "result.json or solution.py missing in $SRC" >&2; exit 1; }
    # Always gate what leaves this host.  verify-run permits genuinely
    # pre-cutover archives, but rejects a post-cutover run stripped of its
    # bundle metadata.
    LOCAL_PROVENANCE="$(python3 "$HERE/scripts/submission_bundle.py" verify-run "$SRC")"
    ensure_reachable
    REMOTE_RUN="$REMOTE_DIR/outputs/imported-regrades/$RID"
    echo "[regrade] $RID -> bundle-aware sequential replay"
    "${S[@]}" "$NAME" "rm -rf ~/$REMOTE_RUN && mkdir -p ~/$REMOTE_RUN"
    # These are run-root work directories.  Root anchoring is important:
    # submission_bundle/files/cache and .../tmp may be authored sidecars.
    rsync -az -e "${S[*]}" --exclude '/.venv' --exclude '/cache' --exclude '/tmp' \
      --exclude '/replays' --exclude '/regrade_replays' --exclude '/regrade-reviews' \
      "$SRC/" "$NAME:$REMOTE_RUN/"
    REMOTE_STATUS=0
    "${S[@]}" "$NAME" "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/$REMOTE_DIR \
      && env KBH_HARDWARE=${KBH_HARDWARE:-H100} ./scripts/regrade_sequential.sh outputs/imported-regrades/$RID \
        > outputs/imported-regrades/$RID/regrade.log 2>&1" || REMOTE_STATUS=$?

    REVIEW_ROOT="$SRC/regrade-reviews/$(date -u +%Y%m%dT%H%M%SZ)-$$"
    REVIEW="$REVIEW_ROOT/$RID"
    mkdir -p "$REVIEW"
    rsync -az -e "${S[*]}" \
      --include '/result.json' --include '/check*.log' --include '/benchmark*.log' \
      --include '/regrade*.log' --include '/submission_bundle/' \
      --include '/submission_bundle/***' --exclude '*' \
      "$NAME:$REMOTE_RUN/" "$REVIEW/"
    if [ "$REMOTE_STATUS" -ne 0 ]; then
      echo "ERROR: remote regrader failed (status $REMOTE_STATUS); review: $REVIEW" >&2
      exit 1
    fi
    [ -f "$REVIEW/result.json" ] \
      || { echo "ERROR: remote regrader returned no result.json; review: $REVIEW" >&2; exit 1; }
    if ! python3 - "$SRC/result.json" "$REVIEW/result.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    original = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    candidate = json.load(stream)
candidate_regrade = candidate.get("regrade")
if not isinstance(candidate_regrade, dict):
    raise SystemExit("remote result has no regrade provenance")
if candidate_regrade == original.get("regrade"):
    raise SystemExit("remote result has no new regrade provenance")
if candidate_regrade.get("mode") != "sequential_isolated":
    raise SystemExit("remote result is not a sequential isolated regrade")
bundled = any(
    key in candidate
    for key in (
        "submission_bundle",
        "submission_replay",
        "submission_bundle_sha256",
        "submission_replay_status",
    )
)
expected_mode, expected_status = ("bundled", "verified") if bundled else ("legacy", "legacy")
if candidate_regrade.get("submission_mode") != expected_mode:
    raise SystemExit("remote regrade submission mode does not match the result")
if candidate_regrade.get("status") != expected_status:
    raise SystemExit("remote regrade status is not publishable")
PY
    then
      echo "ERROR: remote result was not freshly regraded; review: $REVIEW" >&2
      exit 1
    fi
    if ! PROVENANCE="$(python3 "$HERE/scripts/submission_bundle.py" verify-run "$REVIEW")"; then
      echo "ERROR: returned regrade failed provenance verification; review: $REVIEW" >&2
      exit 1
    fi
    if [ "$PROVENANCE" != "$LOCAL_PROVENANCE" ]; then
      echo "ERROR: returned regrade is bound to different submission provenance; review: $REVIEW" >&2
      exit 1
    fi
    printf '%s\n' "$PROVENANCE" > "$REVIEW/provenance.json"
    echo "  review candidate: $REVIEW (checks passed; archive not promoted)"
    ;;

  pull)
    ensure_reachable
    DEST="$BENCH_DIR/outputs/runs-brev-$NAME"
    mkdir -p "$DEST"
    echo "[pull] $NAME:$REMOTE_DIR/outputs/runs/ -> $DEST (thin)"
    rsync -az -e "${S[*]}" \
      --exclude '.venv' --exclude 'cache' --exclude 'tmp' --exclude 'container_uv_cache' \
      "$NAME:$REMOTE_DIR/outputs/runs/" "$DEST/"
    ;;

  down)
    exec "$HERE/scripts/brev_teardown.sh" "$NAME"
    ;;

  *)
    echo "unknown subcommand: $CMD" >&2
    exit 2
    ;;
esac
