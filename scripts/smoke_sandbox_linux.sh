#!/bin/bash
# End-to-end smoke of the contamination sandbox (scripts/lib/sandbox.sh) on a
# Linux box with bwrap. Touches NO GPU and NO real archives: it builds a
# throwaway fake monorepo under <repo>/out/ (gitignored) and verifies:
#   1. kbh_sandbox_init builds the bwrap argv, plants the honeytoken, and the
#      launch canary PASSES on the correctly hidden tree;
#   2. inside the sandbox, planted foreign archives / runs-remote-* /
#      public/data/*/code are invisible while the own run dir, gpu lock dir,
#      and honeytoken decoy (with beacon) are visible;
#   3. the canary FAILS refuse-closed when the same tree is swept WITHOUT the
#      hide (bwrap with only --dev-bind / /).
#
# Run it on any Linux worker (Lambda/Brev/anvil) without taking the GPU:
#   ./scripts/smoke_sandbox_linux.sh
set -euo pipefail

if [ "$(uname -s)" != "Linux" ]; then
    echo "SKIP: Linux-only smoke (bwrap does not exist on $(uname -s))" >&2
    exit 2
fi
if ! command -v bwrap >/dev/null 2>&1; then
    echo "SKIP: bwrap not installed on this box" >&2
    exit 2
fi
if ! bwrap --dev-bind / / true 2>/dev/null; then
    echo "SKIP: bwrap present but user namespaces denied on this box" >&2
    exit 2
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="$HERE/out/sandbox_smoke.$$"
trap 'rm -rf "$SCRATCH"' EXIT

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

# --- fake monorepo with juicy foreign material ------------------------------
FAKE_ROOT="$SCRATCH/fakerepo"
mkdir -p "$FAKE_ROOT/benchmarks" "$FAKE_ROOT/kbtool"   # hide-root markers
REPO_ROOT="$FAKE_ROOT/benchmarks/fakebench"
FOREIGN_ID="20260719_121747_or-fable_anthropic_claude-fable-5_01_smoke"
mkdir -p "$REPO_ROOT/outputs/runs/$FOREIGN_ID" \
         "$REPO_ROOT/outputs/runs-remote-pro/$FOREIGN_ID" \
         "$FAKE_ROOT/public/data/mega/code" \
         "$FAKE_ROOT/public/runs"
echo "the 24x kernel" > "$REPO_ROOT/outputs/runs/$FOREIGN_ID/solution.py"
echo '{"peak_fraction": 24.6091}' > "$REPO_ROOT/outputs/runs/$FOREIGN_ID/result.json"
echo "pulled copy" > "$REPO_ROOT/outputs/runs-remote-pro/$FOREIGN_ID/solution.py"
echo "published kernel" > "$FAKE_ROOT/public/runs/${FOREIGN_ID}_solution.py.txt"

# --- this run ---------------------------------------------------------------
RUN_ID="$(date +%Y%m%d_%H%M%S)_claude_smoke_01_smoke"
RUN_DIR="$REPO_ROOT/outputs/runs/$RUN_ID"
PROBLEM_NAME="01_smoke"
PROBLEM_DIR="$RUN_DIR/repo/problems/$PROBLEM_NAME"
mkdir -p "$PROBLEM_DIR"
export KBH_GPU_LOCK="$REPO_ROOT/outputs/gpu_lock/gpu.lock"
export REPO_ROOT RUN_DIR RUN_ID PROBLEM_DIR PROBLEM_NAME

# --- 1. init: argv + honeytoken + canary must pass ---------------------------
# shellcheck source=lib/sandbox.sh
. "$HERE/scripts/lib/sandbox.sh"
kbh_sandbox_init    # exits 4 (killing this smoke) if the canary fires
[ "$KBH_SANDBOX_ACTIVE" = "1" ] || fail "sandbox did not activate"
grep -q '"canary": "pass"' "$RUN_DIR/sandbox.json" || fail "sandbox.json missing canary pass"
echo "PASS: kbh_sandbox_init + launch canary clean"

# --- 2. inside the sandbox: foreign hidden, own + honeytoken visible ---------
sbx() { "${KBH_SBX[@]}" "$@"; }
sbx test ! -e "$REPO_ROOT/outputs/runs/$FOREIGN_ID" \
    || fail "foreign run dir visible inside sandbox"
sbx test ! -e "$REPO_ROOT/outputs/runs-remote-pro" \
    || fail "runs-remote-pro visible inside sandbox"
sbx test ! -e "$FAKE_ROOT/public/data/mega/code" \
    || fail "public/data/mega/code visible inside sandbox"
sbx test ! -e "$FAKE_ROOT/public/runs/${FOREIGN_ID}_solution.py.txt" \
    || fail "published kernel text visible inside sandbox"
sbx test -d "$PROBLEM_DIR" || fail "own problem workspace not visible"
sbx test -d "$(dirname "$KBH_GPU_LOCK")" || fail "gpu lock dir not visible"
sbx touch "$PROBLEM_DIR/write_probe" || fail "own workspace not writable"
[ -e "$PROBLEM_DIR/write_probe" ] || fail "own write did not persist to host"
HP_GLOB=("$RUN_DIR/.sandbox/honeypot"/*)
HP_ID="$(basename "${HP_GLOB[0]}")"
sbx grep -q "$KBH_SANDBOX_BEACON" "$REPO_ROOT/outputs/runs/$HP_ID/solution.py" \
    || fail "honeytoken decoy not visible (or beacon missing) inside sandbox"
if sbx sh -c "echo tamper >> '$REPO_ROOT/outputs/runs/$HP_ID/solution.py'" 2>/dev/null; then
    fail "honeytoken decoy is writable (must be ro-bind)"
fi
echo "PASS: hide-the-tree holds; own run + honeytoken visible, decoy read-only"

# --- 3. refuse-closed: canary must FAIL without the hide ---------------------
if bwrap --die-with-parent --dev-bind / / \
        python3 "$RUN_DIR/.sandbox/canary.py" \
        --run-id "$RUN_ID" --allow "$HP_ID" --walk "$REPO_ROOT" \
        > "$SCRATCH/canary_unhidden.log" 2>&1; then
    fail "canary passed on an UNHIDDEN tree (must refuse)"
fi
grep -q "REFUSING LAUNCH" "$SCRATCH/canary_unhidden.log" \
    || fail "canary refusal message missing"
echo "PASS: canary refuses an unhidden tree"

echo "SMOKE OK: sandbox hide-the-tree + honeytoken + refuse-closed canary all hold on $(hostname)"
