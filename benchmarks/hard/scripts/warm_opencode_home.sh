#!/usr/bin/env bash
# Build a clean pre-warmed opencode agent home template for container runs.
# The template contains only a freshly migrated opencode sqlite DB -- no host
# sessions, storage, or auth -- so copying it into per-run agent homes leaks
# nothing while skipping the first-launch migration inside the run budget.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCODE_BIN="${KBH_AGENT_CONTAINER_OPENCODE_BIN:-$HOME/.opencode/bin/opencode}"
TEMPLATE_DIR="${KBH_OPENCODE_HOME_TEMPLATE:-$REPO_ROOT/outputs/opencode_home_template}"

if [ ! -x "$OPENCODE_BIN" ]; then
    echo "opencode binary not found: $OPENCODE_BIN" >&2
    exit 127
fi

tmp_home="$(mktemp -d)"
trap 'rm -rf "$tmp_home"' EXIT

env -u XDG_DATA_HOME -u XDG_CONFIG_HOME HOME="$tmp_home" \
    timeout 600 "$OPENCODE_BIN" models > /dev/null 2>&1 || true

if [ ! -f "$tmp_home/.local/share/opencode/opencode.db" ]; then
    echo "migration did not produce opencode.db under $tmp_home" >&2
    exit 1
fi

rm -rf "$TEMPLATE_DIR"
mkdir -p "$TEMPLATE_DIR/.local/share"
cp -a "$tmp_home/.local/share/opencode" "$TEMPLATE_DIR/.local/share/opencode"
echo "opencode home template written: $TEMPLATE_DIR"
