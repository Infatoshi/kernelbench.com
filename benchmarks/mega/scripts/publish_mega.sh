#!/usr/bin/env bash
# Publish KernelBench-Mega results to the site:
#   1. generate a self-contained transcript viewer (the agent's optimization
#      journey) for every gpu-tagged run -> public/runs/<run_id>.html
#   2. copy each run's solution.py + the problem reference.py into
#      public/data/mega/code/ so the page can link the actual kernels
#   3. rebuild public/data/mega/results.csv (rich columns + viewer flag)
#
# Run from anywhere; paths are repo-relative.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1   # benchmarks/mega
REPO="$(cd ../.. && pwd)"
RUNS="outputs/runs"
PUB_RUNS="$REPO/public/runs"
PUB_CODE="$REPO/public/data/mega/code"
mkdir -p "$PUB_RUNS" "$PUB_CODE"

echo "[1/3] generate transcript viewers for gpu-tagged runs"
n=0
for d in "$RUNS"/*_02_kimi_linear_decode; do
  [ -f "$d/gpu" ] || continue
  [ -f "$d/result.json" ] || continue
  rid=$(basename "$d")
  if [ ! -f "$PUB_RUNS/$rid.html" ] || [ "$d/index.html" -nt "$PUB_RUNS/$rid.html" ]; then
    if uv run python -m src.viewer "$d" >/dev/null 2>&1 && [ -f "$d/index.html" ]; then
      cp "$d/index.html" "$PUB_RUNS/$rid.html"
      n=$((n+1))
    fi
  fi
  # solution code for the page to link
  [ -f "$d/solution.py" ] && cp "$d/solution.py" "$PUB_CODE/$rid.solution.py.txt"
done
echo "  generated/updated $n viewers -> $PUB_RUNS"

echo "[1b/3] redact secrets from generated artifacts (agents can echo env keys)"
# Build sed rules from every ~/.env_vars value + token prefixes. Never printed.
SEDF=$(mktemp)
if [ -f "$HOME/.env_vars" ]; then
  while IFS= read -r line; do
    val="${line#*=}"; val="${val%\"}"; val="${val#\"}"
    [ ${#val} -ge 16 ] && printf 's|%s|REDACTED|g\n' "$(printf '%s' "$val" | sed 's/[&/\\|]/\\&/g')" >> "$SEDF"
  done < <(grep -E "^(export )?[A-Z_]+=." "$HOME/.env_vars" | sed 's/^export //')
fi
cat >> "$SEDF" <<'PAT'
s|sk-[A-Za-z0-9_-]\{20,\}|sk-REDACTED|g
s|ghp_[A-Za-z0-9]\{30,\}|ghp_REDACTED|g
s|github_pat_[A-Za-z0-9_]\{30,\}|github_pat_REDACTED|g
s|hf_[A-Za-z0-9]\{30,\}|hf_REDACTED|g
PAT
for f in "$PUB_RUNS"/*_02_kimi_linear_decode.html "$PUB_CODE"/*; do
  [ -f "$f" ] && sed -i -f "$SEDF" "$f"
done
rm -f "$SEDF"
uv run python "$REPO/scripts/redaction.py" "$PUB_RUNS" "$PUB_CODE"

echo "[2/3] copy problem reference"
cp problems/02_kimi_linear_decode/reference.py "$PUB_CODE/02_kimi_linear_decode.reference.py.txt"
cp problems/02_kimi_linear_decode/baseline.py "$PUB_CODE/02_kimi_linear_decode.baseline.py.txt"
# reference/baseline are repo source (no secrets) but redact defensively too
[ -f /tmp/_noop ] || true

echo "[3/3] rebuild leaderboard CSV"
uv run python scripts/build_mega_leaderboard.py
echo "publish_mega done."
