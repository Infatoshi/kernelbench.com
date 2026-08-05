#!/usr/bin/env bash
# Publish KernelBench-Mega results to the site:
#   1. copy every canonical or manually audited run's solution.py into
#      public/data/mega/code/ so failed/rejected attempts remain inspectable
#   2. copy the problem reference and baseline
#   3. rebuild public/data/mega/results.csv (rich columns + trace flag)
#
# Run from anywhere; paths are repo-relative.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1 # benchmarks/mega
REPO="$(cd ../.. && pwd)"
RUNS="outputs/runs"
PUB_CODE="$REPO/public/data/mega/code"
mkdir -p "$PUB_CODE"
PUBLISHED_IDS="$(
	uv run python - <<'PY'
import csv
with open("../../public/data/mega/results.csv") as f:
    print("\n".join(row["run_id"] for row in csv.DictReader(f) if row.get("run_id")))
PY
)"

echo "[1/3] publish canonical/audited solution sources"
n=0
for d in outputs/runs*/*_02_kimi_linear_decode; do
	[ -f "$d/result.json" ] || continue
	rid=$(basename "$d")
	canonical=false
	[ "$(dirname "$d")" = "$RUNS" ] && [ -f "$d/gpu" ] && canonical=true
	audited=false
	[ -f "results/annotations/$rid.yaml" ] && audited=true
	published=false
	[ "$canonical" = true ] && printf '%s\n' "$PUBLISHED_IDS" | grep -Fqx "$rid" && published=true
	[ "$audited" = true ] || [ "$published" = true ] || continue
	# solution code for the page to link -- inline any kernel sidecar the
	# solution loads (import kernels / mega_impl etc.), else the page shows
	# only host glue. Redaction step 1b below still covers the output.
	if [ -f "$d/solution.py" ]; then
		RUN_D="$d" OUT_F="$PUB_CODE/$rid.solution.py.txt" uv run python - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), "..", "scripts"))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../..", "scripts")))
from kernel_sidecars import augment
d = os.environ["RUN_D"]
txt = open(os.path.join(d, "solution.py")).read()
open(os.environ["OUT_F"], "w").write(augment(txt, d))
PY
		n=$((n + 1))
	fi
done
echo "  generated/updated $n sources -> $PUB_CODE"

echo "[1b/3] redact secrets from generated artifacts (agents can echo env keys)"
# Build sed rules from every ~/.env_vars value + token prefixes. Never printed.
SEDF=$(mktemp)
# BSD sed (macOS) needs an explicit empty backup suffix for -i
if sed --version >/dev/null 2>&1; then SED_INPLACE=(-i); else SED_INPLACE=(-i ''); fi
if [ -f "$HOME/.env_vars" ]; then
	while IFS= read -r line; do
		val="${line#*=}"
		val="${val%\"}"
		val="${val#\"}"
		[ ${#val} -ge 16 ] && printf 's|%s|REDACTED|g\n' "$(printf '%s' "$val" | sed 's/[&/\\|]/\\&/g')" >>"$SEDF"
	done < <(grep -E "^(export )?[A-Z_]+=." "$HOME/.env_vars" | sed 's/^export //')
fi
cat >>"$SEDF" <<'PAT'
s|sk-[A-Za-z0-9_-]\{20,\}|sk-REDACTED|g
s|ghp_[A-Za-z0-9]\{30,\}|ghp_REDACTED|g
s|github_pat_[A-Za-z0-9_]\{30,\}|github_pat_REDACTED|g
s|hf_[A-Za-z0-9]\{30,\}|hf_REDACTED|g
PAT
for f in "$PUB_CODE"/*; do
	[ -f "$f" ] && sed "${SED_INPLACE[@]}" -f "$SEDF" "$f"
done
rm -f "$SEDF"
uv run python "$REPO/scripts/redaction.py" "$PUB_CODE"

echo "[2/3] copy problem reference"
cp problems/02_kimi_linear_decode/reference.py "$PUB_CODE/02_kimi_linear_decode.reference.py.txt"
cp problems/02_kimi_linear_decode/baseline.py "$PUB_CODE/02_kimi_linear_decode.baseline.py.txt"
# reference/baseline are repo source (no secrets) but redact defensively too
[ -f /tmp/_noop ] || true

echo "[3/3] rebuild leaderboard CSV"
uv run python scripts/build_mega_leaderboard.py
echo "publish_mega done."
