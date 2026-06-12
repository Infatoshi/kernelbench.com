#!/usr/bin/env bash
# Rebuild the v2 leaderboard + transcript viewers from the run archives.
# Run from benchmarks/hard (or anywhere; paths resolve from script location).
# Writes results/leaderboard.json (site data) and public/runs/*.html (viewers).
set -euo pipefail
HARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$HARD_DIR/../.." && pwd)"
cd "$HARD_DIR"

echo "[1/3] rebuilding leaderboard_v2.json from archives..."
uv run python scripts/build_v2_leaderboard.py | head -1

echo "[2/3] reshaping to site schema (results/leaderboard.json)..."
uv run python - <<'PY'
import json
from pathlib import Path
v = json.load(open("results/leaderboard_v2.json"))
models = [{"label":m["label"],"harness":m["harness"],"model":m["model"],"effort":m["effort"],
          "results":m["results"],"pass_count":m["valid_pass_count"],"total_runs":m["total_problems"]}
         for m in v["models"]]
pp = {p:{"n_attempted":d["n_models"],"n_passed":d["n_valid_passes"],
         "best_peak_fraction":d["best_peak_fraction"],"best_model":d["best_model"],
         "ranked_passes":[{"model":r["model"],"peak_fraction":r["peak_fraction"]} for r in d["ranked_valid_passes"]]}
      for p,d in v["per_problem"].items()}
out = {"schema_version":1,"environment":"v2_containerized","hardware":v["hardware"],
       "problems":v["problems"],"models":models,"per_problem":pp,
       "generated_from_summary":{"input":"benchmarks/hard/outputs/runs","tag":"v2","imported_rows":len(models)}}
json.dump(out, open("results/leaderboard.json","w"), indent=2)
print(f"  wrote results/leaderboard.json ({len(models)} models)")
PY

echo "[3/3] regenerating transcript viewers into public/runs..."
mkdir -p "$REPO_ROOT/public/runs"
RIDS=$(uv run python -c "import json;d=json.load(open('results/leaderboard.json'));print(' '.join(sorted({c['run_id'] for m in d['models'] for c in m['results'].values()})))")
n=0
for rid in $RIDS; do
  [ -d "outputs/runs/$rid" ] || continue
  KB_VIEWER_THEME=phosphor KB_ANNOTATIONS_DIR="$HARD_DIR/results/annotations" \
    uv run python -m src.viewer "outputs/runs/$rid" --out "$REPO_ROOT/public/runs/$rid.html" >/dev/null 2>&1 && n=$((n+1)) || echo "  WARN viewer failed: $rid"
done
echo "  generated $n viewers"
echo "done. review, then: git push (or: kb deploy \"msg\")"
