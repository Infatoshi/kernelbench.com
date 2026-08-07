#!/usr/bin/env bash
# Rebuild the v2 leaderboard + transcript viewers from the run archives.
# Run from benchmarks/cuda (or anywhere; paths resolve from script location).
# Writes results/leaderboard.json (site data) and public/runs/*.html (viewers).
set -euo pipefail
if [[ "${KBH_TRUST_ARCHIVE_LOCK_FD:-}" =~ ^[0-9]+$ ]] \
  && [ "/dev/fd/$KBH_TRUST_ARCHIVE_LOCK_FD" -ef / ]; then
  /usr/bin/flock -x -w 7200 "$KBH_TRUST_ARCHIVE_LOCK_FD"
else
  exec {TRUST_PHASE_LOCK_FD}</
  /usr/bin/flock -x -w 7200 "$TRUST_PHASE_LOCK_FD"
  export KBH_TRUST_ARCHIVE_LOCK_FD="$TRUST_PHASE_LOCK_FD"
fi
HARD_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$HARD_DIR/../.." && pwd)"
cd "$HARD_DIR"

echo "[1/3] rebuilding leaderboard_v2.json from archives..."
uv run python scripts/build_v2_leaderboard.py | tail -1

echo "[2/3] reshaping to site schema (results/leaderboard.json)..."
uv run python - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath("../.."))
import json
from scripts.published_submission import atomic_write_text, read_json_file
v = read_json_file("results/leaderboard_v2.json")
models = [{"label":m["label"],"harness":m["harness"],"model":m["model"],"effort":m["effort"],
          "results":m["results"],"pass_count":m["valid_pass_count"],"total_runs":m["total_problems"]}
         for m in v["models"]]
pp = {p:{"n_attempted":d["n_models"],"n_passed":d["n_valid_passes"],
         "best_peak_fraction":d["best_peak_fraction"],"best_model":d["best_model"],
         "ranked_passes":[{"model":r["model"],"peak_fraction":r["peak_fraction"]} for r in d["ranked_valid_passes"]]}
      for p,d in v["per_problem"].items()}
out = {"schema_version":1,"environment":"v2_containerized","hardware":v["hardware"],
       "problems":v["problems"],"models":models,"per_problem":pp,
       "generated_from_summary":{"input":"benchmarks/cuda/outputs/runs","tag":"v2","imported_rows":len(models)}}
atomic_write_text("results/leaderboard.json", json.dumps(out, indent=2) + "\n")
print(f"  wrote results/leaderboard.json ({len(models)} models)")
PY

echo "[3/3] emitting redacted solution files into public/runs (transcripts live on HuggingFace)..."
PUB="$REPO_ROOT/public/runs" uv run python - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath("../.."))
from scripts.published_submission import publish_selected_solutions

count = publish_selected_solutions(
    "results/leaderboard.json",
    "outputs/runs",
    os.environ["PUB"],
)
print(f"  wrote {count} solution files")
PY

echo "[3b/3] validating and redacting the complete publication tree"
uv run python "$REPO_ROOT/scripts/redaction.py" "$REPO_ROOT/public/runs"
echo "done. review, then: git push (or: kb deploy \"msg\")"
