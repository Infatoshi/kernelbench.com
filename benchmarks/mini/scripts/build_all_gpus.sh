#!/usr/bin/env bash
# Build every per-GPU leaderboard from its isolated runs dir using KBH_RUNS_DIR.
# Run from benchmarks/hard. Writes results/leaderboard[.<gpu>].json.
# Pass a space-list of GPU keys to limit (default: all present).
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# gpu_key : KBH_HARDWARE : runs_dir : out_file
ROWS=(
  "rtx:RTX_PRO_6000:outputs/runs:results/leaderboard.json"
  "h100:H100:outputs/runs-h100:results/leaderboard.h100.json"
  "b200:B200:outputs/runs-b200:results/leaderboard.b200.json"
)
WANT="${*:-rtx h100 b200}"

reshape() {  # $1 = out file
  uv run python - "$1" <<'PY'
import json, sys
out_file = sys.argv[1]
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
json.dump(out, open(out_file,"w"), indent=2)
print(f"  wrote {out_file} ({len(models)} models)")
PY
}

for row in "${ROWS[@]}"; do
  IFS=: read -r key hw runs out <<<"$row"
  case " $WANT " in *" $key "*) ;; *) continue ;; esac
  if [ ! -d "$runs" ] || [ -z "$(find "$runs" -maxdepth 1 -name '2026*' 2>/dev/null | head -1)" ]; then
    echo "[$key] no runs in $runs, skipping"; continue
  fi
  echo "[$key] build hw=$hw runs=$runs -> $out"
  # The published-run curation manifest is RTX_PRO_6000-specific; the other GPU
  # boards keep date-gated collection from their own runs dir. The rtx row uses
  # the default manifest (results/published_runs.json); others disable it.
  manifest_env=""
  [ "$key" = "rtx" ] || manifest_env="KBH_PUBLISHED_MANIFEST="
  env $manifest_env KBH_HARDWARE="$hw" KBH_RUNS_DIR="$runs" uv run python scripts/build_v2_leaderboard.py | tail -1
  reshape "$out"
done
echo "ALL_GPU_BUILDS_DONE"
