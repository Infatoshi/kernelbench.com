#!/usr/bin/env bash
# Rebuild the v2 leaderboard + transcript viewers from the run archives.
# Run from benchmarks/cuda (or anywhere; paths resolve from script location).
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
       "generated_from_summary":{"input":"benchmarks/cuda/outputs/runs","tag":"v2","imported_rows":len(models)}}
json.dump(out, open("results/leaderboard.json","w"), indent=2)
print(f"  wrote results/leaderboard.json ({len(models)} models)")
PY

echo "[3/3] emitting redacted solution files into public/runs (transcripts live on HuggingFace)..."
mkdir -p "$REPO_ROOT/public/runs"
PUB="$REPO_ROOT/public/runs" uv run python - <<'PY'
import json, os, re
pub = os.environ["PUB"]
rids = sorted({c["run_id"] for m in json.load(open("results/leaderboard.json"))["models"]
               for c in m["results"].values() if c.get("run_id")})
vals = []
envf = os.path.expanduser("~/.env_vars")
if os.path.exists(envf):
    for ln in open(envf):
        if "=" in ln and "export" in ln:
            v = ln.split("=", 1)[1].strip().strip('"').strip("'")
            if len(v) >= 12:
                vals.append(v)
vals = sorted(set(vals), key=len, reverse=True)
pats = [re.compile(p) for p in [r"sk-ant-oat01-[A-Za-z0-9_\-]+", r"sk-proj-[A-Za-z0-9_\-]+",
        r"AIzaSy[A-Za-z0-9_\-]{20,}", r"sk-[a-z]{2,}-[A-Za-z0-9_\-]{16,}", r"hf_[A-Za-z0-9]{20,}"]]
def red(s):
    for v in vals: s = s.replace(v, "REDACTED")
    for p in pats: s = p.sub("REDACTED", s)
    return s
n = 0
for rid in rids:
    sp = f"outputs/runs/{rid}/solution.py"
    if os.path.isfile(sp):
        open(f"{pub}/{rid}_solution.py.txt", "w").write(red(open(sp).read()))
        n += 1
print(f"  wrote {n} solution files")
PY

echo "[3b/3] redacting secrets from viewers (agents can echo env keys)"
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
# Legacy HTML viewers moved to HuggingFace; redact any that still exist locally.
# RIDS may be unset (no local *.html viewers), so default to empty to stay set -u safe.
for rid in ${RIDS:-}; do
  [ -f "$REPO_ROOT/public/runs/$rid.html" ] && sed -i -f "$SEDF" "$REPO_ROOT/public/runs/$rid.html"
done
rm -f "$SEDF"
echo "  redacted secrets from viewers"
uv run python "$REPO_ROOT/scripts/redaction.py" "$REPO_ROOT/public/runs"
echo "done. review, then: git push (or: kb deploy \"msg\")"
