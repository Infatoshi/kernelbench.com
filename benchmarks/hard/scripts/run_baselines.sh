#!/bin/bash
# Run benchmark.py for every problem, parse the per-variant output, and emit
# results/problem_baselines.json with reference + SOTA timings.
#
# benchmark.py already runs four variants per shape (eager, compiled, sota,
# solution). We only need to capture eager + compiled + sota — the solution
# column is whatever happens to be in the workspace.
#
# To get a benchmarkable solution.py (some scripts crash without one), we
# write a stub that re-exports from reference.py. The solution numbers are
# baseline-only bookkeeping; eager / compiled / sota are the rows we care about.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="results/problem_baselines.json"
mkdir -p results
TMP_DIR="$(mktemp -d)"
declare -a touched=()

cleanup() {
    for p in "${touched[@]}"; do
        pdir="problems/$p"
        backup="$TMP_DIR/$p.solution.py"
        marker="$TMP_DIR/$p.had_solution"
        if [ -f "$marker" ]; then
            cp "$backup" "$pdir/solution.py"
        else
            rm -f "$pdir/solution.py"
        fi
    done
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

declare -a problems=(
    01_fp8_gemm 02_kda_cutlass 03_paged_attention 04_kahan_softmax
    05_topk_bitonic 06_sonic_moe_swiglu 07_w4a16_gemm
)

# Build the JSON via python so we can compute geomeans + handle missing data.
out_lines=()
out_lines+=("{")
out_lines+=("  \"_schema\": \"per-problem reference and SOTA timings, geomean across shapes\",")
out_lines+=("  \"_generated\": \"$(date -Iseconds)\",")
out_lines+=("  \"_hardware\": \"RTX_PRO_6000_BLACKWELL_SM120\",")
out_lines+=("  \"problems\": {")

for i in "${!problems[@]}"; do
    p="${problems[$i]}"
    pdir="problems/$p"
    [ -f "$pdir/benchmark.py" ] || { echo "skip $p (no benchmark.py)"; continue; }
    touched+=("$p")

    backup="$TMP_DIR/$p.solution.py"
    marker="$TMP_DIR/$p.had_solution"
    if [ -f "$pdir/solution.py" ]; then
        cp "$pdir/solution.py" "$backup"
        touch "$marker"
    fi

    # Stub solution: re-export the reference Model so benchmark.py has something
    # to import. The 'solution' variant is included for traceability, but only
    # 'eager', 'compiled', and 'sota' are baseline-relevant.
    cat > "$pdir/solution.py" <<EOF
# Auto-generated stub for baseline run; re-exports the reference Model.
from reference import Model, get_inputs, get_init_inputs  # noqa: F401
EOF

    echo "==> running $p (timeout 5 min)"
    raw=$(cd "$pdir" && timeout 300 uv run python benchmark.py 2>&1 || true)
    # Save raw output for debugging
    mkdir -p results/raw_baselines
    echo "$raw" > "results/raw_baselines/$p.txt"

    # Parse: variant lines look like "shape=N variant=X tflops=N gbps=N ms=N"
    # For each variant, compute geomean of ms across shapes.
    parsed=$(uv run python - "$pdir/benchmark.py" <<PYEOF
import json
import math
import re

text = open("results/raw_baselines/$p.txt").read()
variants = {}
for line in text.splitlines():
    m = re.match(r"shape=(\d+) variant=(\w+) tflops=([\d.]+) gbps=([\d.]+) ms=([\d.]+)", line)
    if m:
        v = m.group(2)
        ms = float(m.group(5))
        tflops = float(m.group(3))
        gbps = float(m.group(4))
        variants.setdefault(v, []).append({"ms": ms, "tflops": tflops, "gbps": gbps})

def geomean(xs):
    if not xs: return None
    return math.exp(sum(math.log(max(x, 1e-9)) for x in xs) / len(xs))

out = {}
for v, rows in variants.items():
    out[v] = {
        "ms": round(geomean([r["ms"] for r in rows]), 4),
        "tflops": round(geomean([r["tflops"] for r in rows]), 3),
        "gbps": round(geomean([r["gbps"] for r in rows]), 3),
        "n_shapes": len(rows),
    }
print(json.dumps(out))
PYEOF
)
    out_lines+=("    \"$p\": $parsed$([ $i -lt $((${#problems[@]} - 1)) ] && echo ',' || echo '')")
done

out_lines+=("  }")
out_lines+=("}")

printf '%s\n' "${out_lines[@]}" > "$OUT"
echo "wrote $OUT"
cat "$OUT"
