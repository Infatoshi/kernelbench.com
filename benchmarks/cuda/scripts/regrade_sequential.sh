#!/usr/bin/env bash
# Sequential isolated re-benchmark (standing rule, 2026-07-19).
#
# When a wave ran with many agents concurrent, in-run harness timings are
# time-contaminated even with the path-wrapper GPU lock (per-bench lock dirs,
# absolute-path bypasses, overlapping compile/check). Published peak_fraction /
# ms must come from a re-grade where this process is the ONLY GPU owner.
#
# This replays the exact graded path run_hard.sh uses -- check.py then
# benchmark.py, from the run's own archive workspace, with the run's own
# isolated caches -- one run at a time, refusing to start while another CUDA
# process holds the GPU.
#
# Usage:
#   scripts/regrade_sequential.sh outputs/runs/<run_id> [<run_id> ...]
#   scripts/regrade_sequential.sh outputs/runs/*or-opus*/
#
# Env:
#   KBH_REGRADE_GPU=0            GPU index to grade on (default 0)
#   KBH_REGRADE_ALLOW_BUSY=1     skip the idle-GPU precondition (debug only)
#   KBH_REGRADE_DRY_RUN=1        show what would run, touch nothing
#   KBH_REGRADE_DECK=<dir>       canonical deck root to restore template files
#                                from, e.g. problems-h100. Grading against the
#                                deck rather than whatever the workspace holds
#                                means a corrected problem.yaml (hardware key,
#                                tolerances) applies on re-grade, and any agent
#                                edit to the graded surface that survived the
#                                in-session guard is reverted and reported.
#
# Writes into each result.json, preserving the contended originals:
#   peak_fraction / correct / check_* / benchmark_*   <- clean values
#   regrade: {at, host, gpu, contended: {...}}        <- provenance + originals
#
# check.log / benchmark.log are REPLACED with the clean run and the originals
# moved to *.contended.log. That matters beyond tidiness: the cuda headline
# (per-shape ms -> geomean speedup) is computed downstream from benchmark.log,
# not from result.json, so leaving the contended log in place would publish
# contended milliseconds even with a clean peak_fraction.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

GPU="${KBH_REGRADE_GPU:-0}"
DRY="${KBH_REGRADE_DRY_RUN:-0}"
CHECK_TIMEOUT="${KBH_CHECK_TIMEOUT_SECONDS:-1800}"

KBH_CUDA_HOME="${KBH_CUDA_HOME:-/usr/local/cuda-13}"
[ -d "$KBH_CUDA_HOME" ] && export CUDA_HOME="$KBH_CUDA_HOME"

# The whole point is that we own the GPU, so bypass the lock wrapper rather
# than queue behind it.
export KBH_GPU_LOCK_HELD=1
export CUDA_VISIBLE_DEVICES="$GPU"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <run_dir> [<run_dir> ...]" >&2
    exit 2
fi

# Refuse to grade while anything else is computing on this GPU. A re-grade that
# races another job is exactly the contamination we are here to remove.
require_idle_gpu() {
    [ "${KBH_REGRADE_ALLOW_BUSY:-0}" = "1" ] && return 0
    local waited=0
    while true; do
        local busy
        # grep -c prints 0 AND exits 1 when nothing matches, so a `|| echo 0`
        # fallback would append a SECOND zero and break the integer test below.
        busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
        busy=${busy:-0}
        if [ "$busy" -eq 0 ]; then
            [ "$waited" -gt 0 ] && echo "    GPU $GPU idle after ${waited}s"
            return 0
        fi
        if [ "$waited" -eq 0 ]; then
            echo "    waiting for GPU $GPU to go idle ($busy compute app(s))..."
        fi
        sleep 30
        waited=$((waited + 30))
        if [ "$waited" -ge 3600 ]; then
            echo "    GPU $GPU still busy after 1h; skipping" >&2
            return 1
        fi
    done
}

PASS=0; FAIL=0; SKIP=0

for RUN_DIR in "$@"; do
    RUN_DIR="${RUN_DIR%/}"
    RID="$(basename "$RUN_DIR")"

    if [ ! -f "$RUN_DIR/result.json" ]; then
        echo "[skip] $RID: no result.json (run never scored)"; SKIP=$((SKIP+1)); continue
    fi
    if [ ! -f "$RUN_DIR/solution.py" ]; then
        echo "[skip] $RID: no solution.py"; SKIP=$((SKIP+1)); continue
    fi

    PROBLEM=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['problem'])" "$RUN_DIR/result.json")
    PROBLEM_DIR="$RUN_DIR/repo/problems/$PROBLEM"
    if [ ! -d "$PROBLEM_DIR" ]; then
        echo "[skip] $RID: archive workspace missing ($PROBLEM_DIR)"; SKIP=$((SKIP+1)); continue
    fi

    echo "=== $RID ($PROBLEM) ==="

    if [ "$DRY" = "1" ]; then
        echo "    [dry-run] would grade in $PROBLEM_DIR"; continue
    fi

    # Restore the graded surface from the canonical deck. Anything that differs
    # is either a deck correction made since the run (which SHOULD apply) or an
    # agent edit that outlived the in-session guard (which must not stand);
    # either way the re-grade must score against the deck, so report and replace.
    if [ -n "${KBH_REGRADE_DECK:-}" ]; then
        SRC_DECK="$REPO_ROOT/$KBH_REGRADE_DECK/$PROBLEM"
        if [ -d "$SRC_DECK" ]; then
            for t in reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt; do
                if [ -f "$SRC_DECK/$t" ] && ! cmp -s "$SRC_DECK/$t" "$PROBLEM_DIR/$t"; then
                    echo "    restoring $t from $KBH_REGRADE_DECK (workspace copy differed)"
                    cp "$SRC_DECK/$t" "$PROBLEM_DIR/$t"
                fi
            done
        else
            echo "    WARN: KBH_REGRADE_DECK set but $SRC_DECK missing; grading workspace as-is" >&2
        fi
    fi

    # run_hard.sh clears non-template files from the workspace after archiving,
    # so restore the solution and any sidecars it depended on (.cu files, helper
    # modules) before replaying the graded path.
    cp "$RUN_DIR/solution.py" "$PROBLEM_DIR/solution.py"
    if [ -d "$RUN_DIR/scratch" ]; then
        cp -r "$RUN_DIR/scratch/." "$PROBLEM_DIR/" 2>/dev/null || true
    fi

    require_idle_gpu || { SKIP=$((SKIP+1)); continue; }

    # Same isolated caches the original run used, so a compiled extension
    # resolves the way it did in-session.
    export TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
    export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
    export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
    export TMPDIR="$RUN_DIR/tmp" TEMP="$RUN_DIR/tmp" TMP="$RUN_DIR/tmp"
    mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"

    BENCH_TIMEOUT="${KBH_BENCHMARK_TIMEOUT_SECONDS:-1800}"
    [ "$PROBLEM" = "02_kda_cutlass" ] && BENCH_TIMEOUT="${KBH_BENCHMARK_TIMEOUT_SECONDS:-7200}"

    # Park the contended logs once, then write the clean run to the canonical
    # names so every downstream consumer (ms/speedup extraction, viewers) reads
    # single-owner data. Guarded so a second re-grade cannot clobber the true
    # in-session original.
    for L in check benchmark; do
        if [ -f "$RUN_DIR/$L.log" ] && [ ! -f "$RUN_DIR/$L.contended.log" ]; then
            mv "$RUN_DIR/$L.log" "$RUN_DIR/$L.contended.log"
        fi
    done
    CLOG="$RUN_DIR/check.log"
    BLOG="$RUN_DIR/benchmark.log"

    echo "    check.py..."
    C0=$(date +%s); CEXIT=0
    (cd "$PROBLEM_DIR" && timeout "$CHECK_TIMEOUT" uv run python check.py) > "$CLOG" 2>&1 || CEXIT=$?
    CEL=$(( $(date +%s) - C0 ))

    CORRECT=false; SCORE=null; BEXIT=null; BEL=null
    # Same gate as run_hard.sh: exit 0 AND a PASS marker. Neither alone is
    # sufficient (solution stdout can glue onto the marker).
    if [ "$CEXIT" -eq 0 ] && grep -aq "PASS" "$CLOG"; then
        CORRECT=true
        echo "    benchmark.py..."
        B0=$(date +%s); BEXIT=0
        (cd "$PROBLEM_DIR" && timeout "$BENCH_TIMEOUT" uv run python benchmark.py) > "$BLOG" 2>&1 || BEXIT=$?
        BEL=$(( $(date +%s) - B0 ))
        SCORE=$(grep -oP 'peak_fraction:\s*\K[0-9.]+' "$BLOG" | head -1 || true)
        [ -z "$SCORE" ] && SCORE=null
    else
        echo "    check FAILED (exit $CEXIT) -- see $CLOG"
    fi

    RID="$RID" CORRECT="$CORRECT" SCORE="$SCORE" CEXIT="$CEXIT" CEL="$CEL" \
    BEXIT="$BEXIT" BEL="$BEL" GPU="$GPU" \
    python3 - "$RUN_DIR/result.json" <<'PY'
import json, os, socket, subprocess, sys

path = sys.argv[1]
with open(path) as f:
    r = json.load(f)

def num(v):
    return None if v in (None, "", "null") else (float(v) if "." in str(v) else int(v))

# Keep the contended originals so an audit can always see what shifted.
contended = {k: r.get(k) for k in (
    "correct", "peak_fraction", "check_exit_code", "benchmark_exit_code",
    "check_elapsed_seconds", "benchmark_elapsed_seconds")}

try:
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "-i", os.environ["GPU"], "--query-gpu=name",
         "--format=csv,noheader"], text=True).strip()
except Exception:
    gpu_name = None

r["correct"] = os.environ["CORRECT"] == "true"
r["peak_fraction"] = num(os.environ["SCORE"])
r["check_exit_code"] = num(os.environ["CEXIT"])
r["benchmark_exit_code"] = num(os.environ["BEXIT"])
r["check_elapsed_seconds"] = num(os.environ["CEL"])
r["benchmark_elapsed_seconds"] = num(os.environ["BEL"])
r["regrade"] = {
    "at": subprocess.check_output(["date", "-Is"], text=True).strip(),
    "host": socket.gethostname(),
    "gpu_index": int(os.environ["GPU"]),
    "gpu_name": gpu_name,
    "mode": "sequential_isolated",
    "contended": contended,
}

with open(path, "w") as f:
    json.dump(r, f, indent=4)

old, new = contended["peak_fraction"], r["peak_fraction"]
delta = ""
if isinstance(old, (int, float)) and isinstance(new, (int, float)) and old:
    delta = "  (%+.1f%%)" % ((new - old) / old * 100)
print("    correct=%s  peak %s -> %s%s" % (r["correct"], old, new, delta))
PY

    if [ "$CORRECT" = "true" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
done

echo "========================================"
echo "re-graded: $PASS correct, $FAIL failed, $SKIP skipped"
echo "========================================"
