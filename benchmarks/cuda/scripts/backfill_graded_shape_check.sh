#!/usr/bin/env bash
# Re-check every published cell of the problems whose check.py was widened
# (2026-09-17: 01_glm52_fused_moe, 02_deepseek_nsa now verify the graded shapes
# instead of a T<=256 / S<=384 subset).
#
# This replays ONLY check.py -- it does not re-benchmark. The graded number is
# untouched by this change (the timing path is identical); what is being
# re-established is that each published solution is correct at the sizes it
# was timed on. Runs one cell at a time on an idle GPU.
#
# Usage:
#   benchmarks/cuda/scripts/backfill_graded_shape_check.sh            # all, dry run
#   benchmarks/cuda/scripts/backfill_graded_shape_check.sh --go       # all, for real
#   benchmarks/cuda/scripts/backfill_graded_shape_check.sh --go 01    # one problem
#
# Env:
#   KBH_REGRADE_GPU=0            GPU index (default 0)
#   KBH_BACKFILL_DECK=<dir>      canonical deck to restore check.py from
#                                (default problems-rtxpro6000)
#   KBH_REGRADE_ALLOW_BUSY=1     skip the idle-GPU precondition (debug only)
#
# Runs strictly one cell at a time: the point is check wall-clock against the
# 1800s budget, which a shared GPU makes unreliable.
#
# Output: a per-cell PASS/FAIL table plus a JSON summary under
# benchmarks/cuda/results/backfill_<date>.json. A FAIL means the cell was
# passing under the old narrow check and fails at a graded shape -- withdraw it
# and write an annotation; do NOT re-benchmark it.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

DECK="${KBH_BACKFILL_DECK:-problems-rtxpro6000}"
GPU="${KBH_REGRADE_GPU:-0}"
GO=0
PROBLEM_FILTER=""

for arg in "$@"; do
    case "$arg" in
        --go) GO=1 ;;
        -*)
            echo "unknown flag: $arg" >&2
            exit 2
            ;;
        *) PROBLEM_FILTER="$arg" ;;
    esac
done

# The two problems whose check.py changed. Keep this list explicit: a wildcard
# would silently re-check a deck that did not change and confuse the record.
PROBLEMS=("01_glm52_fused_moe" "02_deepseek_nsa")
if [ -n "$PROBLEM_FILTER" ]; then
    case "$PROBLEM_FILTER" in
        01|01_glm52_fused_moe) PROBLEMS=("01_glm52_fused_moe") ;;
        02|02_deepseek_nsa)    PROBLEMS=("02_deepseek_nsa") ;;
        *)
            echo "unknown problem: $PROBLEM_FILTER (expected 01, 02, or a full name)" >&2
            exit 2
            ;;
    esac
fi

if [ ! -f "$DECK/01_glm52_fused_moe/check.py" ]; then
    echo "FATAL: deck not found at $HERE/$DECK" >&2
    exit 3
fi

# Collect the cells: every published run_id for the affected problems, taken
# from the leaderboard so the set matches the board exactly rather than a glob
# over the archive (which also holds superseded and experimental sweeps).
mapfile -t RUNS < <(python3 - "$DECK" "${PROBLEMS[@]}" <<'PY'
import json, sys
from pathlib import Path
deck, probs = sys.argv[1], set(sys.argv[2:])
lb = json.loads(Path("results/leaderboard.json").read_text())
seen = []
for model in lb.get("models", []):
    for prob, cell in (model.get("results") or {}).items():
        if prob in probs and cell.get("run_id"):
            if cell["run_id"] not in seen:
                seen.append(cell["run_id"])
print("\n".join(seen))
PY
)

if [ "${#RUNS[@]}" -eq 0 ]; then
    echo "no published cells found for: ${PROBLEMS[*]}" >&2
    exit 1
fi

echo "deck:     $DECK"
echo "problems: ${PROBLEMS[*]}"
echo "gpu:      $GPU"
echo "cells:    ${#RUNS[@]}"
echo

if [ "$GO" -ne 1 ]; then
    echo "DRY RUN -- would re-check, one at a time:"
    for r in "${RUNS[@]}"; do
        echo "  $r"
    done
    echo
    echo "Re-run with --go to execute."
    exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | grep -c . || true)"
    if [ "${busy:-0}" -gt 0 ] && [ "${KBH_REGRADE_ALLOW_BUSY:-0}" != "1" ]; then
        echo "STOP: GPU $GPU has $busy compute process(es); check wall-clock would be unreliable." >&2
        echo "      Wait for an idle GPU, or set KBH_REGRADE_ALLOW_BUSY=1 to override (debug only)." >&2
        exit 4
    fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/backfill_${STAMP}.json"
PASS=0
FAILED=0
pass_list=()
fail_list=()

for rid in "${RUNS[@]}"; do
    printf '=== %s\n' "$rid"
    if KBH_REGRADE_DECK="$DECK" KBH_REGRADE_GPU="$GPU" \
        scripts/regrade_sequential.sh "outputs/runs/$rid" > "/tmp/backfill_${rid}.log" 2>&1; then
        echo "    PASS"
        PASS=$((PASS + 1))
        pass_list+=("$rid")
    else
        echo "    FAIL (see /tmp/backfill_${rid}.log)"
        tail -n 5 "/tmp/backfill_${rid}.log" | sed 's/^/      /'
        FAILED=$((FAILED + 1))
        fail_list+=("$rid")
    fi
done

python3 - "$OUT" "$DECK" "$PASS" "$FAILED" "${pass_list[@]:-}" -- "${fail_list[@]:-}" <<'PY'
import json, sys
from datetime import datetime
out, deck, npass, nfail = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rest = sys.argv[5:]
idx = rest.index("--") if "--" in rest else len(rest)
p, f = [x for x in rest[:idx] if x], [x for x in rest[idx + 1 :] if x]
payload = {
    "_schema": "graded-shape check backfill (2026-09-17 check.py widening)",
    "_generated": datetime.now().isoformat(timespec="seconds"),
    "deck": deck,
    "problems": ["01_glm52_fused_moe", "02_deepseek_nsa"],
    "n_pass": npass,
    "n_fail": nfail,
    "passed": p,
    "failed": f,
    "note": (
        "A fail means the cell passed under the old narrow check and fails at a "
        "graded shape. Withdraw it and write an annotation; do not re-benchmark."
    ),
}
with open(out, "w") as fh:
    json.dump(payload, fh, indent=1, sort_keys=True)
    fh.write("\n")
print(f"\nwrote {out}")
PY

echo
echo "pass: $PASS   fail: $FAILED"
if [ "$FAILED" -gt 0 ]; then
    echo "withdraw the failing cells and annotate them; the board is not correct until then."
    exit 1
fi