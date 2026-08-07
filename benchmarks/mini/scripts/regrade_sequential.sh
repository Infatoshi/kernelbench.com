#!/usr/bin/env bash
# Sequential isolated re-benchmark (standing rule, 2026-07-19).
#
# When a wave ran with many agents concurrent, in-run harness timings are
# time-contaminated even with the path-wrapper GPU lock (per-bench lock dirs,
# absolute-path bypasses, overlapping compile/check). Published peak_fraction /
# ms must come from a re-grade where this process is the ONLY GPU owner.
#
# This replays the exact graded path run_hard.sh uses -- check.py then
# benchmark.py, from separate digest-bound bundle extractions with empty
# compiler caches and no network -- one run at a time, refusing to start while
# another CUDA process holds the GPU. Pre-bundle archives keep their explicit
# legacy replay path; new bundle-aware runs never fall back to it.
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

TRUST_FLOCK_BIN="/usr/bin/flock"
[ -x "$TRUST_FLOCK_BIN" ] || exit 3
exec {TRUST_PHASE_LOCK_FD}</
if ! "$TRUST_FLOCK_BIN" -x -w 7200 "$TRUST_PHASE_LOCK_FD"; then
    echo "FATAL: timed out waiting for the trusted archive lock" >&2
    exit 3
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# Full monorepo and thin-worker layouts both ship this helper beside the
# shared harness support files.
for REPLAY_HELPER in \
    "$REPO_ROOT/../../scripts/lib/submission_replay.sh" \
    "$REPO_ROOT/scripts/lib/submission_replay.sh"; do
    if [ -f "$REPLAY_HELPER" ]; then
        # shellcheck source=../../../scripts/lib/submission_replay.sh
        . "$REPLAY_HELPER"
        break
    fi
done
if ! type submission_bundle_verify >/dev/null 2>&1; then
    echo "FATAL: submission replay helper not found" >&2
    exit 3
fi
SUBMISSION_BUNDLE_PYTHON="/usr/bin/python3"
[ -x "$SUBMISSION_BUNDLE_PYTHON" ] || exit 3
export SUBMISSION_BUNDLE_PYTHON
submission_bind_bundle_tool || exit 3
submission_bind_trusted_stage_tool || exit 3
REAL_UV="$(command -v uv)"
REAL_TIMEOUT="$(command -v timeout)"
REAL_UV_IDENTITY="$(submission_executable_identity "$REAL_UV")" || exit 3
REAL_PYTHON="$("$REAL_UV" python find --project "$REPO_ROOT")" || exit 3
REAL_PYTHON_IDENTITY="$(submission_executable_identity "$REAL_PYTHON")" || exit 3
REAL_TIMEOUT_IDENTITY="$(submission_executable_identity "$REAL_TIMEOUT")" || exit 3
submission_add_isolation_readonly_executable \
    "$REAL_UV" "$REAL_PYTHON" "$REAL_TIMEOUT" || exit 3
submission_resolve_isolation_tools || true

GPU="${KBH_REGRADE_GPU:-0}"
DRY="${KBH_REGRADE_DRY_RUN:-0}"
CHECK_TIMEOUT="${KBH_CHECK_TIMEOUT_SECONDS:-1800}"

KBH_CUDA_HOME="${KBH_CUDA_HOME:-/usr/local/cuda-13}"
[ -d "$KBH_CUDA_HOME" ] && export CUDA_HOME="$KBH_CUDA_HOME"
PATH="/usr/sbin:/usr/bin:/sbin:/bin"
if [[ "${CUDA_HOME:-}" = /* ]] && [ -d "$CUDA_HOME/bin" ]; then
    PATH="$PATH:$CUDA_HOME/bin"
fi
export PATH

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
        # timeout guard: on a wedged driver nvidia-smi never returns, and an
        # ungated call here hung a full 11-cell chain for a day (2026-07-26,
        # node e). A hung probe must fail loudly, not sleep forever.
        smi_out=$(timeout 15 nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        smi_rc=$?
        if [ "$smi_rc" -eq 124 ]; then
            echo "FATAL: nvidia-smi timed out -- GPU/driver wedged; reboot the box" >&2
            exit 3
        fi
        busy=$(printf '%s' "$smi_out" | grep -c .)
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
    # Canonicalize: TMPDIR/cache paths are exported from RUN_DIR, and nvcc
    # resolves a relative TMPDIR against the workspace cwd, not ours --
    # "nvcc fatal: Could not open output file .../tmp/tmpxft_*" (2026-07-26).
    RUN_DIR="$(cd "$RUN_DIR" 2>/dev/null && pwd)" || { echo "  SKIP: cannot resolve $RUN_DIR"; SKIP=$((SKIP+1)); continue; }
    RID="$(basename "$RUN_DIR")"
    CHECK_SURFACE_DIGEST=""
    BENCH_SURFACE_DIGEST=""
    EXPECTED_SURFACE_DIGEST=""
    ISOLATION_APPLIED=false
    REGRADE_STAGE_COUNT=0

    if [ ! -f "$RUN_DIR/result.json" ]; then
        echo "[skip] $RID: no result.json (run never scored)"; SKIP=$((SKIP+1)); continue
    fi
    if [ ! -f "$RUN_DIR/solution.py" ] && [ ! -d "$RUN_DIR/submission_bundle" ]; then
        echo "[skip] $RID: no solution.py"; SKIP=$((SKIP+1)); continue
    fi

    IFS=$'\t' read -r BUNDLE_KIND BUNDLE_DIGEST REPLAY_STATUS EXPECTED_SURFACE_DIGEST < <(
        "$SUBMISSION_BUNDLE_PYTHON" - "$RUN_DIR/result.json" <<'PY'
import datetime
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
r = json.loads(path.read_text())
b = r.get("submission_bundle") if isinstance(r.get("submission_bundle"), dict) else {}
p = r.get("submission_replay") if isinstance(r.get("submission_replay"), dict) else {}
digest = r.get("submission_bundle_sha256") or b.get("bundle_sha256") or ""
status = r.get("submission_replay_status") or p.get("status") or ""
aware = any(k in r for k in ("submission_bundle", "submission_replay", "submission_bundle_sha256", "submission_replay_status"))
run_id = r.get("run_id")
kind = "bundled" if aware else "legacy"
if not isinstance(run_id, str) or run_id != path.parent.name:
    kind = "invalid"
elif not aware:
    match = re.fullmatch(r"(\d{8}_\d{6})(?:_.+)?", run_id)
    try:
        stamp = datetime.datetime.strptime(match.group(1), "%Y%m%d_%H%M%S") if match else None
    except ValueError:
        stamp = None
    if stamp is None or stamp >= datetime.datetime(2026, 8, 8):
        kind = "invalid"
surface = p.get("grader_surface_sha256") or ""
print(kind, digest, status, surface, sep="\t")
PY
    )
    if [ "$BUNDLE_KIND" = "invalid" ]; then
        echo "[skip] $RID: invalid or post-cutover legacy provenance"
        SKIP=$((SKIP+1)); continue
    fi
    if [ "$BUNDLE_KIND" = "bundled" ]; then
        if [ "$REPLAY_STATUS" != "verified" ] || [ -z "$BUNDLE_DIGEST" ]; then
            echo "[skip] $RID: immutable replay is not verified (status=${REPLAY_STATUS:-missing})"
            SKIP=$((SKIP+1)); continue
        fi
        if ! submission_bundle_verify "$RUN_DIR/submission_bundle" \
            "$BUNDLE_DIGEST" >/dev/null 2>&1; then
            echo "[skip] $RID: bundle does not match result.json digest"
            SKIP=$((SKIP+1)); continue
        fi
        if ! "$SUBMISSION_BUNDLE_PYTHON" "$SUBMISSION_BUNDLE_TOOL" \
            verify-run "$RUN_DIR" >/dev/null 2>&1; then
            echo "[skip] $RID: result replay provenance is not publishable"
            SKIP=$((SKIP+1)); continue
        fi
    fi

    PROBLEM=$("$SUBMISSION_BUNDLE_PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1]))['problem'])" "$RUN_DIR/result.json")
    case "$PROBLEM" in
        ""|.|..|*/*|*\\*)
            echo "[skip] $RID: unsafe problem name in result.json"; SKIP=$((SKIP+1)); continue ;;
    esac
    PROBLEM_DIR="$RUN_DIR/repo/problems/$PROBLEM"
    if [ "$BUNDLE_KIND" = "legacy" ] && [ ! -d "$PROBLEM_DIR" ]; then
        echo "[skip] $RID: archive workspace missing ($PROBLEM_DIR)"; SKIP=$((SKIP+1)); continue
    fi

    echo "=== $RID ($PROBLEM) ==="

    if [ "$DRY" = "1" ]; then
        echo "    [dry-run] would grade in $PROBLEM_DIR"; continue
    fi
    if ! submission_reset_run_subdirectory "$RUN_DIR" regrade_replays \
        || ! submission_prepare_output_file "$RUN_DIR/regrade_setup.log"; then
        echo "[skip] $RID: unsafe archive-owned replay/output path"
        SKIP=$((SKIP+1)); continue
    fi
    if ! submission_configure_isolation "$RUN_DIR"; then
        echo "[skip] $RID: full user/mount/PID/network isolation is unavailable"
        SKIP=$((SKIP+1)); continue
    fi

    ISOLATION_TEMPLATE_NAMES=(
        reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt baseline.py
    )

    # Restore the graded surface from the canonical deck. Anything that differs
    # is either a deck correction made since the run (which SHOULD apply) or an
    # agent edit that outlived the in-session guard (which must not stand);
    # either way the re-grade must score against the deck, so report and replace.
    if [ "$BUNDLE_KIND" = "legacy" ] && [ -n "${KBH_REGRADE_DECK:-}" ]; then
        SRC_DECK="$REPO_ROOT/$KBH_REGRADE_DECK/$PROBLEM"
        if [ -d "$SRC_DECK" ]; then
            for t in reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt baseline.py; do
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
    if [ "$BUNDLE_KIND" = "legacy" ]; then
        cp "$RUN_DIR/solution.py" "$PROBLEM_DIR/solution.py"
        if [ -d "$RUN_DIR/scratch" ]; then
            cp -r "$RUN_DIR/scratch/." "$PROBLEM_DIR/" 2>/dev/null || true
        fi
    else
        TEMPLATE_ROOT="$RUN_DIR/template_files"
        if [ -n "${KBH_REGRADE_DECK:-}" ] \
            && [ -d "$REPO_ROOT/$KBH_REGRADE_DECK/$PROBLEM" ]; then
            TEMPLATE_ROOT="$REPO_ROOT/$KBH_REGRADE_DECK/$PROBLEM"
        fi
        REPLAY_TEMPLATES=(
            reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt
        )
        if [ -e "$TEMPLATE_ROOT/baseline.py" ]; then
            REPLAY_TEMPLATES+=(baseline.py)
        fi
        REPLAY_ROOT="$RUN_DIR/regrade_replays/check"
        if ! submission_prepare_replay "$RUN_DIR/submission_bundle" \
            "$BUNDLE_DIGEST" "$REPO_ROOT" "$TEMPLATE_ROOT" "$REPLAY_ROOT" \
            "$PROBLEM" "${REPLAY_TEMPLATES[@]}"; then
            echo "[skip] $RID: cannot prepare verified replay workspace"
            SKIP=$((SKIP+1)); continue
        fi
        PROBLEM_DIR="$REPLAY_ROOT/repo/problems/$PROBLEM"
    fi

    require_idle_gpu || { SKIP=$((SKIP+1)); continue; }

    # A regrade must not inherit compiled artifacts from the agent session or a
    # previous replay. Provision trusted dependencies first, then run the
    # submission with empty compilation caches and network isolation.
    REPLAY_CACHE_ROOT="$RUN_DIR/regrade_replays/check/cache"
    if ! submission_reset_caches "$REPLAY_CACHE_ROOT"; then
        echo "    cannot reset check caches"
        FAIL=$((FAIL+1)); continue
    fi
    if ! submission_executable_matches "$REAL_UV" "$REAL_UV_IDENTITY" \
        || ! submission_executable_matches "$REAL_PYTHON" "$REAL_PYTHON_IDENTITY"; then
        echo "    resolved uv/Python executable changed; refusing regrade"
        FAIL=$((FAIL+1)); continue
    fi
    if [ "$BUNDLE_KIND" = "bundled" ]; then
        PROJECT_ROOT="$RUN_DIR/regrade_replays/check/repo"
        CHECK_SURFACE_DIGEST="$(submission_trusted_surface_digest \
            "$PROJECT_ROOT" "$PROBLEM" "${REPLAY_TEMPLATES[@]}")" || {
            echo "    cannot snapshot trusted check surface"; FAIL=$((FAIL+1)); continue;
        }
        if [ -z "$EXPECTED_SURFACE_DIGEST" ] \
            || [ "$CHECK_SURFACE_DIGEST" != "$EXPECTED_SURFACE_DIGEST" ]; then
            echo "    archived grader surface differs from the original verified replay"
            FAIL=$((FAIL+1)); continue
        fi
    else
        PROJECT_ROOT="$RUN_DIR/repo"
    fi
    if ! "$REAL_UV" sync --frozen --python "$REAL_PYTHON" --project "$PROJECT_ROOT" \
        > "$RUN_DIR/regrade_setup.log" 2>&1; then
        echo "    replay environment setup failed -- see regrade_setup.log"
        FAIL=$((FAIL+1)); continue
    fi
    if ! submission_select_network_isolation \
        || ! submission_select_clean_environment \
            "$RUN_DIR/regrade_replays/check/home"; then
        echo "    full replay namespace isolation unavailable; refusing regrade"
        SKIP=$((SKIP+1)); continue
    fi
    if [ "$BUNDLE_KIND" = "bundled" ]; then
        AFTER_SYNC_SURFACE_DIGEST="$(submission_trusted_surface_digest \
            "$PROJECT_ROOT" "$PROBLEM" "${REPLAY_TEMPLATES[@]}")" || {
            echo "    cannot snapshot trusted check surface"; FAIL=$((FAIL+1)); continue;
        }
        if [ "$AFTER_SYNC_SURFACE_DIGEST" != "$CHECK_SURFACE_DIGEST" ]; then
            echo "    trusted check surface changed during environment setup"
            FAIL=$((FAIL+1)); continue
        fi
    fi

    BENCH_TIMEOUT="${KBH_BENCHMARK_TIMEOUT_SECONDS:-1800}"
    [ "$PROBLEM" = "02_kda_cutlass" ] && BENCH_TIMEOUT="${KBH_BENCHMARK_TIMEOUT_SECONDS:-7200}"

    # Park the contended logs once, then write the clean run to the canonical
    # names so every downstream consumer (ms/speedup extraction, viewers) reads
    # single-owner data. Guarded so a second re-grade cannot clobber the true
    # in-session original.
    for L in check benchmark; do
        if [ -f "$RUN_DIR/$L.log" ] && [ ! -L "$RUN_DIR/$L.log" ] \
            && [ ! -e "$RUN_DIR/$L.contended.log" ] \
            && [ ! -L "$RUN_DIR/$L.contended.log" ]; then
            mv -- "$RUN_DIR/$L.log" "$RUN_DIR/$L.contended.log"
        fi
    done
    CLOG="$RUN_DIR/check.log"
    BLOG="$RUN_DIR/benchmark.log"
    if ! submission_prepare_output_file "$CLOG" \
        || ! submission_prepare_output_file "$BLOG"; then
        echo "    cannot prepare safe regrade logs"
        FAIL=$((FAIL+1)); continue
    fi

    echo "    check.py..."
    C0=$(date +%s); CEXIT=125
    if submission_executable_matches "$REAL_UV" "$REAL_UV_IDENTITY" \
        && submission_executable_matches "$REAL_PYTHON" "$REAL_PYTHON_IDENTITY" \
        && submission_executable_matches "$REAL_TIMEOUT" "$REAL_TIMEOUT_IDENTITY" \
        && submission_build_isolated_command \
        "$RUN_DIR/regrade_replays/check" "$PROJECT_ROOT" "$PROBLEM_DIR" \
        "${ISOLATION_TEMPLATE_NAMES[@]}" -- \
        "$PROJECT_ROOT/.venv/bin/python" -P "$SUBMISSION_TRUSTED_STAGE_TOOL" \
        --preload-module torch --preload-module yaml check.py; then
        ISOLATION_APPLIED=true
        CEXIT=0
        REGRADE_STAGE_COUNT=1
        "$REAL_TIMEOUT" "$CHECK_TIMEOUT" \
            "${SUBMISSION_ISOLATED_COMMAND[@]}" > "$CLOG" 2>&1 || CEXIT=$?
    else
        echo "failed to construct isolated check replay" > "$CLOG"
    fi
    CEL=$(( $(date +%s) - C0 ))
    if [ "$BUNDLE_KIND" = "bundled" ] && ! submission_bundle_verify \
        "$RUN_DIR/submission_bundle" "$BUNDLE_DIGEST" >/dev/null 2>&1; then
        echo "bundle verification FAILED after check" >> "$CLOG"
        CEXIT=125
    fi
    if [ "$BUNDLE_KIND" = "bundled" ] && [ "$(submission_trusted_surface_digest \
        "$PROJECT_ROOT" "$PROBLEM" "${REPLAY_TEMPLATES[@]}" \
        2>/dev/null || true)" != "$CHECK_SURFACE_DIGEST" ]; then
        echo "trusted replay surface changed during check" >> "$CLOG"
        CEXIT=126
    fi

    CORRECT=false; SCORE=null; BEXIT=null; BEL=null
    # Same preliminary gate as run_hard.sh: exit 0 AND one PASS marker. The
    # in-process receipt remains forgeable, so bundle-era publication also
    # requires independent audit approval and the automatic static HACK veto.
    if [ "$CEXIT" -eq 0 ] && submission_check_passed "$CLOG"; then
        CORRECT=true
        echo "    benchmark.py..."
        BENCH_READY=true
        if [ "$BUNDLE_KIND" = "bundled" ]; then
            REPLAY_ROOT="$RUN_DIR/regrade_replays/benchmark"
            if ! submission_prepare_replay "$RUN_DIR/submission_bundle" \
                "$BUNDLE_DIGEST" "$REPO_ROOT" "$TEMPLATE_ROOT" "$REPLAY_ROOT" \
                "$PROBLEM" "${REPLAY_TEMPLATES[@]}"; then
                BENCH_READY=false
            fi
            PROBLEM_DIR="$REPLAY_ROOT/repo/problems/$PROBLEM"
            PROJECT_ROOT="$REPLAY_ROOT/repo"
        fi
        if ! submission_reset_caches \
            "$RUN_DIR/regrade_replays/benchmark/cache"; then
            BENCH_READY=false
        fi
        if ! submission_executable_matches "$REAL_UV" "$REAL_UV_IDENTITY" \
            || ! submission_executable_matches "$REAL_PYTHON" "$REAL_PYTHON_IDENTITY"; then
            BENCH_READY=false
        fi
        if [ "$BENCH_READY" = "true" ] && ! "$REAL_UV" sync --frozen \
            --python "$REAL_PYTHON" --project "$PROJECT_ROOT" \
            >> "$RUN_DIR/regrade_setup.log" 2>&1; then
            BENCH_READY=false
        fi
        if [ "$BENCH_READY" = "true" ] && [ "$BUNDLE_KIND" = "bundled" ]; then
            BENCH_SURFACE_DIGEST="$(submission_trusted_surface_digest \
                "$PROJECT_ROOT" "$PROBLEM" "${REPLAY_TEMPLATES[@]}")" \
                || BENCH_READY=false
            if [ "$BENCH_READY" = "true" ] \
                && [ "$BENCH_SURFACE_DIGEST" != "$CHECK_SURFACE_DIGEST" ]; then
                echo "check/benchmark trusted surfaces differ" \
                    >> "$RUN_DIR/regrade_setup.log"
                BENCH_READY=false
            fi
        fi
        if [ "$BENCH_READY" = "true" ] \
            && ! submission_select_clean_environment \
                "$RUN_DIR/regrade_replays/benchmark/home"; then
            BENCH_READY=false
        fi
        if [ "$BENCH_READY" = "true" ]; then
            B0=$(date +%s); BEXIT=125
            if submission_executable_matches "$REAL_UV" "$REAL_UV_IDENTITY" \
                && submission_executable_matches "$REAL_PYTHON" "$REAL_PYTHON_IDENTITY" \
                && submission_executable_matches "$REAL_TIMEOUT" "$REAL_TIMEOUT_IDENTITY" \
                && submission_build_isolated_command \
                "$RUN_DIR/regrade_replays/benchmark" "$PROJECT_ROOT" "$PROBLEM_DIR" \
                "${ISOLATION_TEMPLATE_NAMES[@]}" -- \
                "$PROJECT_ROOT/.venv/bin/python" -P "$SUBMISSION_TRUSTED_STAGE_TOOL" \
                --preload-module torch --preload-module yaml benchmark.py; then
                BEXIT=0
                REGRADE_STAGE_COUNT=2
                "$REAL_TIMEOUT" "$BENCH_TIMEOUT" \
                    "${SUBMISSION_ISOLATED_COMMAND[@]}" > "$BLOG" 2>&1 || BEXIT=$?
            else
                echo "failed to construct isolated benchmark replay" > "$BLOG"
            fi
            BEL=$(( $(date +%s) - B0 ))
            if [ "$BUNDLE_KIND" = "bundled" ] && ! submission_bundle_verify \
                "$RUN_DIR/submission_bundle" "$BUNDLE_DIGEST" >/dev/null 2>&1; then
                echo "bundle verification FAILED after benchmark" >> "$BLOG"
                BEXIT=125
            fi
            if [ "$BUNDLE_KIND" = "bundled" ] && [ "$(submission_trusted_surface_digest \
                "$PROJECT_ROOT" "$PROBLEM" "${REPLAY_TEMPLATES[@]}" \
                2>/dev/null || true)" != "$BENCH_SURFACE_DIGEST" ]; then
                echo "trusted replay surface changed during benchmark" >> "$BLOG"
                BEXIT=126
            fi
            if [ "$BEXIT" -eq 0 ]; then
                if ! SCORE="$(submission_extract_peak_fraction "$BLOG")"; then
                    SCORE=null
                    BEXIT=65
                    echo "ambiguous or invalid final metric" >> "$BLOG"
                fi
            fi
            if [ "$BEXIT" -ne 0 ]; then
                CORRECT=false
                SCORE=null
            fi
        else
            BEXIT=125
            CORRECT=false
            echo "fresh benchmark replay setup FAILED" > "$BLOG"
        fi
    else
        echo "    check FAILED (exit $CEXIT) -- see $CLOG"
    fi

    # Drop both the original and replay venvs before atomically committing the
    # updated result.json; the result file remains the last archive write.
    STRIP_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/lib/strip_run_venv.sh"
    if [ ! -f "$STRIP_HELPER" ]; then
        STRIP_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/strip_run_venv.sh"
    fi
    # shellcheck source=../../../scripts/lib/strip_run_venv.sh
    . "$STRIP_HELPER"
    strip_run_venv "$RUN_DIR"
    rm -rf -- "$RUN_DIR/regrade_replays/check" \
        "$RUN_DIR/regrade_replays/benchmark"

    REGRADE_STATUS="verified"
    if [ "$BUNDLE_KIND" = "bundled" ]; then
        case "$CEXIT:${BEXIT:-}" in
            *125*|*126*) REGRADE_STATUS="verification_failed" ;;
        esac
    else
        REGRADE_STATUS="legacy"
    fi
    GRADER_SURFACE_DIGEST="${BENCH_SURFACE_DIGEST:-$CHECK_SURFACE_DIGEST}"

    RID="$RID" CORRECT="$CORRECT" SCORE="$SCORE" CEXIT="$CEXIT" CEL="$CEL" \
    BEXIT="$BEXIT" BEL="$BEL" GPU="$GPU" BUNDLE_KIND="$BUNDLE_KIND" \
    BUNDLE_DIGEST="$BUNDLE_DIGEST" REGRADE_STATUS="$REGRADE_STATUS" \
    GRADER_SURFACE_DIGEST="$GRADER_SURFACE_DIGEST" \
    REGRADE_STAGE_COUNT="$REGRADE_STAGE_COUNT" \
    NETWORK_MODE="$SUBMISSION_NETWORK_MODE" \
    ISOLATION_APPLIED="$ISOLATION_APPLIED" \
    "$SUBMISSION_BUNDLE_PYTHON" - "$RUN_DIR/result.json" <<'PY' | \
        submission_atomic_write_json "$RUN_DIR/result.json"
import json, os, socket, subprocess, sys

path = sys.argv[1]
with open(path) as f:
    r = json.load(f)

def num(v):
    return None if v in (None, "", "null") else (float(v) if "." in str(v) else int(v))

active_before = {k: r.get(k) for k in (
    "correct", "peak_fraction", "check_exit_code", "benchmark_exit_code",
    "check_elapsed_seconds", "benchmark_elapsed_seconds")}
previous_regrade = r.get("regrade")
if (
    isinstance(previous_regrade, dict)
    and isinstance(previous_regrade.get("contended"), dict)
):
    # A repeated regrade must retain the first in-session/contended snapshot,
    # not silently replace it with the previous clean regrade.
    contended = previous_regrade["contended"]
else:
    contended = active_before

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
stage_count = int(os.environ["REGRADE_STAGE_COUNT"])
r["regrade"] = {
    "at": subprocess.check_output(["date", "-Is"], text=True).strip(),
    "host": socket.gethostname(),
    "gpu_index": int(os.environ["GPU"]),
    "gpu_name": gpu_name,
    "mode": "sequential_isolated",
    "status": os.environ["REGRADE_STATUS"],
    "submission_mode": os.environ["BUNDLE_KIND"],
    "submission_bundle_sha256": os.environ["BUNDLE_DIGEST"] or None,
    "grader_surface_sha256": os.environ["GRADER_SURFACE_DIGEST"] or None,
    "stage_count": stage_count,
    "check_exit_code": num(os.environ["CEXIT"]) if stage_count >= 1 else None,
    "benchmark_exit_code": num(os.environ["BEXIT"]) if stage_count >= 2 else None,
    "fresh_extraction": os.environ["BUNDLE_KIND"] == "bundled",
    "fresh_caches": True,
    "network_isolation": os.environ["NETWORK_MODE"],
    "network_isolated": os.environ["ISOLATION_APPLIED"] == "true",
    "mount_isolated": os.environ["ISOLATION_APPLIED"] == "true",
    "root_isolated": os.environ["ISOLATION_APPLIED"] == "true",
    "pid_isolated": os.environ["ISOLATION_APPLIED"] == "true",
    "clean_environment": os.environ["ISOLATION_APPLIED"] == "true",
    "in_process_completion_guard": os.environ["ISOLATION_APPLIED"] == "true",
    "contended": contended,
}

old, new = active_before["peak_fraction"], r["peak_fraction"]
delta = ""
if isinstance(old, (int, float)) and isinstance(new, (int, float)) and old:
    delta = "  (%+.1f%%)" % ((new - old) / old * 100)
print("    correct=%s  peak %s -> %s%s" % (r["correct"], old, new, delta), file=sys.stderr)
json.dump(r, sys.stdout)
PY
    if [ "$?" -ne 0 ]; then
        echo "FATAL: could not atomically update $RUN_DIR/result.json" >&2
        exit 4
    fi

    if [ "$CORRECT" = "true" ] && [ "$BEXIT" = "0" ] && [ "$SCORE" != "null" ]; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
    fi
done

echo "========================================"
echo "re-graded: $PASS correct, $FAIL failed, $SKIP skipped"
echo "========================================"
[ "$FAIL" -eq 0 ] && [ "$SKIP" -eq 0 ]
