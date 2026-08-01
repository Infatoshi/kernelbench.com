#!/usr/bin/env bash
# KernelBench-mini harness entry — thin wrapper over the shared runner.
# ALL harness logic lives in scripts/lib/run_harness.sh at the monorepo root;
# this file only pins the bench identity. Edit the lib, not this file.
set -euo pipefail
KB_BENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export KB_BENCH_DIR
export KB_BENCH_BANNER="KERNELBENCH-MINI RUN"
export KB_BUDGET_SECONDS_DEFAULT="1800"
# Lib resolution: monorepo layout first; on a thin-synced remote worker the
# lib is shipped INTO the bench dir at scripts/lib/ by `kb lambda sync`.
for LIB in "$KB_BENCH_DIR/../../scripts/lib/run_harness.sh" \
           "$KB_BENCH_DIR/scripts/lib/run_harness.sh"; do
    [ -f "$LIB" ] && exec bash "$LIB" "$@"
done
echo "STOP: shared runner scripts/lib/run_harness.sh not found (monorepo or bench-local)" >&2
exit 3
