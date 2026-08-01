#!/usr/bin/env bash
# Strip per-run Python virtualenvs from KernelBench run archives.
#
# Why: each scored run leaves repo/.venv (torch + CUDA wheels, often ~0.5-5G
# per run; libcublasLt alone ~517M). Keeping them is pure bloat — uv.lock +
# `uv run` recreate the env for regrade. lambda_worker/brev_worker pull already
# --exclude '.venv'; this covers local archives and post-regrade residue.
#
# Usage (sourceable or executable):
#   source scripts/lib/strip_run_venv.sh && strip_run_venv "$RUN_DIR"
#   scripts/lib/strip_run_venv.sh outputs/runs/<run_id>
#   scripts/lib/strip_run_venv.sh --tree benchmarks/hard/outputs
#   scripts/lib/strip_run_venv.sh --tree rescue
#
# Opt out for a single run: KBH_KEEP_RUN_VENV=1

strip_run_venv() {
    local root="${1:?strip_run_venv: path required}"
    if [ "${KBH_KEEP_RUN_VENV:-0}" = "1" ]; then
        return 0
    fi
    if [ ! -d "$root" ]; then
        return 0
    fi
    # -prune so find does not walk into dirs we are about to delete.
    local found=0
    while IFS= read -r -d '' d; do
        found=1
        rm -rf "$d"
    done < <(find "$root" -type d -name .venv -prune -print0 2>/dev/null)
    if [ "$found" -eq 1 ]; then
        echo "stripped .venv under $root" >&2
    fi
    return 0
}

strip_run_venv_tree() {
    local tree="${1:?strip_run_venv_tree: path required}"
    if [ ! -d "$tree" ]; then
        echo "strip_run_venv_tree: not a directory: $tree" >&2
        return 1
    fi
    strip_run_venv "$tree"
}

# When executed (not sourced), treat args as paths; --tree is a no-op flag
# (any path is walked recursively for .venv dirs either way).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    if [ "$#" -eq 0 ]; then
        echo "usage: $0 <run_dir_or_tree> [...]" >&2
        echo "  (alias: $0 --tree <path> — same as bare path)" >&2
        exit 2
    fi
    for p in "$@"; do
        if [ "$p" = "--tree" ]; then
            continue
        fi
        strip_run_venv "$p"
    done
fi
