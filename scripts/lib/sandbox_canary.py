#!/usr/bin/env python3
"""Launch-time sandbox canary — refuse-closed foreign-archive detector.

scripts/lib/sandbox.sh runs this INSIDE the agent bwrap sandbox, before the
model starts. It sweeps the visible tree for anything a live agent must never
see: another run's directory, a stray result.json or solution.py, a pulled
runs-remote-*/runs-lambda-* archive, or the published public/data/*/code tree.
Any hit fails the launch (exit 1). The inverse of the old denylist: instead of
enumerating paths to hide, prove nothing archive-shaped is visible.

Allowed by name: this run's own directory and the deliberately planted
honeytoken decoy (see sandbox.sh; beacon documented in kbtool/kb/contamination.py).

Stdlib only, no bwrap needed to unit-test: kbtool/tests/test_sandbox_canary.py.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_")
REMOTE_ARCHIVE_RE = re.compile(r"^runs-(remote|lambda)-")
# Never descend into these; they are huge and cannot legally hold archives
# (the strict rule that every KernelBench artifact stays in outputs/runs).
PRUNE = {".git", ".venv", "node_modules", "__pycache__", ".cache", ".npm", ".bun"}


def _is_published_code_dir(dirpath: str, name: str) -> bool:
    parts = dirpath.split(os.sep)
    return name == "code" and "public" in parts and "data" in parts


def scan(
    walk_roots: list[str],
    must_be_hidden: list[str],
    run_id: str,
    allow: list[str],
) -> list[str]:
    """Return a list of human-readable violations; empty means clean."""
    allowed_names = set(allow) | {run_id}
    hits: list[str] = []
    for root in walk_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            keep: list[str] = []
            for name in sorted(dirnames):
                full = os.path.join(dirpath, name)
                if name in allowed_names:
                    continue  # own run dir / honeytoken: skip subtree
                if name in PRUNE:
                    continue
                if REMOTE_ARCHIVE_RE.match(name):
                    hits.append(f"visible pulled-archive dir: {full}")
                    continue
                if RUN_DIR_RE.match(name):
                    hits.append(f"foreign run dir: {full}")
                    continue
                if _is_published_code_dir(dirpath, name):
                    hits.append(f"visible published-code dir: {full}")
                    continue
                keep.append(name)
            dirnames[:] = keep
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                if name == "result.json":
                    hits.append(f"foreign result.json: {full}")
                elif "solution.py" in name:
                    hits.append(f"foreign solution file: {full}")
    for p in must_be_hidden:
        if os.path.isdir(p):
            try:
                entries = os.listdir(p)
            except OSError:
                entries = []
            if entries:
                hits.append(f"must-be-hidden dir visible and non-empty: {p}")
        elif os.path.exists(p):
            hits.append(f"must-be-hidden path visible: {p}")
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True, help="this run's directory basename")
    ap.add_argument("--allow", action="append", default=[],
                    help="additional allowed dir basename (honeytoken decoy)")
    ap.add_argument("--walk", action="append", default=[],
                    help="root to sweep recursively")
    ap.add_argument("--must-be-hidden", action="append", default=[],
                    help="path that must be absent or an empty tmpfs stub")
    args = ap.parse_args(argv)

    hits = scan(args.walk, args.must_be_hidden, args.run_id, args.allow)
    if hits:
        print(f"sandbox canary: {len(hits)} violation(s) — REFUSING LAUNCH")
        for h in hits:
            print(f"  {h}")
        return 1
    print("sandbox canary: CLEAN (no foreign archive material visible)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
