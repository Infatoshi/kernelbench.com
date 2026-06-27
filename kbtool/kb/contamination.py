"""Cross-run contamination audit for KernelBench runs (hard or mega).

The harness does NOT sandbox the agent filesystem: an agent has bash + absolute
paths, so it can read the shared `outputs/runs/` archive -- every prior winning
solution -- and reverse-engineer a known answer instead of writing its own
kernel. A run is CONTAMINATED if its agent transcript references another run's
archive (`outputs/runs/<other_ts>`).

This is the audit `kb lint` does NOT do (lint only scans a single solution.py
for in-solution reward-hacks). Run it before publishing; the leaderboard
builders also exclude contaminated runs automatically.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_TS = re.compile(r"outputs/runs/(\d{8}_\d{6})")


def other_archives(run_dir: Path) -> set[str]:
    """Distinct OTHER run timestamps referenced by this run's AGENT transcript.

    Only the agent transcript (transcript.jsonl / codex_session.jsonl) is
    scanned -- NOT stderr.log/scratch, which carry harness orchestration noise.
    """
    m = re.match(r"(\d{8}_\d{6})", run_dir.name)
    self_ts = m.group(1) if m else ""
    seen: set[str] = set()
    for fn in ("transcript.jsonl", "codex_session.jsonl"):
        p = run_dir / fn
        if p.exists():
            for ts in _TS.findall(p.read_text(errors="ignore")):
                if ts != self_ts:
                    seen.add(ts)
    return seen


def run(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kb contamination")
    ap.add_argument(
        "runs",
        help="path to an outputs/runs directory, or a bench name (hard|mega|v3)",
    )
    ap.add_argument("--published", help="leaderboard.json to flag contaminated PUBLISHED cells")
    args = ap.parse_args(argv)

    runs = Path(args.runs)
    # Convenience: accept a bench name and resolve against the repo.
    if repo_root is not None and not runs.exists() and args.runs in ("hard", "mega", "v3"):
        runs = repo_root / "benchmarks" / args.runs / "outputs" / "runs"
    if not runs.is_dir():
        print(f"no such runs dir: {runs}")
        return 1

    dirty: dict[str, int] = {}
    total = 0
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        total += 1
        n = other_archives(d)
        if n:
            dirty[d.name] = len(n)

    print(f"=== contamination audit: {len(dirty)} / {total} runs read other archives ===")
    for name, cnt in sorted(dirty.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>3} other archives  {name}")

    if args.published:
        lb = json.loads(Path(args.published).read_text())
        pub_dirty = 0
        pub_total = 0
        for m in lb.get("models", []):
            for prob, cell in m.get("results", {}).items():
                rid = cell.get("run_id")
                if not rid:
                    continue
                pub_total += 1
                if rid in dirty:
                    pub_dirty += 1
                    print(f"  PUBLISHED-CONTAMINATED  {m.get('label')} {prob}  ({dirty[rid]} archives)")
        print(f"=== PUBLISHED cells contaminated: {pub_dirty} / {pub_total} ===")
    return 0
