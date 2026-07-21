#!/usr/bin/env python3
"""Emit redacted solution text for the non-canonical GPU boards.

The canonical (RTX PRO 6000) board's solutions are emitted flat into
public/runs/<rid>_solution.py.txt by publish_v2.sh; per-GPU boards namespace
under public/runs/<gpu>/ so a colliding rid never claims the canonical file
(see scripts/build_run_detail.py for the same rule on rundetail JSON).

Run from benchmarks/hard (publish_v2.sh calls it):
    uv run python scripts/emit_board_solutions.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

HARD = Path(__file__).resolve().parent.parent
REPO = HARD.parent.parent
PUB = REPO / "public" / "runs"

# (bench, gpu key, leaderboard file, runs dirs to try in order).
BOARDS = [
    ("hard", "h100", "results/leaderboard.h100.json", ["outputs/runs-h100"]),
    ("hard", "b200", "results/leaderboard.b200.json", ["outputs/runs-b200"]),
    ("cuda", "b200", "results/leaderboard.b200.json", ["outputs/runs-b200"]),
]


def _redactor():
    """Same rules as publish_v2.sh: every long ~/.env_vars value + token
    prefix patterns."""
    vals: list[str] = []
    envf = os.path.expanduser("~/.env_vars")
    if os.path.exists(envf):
        for ln in open(envf):
            if "=" in ln and "export" in ln:
                v = ln.split("=", 1)[1].strip().strip('"').strip("'")
                if len(v) >= 12:
                    vals.append(v)
    vals = sorted(set(vals), key=len, reverse=True)
    pats = [re.compile(p) for p in (
        r"sk-ant-oat01-[A-Za-z0-9_\-]+", r"sk-proj-[A-Za-z0-9_\-]+",
        r"AIzaSy[A-Za-z0-9_\-]{20,}", r"sk-[a-z]{2,}-[A-Za-z0-9_\-]{16,}",
        r"hf_[A-Za-z0-9]{20,}",
    )]

    def red(s: str) -> str:
        for v in vals:
            s = s.replace(v, "REDACTED")
        for p in pats:
            s = p.sub("REDACTED", s)
        return s

    return red


def main() -> None:
    red = _redactor()
    for bench, gpu, lb_rel, runs_rels in BOARDS:
        bench_dir = REPO / "benchmarks" / bench
        lb = bench_dir / lb_rel
        if not lb.exists():
            continue
        data = json.loads(lb.read_text())
        rids = sorted({c["run_id"] for m in data.get("models", [])
                       for c in m.get("results", {}).values() if c.get("run_id")})
        out_dir = PUB / gpu
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for rid in rids:
            for runs_rel in runs_rels:
                sp = bench_dir / runs_rel / rid / "solution.py"
                if sp.is_file():
                    (out_dir / f"{rid}_solution.py.txt").write_text(red(sp.read_text()))
                    n += 1
                    break
        print(f"  [{bench}/{gpu}] wrote {n}/{len(rids)} board solutions -> public/runs/{gpu}/")


if __name__ == "__main__":
    main()
