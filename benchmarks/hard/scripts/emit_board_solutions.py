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

from pathlib import Path

HARD = Path(__file__).resolve().parent.parent
REPO = HARD.parent.parent
PUB = REPO / "public" / "runs"

# (bench, gpu key, leaderboard file, runs dirs to try in order).
BOARDS = [
    ("hard", "h100", "results/leaderboard.h100.json", ["outputs/runs-h100"]),
    ("hard", "b200", "results/leaderboard.b200.json", ["outputs/runs-b200"]),
    ("cuda", "h100", "results/leaderboard.h100.json", ["outputs/runs-h100"]),
    ("cuda", "b200", "results/leaderboard.b200.json", ["outputs/runs-b200"]),
]

import sys  # noqa: E402

sys.path.insert(0, str(REPO))
from scripts.published_submission import (  # noqa: E402
    atomic_write_text,
    prepare_selected_solution_outputs,
    trusted_archive_lock,
)


def _emit() -> None:
    staged: list[tuple[Path, str]] = []
    summaries: list[tuple[str, str, int]] = []
    destinations: set[Path] = set()
    for bench, gpu, lb_rel, runs_rels in BOARDS:
        bench_dir = REPO / "benchmarks" / bench
        lb = bench_dir / lb_rel
        if not lb.exists() and not lb.is_symlink():
            continue
        if len(runs_rels) != 1:
            raise RuntimeError("board publication requires exactly one runs root")
        outputs = prepare_selected_solution_outputs(
            lb,
            bench_dir / runs_rels[0],
            PUB / gpu,
        )
        for destination, contents in outputs:
            if destination in destinations:
                raise RuntimeError(f"duplicate board publication target: {destination}")
            destinations.add(destination)
            staged.append((destination, contents))
        summaries.append((bench, gpu, len(outputs)))

    # Every board and selected archive has been validated before the first
    # public file changes, avoiding a silently partial per-GPU publication.
    for destination, contents in staged:
        atomic_write_text(destination, contents)
    for bench, gpu, count in summaries:
        print(
            f"  [{bench}/{gpu}] wrote {count} board solutions "
            f"-> public/runs/{gpu}/"
        )


def main() -> None:
    with trusted_archive_lock():
        _emit()


if __name__ == "__main__":
    main()
