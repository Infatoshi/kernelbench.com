"""Report manifest rows that exited without a matching result.json.

Usage:
    uv run python scripts/check_sweep_health.py --run-group NAME

This is intentionally read-only. It cross-checks the sweep manifest against
archived result.json files and live PIDs, which catches infrastructure rows
that died before run_hard.sh could write result metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_results(run_group: str) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for path in sorted((ROOT / "outputs" / "runs").glob("*/result.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("run_group") != run_group:
            continue
        key = (
            str(data.get("harness") or ""),
            str(data.get("model") or ""),
            str(data.get("problem") or ""),
        )
        by_key.setdefault(key, []).append({"run_id": path.parent.name, **data})
    return by_key


def pid_running(pid: str) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-group", required=True)
    args = parser.parse_args()

    manifest = ROOT / "outputs" / "sweeps" / args.run_group / "manifest.tsv"
    results = load_results(args.run_group)
    exited_no_result = []
    running = 0
    rows = []
    with manifest.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(row)
            is_running = pid_running(row.get("pid", ""))
            running += int(is_running)
            key = (
                row.get("harness", ""),
                row.get("model", ""),
                Path(row.get("problem", "")).name,
            )
            if not is_running and key not in results:
                exited_no_result.append(row)

    print(f"manifest_rows={len(rows)}")
    print(f"result_rows={sum(len(v) for v in results.values())}")
    print(f"running_rows={running}")
    print(f"exited_no_result={len(exited_no_result)}")
    for row in exited_no_result:
        print(
            "\t".join(
                [
                    row.get("name", ""),
                    row.get("problem", ""),
                    f"pid={row.get('pid', '')}",
                    f"log={row.get('log', '')}",
                ]
            )
        )
    return 1 if exited_no_result else 0


if __name__ == "__main__":
    raise SystemExit(main())
