"""Summarize KernelBench-Mega archived run results.

Usage:
    uv run python scripts/summarize_runs.py [--run-group NAME] [--output-dir DIR]

Reads outputs/runs/*/result.json and writes summary.json plus summary.csv.
The flattened rows are intentionally close to the website leaderboard cell
shape, with token and queue metadata kept beside scoring fields.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "summaries"

FIELDS = [
    "run_id",
    "run_group",
    "problem",
    "harness",
    "model",
    "reasoning_effort",
    "correct",
    "has_solution",
    "failure_reason",
    "retryable_infra_failure",
    "minimum_useful_output_tokens",
    "peak_fraction",
    "elapsed_seconds",
    "total_elapsed_seconds",
    "check_elapsed_seconds",
    "benchmark_elapsed_seconds",
    "check_exit_code",
    "benchmark_exit_code",
    "session_complete",
    "harness_exit_code",
    "template_mutated",
    "agent_cuda_disabled",
    "gpu_queue_mode",
    "gpu_lock_calls",
    "gpu_lock_wait_seconds_total",
    "gpu_lock_active_seconds_total",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
    "total_cost_usd",
    "output_tokens_per_second",
    "started_at",
    "finished_at",
    "path",
]


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("run_id", path.parent.name)
    data["path"] = str(path.parent.relative_to(ROOT))
    return data


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    row = {field: data.get(field) for field in FIELDS}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
        "total_cost_usd",
    ):
        row[key] = usage.get(key)
    row["run_id"] = data.get("run_id")
    row["path"] = data.get("path")
    lock = _parse_gpu_lock(ROOT / str(row["path"]) / "gpu_lock.log")
    row.update(lock)
    return row


def _parse_gpu_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "gpu_lock_calls": 0,
            "gpu_lock_wait_seconds_total": 0,
            "gpu_lock_active_seconds_total": 0,
        }
    waits: dict[str, int] = {}
    calls = 0
    wait_total = 0
    active_total = 0
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            ts = _parse_log_time(parts[0])
        except ValueError:
            continue
        pid = _kv(parts, "pid")
        if not pid:
            continue
        if " wait " in f" {line} ":
            waits[pid] = ts
        elif " start " in f" {line} ":
            calls += 1
            if pid in waits:
                wait_total += max(0, ts - waits.pop(pid))
        elif " end " in f" {line} ":
            elapsed = _kv(parts, "elapsed_s")
            if elapsed is not None:
                try:
                    active_total += int(elapsed)
                except ValueError:
                    pass
    return {
        "gpu_lock_calls": calls,
        "gpu_lock_wait_seconds_total": wait_total,
        "gpu_lock_active_seconds_total": active_total,
    }


def _kv(parts: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _parse_log_time(value: str) -> int:
    from datetime import datetime

    return int(datetime.fromisoformat(value).timestamp())


def collect(run_group: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted((ROOT / "outputs" / "runs").glob("*/result.json")):
        data = _load_result(result_path)
        if data is None:
            continue
        if run_group and data.get("run_group") != run_group:
            continue
        rows.append(_flatten(data))
    rows.sort(
        key=lambda r: (
            str(r.get("run_group") or ""),
            str(r.get("problem") or ""),
            str(r.get("harness") or ""),
            str(r.get("model") or ""),
            str(r.get("run_id") or ""),
        )
    )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({"runs": rows}, indent=2) + "\n")
    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-group")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        name = args.run_group or "all"
        output_dir = DEFAULT_OUTPUT_ROOT / name

    rows = collect(args.run_group)
    write_outputs(rows, output_dir)
    print(f"wrote {len(rows)} rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
