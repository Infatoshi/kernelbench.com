"""Aggregate archived runs into a long, scroll-friendly metrics table.

Reads outputs/runs/*/result.json and prints one row per metric. Keeping
``problem`` as a regular column avoids the old problem-pivot layout that grew
wide enough to require horizontal scrolling.
"""
from __future__ import annotations

import argparse
import json
from math import exp, log
from pathlib import Path
from typing import Any

IDENTITY_FIELDS = ("problem", "harness", "model", "reasoning_effort", "run_id")
TABLE_FIELDS = (*IDENTITY_FIELDS, "scope", "metric", "value")

AGGREGATE_METRICS = (
    "runs",
    "correct",
    "gmean_peak_fraction",
)


def geomean(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return exp(sum(log(v) for v in positive) / len(positive))


def load_results(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(runs_dir.glob("*/result.json")):
        try:
            result = json.loads(result_path.read_text())
        except Exception:
            continue
        if not isinstance(result, dict):
            continue
        result.setdefault("run_id", result_path.parent.name)
        rows.append(result)
    return rows


def long_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    table: list[dict[str, str]] = []

    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            _cell(result.get("problem")),
            _cell(result.get("harness")),
            _cell(result.get("model")),
            _cell(result.get("reasoning_effort")),
        )
        by_key.setdefault(key, []).append(result)

    for (problem, harness, model, effort), runs in sorted(by_key.items()):
        n = len(runs)
        correct = sum(1 for r in runs if r.get("correct"))
        scores = [r.get("peak_fraction") for r in runs if r.get("correct") and r.get("peak_fraction")]
        gm = geomean(scores) if scores else 0.0
        base = {
            "problem": problem,
            "harness": harness,
            "model": model,
            "reasoning_effort": effort,
            "run_id": "",
            "scope": "aggregate",
        }
        for metric, value in zip(AGGREGATE_METRICS, (n, correct, gm), strict=True):
            table.append({**base, "metric": metric, "value": _cell(value)})

    for result in sorted(results, key=_run_sort_key):
        base = {
            "problem": _cell(result.get("problem")),
            "harness": _cell(result.get("harness")),
            "model": _cell(result.get("model")),
            "reasoning_effort": _cell(result.get("reasoning_effort")),
            "run_id": _cell(result.get("run_id")),
            "scope": "run",
        }
        for metric, value in _iter_metrics(result):
            table.append({**base, "metric": metric, "value": _cell(value)})

    return table


def render_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(TABLE_FIELDS) + " |",
        "| " + " | ".join("---" for _ in TABLE_FIELDS) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown(row.get(field, "")) for field in TABLE_FIELDS) + " |")
    return "\n".join(lines)


def _iter_metrics(result: dict[str, Any]) -> list[tuple[str, Any]]:
    metrics: list[tuple[str, Any]] = []
    for key in sorted(result):
        if key in IDENTITY_FIELDS:
            continue
        value = result[key]
        if isinstance(value, dict):
            for subkey in sorted(value):
                metrics.append((f"{key}.{subkey}", value[subkey]))
        else:
            metrics.append((key, value))
    return metrics


def _run_sort_key(result: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _cell(result.get("problem")),
        _cell(result.get("harness")),
        _cell(result.get("model")),
        _cell(result.get("reasoning_effort")),
        _cell(result.get("run_id")),
    )


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool | int | str):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path, default=Path("outputs/runs"), nargs="?")
    args = parser.parse_args()

    print(render_markdown(long_rows(load_results(args.runs_dir))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
