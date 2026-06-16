"""Aggregate results across problems and runs into a compact report.

Reads outputs/runs/*/result.json, groups by (harness, model) and by problem,
prints a geomean table.
"""
from __future__ import annotations

import argparse
import json
from math import exp, log
from pathlib import Path


def geomean(values: list[float]) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return exp(sum(log(v) for v in positive) / len(positive))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path, default=Path("outputs/runs"), nargs="?")
    args = parser.parse_args()

    rows: list[dict] = []
    for result_path in sorted(args.runs_dir.glob("*/result.json")):
        try:
            rows.append(json.loads(result_path.read_text()))
        except Exception:
            continue

    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        k = (r.get("harness", ""), r.get("model", ""))
        by_key.setdefault(k, []).append(r)

    print(f"{'harness':<14} {'model':<24} {'n':>4} {'correct':>8} {'gmean_peak':>12}")
    print("-" * 68)
    for (harness, model), runs in sorted(by_key.items()):
        n = len(runs)
        correct = sum(1 for r in runs if r.get("correct"))
        scores = [r.get("peak_fraction") for r in runs if r.get("correct") and r.get("peak_fraction")]
        gm = geomean(scores) if scores else 0.0
        print(f"{harness:<14} {model:<24} {n:>4} {correct:>3}/{n:<3} {gm:>12.3f}")


if __name__ == "__main__":
    main()
