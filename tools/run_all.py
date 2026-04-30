#!/usr/bin/env python3
"""
Master benchmark runner for KernelBench-v3.

Automatically selects platform-compatible GPUs for each benchmark.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from typing import Any


BENCHMARK_CONFIG: dict[str, dict[str, Any]] = {
    "cuda": {
        "script": "batch_eval.py",
        "gpus": ["RTX3090", "H100", "B200"],
        "levels": [1, 2, 3, 4],
    },
    "triton": {
        "script": "triton_batch_eval.py",
        "gpus": ["RTX3090", "H100", "B200"],
        "levels": [1, 2, 3, 4],
    },
    "cutlass": {
        "script": "cutlass_batch_eval.py",
        "gpus": ["H100", "B200"],
        "levels": [1, 2, 3, 4],
    },
    "cute": {
        "script": "cute_batch_eval.py",
        "gpus": ["H100", "B200"],
        "levels": [1, 2, 3, 4],
    },
    "cutile": {
        "script": "cutile_batch_eval.py",
        "gpus": ["B200"],
        "levels": [1, 2, 3, 4],
    },
    "metal": {
        "script": "metal_batch_eval.py",
        "gpus": ["M4MAX"],
        "levels": [1, 2, 3, 4],
    },
    "graphics": {
        "script": "graphics_batch_eval.py",
        "gpus": ["RTX3090"],
        "levels": [1],
    },
}


def _gpu_matches_host_platform(gpu: str, host_system: str) -> bool:
    if gpu == "M4MAX":
        return host_system == "Darwin"
    return host_system != "Darwin"


def run_benchmark(
    name: str,
    model: str,
    requested_gpus: list[str] | None = None,
    max_turns: int = 10,
    dry_run: bool = False,
) -> int:
    config = BENCHMARK_CONFIG[name]
    host_system = platform.system()
    candidate_gpus = requested_gpus or config["gpus"]
    allowed_gpus = config["gpus"]

    valid_gpus = [gpu for gpu in candidate_gpus if gpu in allowed_gpus]
    valid_gpus = [gpu for gpu in valid_gpus if _gpu_matches_host_platform(gpu, host_system)]

    if not valid_gpus:
        print(
            f"Skipping {name}: no platform-compatible GPUs "
            f"(requested={candidate_gpus}, allowed={allowed_gpus}, host={host_system})"
        )
        return 0

    cmd = [
        "uv",
        "run",
        "python",
        config["script"],
        "--models",
        model,
        "--gpus",
        ",".join(valid_gpus),
        "--levels",
        ",".join(str(level) for level in config["levels"]),
        "--max-turns",
        str(max_turns),
        "--sequential",
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'=' * 60}")
    print(f"Running {name} on {valid_gpus}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all KernelBench-v3 benchmark families")
    parser.add_argument("model", nargs="?", default="minimax/minimax-m2.5", help="Model key or OpenRouter model ID")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum turns per problem")
    parser.add_argument("--dry-run", action="store_true", help="Run dry-run validation only")
    args = parser.parse_args()

    failures: list[tuple[str, int]] = []
    for benchmark_name in BENCHMARK_CONFIG:
        rc = run_benchmark(
            benchmark_name,
            model=args.model,
            max_turns=args.max_turns,
            dry_run=args.dry_run,
        )
        if rc != 0:
            failures.append((benchmark_name, rc))

    if not args.dry_run:
        aggregate_cmd = ["uv", "run", "python", "aggregate_results.py", "--output", "full_eval.csv"]
        print(f"\nRunning aggregate step: {' '.join(aggregate_cmd)}")
        aggregate_rc = subprocess.run(aggregate_cmd, check=False).returncode
        if aggregate_rc != 0:
            failures.append(("aggregate", aggregate_rc))

    if failures:
        print("\nCompleted with failures:")
        for name, rc in failures:
            print(f"- {name}: exit code {rc}")
        sys.exit(1)

    print("\nAll requested benchmark runs completed successfully.")


if __name__ == "__main__":
    main()
