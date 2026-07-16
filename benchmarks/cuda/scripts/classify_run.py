"""Classify a KernelBench-Hard run from run_hard.sh environment variables."""
from __future__ import annotations

import json
import os

from src.harness.classification import classify_run


def _as_bool(value: str) -> bool:
    return value == "true"


def _as_int(value: str | None) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def main() -> None:
    minimum = int(os.environ.get("MIN_USEFUL_OUTPUT_TOKENS") or "500")
    print(
        json.dumps(
            classify_run(
                usage=json.loads(os.environ.get("USAGE_JSON") or "{}"),
                correct=_as_bool(os.environ["CORRECT"]),
                template_mutated=_as_bool(os.environ["TEMPLATE_MUTATED"]),
                has_solution=_as_bool(os.environ["HAS_SOLUTION"]),
                session_complete=_as_bool(os.environ["SESSION_COMPLETE"]),
                harness_exit=_as_int(os.environ.get("HARNESS_EXIT")),
                check_exit=_as_int(os.environ.get("CHECK_EXIT_CODE")),
                bench_exit=_as_int(os.environ.get("BENCH_EXIT_CODE")),
                log_file=os.environ.get("LOG_FILE"),
                stderr_file=os.environ.get("STDERR_FILE"),
                minimum_useful_output_tokens=minimum,
            ),
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
