"""Run a grading script and reject candidate-triggered successful exits.

Checkers and benchmarks execute ``solution.py`` in-process.  A solution that
raises ``SystemExit(0)`` would otherwise make Python report success before the
trusted script reaches its final PASS or score emission.  Normal fallthrough
is the only successful completion; non-zero ``SystemExit`` values retain their
usual failure semantics.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_grading_script(script: str | Path) -> int:
    """Run *script* as ``__main__`` and return zero only on fallthrough."""
    script_path = Path(script).resolve()
    original_path = sys.path.copy()
    sys.path.insert(0, str(script_path.parent))
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
        print(
            f"FAIL: {script_path.name} exited successfully before normal completion",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        sys.path[:] = original_path
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: trusted_entrypoint.py <check.py|benchmark.py>", file=sys.stderr)
        return 2
    return run_grading_script(args[0])


if __name__ == "__main__":
    raise SystemExit(main())
