"""KernelBench-CUDA CLI entrypoint.

The first migration step keeps the battle-tested shell runner as the execution
backend while moving the public command surface to `uv run kbh ...`. That gives
us one stable entrypoint to preserve while the runner internals move into Python.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HARNESS_ALIASES = {
    "claude-code": "claude",
    "claude_code": "claude",
    # hy3-claude was the old OpenRouter+Claude-Code preview route; TokenHub
    # Hy3 is OpenCode. Alias keeps old muscle memory working.
    "hy3-claude": "hy3",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbh", description="KernelBench-CUDA runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="run one harness/model/problem combination",
        description="Run one KernelBench-CUDA harness/model/problem combination.",
    )
    run.add_argument("harness", help="harness name, for example claude, codex, opencode")
    run.add_argument("model", help="model identifier passed to the harness")
    run.add_argument(
        "problem_dir",
        type=Path,
        help="problem directory, for example problems-rtxpro6000/01_fp8_gemm",
    )
    run.add_argument("reasoning_effort", nargs="?", help="optional reasoning/effort argument")
    run.add_argument(
        "--runner",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the backend command without executing it",
    )
    run.set_defaults(func=run_command)

    return parser


def run_command(args: argparse.Namespace) -> int:
    root = repo_root()
    runner = args.runner or root / "scripts" / "run_hard.sh"
    runner = runner if runner.is_absolute() else root / runner
    if not runner.exists():
        print(f"runner not found: {runner}", file=sys.stderr)
        return 1

    problem_dir = args.problem_dir
    if not problem_dir.is_absolute():
        problem_dir = root / problem_dir

    command = [
        str(runner),
        normalize_harness(args.harness),
        args.model,
        str(problem_dir),
    ]
    if args.reasoning_effort:
        command.append(args.reasoning_effort)

    if args.dry_run:
        print(" ".join(_quote(part) for part in command))
        return 0

    completed = subprocess.run(command, cwd=root, check=False)
    return completed.returncode


def _quote(value: str) -> str:
    return shlex.quote(value)


def normalize_harness(harness: str) -> str:
    return HARNESS_ALIASES.get(harness, harness)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
