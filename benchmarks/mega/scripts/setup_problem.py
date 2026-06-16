"""Install per-problem SOTA dependencies.

Reads problem.yaml, looks at the `sota.deps` field, and installs via uv pip.
Run once per problem before sweeping. Safe to re-run.

Usage:
    uv run python scripts/setup_problem.py problems/01_fp8_gemm
    uv run python scripts/setup_problem.py --all
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def install_deps(problem_dir: Path) -> bool:
    yaml_path = problem_dir / "problem.yaml"
    if not yaml_path.exists():
        print(f"  [skip] no problem.yaml in {problem_dir}")
        return False

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    deps = meta.get("sota", {}).get("deps", [])
    if not deps:
        print(f"  [skip] no sota.deps declared in {problem_dir.name}")
        return True

    print(f"  installing sota deps for {problem_dir.name}: {deps}")
    cmd = ["uv", "pip", "install", *deps]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [err] install failed:\n{result.stderr}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_dir", nargs="?", type=Path)
    parser.add_argument("--all", action="store_true", help="install deps for every problem")
    args = parser.parse_args()

    if args.all:
        root = Path(__file__).resolve().parent.parent
        for p in sorted((root / "problems").iterdir()):
            if p.is_dir():
                print(f"--- {p.name} ---")
                install_deps(p)
    elif args.problem_dir:
        install_deps(args.problem_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
