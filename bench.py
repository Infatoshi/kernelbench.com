#!/usr/bin/env python3
"""KernelBench v3 — Hardware-centric GPU kernel optimization benchmark."""

import argparse
import sys
from pathlib import Path


def cmd_run(args):
    from src.batch import run_batch
    from src.hardware import get_target
    from src.models import MODELS, get_model_config

    target = get_target(args.hardware)

    if args.models == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",")]

    for key in model_keys:
        cfg = get_model_config(key)
        if cfg is None:
            print(f"Error: unknown model '{key}'")
            sys.exit(1)

    levels = [int(x) for x in args.levels.split(",")]

    run_batch(
        target=target,
        model_keys=model_keys,
        levels=levels,
        workers=args.workers,
        problems_per_level=args.problems_per_level,
        problem_filter=args.problem,
        dry_run=args.dry_run,
        resume_dir=args.resume,
        judge_model=args.judge_model,
    )


def cmd_list_models(args):
    from src.models import MODELS
    print(f"{'Key':<40} {'Name':<30} {'Provider':<12}")
    print("-" * 82)
    for key, cfg in sorted(MODELS.items()):
        print(f"{key:<40} {cfg.name:<30} {cfg.provider:<12}")


def cmd_list_hardware(args):
    from src.hardware import TARGETS
    print(f"{'Target':<12} {'Display':<15} {'GPU':<10} {'VRAM':<8} {'Problems'}")
    print("-" * 70)
    for name, t in sorted(TARGETS.items()):
        n_problems = len(t.find_problems(Path(".")))
        print(f"{name:<12} {t.display_name:<15} {t.gpu_sku:<10} {t.vram_gb:<8} {n_problems}")


def cmd_list_problems(args):
    from src.hardware import get_target
    target = get_target(args.hardware)
    problems = target.find_problems(Path("."))
    for level, path in problems:
        print(f"L{level} {path.name}")
    print(f"\nTotal: {len(problems)} problems")


def cmd_summary(args):
    import json
    results_path = Path(args.run_dir) / "results.jsonl"
    if not results_path.exists():
        print(f"No results found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        results = [json.loads(line) for line in f if line.strip()]

    total = len(results)
    compiled = sum(1 for r in results if r.get("compiled"))
    correct = sum(1 for r in results if r.get("correct"))
    speedups = [r["speedup"] for r in results if r.get("correct") and r.get("speedup")]
    avg_sp = sum(speedups) / len(speedups) if speedups else 0
    total_cost = sum(r.get("estimated_cost_usd", 0) or 0 for r in results)

    print(f"Run: {args.run_dir}")
    print(f"Total: {total} | Compiled: {compiled} | Correct: {correct} | Avg speedup: {avg_sp:.2f}x | Cost: ${total_cost:.2f}")


def main():
    parser = argparse.ArgumentParser(description="KernelBench v3")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run benchmark evaluations")
    p_run.add_argument("hardware", help="Hardware target: rtx3090, h100, b200, m4max")
    p_run.add_argument("--models", required=True, help="Model key(s), comma-separated, or 'all'")
    p_run.add_argument("--levels", default="1,2,3,4", help="Comma-separated levels")
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument("--problems-per-level", type=int, default=None)
    p_run.add_argument("--problem", type=str, default=None, help="Run a single problem by filename")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--resume", type=str, default=None, help="Resume from run directory")
    p_run.add_argument("--judge-model", type=str, default=None, help="Model key for post-benchmark judge")

    sub.add_parser("list-models", help="List registered models")
    sub.add_parser("list-hardware", help="List hardware targets")

    p_problems = sub.add_parser("list-problems", help="List problems for a hardware target")
    p_problems.add_argument("hardware")

    p_summary = sub.add_parser("summary", help="Print summary for a completed run")
    p_summary.add_argument("run_dir")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    elif args.command == "list-hardware":
        cmd_list_hardware(args)
    elif args.command == "list-problems":
        cmd_list_problems(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
