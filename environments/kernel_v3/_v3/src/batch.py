"""Batch evaluation orchestrator for KernelBench benchmarks."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from src.eval.results import EvalResult
from src.models import get_model_config


def get_all_tasks(
    target_name: str,
    model_keys: List[str],
    levels: List[int],
    problems_per_level: Optional[int] = None,
    problem_filter: Optional[str] = None,
) -> List[Tuple[str, int, Path]]:
    """Generate (model_key, level, problem_path) combinations."""
    from src.hardware import get_target

    target = get_target(target_name)
    all_problems = target.find_problems(Path("."))
    level_set = set(levels)
    problems = [(lv, p) for lv, p in all_problems if lv in level_set]

    if problem_filter is not None:
        problems = [(lv, p) for lv, p in problems if problem_filter in p.name]

    if problems_per_level is not None:
        filtered = []
        counts: dict = {}
        for lv, p in problems:
            counts[lv] = counts.get(lv, 0) + 1
            if counts[lv] <= problems_per_level:
                filtered.append((lv, p))
        problems = filtered

    tasks = []
    for model_key in model_keys:
        for lv, p in problems:
            tasks.append((model_key, lv, p))
    return tasks


def load_completed(run_dir: Path) -> set:
    completed = set()
    results_file = run_dir / "results.jsonl"
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add(f"{r['model']}_{r['gpu']}_{r['problem']}")
                except Exception:
                    pass
    return completed


def run_single_eval(
    target_name: str,
    model_key: str,
    level: int,
    problem_path: Path,
    max_turns: Optional[int] = None,
    turn_artifact_dir: Optional[Path] = None,
    judge_model_key: Optional[str] = None,
) -> EvalResult:
    from src.eval.agent import run_eval
    from src.hardware import get_target

    target = get_target(target_name)
    model_config = get_model_config(model_key)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_key}")

    if max_turns is None:
        max_turns = target.max_turns(level)
    max_time = target.max_time(level)

    with open(problem_path) as f:
        problem_code = f.read()

    print(f"[START] {model_config.name} | {target.display_name} | {problem_path.name}", flush=True)

    prev_turn_dir = os.environ.get("KB_TURN_ARTIFACT_DIR")
    try:
        if turn_artifact_dir is not None:
            turn_artifact_dir.mkdir(parents=True, exist_ok=True)
            os.environ["KB_TURN_ARTIFACT_DIR"] = str(turn_artifact_dir)
        elif "KB_TURN_ARTIFACT_DIR" in os.environ:
            del os.environ["KB_TURN_ARTIFACT_DIR"]

        result = run_eval(
            hardware_target=target,
            model_config=model_config,
            problem_code=problem_code,
            problem_name=problem_path.name,
            level=level,
            max_turns=max_turns,
            max_time=max_time,
            judge_model_key=judge_model_key,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        result = EvalResult(
            model=model_config.name, gpu=target.gpu_sku,
            problem=problem_path.name, level=level, error=str(e),
        )
    finally:
        if prev_turn_dir is None:
            os.environ.pop("KB_TURN_ARTIFACT_DIR", None)
        else:
            os.environ["KB_TURN_ARTIFACT_DIR"] = prev_turn_dir

    status = "OK" if result.correct else ("FAIL" if result.compiled else "ERR")
    speedup = f"{result.speedup:.2f}x" if result.speedup else "N/A"
    kernels = f"k:{result.ref_kernels}->{result.sol_kernels}" if result.ref_kernels is not None else ""
    print(f"[{status}] {model_config.name} | {target.display_name} | {problem_path.name} | {speedup} {kernels}", flush=True)
    return result


def run_batch(
    target,
    model_keys: List[str],
    levels: List[int],
    workers: int = 4,
    problems_per_level: Optional[int] = None,
    problem_filter: Optional[str] = None,
    dry_run: bool = False,
    resume_dir: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> None:
    target_name = target.name
    tasks = get_all_tasks(target_name, model_keys, levels, problems_per_level, problem_filter)

    if resume_dir:
        run_dir = Path(resume_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("outputs/batch_eval") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    completed = load_completed(run_dir) if resume_dir else set()

    print("=" * 80)
    print(f"KERNELBENCH — {target.display_name}")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Models: {model_keys}")
    print(f"Hardware: {target.display_name} ({target.gpu_sku})")
    print(f"Levels: {levels}")
    print(f"Total tasks: {len(tasks)}")
    time_limits = {lv: target.max_time(lv) for lv in range(1, 5)}
    print(f"Time limits: L1={time_limits[1]//60}min, L2={time_limits[2]//60}min, L3={time_limits[3]//60}min, L4={time_limits[4]//60}min")
    print(f"Workers: {workers}")
    if judge_model:
        print(f"Judge model: {judge_model}")
    print("=" * 80)

    if dry_run:
        print("\n[DRY RUN] Would evaluate:")
        for model_key, lv, problem_path in tasks[:5]:
            cfg = get_model_config(model_key)
            print(f"  {cfg.name if cfg else model_key} | {target.display_name} | L{lv} | {problem_path.name}")
        if len(tasks) > 5:
            print(f"  ... and {len(tasks) - 5} more")
        return

    max_concurrent_caps = []
    for model_key in model_keys:
        cfg = get_model_config(model_key)
        if cfg and cfg.max_concurrent is not None:
            max_concurrent_caps.append(cfg.max_concurrent)
    if max_concurrent_caps:
        effective_workers = min(workers, min(max_concurrent_caps))
        if effective_workers < workers:
            print(f"Note: capping workers {workers} -> {effective_workers} (model concurrency limit)")
            workers = effective_workers

    pending = []
    for task in tasks:
        model_key, lv, problem_path = task
        cfg = get_model_config(model_key)
        task_id = f"{cfg.name if cfg else model_key}_{target.gpu_sku}_{problem_path.name}"
        if task_id not in completed:
            pending.append(task)

    print(f"Total tasks: {len(tasks)}")
    print(f"Already completed: {len(completed)}")
    print(f"Pending: {len(pending)}")
    print()

    if not pending:
        print("All tasks completed!")
        _print_summary(run_dir)
        return

    start_time = time.time()
    results_file = run_dir / "results.jsonl"

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for task in pending:
            model_key, lv, problem_path = task
            cfg = get_model_config(model_key)
            task_id = f"{cfg.name if cfg else model_key}_{target.gpu_sku}_{problem_path.name}"
            turn_dir = run_dir / "turns" / task_id.replace("/", "-").replace(" ", "_")
            future = executor.submit(run_single_eval, target_name, model_key, lv, problem_path, None, turn_dir, judge_model)
            futures[future] = task

        done_count = len(completed)
        for future in as_completed(futures):
            task = futures[future]
            model_key, lv, problem_path = task
            try:
                result = future.result()
            except Exception as e:
                cfg = get_model_config(model_key)
                result = EvalResult(
                    model=cfg.name if cfg else model_key, gpu=target.gpu_sku,
                    problem=problem_path.name, level=lv, error=str(e),
                )

            result_dict = asdict(result)
            with open(results_file, "a") as f:
                f.write(json.dumps(result_dict, default=str) + "\n")
            done_count += 1
            print(f"Progress: {done_count}/{len(tasks)} ({100 * done_count / len(tasks):.1f}%)")

    elapsed_hours = (time.time() - start_time) / 3600
    _print_summary(run_dir)
    print(f"\nTotal time: {elapsed_hours:.1f} hours")
    print(f"Results saved to: {run_dir}")


def _print_summary(run_dir: Path) -> None:
    results_file = run_dir / "results.jsonl"
    if not results_file.exists():
        return

    with open(results_file) as f:
        results = [json.loads(line) for line in f if line.strip()]

    total = len(results)
    total_in = sum(r.get("input_tokens", 0) for r in results)
    total_out = sum(r.get("output_tokens", 0) for r in results)
    total_cost = sum(r.get("estimated_cost_usd", 0) or 0 for r in results)

    print("\n" + "=" * 100)
    print("EVALUATION SUMMARY")
    print("=" * 100)
    print(f"\nTotal runs: {total}")
    print(f"Total tokens: {total_in:,} in / {total_out:,} out / {total_in + total_out:,} total")
    print(f"Total cost: ${total_cost:.2f}")

    by_model: dict = {}
    for r in results:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"total": 0, "compiled": 0, "correct": 0, "speedups": [], "tokens": 0, "cost": 0.0}
        by_model[m]["total"] += 1
        if r.get("compiled"):
            by_model[m]["compiled"] += 1
        if r.get("correct"):
            by_model[m]["correct"] += 1
        if r.get("speedup"):
            by_model[m]["speedups"].append(r["speedup"])
        by_model[m]["tokens"] += r.get("input_tokens", 0) + r.get("output_tokens", 0)
        by_model[m]["cost"] += r.get("estimated_cost_usd", 0) or 0

    print("\n--- BY MODEL ---")
    print(f"{'Model':<27} {'Total':>5} {'Compiled':>8} {'Correct':>8} {'Speedup':>10} {'Tokens':>12} {'Cost':>10}")
    print("-" * 90)
    for m, s in sorted(by_model.items()):
        sp = f"{sum(s['speedups'])/len(s['speedups']):.2f}x" if s["speedups"] else "N/A"
        cost_str = f"${s['cost']:.2f}"
        print(f"{m:<27} {s['total']:>5} {s['compiled']:>8} {s['correct']:>8} {sp:>10} {s['tokens']:>12,} {cost_str:>10}")

    print("\n--- BY GPU ---")
    by_gpu: dict = {}
    for r in results:
        g = r["gpu"]
        if g not in by_gpu:
            by_gpu[g] = {"total": 0, "compiled": 0, "correct": 0, "speedups": []}
        by_gpu[g]["total"] += 1
        if r.get("compiled"):
            by_gpu[g]["compiled"] += 1
        if r.get("correct"):
            by_gpu[g]["correct"] += 1
        if r.get("speedup"):
            by_gpu[g]["speedups"].append(r["speedup"])

    print(f"{'GPU':<15} {'Total':>8} {'Compiled':>10} {'Correct':>10} {'Avg Speedup':>12}")
    print("-" * 55)
    for g, s in sorted(by_gpu.items()):
        sp = f"{sum(s['speedups'])/len(s['speedups']):.2f}x" if s["speedups"] else "N/A"
        print(f"{g:<15} {s['total']:>8} {s['compiled']:>10} {s['correct']:>10} {sp:>12}")

    print("\n--- BY LEVEL ---")
    by_level: dict = {}
    for r in results:
        lv = r["level"]
        if lv not in by_level:
            by_level[lv] = {"total": 0, "compiled": 0, "correct": 0, "speedups": []}
        by_level[lv]["total"] += 1
        if r.get("compiled"):
            by_level[lv]["compiled"] += 1
        if r.get("correct"):
            by_level[lv]["correct"] += 1
        if r.get("speedup"):
            by_level[lv]["speedups"].append(r["speedup"])

    print(f"{'Level':>8} {'Total':>8} {'Compiled':>10} {'Correct':>10} {'Avg Speedup':>12}")
    print("-" * 50)
    for lv, s in sorted(by_level.items()):
        sp = f"{sum(s['speedups'])/len(s['speedups']):.2f}x" if s["speedups"] else "N/A"
        print(f"{lv:>8} {s['total']:>8} {s['compiled']:>10} {s['correct']:>10} {sp:>12}")
