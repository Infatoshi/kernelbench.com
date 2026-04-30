"""Aggregate all KernelBench-v3 evaluation results into a single CSV."""

import csv
import json
import shutil
from pathlib import Path

BASE = Path("/home/infatoshi/cuda/KernelBench-v3")
BATCH_DIR = BASE / "outputs" / "batch_eval"
OUT_DIR = BASE / "outputs" / "aggregated"
CSV_PATH = OUT_DIR / "results_v3.csv"
SOL_DIR = OUT_DIR / "solutions"

# Models to EXCLUDE
EXCLUDE_MODELS = {
    "DeepSeek V3.2",
    "Qwen3 Coder Next",
    "Qwen: Qwen3.5-35B-A3B",
    "Qwen: Qwen3.5-122B-A10B",
}

# Build problem -> level_dir mapping
LEVEL_DIRS = ["level1", "level2", "level3", "level4", "graphics", "tile_specialized", "cutile"]
PROBLEM_TO_LEVEL_DIR = {}
for ld in LEVEL_DIRS:
    d = BASE / "problems" / ld
    if d.exists():
        for f in d.iterdir():
            if f.suffix == ".py":
                PROBLEM_TO_LEVEL_DIR[f.name] = ld

# Model name -> short name mapping
MODEL_SHORT = {
    "Gemini 3 Flash Preview": "Gemini 3 Flash",
    "Gemini 3.1 Pro Preview": "Gemini 3.1 Pro",
    "Claude Opus 4.6": "Claude Opus 4.6",
    "Claude Sonnet 4.6": "Claude Sonnet 4.6",
    "GPT-5.4": "GPT-5.4",
    "GPT-5.3": "GPT-5.3",
    "GLM-5": "GLM-5",
    "Kimi K2.5": "Kimi K2.5",
    "MiniMax M2.5": "MiniMax M2.5",
    "Qwen3.5 397B A17B": "Qwen3.5-397B",
}

# Model name -> provider
MODEL_PROVIDER = {
    "Gemini 3 Flash Preview": "Google",
    "Gemini 3.1 Pro Preview": "Google",
    "Claude Opus 4.6": "Anthropic",
    "Claude Sonnet 4.6": "Anthropic",
    "GPT-5.4": "OpenAI",
    "GPT-5.3": "OpenAI",
    "GLM-5": "Z-AI",
    "Kimi K2.5": "Moonshot AI",
    "MiniMax M2.5": "MiniMax",
    "Qwen3.5 397B A17B": "Alibaba",
}

# op_type -> category
OP_CATEGORIES = {
    "gemm": "GEMM",
    "conv": "Convolution",
    "softmax": "Softmax",
    "elementwise": "Elementwise",
    "norm": "Normalization",
    "pool": "Pooling",
    "reduce": "Reduction",
    "attention": "Attention",
    "transformer": "Transformer",
    "moe": "MoE",
    "mla": "MLA",
    "gqa": "GQA",
    "fused": "Fused",
    "matmul": "GEMM",
    "graphics": "Graphics",
    "quantized": "Quantized",
}


def categorize_op(op_type):
    if not op_type:
        return "Other"
    op_lower = op_type.lower()
    for key, cat in OP_CATEGORIES.items():
        if key in op_lower:
            return cat
    return op_type.title() if op_type else "Other"


def problem_name(filename):
    """Human-readable name from problem filename."""
    name = filename.replace(".py", "")
    # Strip leading number and underscore
    parts = name.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        name = parts[1]
    return name.replace("_", " ")


def main():
    # Collect all results, keyed by (model, gpu, problem) -> (run_timestamp, record, run_dir)
    best = {}  # (model, gpu, problem) -> (run_name, record, run_dir_path)

    run_dirs = sorted(BATCH_DIR.glob("run_*"))
    total_read = 0
    skipped_models = 0

    for run_dir in run_dirs:
        results_file = run_dir / "results.jsonl"
        if not results_file.exists():
            continue
        run_name = run_dir.name  # e.g. run_20260226_235356

        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                model = rec.get("model", "")
                if model in EXCLUDE_MODELS:
                    skipped_models += 1
                    continue

                total_read += 1
                gpu = rec.get("gpu", "")
                problem = rec.get("problem", "")
                key = (model, gpu, problem)

                # Keep latest run (run_dirs are sorted by timestamp)
                if key not in best or run_name > best[key][0]:
                    best[key] = (run_name, rec, run_dir)

    print(f"Read {total_read} records from {len(run_dirs)} runs, skipped {skipped_models} excluded-model records")
    print(f"After dedup: {len(best)} unique (model, gpu, problem) entries")

    # Build CSV rows
    rows = []
    for (model, gpu, problem), (run_name, rec, run_dir) in best.items():
        level = rec.get("level", "")
        level_dir = PROBLEM_TO_LEVEL_DIR.get(problem, f"level{level}" if level else "unknown")
        speedup = rec.get("speedup")
        correct = rec.get("correct", False)
        model_short = MODEL_SHORT.get(model, model)

        row = {
            "run": run_name,
            "model": model,
            "model_short": model_short,
            "provider": MODEL_PROVIDER.get(model, "Unknown"),
            "gpu": gpu,
            "level": level,
            "level_dir": level_dir,
            "problem": problem,
            "problem_name": problem_name(problem),
            "op_type": rec.get("op_type", ""),
            "problem_category": categorize_op(rec.get("op_type", "")),
            "compiled": rec.get("compiled", False),
            "correct": correct,
            "speedup": speedup if speedup is not None else "",
            "beats_baseline": (speedup is not None and speedup > 1.0) if correct else False,
            "baseline_type": rec.get("baseline_type", ""),
            "ref_ms": rec.get("ref_ms", ""),
            "sol_ms": rec.get("sol_ms", ""),
            "precision_used": rec.get("precision_used", ""),
            "input_tokens": rec.get("input_tokens", ""),
            "output_tokens": rec.get("output_tokens", ""),
            "total_tokens": rec.get("total_tokens", ""),
            "turns": rec.get("turns", ""),
            "estimated_cost_usd": rec.get("estimated_cost_usd", ""),
            "elapsed_seconds": rec.get("elapsed_seconds", ""),
            "baseline_link": f"https://github.com/Infatoshi/KernelBench-v3/blob/master/problems/{level_dir}/{problem}",
            "solution_link": f"/data/kernelbench-v3/solutions/{model_short}_{gpu}_{problem}.txt",
        }
        rows.append(row)

    # Sort by model, gpu, level, problem
    def sort_key(r):
        lev = r["level"] if isinstance(r["level"], int) else 999
        return (r["model"], r["gpu"], lev, r["problem"])

    rows.sort(key=sort_key)

    # Write CSV
    fieldnames = [
        "run", "model", "model_short", "provider", "gpu", "level", "level_dir",
        "problem", "problem_name", "op_type", "problem_category",
        "compiled", "correct", "speedup", "beats_baseline",
        "baseline_type", "ref_ms", "sol_ms", "precision_used",
        "input_tokens", "output_tokens", "total_tokens", "turns",
        "estimated_cost_usd", "elapsed_seconds",
        "baseline_link", "solution_link",
    ]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {CSV_PATH}")

    # Copy solution files for correct results
    SOL_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for row in rows:
        if not row["correct"]:
            continue
        model_short = row["model_short"]
        gpu = row["gpu"]
        problem = row["problem"]
        run_name = row["run"]

        # Build turns directory name: {Model_Name}_{GPU}_{problem}
        model_underscore = row["model"].replace(" ", "_")
        turns_dir_name = f"{model_underscore}_{gpu}_{problem}"
        run_dir = BATCH_DIR / run_name / "turns" / turns_dir_name

        # Find the latest solution file
        solution_src = None
        if run_dir.exists():
            # Get highest numbered turn solution
            sol_files = sorted(run_dir.glob("turn_*_solution.py"))
            if sol_files:
                solution_src = sol_files[-1]

        if solution_src and solution_src.exists():
            dest = SOL_DIR / f"{model_short}_{gpu}_{problem}.txt"
            shutil.copy2(solution_src, dest)
            copied += 1
        else:
            missing += 1

    print(f"Copied {copied} solution files, {missing} missing")

    # Print sample rows
    print("\n--- Sample rows ---")
    for i, row in enumerate(rows[:5]):
        print(f"  {row['model_short']:20s} {row['gpu']:8s} L{row['level']} {row['problem']:40s} correct={row['correct']}  speedup={row['speedup']}")
    print("  ...")
    for row in rows[-3:]:
        print(f"  {row['model_short']:20s} {row['gpu']:8s} L{row['level']} {row['problem']:40s} correct={row['correct']}  speedup={row['speedup']}")


if __name__ == "__main__":
    main()
