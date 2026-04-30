#!/usr/bin/env python3
"""
Aggregate all benchmark results into a comprehensive CSV for visualization and analysis.

Generates:
1. Full CSV with all metrics and link columns
2. Summary statistics
3. Matplotlib visualizations

Usage:
    uv run python aggregate_results.py [--output-dir outputs/aggregated]
"""

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

# =============================================================================
# PROBLEM METADATA
# =============================================================================

# Map problem files to categories
PROBLEM_CATEGORIES = {
    # Level 1 - Simple operators
    "1_Square_matrix_multiplication_.py": "matmul",
    "2_Standard_matrix_multiplication_.py": "matmul",
    "3_Batched_matrix_multiplication.py": "matmul",
    "4_Matrix_vector_multiplication_.py": "matmul",
    "8_Matmul_with_irregular_shapes_.py": "matmul",
    "9_Tall_skinny_matrix_multiplication_.py": "matmul",
    "23_Softmax.py": "activation",
    "26_GELU_.py": "activation",
    "36_RMSNorm_.py": "norm",
    "40_LayerNorm.py": "norm",
    "42_Max_Pooling_2D.py": "pooling",
    "47_Sum_reduction_over_a_dimension.py": "reduction",
    "63_conv_standard_2D__square_input__square_kernel.py": "conv",
    "82_conv_depthwise_2D_square_input_square_kernel.py": "conv",
    "95_CrossEntropyLoss.py": "loss",

    # Level 2 - Fused operations
    "6_Conv3d_Softmax_MaxPool_MaxPool.py": "fused_conv",
    "17_Conv2d_InstanceNorm_Divide.py": "fused_conv",
    "37_Matmul_Swish_Sum_GroupNorm.py": "fused_matmul",
    "40_Matmul_Scaling_ResidualAdd.py": "fused_matmul",
    "46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py": "fused_conv",
    "52_Conv2d_Activation_BatchNorm.py": "fused_conv",
    "55_Matmul_MaxPool_Sum_Scale.py": "fused_matmul",
    "59_Matmul_Swish_Scaling.py": "fused_matmul",
    "66_Matmul_Dropout_Mean_Softmax.py": "fused_matmul",
    "73_Conv2d_BatchNorm_Scaling.py": "fused_conv",
    "82_Conv2d_Tanh_Scaling_BiasAdd_Max.py": "fused_conv",
    "85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py": "fused_conv",
    "86_Matmul_Divide_GELU.py": "fused_matmul",
    "98_Matmul_AvgPool_GELU_Scale_Max.py": "fused_matmul",
    "99_Matmul_GELU_Softmax.py": "fused_matmul",

    # Level 3 - Single blocks
    "31_VisionAttention.py": "attention",
    "43_MinGPTCausalAttention.py": "attention",
    "44_MiniGPTBlock.py": "transformer",

    # Level 4 - Novel layers
    "1_DeepSeek_MLA.py": "attention",
    "2_DeepSeek_MoE.py": "moe",
    "3_GroupedQueryAttention.py": "attention",
    "4_FP8_Matmul.py": "quantized",
    "5_MoE_GatedGEMM.py": "moe",
    "6_INT4_Quantized_GEMM.py": "quantized",
    "7_GatedDeltaNet.py": "linear_attention",
    "8_KimiDeltaAttention.py": "linear_attention",
}

# Human-readable problem names
PROBLEM_NAMES = {
    # Level 1
    "1_Square_matrix_multiplication_.py": "Square Matrix Multiplication",
    "2_Standard_matrix_multiplication_.py": "Standard Matrix Multiplication",
    "3_Batched_matrix_multiplication.py": "Batched Matrix Multiplication",
    "4_Matrix_vector_multiplication_.py": "Matrix-Vector Multiplication",
    "8_Matmul_with_irregular_shapes_.py": "Irregular Shape Matmul",
    "9_Tall_skinny_matrix_multiplication_.py": "Tall-Skinny Matmul",
    "23_Softmax.py": "Softmax",
    "26_GELU_.py": "GELU Activation",
    "36_RMSNorm_.py": "RMSNorm",
    "40_LayerNorm.py": "LayerNorm",
    "42_Max_Pooling_2D.py": "Max Pooling 2D",
    "47_Sum_reduction_over_a_dimension.py": "Sum Reduction",
    "63_conv_standard_2D__square_input__square_kernel.py": "Conv2D Standard",
    "82_conv_depthwise_2D_square_input_square_kernel.py": "Depthwise Conv2D",
    "95_CrossEntropyLoss.py": "Cross-Entropy Loss",

    # Level 2
    "6_Conv3d_Softmax_MaxPool_MaxPool.py": "Conv3D + Softmax + MaxPool",
    "17_Conv2d_InstanceNorm_Divide.py": "Conv2D + InstanceNorm + Divide",
    "37_Matmul_Swish_Sum_GroupNorm.py": "Matmul + Swish + Sum + GroupNorm",
    "40_Matmul_Scaling_ResidualAdd.py": "Matmul + Scale + Residual",
    "46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py": "Conv2D + Tanh + AvgPool",
    "52_Conv2d_Activation_BatchNorm.py": "Conv2D + Activation + BatchNorm",
    "55_Matmul_MaxPool_Sum_Scale.py": "Matmul + MaxPool + Sum + Scale",
    "59_Matmul_Swish_Scaling.py": "Matmul + Swish + Scale",
    "66_Matmul_Dropout_Mean_Softmax.py": "Matmul + Dropout + Mean + Softmax",
    "73_Conv2d_BatchNorm_Scaling.py": "Conv2D + BatchNorm + Scale",
    "82_Conv2d_Tanh_Scaling_BiasAdd_Max.py": "Conv2D + Tanh + Scale + Bias + Max",
    "85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py": "Conv2D + GroupNorm + MaxPool + Clamp",
    "86_Matmul_Divide_GELU.py": "Matmul + Divide + GELU",
    "98_Matmul_AvgPool_GELU_Scale_Max.py": "Matmul + AvgPool + GELU + Scale + Max",
    "99_Matmul_GELU_Softmax.py": "Matmul + GELU + Softmax",

    # Level 3
    "31_VisionAttention.py": "Vision Attention",
    "43_MinGPTCausalAttention.py": "MinGPT Causal Attention",
    "44_MiniGPTBlock.py": "MiniGPT Transformer Block",

    # Level 4
    "1_DeepSeek_MLA.py": "DeepSeek MLA",
    "2_DeepSeek_MoE.py": "DeepSeek MoE",
    "3_GroupedQueryAttention.py": "Grouped Query Attention (GQA)",
    "4_FP8_Matmul.py": "FP8 Matmul",
    "5_MoE_GatedGEMM.py": "MoE Gated GEMM",
    "6_INT4_Quantized_GEMM.py": "INT4 Quantized GEMM",
    "7_GatedDeltaNet.py": "Gated DeltaNet",
    "8_KimiDeltaAttention.py": "Kimi Delta Attention",
}

# Max turns per level
MAX_TURNS_PER_LEVEL = {1: 10, 2: 12, 3: 15, 4: 15}

# =============================================================================
# MODEL METADATA
# =============================================================================

MODEL_METADATA = {
    "Claude Opus 4.5": {
        "model_key": "claude-opus-4.5",
        "model_id": "claude-opus-4-5-20251101",
        "provider": "anthropic",
        "tier": "frontier",
        "input_price_per_m": 5.00,
        "output_price_per_m": 25.00,
    },
    "Claude Sonnet 4.5": {
        "model_key": "claude-sonnet-4.5",
        "model_id": "claude-sonnet-4-5-20250929",
        "provider": "anthropic",
        "tier": "frontier",
        "input_price_per_m": 3.00,
        "output_price_per_m": 15.00,
    },
    "GPT-5.2": {
        "model_key": "gpt-5.2",
        "model_id": "gpt-5.2",
        "provider": "openai",
        "tier": "frontier",
        "input_price_per_m": 1.75,
        "output_price_per_m": 14.00,
    },
    "GPT-5.2 Codex": {
        "model_key": "gpt-5.2-codex",
        "model_id": "gpt-5.2-codex",
        "provider": "openai",
        "tier": "frontier",
        "input_price_per_m": 1.75,
        "output_price_per_m": 14.00,
    },
    "Gemini 3 Flash": {
        "model_key": "gemini-3-flash",
        "model_id": "gemini-3-flash-preview",
        "provider": "gemini",
        "tier": "frontier",
        "input_price_per_m": 0.50,
        "output_price_per_m": 3.00,
    },
    "Gemini 3 Pro": {
        "model_key": "gemini-3-pro",
        "model_id": "gemini-3-pro-preview",
        "provider": "gemini",
        "tier": "frontier",
        "input_price_per_m": 2.00,
        "output_price_per_m": 12.00,
    },
    "Grok 4.1 Fast Reasoning": {
        "model_key": "grok-4.1",
        "model_id": "grok-4-1-fast-reasoning",
        "provider": "xai",
        "tier": "frontier",
        "input_price_per_m": 0.20,
        "output_price_per_m": 0.50,
    },
    "GLM-4.7": {
        "model_key": "glm-4.7",
        "model_id": "z-ai/glm-4.7",
        "provider": "openrouter",
        "tier": "open",
        "input_price_per_m": 0.40,
        "output_price_per_m": 1.50,
    },
    "DeepSeek V3.2": {
        "model_key": "deepseek-v3.2",
        "model_id": "deepseek/deepseek-v3.2",
        "provider": "openrouter",
        "tier": "open",
        "input_price_per_m": 0.25,
        "output_price_per_m": 0.38,
    },
    "Kimi K2 Thinking": {
        "model_key": "kimi-k2-thinking",
        "model_id": "moonshotai/kimi-k2-thinking",
        "provider": "openrouter",
        "tier": "open",
        "input_price_per_m": 0.40,
        "output_price_per_m": 1.75,
    },
    "MiniMax M2.1": {
        "model_key": "minimax-m2.1",
        "model_id": "minimax/minimax-m2.1",
        "provider": "openrouter",
        "tier": "open",
        "input_price_per_m": 0.27,
        "output_price_per_m": 1.12,
    },
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_results(batch_eval_dir: Path) -> list[dict]:
    """Load all results from batch_eval runs."""
    results = []
    for run_dir in sorted(batch_eval_dir.glob("run_*")):
        results_file = run_dir / "results.jsonl"
        if results_file.exists():
            with open(results_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        data["run_id"] = run_dir.name
                        data["run_timestamp"] = run_dir.name.replace("run_", "")
                        results.append(data)
    return results


def load_results(results_path: Path) -> list[dict]:
    """Load JSONL results file."""
    rows: list[dict] = []
    if not results_path.exists():
        return rows
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate_run(run_dir: Path) -> dict:
    """Aggregate metrics from a single batch run directory."""
    results = load_results(run_dir / "results.jsonl")
    if not results:
        return {
            "run_id": run_dir.name,
            "model": "",
            "hardware": "",
            "benchmark": "unknown",
            "pass_rate": 0.0,
            "fast_1_0": 0.0,
            "fast_1_5": 0.0,
            "fast_2_0": 0.0,
            "median_speedup": None,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "avg_turns": 0.0,
            "median_tflops": None,
            "num_results": 0,
        }

    total = len(results)
    speedups = [r["speedup"] for r in results if r.get("speedup") is not None]
    tflops = [r["achieved_tflops"] for r in results if r.get("achieved_tflops") is not None]
    models = sorted({r.get("model", "") for r in results if r.get("model")})
    gpus = sorted({r.get("gpu", "") for r in results if r.get("gpu")})
    benchmarks = sorted({r.get("benchmark", "") for r in results if r.get("benchmark")})

    model_label = ",".join(models)
    gpu_label = ",".join(gpus)
    benchmark_label = ",".join(benchmarks) if benchmarks else "unknown"

    return {
        "run_id": run_dir.name,
        "model": model_label,
        "hardware": gpu_label,
        "benchmark": benchmark_label,
        "pass_rate": sum(1 for r in results if r.get("correct")) / total,
        "fast_1_0": sum(1 for r in results if r.get("speedup") is not None and r["speedup"] > 1.0) / total,
        "fast_1_5": sum(1 for r in results if r.get("speedup") is not None and r["speedup"] > 1.5) / total,
        "fast_2_0": sum(1 for r in results if r.get("speedup") is not None and r["speedup"] > 2.0) / total,
        "median_speedup": statistics.median(speedups) if speedups else None,
        "total_tokens": sum(r.get("total_tokens", 0) or 0 for r in results),
        "total_cost_usd": sum(r.get("estimated_cost_usd", 0) or 0 for r in results),
        "avg_turns": statistics.mean([r.get("turns", 0) for r in results]) if results else 0.0,
        "median_tflops": statistics.median(tflops) if tflops else None,
        "num_results": total,
    }


def aggregate_runs(batch_eval_dir: Path) -> list[dict]:
    """Aggregate every run_* directory under batch_eval outputs."""
    rows: list[dict] = []
    for run_dir in sorted(batch_eval_dir.glob("run_*")):
        if (run_dir / "results.jsonl").exists():
            rows.append(aggregate_run(run_dir))
    return rows


def write_aggregate_csv(rows: list[dict], output_path: Path) -> None:
    """Write aggregate rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(output_path, "w", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def enrich_result(result: dict) -> dict:
    """Add metadata and derived metrics to a result."""
    problem = result.get("problem", "")
    model = result.get("model", "")
    level = result.get("level", 0)

    # Problem metadata
    result["problem_name"] = PROBLEM_NAMES.get(problem, problem)
    result["problem_category"] = PROBLEM_CATEGORIES.get(problem, "unknown")
    result["max_turns"] = MAX_TURNS_PER_LEVEL.get(level, 10)

    # Model metadata
    model_meta = MODEL_METADATA.get(model, {})
    result["model_key"] = model_meta.get("model_key", model.lower().replace(" ", "-"))
    result["model_id"] = model_meta.get("model_id", "")
    result["provider"] = model_meta.get("provider", "unknown")
    result["model_tier"] = model_meta.get("tier", "unknown")
    result["input_price_per_m"] = model_meta.get("input_price_per_m", 0)
    result["output_price_per_m"] = model_meta.get("output_price_per_m", 0)

    # Derived metrics
    compiled = result.get("compiled", False)
    correct = result.get("correct", False)
    speedup = result.get("speedup")

    result["passed"] = compiled and correct
    result["speedup_if_correct"] = speedup if correct else None
    result["beats_baseline"] = speedup > 1.0 if (correct and speedup is not None) else False

    # Token efficiency
    total_tokens = result.get("total_tokens", 0)
    if result["passed"] and total_tokens > 0:
        result["tokens_per_success"] = total_tokens
    else:
        result["tokens_per_success"] = None

    # Cost efficiency
    cost = result.get("estimated_cost_usd", 0)
    if result["passed"] and cost and cost > 0:
        result["cost_per_success"] = cost
    else:
        result["cost_per_success"] = None

    # Speedup efficiency (speedup per dollar)
    if result["beats_baseline"] and cost and cost > 0:
        result["speedup_per_dollar"] = speedup / cost
    else:
        result["speedup_per_dollar"] = None

    # Turn efficiency
    turns = result.get("turns", 0)
    max_turns = result.get("max_turns", 10)
    result["turns_remaining"] = max_turns - turns
    result["turn_utilization"] = turns / max_turns if max_turns > 0 else 0

    # Links (relative paths for website)
    gpu = result.get("gpu", "")
    # Format: outputs/solutions/MODEL_KEY/GPU/PROBLEM.py
    model_key = result["model_key"]
    result["solution_code_link"] = f"solutions/{model_key}/{gpu}/{problem.replace('.py', '_solution.py')}"
    result["conversation_log_link"] = f"logs/{model_key}/{gpu}/{problem.replace('.py', '.jsonl')}"
    result["error_details_link"] = f"errors/{model_key}/{gpu}/{problem.replace('.py', '.txt')}" if result.get("error") else None

    return result


# =============================================================================
# AGGREGATION
# =============================================================================

def build_dataframe(results: list[dict]) -> pd.DataFrame:
    """Build DataFrame with all columns."""
    enriched = [enrich_result(r) for r in results]
    df = pd.DataFrame(enriched)

    # Reorder columns for clarity
    column_order = [
        # Primary dimensions
        "model", "model_key", "gpu", "level", "problem", "problem_name", "problem_category",
        # Model metadata
        "provider", "model_tier", "model_id", "input_price_per_m", "output_price_per_m",
        # Outcome metrics
        "compiled", "correct", "passed", "submitted",
        "speedup", "speedup_if_correct", "beats_baseline",
        "ref_ms", "sol_ms",
        # Token metrics
        "input_tokens", "output_tokens", "total_tokens",
        "turns", "max_turns", "turns_remaining", "turn_utilization",
        # Cost metrics
        "estimated_cost_usd",
        # Derived efficiency metrics
        "tokens_per_success", "cost_per_success", "speedup_per_dollar",
        # Kernel info
        "ref_kernels", "sol_kernels",
        # Timing
        "elapsed_seconds",
        # Links
        "solution_code_link", "conversation_log_link", "error_details_link",
        # Error info
        "error",
        # Run metadata
        "run_id", "run_timestamp",
    ]

    # Only include columns that exist
    existing_cols = [c for c in column_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in column_order]
    df = df[existing_cols + extra_cols]

    return df


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_evaluations": len(df),
        "unique_models": df["model"].nunique(),
        "unique_gpus": df["gpu"].nunique(),
        "unique_problems": df["problem"].nunique(),

        # Overall metrics
        "overall": {
            "pass_rate": df["passed"].mean(),
            "compile_rate": df["compiled"].mean(),
            "correct_rate": df["correct"].mean(),
            "beats_baseline_rate": df["beats_baseline"].mean(),
            "avg_speedup_when_correct": df[df["correct"]]["speedup"].mean(),
            "median_speedup_when_correct": df[df["correct"]]["speedup"].median(),
            "total_tokens": int(df["total_tokens"].sum()),
            "total_cost_usd": float(df["estimated_cost_usd"].sum()),
        },

        # Per-model stats
        "by_model": {},

        # Per-level stats
        "by_level": {},

        # Per-GPU stats
        "by_gpu": {},

        # Per-category stats
        "by_category": {},
    }

    # Per-model
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        summary["by_model"][model] = {
            "pass_rate": float(model_df["passed"].mean()),
            "compile_rate": float(model_df["compiled"].mean()),
            "correct_rate": float(model_df["correct"].mean()),
            "beats_baseline_rate": float(model_df["beats_baseline"].mean()),
            "avg_speedup_when_correct": float(model_df[model_df["correct"]]["speedup"].mean()) if model_df["correct"].any() else None,
            "total_tokens": int(model_df["total_tokens"].sum()),
            "total_cost_usd": float(model_df["estimated_cost_usd"].sum()),
            "avg_turns": float(model_df["turns"].mean()),
        }

    # Per-level
    for level in sorted(df["level"].unique()):
        level_df = df[df["level"] == level]
        summary["by_level"][f"L{level}"] = {
            "pass_rate": float(level_df["passed"].mean()),
            "beats_baseline_rate": float(level_df["beats_baseline"].mean()),
            "avg_speedup_when_correct": float(level_df[level_df["correct"]]["speedup"].mean()) if level_df["correct"].any() else None,
            "num_problems": level_df["problem"].nunique(),
        }

    # Per-GPU
    for gpu in df["gpu"].unique():
        gpu_df = df[df["gpu"] == gpu]
        summary["by_gpu"][gpu] = {
            "pass_rate": float(gpu_df["passed"].mean()),
            "beats_baseline_rate": float(gpu_df["beats_baseline"].mean()),
            "avg_speedup_when_correct": float(gpu_df[gpu_df["correct"]]["speedup"].mean()) if gpu_df["correct"].any() else None,
        }

    # Per-category
    for cat in df["problem_category"].unique():
        cat_df = df[df["problem_category"] == cat]
        summary["by_category"][cat] = {
            "pass_rate": float(cat_df["passed"].mean()),
            "beats_baseline_rate": float(cat_df["beats_baseline"].mean()),
            "avg_speedup_when_correct": float(cat_df[cat_df["correct"]]["speedup"].mean()) if cat_df["correct"].any() else None,
            "num_problems": cat_df["problem"].nunique(),
        }

    return summary


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create matplotlib visualizations."""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Pass rate by model (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    model_stats = df.groupby("model")["passed"].mean().sort_values(ascending=False)
    colors = ['#2ecc71' if v > 0.5 else '#e74c3c' for v in model_stats.values]
    ax.bar(range(len(model_stats)), model_stats.values, color=colors)
    ax.set_xticks(range(len(model_stats)))
    ax.set_xticklabels(model_stats.index, rotation=45, ha='right')
    ax.set_ylabel("Pass Rate (Compiled + Correct)")
    ax.set_title("Pass Rate by Model")
    ax.set_ylim(0, 1)
    for i, v in enumerate(model_stats.values):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(viz_dir / "pass_rate_by_model.png", dpi=150)
    plt.close()

    # 2. Pass rate by level (grouped bar)
    fig, ax = plt.subplots(figsize=(14, 6))
    pivot = df.pivot_table(index="model", columns="level", values="passed", aggfunc="mean")
    pivot = pivot.reindex(model_stats.index)  # Sort by overall pass rate
    x = np.arange(len(pivot.index))
    width = 0.2
    colors_level = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    for i, level in enumerate(sorted(pivot.columns)):
        offset = (i - 1.5) * width
        ax.bar(x + offset, pivot[level], width, label=f'L{level}', color=colors_level[i])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.set_ylabel("Pass Rate")
    ax.set_title("Pass Rate by Model and Level")
    ax.legend(title="Level")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(viz_dir / "pass_rate_by_model_level.png", dpi=150)
    plt.close()

    # 3. Speedup distribution (when correct)
    fig, ax = plt.subplots(figsize=(12, 6))
    correct_df = df[df["correct"]]
    for model in model_stats.index:
        model_speedups = correct_df[correct_df["model"] == model]["speedup"]
        if len(model_speedups) > 0:
            ax.scatter([model] * len(model_speedups), model_speedups, alpha=0.5, s=50)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline (1.0x)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel("Speedup vs Baseline")
    ax.set_title("Speedup Distribution by Model (Correct Solutions Only)")
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    fig.savefig(viz_dir / "speedup_distribution_by_model.png", dpi=150)
    plt.close()

    # 4. Cost vs Performance scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    model_perf = df.groupby("model").agg({
        "passed": "mean",
        "estimated_cost_usd": "sum",
        "beats_baseline": "mean"
    }).reset_index()
    model_perf.columns = ["model", "pass_rate", "total_cost", "beats_baseline_rate"]

    ax.scatter(
        model_perf["total_cost"],
        model_perf["pass_rate"],
        s=model_perf["beats_baseline_rate"] * 500 + 50,
        alpha=0.7,
        c=range(len(model_perf)),
        cmap='tab10'
    )
    for i, row in model_perf.iterrows():
        ax.annotate(row["model"], (row["total_cost"], row["pass_rate"]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("Pass Rate")
    ax.set_title("Cost vs Performance (bubble size = beats baseline rate)")
    plt.tight_layout()
    fig.savefig(viz_dir / "cost_vs_performance.png", dpi=150)
    plt.close()

    # 5. Tokens used per problem
    fig, ax = plt.subplots(figsize=(14, 6))
    token_by_problem = df.groupby(["problem_name", "level"])["total_tokens"].mean().reset_index()
    token_by_problem = token_by_problem.sort_values(["level", "total_tokens"])
    colors_by_level = {1: '#3498db', 2: '#9b59b6', 3: '#e67e22', 4: '#e74c3c'}
    bar_colors = [colors_by_level[level_value] for level_value in token_by_problem["level"]]
    ax.barh(range(len(token_by_problem)), token_by_problem["total_tokens"], color=bar_colors)
    ax.set_yticks(range(len(token_by_problem)))
    ax.set_yticklabels(token_by_problem["problem_name"], fontsize=7)
    ax.set_xlabel("Average Tokens Used")
    ax.set_title("Average Tokens by Problem (colored by level)")
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_by_level[level_value], label=f"L{level_value}")
        for level_value in [1, 2, 3, 4]
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    fig.savefig(viz_dir / "tokens_by_problem.png", dpi=150)
    plt.close()

    # 6. Heatmap: Model x Problem pass/fail
    fig, ax = plt.subplots(figsize=(16, 10))
    pivot_pass = df.pivot_table(index="model", columns="problem_name", values="passed", aggfunc="mean")
    # Sort columns by level then name
    problem_order = df.drop_duplicates("problem_name").sort_values(["level", "problem_name"])["problem_name"]
    pivot_pass = pivot_pass.reindex(columns=[p for p in problem_order if p in pivot_pass.columns])
    pivot_pass = pivot_pass.reindex(model_stats.index)

    im = ax.imshow(pivot_pass.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot_pass.columns)))
    ax.set_xticklabels(pivot_pass.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(pivot_pass.index)))
    ax.set_yticklabels(pivot_pass.index, fontsize=8)
    ax.set_title("Pass Rate Heatmap: Model x Problem")
    plt.colorbar(im, ax=ax, label="Pass Rate")
    plt.tight_layout()
    fig.savefig(viz_dir / "heatmap_model_problem.png", dpi=150)
    plt.close()

    # 7. GPU comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pass rate by GPU
    gpu_pass = df.groupby(["model", "gpu"])["passed"].mean().unstack()
    gpu_pass = gpu_pass.reindex(model_stats.index)
    x = np.arange(len(gpu_pass.index))
    width = 0.35
    axes[0].bar(x - width/2, gpu_pass.get("H100", [0]*len(x)), width, label='H100', color='#3498db')
    axes[0].bar(x + width/2, gpu_pass.get("B200", [0]*len(x)), width, label='B200', color='#e74c3c')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(gpu_pass.index, rotation=45, ha='right')
    axes[0].set_ylabel("Pass Rate")
    axes[0].set_title("Pass Rate: H100 vs B200")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Speedup by GPU
    gpu_speedup = df[df["correct"]].groupby(["model", "gpu"])["speedup"].mean().unstack()
    gpu_speedup = gpu_speedup.reindex(model_stats.index)
    axes[1].bar(x - width/2, gpu_speedup.get("H100", [0]*len(x)), width, label='H100', color='#3498db')
    axes[1].bar(x + width/2, gpu_speedup.get("B200", [0]*len(x)), width, label='B200', color='#e74c3c')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gpu_speedup.index, rotation=45, ha='right')
    axes[1].set_ylabel("Avg Speedup (when correct)")
    axes[1].set_title("Speedup: H100 vs B200")
    axes[1].legend()
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(viz_dir / "gpu_comparison.png", dpi=150)
    plt.close()

    # 8. Category performance
    fig, ax = plt.subplots(figsize=(12, 6))
    cat_stats = df.groupby("problem_category").agg({
        "passed": "mean",
        "beats_baseline": "mean"
    }).sort_values("passed", ascending=False)

    x = np.arange(len(cat_stats.index))
    width = 0.35
    ax.bar(x - width/2, cat_stats["passed"], width, label='Pass Rate', color='#3498db')
    ax.bar(x + width/2, cat_stats["beats_baseline"], width, label='Beats Baseline', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_stats.index, rotation=45, ha='right')
    ax.set_ylabel("Rate")
    ax.set_title("Performance by Problem Category")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(viz_dir / "category_performance.png", dpi=150)
    plt.close()

    print(f"Created {len(list(viz_dir.glob('*.png')))} visualizations in {viz_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write per-run aggregate CSV to this path (e.g., results_summary.csv)",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/aggregated",
                       help="Output directory for CSV and visualizations")
    parser.add_argument("--batch-eval-dir", type=str, default="outputs/batch_eval",
                       help="Directory containing batch_eval runs")
    parser.add_argument("--no-viz", action="store_true",
                       help="Skip visualization generation")
    args = parser.parse_args()

    batch_eval_dir = Path(args.batch_eval_dir)
    if args.output:
        rows = aggregate_runs(batch_eval_dir)
        output_path = Path(args.output)
        write_aggregate_csv(rows, output_path)
        print(f"Wrote {len(rows)} aggregated run rows to {output_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {batch_eval_dir}...")
    results = load_all_results(batch_eval_dir)
    print(f"Loaded {len(results)} results")

    print("Building DataFrame with all columns...")
    df = build_dataframe(results)
    print(f"DataFrame shape: {df.shape}")

    # Export full CSV
    csv_path = output_dir / "benchmark_results_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"Exported full CSV: {csv_path}")

    # Export compact CSV (most useful columns only)
    compact_cols = [
        "model", "gpu", "level", "problem_name", "problem_category",
        "compiled", "correct", "passed", "speedup", "beats_baseline",
        "input_tokens", "output_tokens", "total_tokens", "turns",
        "estimated_cost_usd", "ref_ms", "sol_ms"
    ]
    compact_df = df[[c for c in compact_cols if c in df.columns]]
    compact_csv_path = output_dir / "benchmark_results_compact.csv"
    compact_df.to_csv(compact_csv_path, index=False)
    print(f"Exported compact CSV: {compact_csv_path}")

    # Compute and export summary
    print("Computing summary statistics...")
    summary = compute_summary_stats(df)
    summary_path = output_dir / "summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Exported summary: {summary_path}")

    # Print key stats
    print("\n" + "="*60)
    print("KEY STATISTICS")
    print("="*60)
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Models: {summary['unique_models']}")
    print(f"GPUs: {summary['unique_gpus']}")
    print(f"Problems: {summary['unique_problems']}")
    print(f"\nOverall pass rate: {summary['overall']['pass_rate']:.1%}")
    print(f"Beats baseline rate: {summary['overall']['beats_baseline_rate']:.1%}")
    print(f"Avg speedup (when correct): {summary['overall']['avg_speedup_when_correct']:.2f}x")
    print(f"Total tokens: {summary['overall']['total_tokens']:,}")
    print(f"Total cost: ${summary['overall']['total_cost_usd']:.2f}")

    print("\nBy Model (sorted by pass rate):")
    sorted_models = sorted(summary['by_model'].items(),
                          key=lambda x: x[1]['pass_rate'], reverse=True)
    for model, stats in sorted_models:
        print(f"  {model}: {stats['pass_rate']:.1%} pass, ${stats['total_cost_usd']:.2f}")

    # Generate visualizations
    if not args.no_viz:
        print("\nGenerating visualizations...")
        create_visualizations(df, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
