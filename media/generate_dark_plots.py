#!/usr/bin/env python3
"""Regenerate public matplotlib figures with the site's dark visual theme."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"

BG = "#050806"
PANEL = "#0b100c"
GRID = "#223026"
TEXT = "#d8f3dc"
MUTED = "#8aa391"
GREEN = "#5ee787"
AMBER = "#f2c94c"
BLUE = "#61afef"
RED = "#ff6b6b"
PURPLE = "#c792ea"
CYAN = "#56b6c2"


def apply_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "legend.facecolor": PANEL,
            "legend.edgecolor": GRID,
            "legend.labelcolor": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def short_model(name: str) -> str:
    replacements = {
        "Gemini 3 Flash Preview": "Gemini 3 Flash",
        "Claude Opus 4.6": "Opus 4.6",
        "Claude Sonnet 4.6": "Sonnet 4.6",
        "Qwen3.5-397B": "Qwen3.5 397B",
        "Qwen: Qwen3.5-122B-A10B": "Qwen3.5 122B",
    }
    return replacements.get(name, name)


def read_v3_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (PUBLIC / "data/v3/results.csv").open() as f:
        for row in csv.DictReader(f):
            row["compiled"] = row["compiled"].lower() == "true"
            row["correct"] = row["correct"].lower() == "true"
            row["level"] = int(row["level"] or 0)
            row["speedup"] = float(row["speedup"]) if row["speedup"] else None
            row["estimated_cost_usd"] = (
                float(row["estimated_cost_usd"]) if row["estimated_cost_usd"] else 0.0
            )
            row["beats_baseline"] = bool(row["correct"] and (row["speedup"] or 0) >= 1.0)
            rows.append(row)
    return rows


def group(rows: list[dict[str, object]], key: str) -> dict[object, list[dict[str, object]]]:
    out: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out


def rate(rows: list[dict[str, object]], key: str) -> float:
    return sum(bool(r[key]) for r in rows) / len(rows) if rows else 0.0


def v3_outputs() -> list[Path]:
    return [
        PUBLIC / base / name
        for base in ("v3", "blog-v3")
        for name in (
            "results_overall.png",
            "results_heatmap.png",
            "speedup_distribution.png",
            "level_breakdown.png",
            "cost_vs_accuracy.png",
            "compilation_funnel.png",
        )
    ]


def write_v3_figures() -> None:
    rows = read_v3_rows()
    by_model = group(rows, "model")
    model_order = sorted(
        by_model,
        key=lambda m: rate(by_model[m], "correct"),
        reverse=True,
    )
    labels = [short_model(str(m)) for m in model_order]
    x = np.arange(len(model_order))

    compiled = [rate(by_model[m], "compiled") for m in model_order]
    correct = [rate(by_model[m], "correct") for m in model_order]
    beats = [rate(by_model[m], "beats_baseline") for m in model_order]

    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.25
    ax.bar(x - width, compiled, width, label="compiled", color=BLUE)
    ax.bar(x, correct, width, label="correct", color=GREEN)
    ax.bar(x + width, beats, width, label="beats baseline", color=AMBER)
    ax.set_title("KernelBench v3 results")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y")
    ax.legend(loc="upper right")
    for i, value in enumerate(correct):
        ax.text(i, value + 0.025, pct(value), ha="center", fontsize=8, color=TEXT)
    save_to_v3_pair(fig, "results_overall.png")

    levels = sorted({int(r["level"]) for r in rows})
    heat = np.array(
        [
            [rate([r for r in by_model[m] if r["level"] == level], "correct") for level in levels]
            for m in model_order
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(heat, vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Correctness by model and level")
    ax.set_xticks(range(len(levels)), [f"L{level}" for level in levels])
    ax.set_yticks(range(len(labels)), labels)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, pct(float(heat[i, j])), ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("correct rate")
    save_to_v3_pair(fig, "results_heatmap.png")

    fig, ax = plt.subplots(figsize=(14, 7))
    rng = np.random.default_rng(7)
    for i, m in enumerate(model_order):
        speedups = [
            float(r["speedup"])
            for r in by_model[m]
            if r["correct"] and r["speedup"] is not None and float(r["speedup"]) > 0
        ]
        if not speedups:
            continue
        jitter = rng.normal(0, 0.05, len(speedups))
        ax.scatter(np.full(len(speedups), i) + jitter, speedups, s=24, alpha=0.65, color=GREEN)
    ax.axhline(1.0, color=AMBER, linestyle="--", linewidth=1.5, label="PyTorch baseline")
    ax.set_yscale("log")
    ax.set_ylabel("speedup vs baseline, log scale")
    ax.set_title("Speedup distribution for correct kernels")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y")
    ax.legend()
    save_to_v3_pair(fig, "speedup_distribution.png")

    by_level = group(rows, "level")
    lx = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        lx - width,
        [rate(by_level[level], "compiled") for level in levels],
        width,
        label="compiled",
        color=BLUE,
    )
    ax.bar(
        lx,
        [rate(by_level[level], "correct") for level in levels],
        width,
        label="correct",
        color=GREEN,
    )
    ax.bar(
        lx + width,
        [rate(by_level[level], "beats_baseline") for level in levels],
        width,
        label="beats baseline",
        color=AMBER,
    )
    ax.set_title("Performance by difficulty level")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(lx, [f"L{level}" for level in levels])
    ax.grid(axis="y")
    ax.legend()
    save_to_v3_pair(fig, "level_breakdown.png")

    fig, ax = plt.subplots(figsize=(11, 7))
    for m in model_order:
        rs = by_model[m]
        cost = sum(float(r["estimated_cost_usd"]) for r in rs)
        acc = rate(rs, "correct")
        bb = rate(rs, "beats_baseline")
        ax.scatter(cost, acc, s=90 + 480 * bb, color=GREEN if acc >= 0.5 else AMBER, alpha=0.8)
        ax.text(cost, acc, " " + short_model(str(m)), va="center", fontsize=8)
    ax.set_title("Cost vs correctness")
    ax.set_xlabel("total estimated API cost, USD")
    ax.set_ylabel("correct rate")
    ax.set_ylim(0, 1)
    ax.grid(True)
    save_to_v3_pair(fig, "cost_vs_accuracy.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    funnel_names = ["attempted", "compiled", "correct", "beats baseline"]
    funnel_values = [
        len(rows),
        sum(bool(r["compiled"]) for r in rows),
        sum(bool(r["correct"]) for r in rows),
        sum(bool(r["beats_baseline"]) for r in rows),
    ]
    colors = [MUTED, BLUE, GREEN, AMBER]
    ax.barh(funnel_names, funnel_values, color=colors)
    ax.invert_yaxis()
    ax.set_title("Compilation funnel")
    ax.set_xlabel("evaluations")
    ax.grid(axis="x")
    for i, value in enumerate(funnel_values):
        ax.text(value + max(funnel_values) * 0.015, i, f"{value:,}", va="center")
    save_to_v3_pair(fig, "compilation_funnel.png")


def save_to_v3_pair(fig: plt.Figure, name: str) -> None:
    for base in ("v3", "blog-v3"):
        fig.savefig(PUBLIC / base / name, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def read_leaderboard() -> dict:
    with (ROOT / "benchmarks/hard/results/leaderboard.json").open() as f:
        return json.load(f)


def hard_label(label: str) -> str:
    for old, new in (
        ("codex/gpt-5.5 [2026-05-28 finish xhigh]", "GPT-5.5"),
        ("claude/claude-opus-4-7 [2026-05-28 finish max]", "Opus 4.7"),
        ("claude/claude-opus-4-8 [2026-05-28 opus48-grok max]", "Opus 4.8"),
        ("claude/claude-opus-4-6 [2026-06-04 opus46 max]", "Opus 4.6"),
        ("cursor/composer-2.5-fast [2026-05-28 finish]", "Composer 2.5"),
        ("gemini/gemini-3.5-flash [2026-05-28 finish]", "Gemini 3.5"),
        ("grok/grok-build [2026-05-28 opus48-grok max]", "Grok Build"),
        ("kimi/kimi-k2.6", "Kimi K2.6"),
        ("opencode/openrouter-pinned/xiaomi/mimo-v2.5-pro", "MiMo v2.5"),
        ("opencode/deepseek/deepseek-v4-flash", "DeepSeek Flash"),
        ("opencode/deepseek/deepseek-v4-pro", "DeepSeek Pro"),
        ("zai-claude/glm-5.1 [2026-05-13]", "GLM-5.1 claude"),
        ("droid/zai/glm-5.1 [2026-05-08]", "GLM-5.1 droid"),
        ("opencode/zai/glm-5.1", "GLM-5.1 open"),
        ("minimax-claude/MiniMax-M3 [2026-06-01]", "MiniMax M3"),
    ):
        if label == old or label.startswith(old):
            return new
    return label.split("/")[-1].split(" [")[0]


def visible_hard_models(lb: dict) -> list[dict]:
    wanted = [
        "codex/gpt-5.5 [2026-05-28 finish xhigh]",
        "claude/claude-opus-4-7 [2026-05-28 finish max]",
        "kimi/kimi-k2.6",
        "opencode/openrouter-pinned/xiaomi/mimo-v2.5-pro",
        "opencode/deepseek/deepseek-v4-flash",
        "opencode/deepseek/deepseek-v4-pro",
        "opencode/zai/glm-5.1",
        "droid/zai/glm-5.1 [2026-05-08]",
        "zai-claude/glm-5.1 [2026-05-13]",
    ]
    prefixes = [
        "claude/claude-opus-4-8 [2026-05-28 opus48-grok",
        "claude/claude-opus-4-6 [2026-06-04 opus46",
        "cursor/composer-2.5-fast [2026-05-28 finish",
        "gemini/gemini-3.5-flash [2026-05-28 finish",
        "grok/grok-build [2026-05-28 opus48-grok",
        "minimax-claude/MiniMax-M3 [2026-06-01",
    ]
    return [
        m
        for m in lb["models"]
        if m["label"] in wanted or any(m["label"].startswith(p) for p in prefixes)
    ]


def write_hard_figures() -> None:
    lb = read_leaderboard()
    models = visible_hard_models(lb)
    problems = lb["problems"]
    labels = [hard_label(m["label"]) for m in models]
    matrix = np.array(
        [
            [
                (m["results"].get(p) or {}).get("peak_fraction")
                if (m["results"].get(p) or {}).get("correct")
                else np.nan
                for p in problems
            ]
            for m in models
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(13, 8))
    cmap = mpl.colormaps["viridis"].copy()
    cmap.set_bad("#182019")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=np.nanmax(matrix))
    ax.set_title("KernelBench Hard peak fraction heatmap")
    ax.set_xticks(range(len(problems)), [p.replace("_", "\n", 1) for p in problems], fontsize=8)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, "-" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("peak fraction")
    save(fig, PUBLIC / "blog-hard/leaderboard_heatmap.png")

    order = sorted(range(len(models)), key=lambda i: models[i]["pass_count"], reverse=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh([labels[i] for i in order], [models[i]["pass_count"] for i in order], color=GREEN)
    ax.invert_yaxis()
    ax.set_title("Passed cells by model")
    ax.set_xlabel("correct benchmark cells")
    ax.set_xlim(0, max(m["total_runs"] for m in models))
    ax.grid(axis="x")
    save(fig, PUBLIC / "blog-hard/pass_count_by_model.png")

    best = []
    for p in problems:
        vals = [float((m["results"].get(p) or {}).get("peak_fraction") or 0) for m in models]
        best.append(max(vals))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(problems)), best, color=[GREEN if v >= 0.5 else AMBER for v in best])
    ax.set_title("Best observed peak fraction per problem")
    ax.set_ylabel("best peak fraction")
    ax.set_xticks(range(len(problems)), [p[:2] for p in problems])
    ax.grid(axis="y")
    for i, value in enumerate(best):
        ax.text(i, value + 0.015, f"{value:.2f}", ha="center", fontsize=8)
    save(fig, PUBLIC / "blog-hard/best_peak_per_problem.png")

    fp8 = "01_fp8_gemm"
    fp8_vals = [
        (
            labels[i],
            float((m["results"].get(fp8) or {}).get("peak_fraction") or 0),
            bool((m["results"].get(fp8) or {}).get("correct")),
        )
        for i, m in enumerate(models)
    ]
    fp8_vals.sort(key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([v[0] for v in fp8_vals], [v[1] for v in fp8_vals], color=[GREEN if v[2] else RED for v in fp8_vals])
    ax.invert_yaxis()
    ax.set_title("FP8 GEMM cluster before stricter verifier")
    ax.set_xlabel("peak fraction")
    ax.grid(axis="x")
    save(fig, PUBLIC / "blog-hard/fp8_gemm_cluster.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.text(0.5, 0.65, "Kahan softmax retired", ha="center", va="center", fontsize=24, color=TEXT)
    ax.text(
        0.5,
        0.4,
        "Problem removed from the active Hard deck; legacy figure kept dark-themed for archival URLs.",
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED,
    )
    save(fig, PUBLIC / "blog-hard/kahan_inversion.png")


def read_fp8_rows() -> list[dict]:
    out: list[dict] = []
    for name in ("fixed-tolerance-summary.json", "recovery-smokes-summary.json"):
        with (PUBLIC / "blog-hard/fp8-constraint-rerun" / name).open() as f:
            out.extend(json.load(f)["runs"])
    return out


def fp8_name(row: dict) -> str:
    model = row["model"].split("/")[-1]
    return {
        "claude-opus-4-6": "Opus 4.6",
        "claude-opus-4-7": "Opus 4.7",
        "claude-opus-4-8": "Opus 4.8",
        "gpt-5.5": "GPT-5.5",
        "kimi-k2.6": "Kimi K2.6",
        "deepseek-v4-flash": "DS Flash",
        "deepseek-v4-pro": "DS Pro",
        "glm-5.1": "GLM-5.1",
    }.get(model, model)


def write_fp8_figures() -> None:
    rows = read_fp8_rows()
    labels = [fp8_name(r) for r in rows]
    output = [int(r.get("output_tokens") or 0) for r in rows]
    reasoning = [int(r.get("reasoning_tokens") or 0) for r in rows]
    cache = [int(r.get("cache_read_tokens") or 0) + int(r.get("cache_creation_tokens") or 0) for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x, cache, color=PURPLE, label="cache")
    ax.bar(x, reasoning, bottom=cache, color=BLUE, label="reasoning")
    ax.bar(x, output, bottom=np.array(cache) + np.array(reasoning), color=GREEN, label="output")
    ax.set_title("FP8 constraint rerun token burn")
    ax.set_ylabel("tokens")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y")
    ax.legend()
    save(fig, PUBLIC / "blog-hard/fp8-constraint-rerun/fp8_token_burn_stacked.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    totals = np.array(cache) + np.array(reasoning) + np.array(output)
    peak = [0 if not r.get("correct") else float(r.get("peak_fraction") or 0) for r in rows]
    ax.scatter(totals, peak, s=90, color=RED, alpha=0.85)
    for total, p, label in zip(totals, peak, labels, strict=False):
        ax.text(total, p + 0.01, label, fontsize=8)
    ax.set_title("Tokens vs effective FP8 peak")
    ax.set_xlabel("total visible/cache tokens")
    ax.set_ylabel("verified peak fraction")
    ax.set_ylim(-0.02, 1)
    ax.grid(True)
    save(fig, PUBLIC / "blog-hard/fp8-constraint-rerun/fp8_tokens_vs_effective_peak.png")

    fig, ax1 = plt.subplots(figsize=(13, 6))
    elapsed = [float(r.get("total_elapsed_seconds") or r.get("elapsed_seconds") or 0) / 60 for r in rows]
    cost = [r.get("total_cost_usd") or 0 for r in rows]
    ax1.bar(x - 0.18, elapsed, 0.36, color=CYAN, label="wall minutes")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, cost, 0.36, color=AMBER, label="cost USD")
    ax1.set_title("FP8 constraint rerun cost before outcome")
    ax1.set_ylabel("wall minutes")
    ax2.set_ylabel("cost USD")
    ax1.set_xticks(x, labels, rotation=35, ha="right")
    ax1.grid(axis="y")
    ax2.tick_params(colors=MUTED)
    ax2.yaxis.label.set_color(TEXT)
    lines, names = ax1.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, names + names2, loc="upper right")
    save(fig, PUBLIC / "blog-hard/fp8-constraint-rerun/fp8_cost_before_outcome.png")


def main() -> None:
    apply_theme()
    write_v3_figures()
    write_hard_figures()
    write_fp8_figures()


if __name__ == "__main__":
    main()
