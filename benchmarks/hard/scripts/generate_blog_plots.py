"""Generate matplotlib plots for the kernelbench-hard blog post.

Outputs PNGs into public/blog-hard/ in the kernelbench.com monorepo.
Style: dark phosphor theme to match the site.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Repo paths — script lives in benchmarks/hard/scripts/, plots target public/blog-hard/
REPO_ROOT = Path(__file__).resolve().parents[3]
LB_PATH = REPO_ROOT / "benchmarks/hard/results/leaderboard.json"
OUT_DIR = REPO_ROOT / "public/blog-hard"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# kernelbench.com palette
COL_BG = "#0a0d0a"
COL_BG_SOFT = "#0d110d"
COL_FG = "#4ade80"
COL_FG_BRIGHT = "#86efac"
COL_FG_DIM = "#166534"
COL_FG_MUTED = "#15803d"
COL_BORDER = "#14532d"
COL_ACCENT = "#fbbf24"
COL_WARN = "#fb923c"
COL_BAD = "#f87171"
COL_BLUE = "#60a5fa"

mpl.rcParams.update({
    "figure.facecolor": COL_BG,
    "axes.facecolor": COL_BG_SOFT,
    "savefig.facecolor": COL_BG,
    "axes.edgecolor": COL_BORDER,
    "axes.labelcolor": COL_FG,
    "axes.titlecolor": COL_FG_BRIGHT,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "xtick.color": COL_FG_MUTED,
    "ytick.color": COL_FG_MUTED,
    "text.color": COL_FG,
    "grid.color": COL_BORDER,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "font.family": "monospace",
    "font.size": 10,
})


def short(label: str) -> str:
    return (label
            .replace("opencode/openrouter-pinned/", "or/")
            .replace("opencode/", "")
            .replace("codex/", "")
            .replace("claude/", "")
            .replace("kimi/", ""))


def main() -> None:
    if not LB_PATH.exists():
        sys.exit(f"missing {LB_PATH}")
    lb = json.loads(LB_PATH.read_text())

    PROBLEMS = lb["problems"]
    SHORT_PROB = {
        "01_fp8_gemm": "01 fp8",
        "02_kda_cutlass": "02 kda",
        "03_paged_attention": "03 paged",
        "04_kahan_softmax": "04 kahan",
        "05_topk_bitonic": "05 topk",
        "06_sonic_moe_swiglu": "06 moe",
        "07_w4a16_gemm": "07 w4a16",
    }

    models = lb["models"]
    model_labels = [short(m["label"]) for m in models]

    # Build matrix: rows=models, cols=problems, entries=peak_fraction or NaN
    grid = np.full((len(models), len(PROBLEMS)), np.nan)
    for i, m in enumerate(models):
        for j, p in enumerate(PROBLEMS):
            cell = m["results"].get(p, {})
            if cell.get("correct") and cell.get("peak_fraction") is not None:
                grid[i, j] = cell["peak_fraction"]

    plot_heatmap(grid, model_labels, [SHORT_PROB[p] for p in PROBLEMS], OUT_DIR / "leaderboard_heatmap.png")
    plot_pass_count(models, model_labels, len(PROBLEMS), OUT_DIR / "pass_count_by_model.png")
    plot_fp8_cluster(models, OUT_DIR / "fp8_gemm_cluster.png")
    plot_kahan_inversion(models, OUT_DIR / "kahan_inversion.png")
    plot_top_peaks_per_problem(lb, [SHORT_PROB[p] for p in PROBLEMS], OUT_DIR / "best_peak_per_problem.png")


def plot_heatmap(grid, model_labels, problem_labels, out_path):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    # Use a custom colormap from dark to bright phosphor
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "phosphor", [COL_BG_SOFT, COL_FG_DIM, COL_FG, COL_FG_BRIGHT, COL_ACCENT]
    )
    cmap.set_bad(color="#1a0e0e", alpha=1.0)
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0.0, vmax=0.65)

    ax.set_xticks(range(len(problem_labels)))
    ax.set_xticklabels(problem_labels, rotation=0)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)
    ax.set_title("peak_fraction across (model, problem). dark = ERR/FAIL, brighter = closer to hardware ceiling.", loc="left")

    # Cell annotations
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", color=COL_FG_DIM, fontsize=8)
            else:
                # Bright text on dark cells, dark text on bright cells
                txt_color = COL_BG if v > 0.30 else COL_FG_BRIGHT
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=txt_color, fontsize=8, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label("peak_fraction", color=COL_FG_MUTED)
    cbar.ax.yaxis.set_tick_params(color=COL_FG_MUTED)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COL_FG_MUTED)
    cbar.outline.set_edgecolor(COL_BORDER)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def plot_pass_count(models, model_labels, n_problems, out_path):
    pass_counts = [m["pass_count"] for m in models]
    # sort by pass count desc
    order = np.argsort(pass_counts)
    pass_counts = [pass_counts[i] for i in order]
    labels = [model_labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, pass_counts, color=COL_FG, edgecolor=COL_FG_BRIGHT, linewidth=1.0)

    # Highlight 7/7 in accent
    for i, v in enumerate(pass_counts):
        if v == n_problems:
            bars[i].set_color(COL_ACCENT)
            bars[i].set_edgecolor(COL_ACCENT)

    ax.set_xlim(0, n_problems + 0.5)
    ax.set_xlabel(f"problems passed (of {n_problems})")
    ax.set_title("pass count by model", loc="left")
    ax.set_xticks(range(n_problems + 1))
    ax.grid(axis="x", alpha=0.3)

    # Annotate counts
    for i, v in enumerate(pass_counts):
        ax.text(v + 0.12, i, f"{v}/{n_problems}", va="center", color=COL_FG_BRIGHT, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def plot_fp8_cluster(models, out_path):
    """Bar chart of all peak ≥ 0.10 cells on 01_fp8_gemm — the bf16 dressup cluster."""
    rows = []
    for m in models:
        cell = m["results"].get("01_fp8_gemm", {})
        if cell.get("correct") and (cell.get("peak_fraction") or 0) >= 0.10:
            lbl = short(m["label"])
            rows.append((lbl, cell["peak_fraction"]))
    rows.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [COL_WARN] * len(rows)
    bars = ax.bar(labels, values, color=colors, edgecolor=COL_BAD, linewidth=1.0)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("peak_fraction (vs SM120 fp8 tensor-core peak)")
    ax.set_title(
        "01 fp8_gemm: every high-peak solution is bf16 in disguise.\n"
        "all 5 cast fp8 → bf16 inside the kernel; opus + gpt-5.5 pin to arch::Sm80 (Ampere CUTLASS).",
        loc="left", fontsize=10, color=COL_FG_BRIGHT,
    )
    plt.xticks(rotation=18, ha="right", color=COL_FG_MUTED)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", color=COL_WARN, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def plot_kahan_inversion(models, out_path):
    """The Kahan softmax story: deepseek-v4-pro at 0.101 (Kahan) vs everyone else higher (naive)."""
    rows = []
    for m in models:
        cell = m["results"].get("04_kahan_softmax", {})
        if cell.get("correct") and cell.get("peak_fraction") is not None:
            lbl = short(m["label"])
            rows.append((lbl, cell["peak_fraction"]))
    rows.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    # All naive in warn color, Kahan implementation in green (clean)
    colors = []
    for lbl in labels:
        if "deepseek-v4-pro" in lbl:
            colors.append(COL_FG_BRIGHT)
        else:
            colors.append(COL_WARN)
    bars = ax.bar(labels, values, color=colors, edgecolor=COL_BORDER, linewidth=0.8)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("peak_fraction (vs DRAM bandwidth)")
    ax.set_title(
        "04 kahan_softmax: model that implemented Kahan compensation scored *lowest* of the seven passes.\n"
        "green = actually implemented Kahan compensated summation. orange = skipped it for naive softmax.",
        loc="left", fontsize=10, color=COL_FG_BRIGHT,
    )
    plt.xticks(rotation=18, ha="right", color=COL_FG_MUTED)
    ax.grid(axis="y", alpha=0.3)
    for b, v, c in zip(bars, values, colors):
        ax.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.3f}",
                ha="center", color=c, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


def plot_top_peaks_per_problem(lb, problem_short_labels, out_path):
    """For each problem: top peak across models (highlights which problems are hard)."""
    problems = lb["problems"]
    bests = []
    for p in problems:
        pp = lb["per_problem"][p]
        bests.append(pp["best_peak_fraction"] or 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(problem_short_labels, bests, color=COL_FG, edgecolor=COL_FG_BRIGHT, linewidth=1.0)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("best peak_fraction (any model)")
    ax.set_title(
        "best peak per problem. 02 kda + 05 topk barely break 5% — the deck's hardest problems.",
        loc="left", fontsize=10, color=COL_FG_BRIGHT,
    )
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, bests):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", color=COL_FG_BRIGHT, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
