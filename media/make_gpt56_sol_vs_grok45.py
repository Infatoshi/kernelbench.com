"""GPT-5.6 Sol (xhigh) vs Grok 4.5 (max) on KernelBench-Hard, RTX PRO 6000.

Two-model engagement chart. Sol Top-k / Sonic stay as hatched reward-hack
markers (not scored).

    uv run python media/make_gpt56_sol_vs_grok45.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, SERIES, apply, tight_square

apply()

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "benchmarks/hard/results/leaderboard.json"
OUT = Path(__file__).resolve().parent / "gpt56_sol_vs_grok45.png"

PROBLEMS = [
    ("01_fp8_gemm", "FP8 GEMM"),
    ("02_kda_cutlass", "KDA"),
    ("03_paged_attention", "Paged attention"),
    ("05_topk_bitonic", "Top-k"),
    ("06_sonic_moe_swiglu", "Sonic MoE"),
    ("07_w4a16_gemm", "W4A16 GEMM"),
]

ROWS = [
    ("Grok 4.5", "grok", "grok-4.5"),
    ("GPT-5.6 Sol", "codex", "gpt-5.6-sol"),
]

COLORS = {
    "Grok 4.5": SERIES[1],
    "GPT-5.6 Sol": C["accent"],
}

# Audited Sol rejects — not leaderboard scores.
SOL_HACK = {"05_topk_bitonic", "06_sonic_moe_swiglu"}


def load_rows() -> dict[tuple[str, str], dict]:
    data = json.loads(BOARD.read_text())
    return {
        (row["harness"], row["model"]): row.get("results", {})
        for row in data["models"]
    }


def peak_fraction(results: dict, problem: str) -> float | None:
    cell = results.get(problem)
    if not isinstance(cell, dict) or cell.get("correct") is False:
        return None
    value = cell.get("peak_fraction")
    return float(value) if value is not None else None


def cell_for(rows, label, harness, model, problem):
    if label == "GPT-5.6 Sol" and problem in SOL_HACK:
        return (None, "hack")
    value = peak_fraction(rows.get((harness, model), {}), problem)
    if value is None:
        return (None, "missing")
    return (value, "clean")


def draw_panel(ax, title, cells):
    y = np.arange(len(ROWS))
    present = [v for v, k in cells if v is not None and k == "clean"]
    xmax = max(present, default=0.05)
    xmax = min(1.0, max(0.12, xmax * 1.22))

    for index, ((label, _, _), (value, kind)) in enumerate(zip(ROWS, cells)):
        if kind == "hack":
            ax.barh(
                index,
                xmax * 0.14,
                height=0.55,
                color=C["bad"],
                edgecolor=C["bg"],
                hatch="////",
                alpha=0.9,
                zorder=3,
            )
            ax.text(
                xmax * 0.16,
                index,
                "hack",
                va="center",
                color=C["bad"],
                fontsize=9,
                fontweight="bold",
            )
            continue
        if value is None:
            ax.text(0.01, index, "—", va="center", color=C["fg_dim"], fontsize=10)
            continue
        ax.barh(
            index,
            value,
            height=0.55,
            color=COLORS[label],
            edgecolor=C["bg"],
            zorder=3,
        )
        ax.text(
            min(value + xmax * 0.02, xmax * 0.95),
            index,
            f"{value:.1%}",
            va="center",
            color=C["fg"],
            fontsize=9,
            fontweight="bold" if label == "GPT-5.6 Sol" else "normal",
        )

    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=6)
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, xmax, 4))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", alpha=0.55, zorder=0)


def main() -> None:
    rows = load_rows()
    fig, axes = tight_square(nrows=3, ncols=2, size=10.0)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.08, hspace=0.36, wspace=0.16)

    for ax, (problem, title) in zip(axes.flat, PROBLEMS):
        cells = [cell_for(rows, *row, problem) for row in ROWS]
        draw_panel(ax, title, cells)

    handles = [
        Patch(facecolor=COLORS["Grok 4.5"], label="Grok 4.5 (max)"),
        Patch(facecolor=COLORS["GPT-5.6 Sol"], label="GPT-5.6 Sol (xhigh)"),
        Patch(facecolor=C["bad"], hatch="////", edgecolor=C["bg"], label="reward hack"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.52, 0.975),
        frameon=True,
        facecolor=C["surface"],
        edgecolor=C["border"],
        labelcolor=C["fg"],
        fontsize=10,
        framealpha=0.96,
    )
    fig.text(
        0.5,
        0.022,
        "KernelBench-Hard · RTX PRO 6000 Blackwell · fraction of hardware roofline ↑ · clean audited only",
        ha="center",
        color=C["fg_muted"],
        fontsize=9,
    )
    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
