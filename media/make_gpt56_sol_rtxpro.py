"""Grok 4.5 / Fable 5 / GLM-5.2 / GPT-5.6 Sol on RTX PRO 6000.

Six compact problem panels, one bar per model. Reads the published
KernelBench-Hard leaderboard. Sol Top-k / Sonic that were audited as
reward_hack stay on the chart as hatched rose "hack" markers (not scored).

    uv run python media/make_gpt56_sol_rtxpro.py
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
OUT = Path(__file__).resolve().parent / "gpt56_sol_rtxpro.png"

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
    ("Fable 5", "claude", "claude-fable-5"),
    ("GLM-5.2", "zai-claude", "glm-5.2"),
    ("GPT-5.6 Sol", "codex", "gpt-5.6-sol"),
]

COLORS = {
    "Grok 4.5": SERIES[1],
    "Fable 5": SERIES[2],
    "GLM-5.2": SERIES[3],
    "GPT-5.6 Sol": C["accent"],
}

# Clean Fable RTX fp8 not yet on the published allowlist row.
FABLE_FP8 = 0.4098

# Sol cells that passed the official checker but failed the manual audit.
# Do not treat these as leaderboard scores; chart them as rejected only.
SOL_HACK: dict[str, str] = {
    "05_topk_bitonic": "hack",  # Gaussian-tail TopK specialization
    "06_sonic_moe_swiglu": "hack",  # numeric-stress prefix detector
}


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


def cell_for(
    rows: dict[tuple[str, str], dict], label: str, harness: str, model: str, problem: str
) -> tuple[float | None, str]:
    """Return (value, kind) where kind is clean | hack | missing."""
    if label == "GPT-5.6 Sol" and problem in SOL_HACK:
        return (None, "hack")
    value = peak_fraction(rows.get((harness, model), {}), problem)
    if label == "Fable 5" and problem == "01_fp8_gemm" and value is None:
        value = FABLE_FP8
    if value is None:
        return (None, "missing")
    return (value, "clean")


def values_for(
    rows: dict[tuple[str, str], dict], problem: str
) -> list[tuple[float | None, str]]:
    return [cell_for(rows, label, harness, model, problem) for label, harness, model in ROWS]


def draw_panel(ax, title: str, cells: list[tuple[float | None, str]]) -> None:
    y = np.arange(len(ROWS))
    present = [value for value, kind in cells if value is not None and kind == "clean"]
    xmax = max(present, default=0.05)
    xmax = min(1.0, max(0.1, xmax * 1.22))

    for index, ((label, _, _), (value, kind)) in enumerate(zip(ROWS, cells)):
        if kind == "hack":
            ax.barh(
                index,
                xmax * 0.12,
                height=0.62,
                color=C["bad"],
                edgecolor=C["bg"],
                linewidth=0.5,
                hatch="////",
                alpha=0.85,
                zorder=3,
            )
            ax.text(
                xmax * 0.14,
                index,
                "hack",
                va="center",
                ha="left",
                color=C["bad"],
                fontsize=8,
                fontweight="bold",
                zorder=4,
            )
            continue
        if value is None:
            ax.text(
                0.008,
                index,
                "—",
                va="center",
                ha="left",
                color=C["fg_dim"],
                fontsize=9,
            )
            continue
        ax.barh(
            index,
            value,
            height=0.62,
            color=COLORS[label],
            edgecolor=C["bg"],
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            min(value + xmax * 0.018, xmax * 0.96),
            index,
            f"{value:.1%}",
            va="center",
            ha="left",
            color=C["fg"],
            fontsize=8,
            fontweight="bold" if label == "GPT-5.6 Sol" else "normal",
            zorder=4,
        )

    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=7)
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, xmax, 4))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, axis="x", alpha=0.55, zorder=0)


def main() -> None:
    rows = load_rows()
    fig, axes = tight_square(nrows=3, ncols=2, size=10.5)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.91, bottom=0.08, hspace=0.34, wspace=0.16)

    for ax, (problem, title) in zip(axes.flat, PROBLEMS):
        draw_panel(ax, title, values_for(rows, problem))

    handles = [Patch(facecolor=COLORS[label], label=label) for label, _, _ in ROWS]
    handles.append(
        Patch(facecolor=C["bad"], hatch="////", edgecolor=C["bg"], label="reward hack")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.52, 0.978),
        frameon=True,
        facecolor=C["surface"],
        edgecolor=C["border"],
        labelcolor=C["fg"],
        fontsize=9,
        framealpha=0.96,
        borderpad=0.5,
        handlelength=1.25,
        columnspacing=1.0,
    )
    fig.text(
        0.5,
        0.022,
        "RTX PRO 6000 Blackwell · fraction of hardware roofline ↑ · clean audited cells only",
        ha="center",
        va="bottom",
        color=C["fg_muted"],
        fontsize=9,
    )

    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")
    for problem, title in PROBLEMS:
        cells = values_for(rows, problem)
        rendered = " ".join(
            kind if value is None else f"{value:.4f}" for value, kind in cells
        )
        print(f"{title:16s} {rendered}")


if __name__ == "__main__":
    main()
