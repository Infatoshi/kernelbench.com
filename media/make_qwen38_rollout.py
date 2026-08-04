"""Qwen 3.8 Max clean KernelBench cells versus each board leader.

Visual-first engagement chart: square, bars + axes + compact legend only.
Audit-rejected, contaminated, failed, and TopK cells are deliberately omitted;
the post copy explains the full 22-cell outcome. TopK uses milliseconds rather
than a roofline headline, so it is not mixed into this normalized score chart.

    uv run --project benchmarks/hard python media/make_qwen38_rollout.py
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
from kbh_theme import C, apply, tight_square

apply()

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "qwen38_rollout.png"
MODEL = "qwen/qwen3.8-max"
POINTS = [
    ("RTX / Sonic MoE", "hard", "leaderboard.json", "06_sonic_moe_swiglu"),
    ("RTX / Paged Attention", "hard", "leaderboard.json", "03_paged_attention"),
    ("RTX / KDA CUTLASS", "hard", "leaderboard.json", "02_kda_cutlass"),
    ("H100 / Paged Attention", "hard", "leaderboard.h100.json", "03_paged_attention"),
    ("RTX / Grid + MinGRU", "cuda", "leaderboard.json", "04_grid_mingru_sps"),
]


def load_point(bench: str, board_name: str, problem: str) -> tuple[float, float]:
    path = ROOT / "benchmarks" / bench / "results" / board_name
    board = json.loads(path.read_text())
    row = next(m for m in board["models"] if m.get("model") == MODEL)
    cell = row["results"][problem]
    if not cell.get("correct") or cell.get("invalid_reason"):
        raise ValueError(f"{path.name}:{problem} is not a clean passing cell")
    score = float(cell["peak_fraction"])
    best = float(board["per_problem"][problem]["best_peak_fraction"])
    return score, best


def main() -> None:
    labels: list[str] = []
    ratios: list[float] = []
    raw: list[tuple[float, float]] = []
    for label, bench, board_name, problem in POINTS:
        score, best = load_point(bench, board_name, problem)
        labels.append(label)
        ratios.append(100.0 * score / best)
        raw.append((score, best))

    y = np.arange(len(labels))
    fig, ax = tight_square(size=9.5)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.96, bottom=0.12)

    ax.barh(y, [100.0] * len(y), height=0.58, color=C["surface_muted"], zorder=1)
    bars = ax.barh(y, ratios, height=0.58, color=C["accent"], zorder=3)
    for bar, ratio in zip(bars, ratios):
        ax.text(
            min(ratio + 1.5, 96.0),
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.0f}%",
            va="center",
            ha="left" if ratio < 94 else "right",
            color=C["fg_bright"],
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of clean board best (%)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.55, zorder=0)
    ax.grid(False, axis="y")
    ax.legend(
        handles=[
            Patch(facecolor=C["accent"], label="Qwen 3.8 Max"),
            Patch(facecolor=C["surface_muted"], label="clean board best"),
        ],
        loc="lower right",
        frameon=True,
        facecolor=C["surface"],
        edgecolor=C["border"],
        labelcolor=C["fg"],
        framealpha=0.95,
        fontsize=9,
    )

    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")
    for label, ratio, (score, best) in zip(labels, ratios, raw):
        print(f"{label:28s} qwen={score:.4f} best={best:.4f} share={ratio:.1f}%")


if __name__ == "__main__":
    main()
