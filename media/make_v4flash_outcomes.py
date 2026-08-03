"""Audited DeepSeek V4 Flash 0731 outcomes across Hard, CUDA, and Mega.

Counts use the final isolated run per problem/GPU. Green is publishable clean;
rose is reward-hack exclusion; amber is a genuine correctness bug. The Hard
cells are the 12 published or-fable rows. CUDA/Mega verdicts are the manual
2026-08-03 annotations under benchmarks/{cuda,mega}/results/annotations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply

apply()

OUT = Path(__file__).resolve().parent / "v4flash_outcomes.png"
BENCHES = ["Hard", "CUDA", "Mega"]
PANELS = [
    (
        "RTX PRO 6000",
        {
            "clean": [6, 4, 0],
            "reward hack": [0, 0, 1],
            "kernel bug": [0, 0, 0],
        },
    ),
    (
        "H100",
        {
            "clean": [6, 3, 0],
            "reward hack": [0, 1, 0],
            "kernel bug": [0, 0, 1],
        },
    ),
]
COLORS = {
    "clean": C["accent"],
    "reward hack": C["bad"],
    "kernel bug": C["warn"],
}


def draw_panel(ax: plt.Axes, gpu: str, data: dict[str, list[int]]) -> None:
    x = np.arange(len(BENCHES))
    bottom = np.zeros(len(BENCHES))
    for status in ("clean", "reward hack", "kernel bug"):
        values = np.asarray(data[status], dtype=float)
        bars = ax.bar(x, values, bottom=bottom, width=0.62, color=COLORS[status])
        for bar, value, base in zip(bars, values, bottom, strict=True):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    color=C["bg"],
                    fontsize=12,
                    fontweight="bold",
                )
        bottom += values

    ax.set_xticks(x, BENCHES)
    ax.set_ylim(0, 6.5)
    ax.set_yticks(range(0, 7))
    ax.set_ylabel("audited cells")
    ax.text(
        0.02,
        0.96,
        gpu,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=C["fg"],
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.42)
    ax.grid(axis="x", visible=False)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 10.5))
    fig.patch.set_facecolor(C["bg"])
    for ax, (gpu, data) in zip(axes, PANELS, strict=True):
        draw_panel(ax, gpu, data)

    fig.legend(
        handles=[Patch(facecolor=COLORS[name], label=name) for name in COLORS],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.91, wspace=0.22)
    fig.savefig(OUT, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
