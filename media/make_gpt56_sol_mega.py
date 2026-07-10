"""GPT-5.6 Sol vs frontier on KernelBench-Mega Kimi-Linear decode (RTX PRO 6000).

    uv run python media/make_gpt56_sol_mega.py
"""
from __future__ import annotations

import csv
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
CSV = ROOT / "public/data/mega/results.csv"
OUT = Path(__file__).resolve().parent / "gpt56_sol_mega.png"

GPU = "RTX PRO 6000 Blackwell"
PROBLEM = "02_kimi_linear_decode"

# Subject set that actually has RTX decode cells on the published mega CSV.
ROWS = [
    ("Fable 5", "claude-fable-5"),
    ("Opus 4.8", "claude-opus-4-8"),
    ("GLM-5.2", "glm-5.2"),
    ("GPT-5.6 Sol", "gpt-5.6-sol"),
]

COLORS = {
    "Fable 5": SERIES[2],
    "Opus 4.8": SERIES[4],
    "GLM-5.2": SERIES[3],
    "GPT-5.6 Sol": C["accent"],
}


def load() -> dict[str, float]:
    out: dict[str, float] = {}
    with CSV.open() as f:
        for row in csv.DictReader(f):
            if (
                row["gpu"] != GPU
                or row["problem"] != PROBLEM
                or row["correct"] != "true"
                or not row.get("score")
            ):
                continue
            out[row["model"]] = float(row["score"])
    return out


def main() -> None:
    data = load()
    values = [data.get(model) for _, model in ROWS]
    if any(v is None for v in values):
        missing = [label for (label, model), v in zip(ROWS, values) if v is None]
        raise SystemExit(f"missing mega cells: {missing}")

    # Sort high → low for feed readability.
    order = sorted(range(len(ROWS)), key=lambda i: values[i], reverse=True)
    labels = [ROWS[i][0] for i in order]
    vals = [values[i] for i in order]

    fig, ax = tight_square(size=9.0)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.12)

    y = np.arange(len(labels))
    xmax = max(vals) * 1.15
    for i, (label, v) in enumerate(zip(labels, vals)):
        ax.barh(
            i,
            v,
            height=0.62,
            color=COLORS[label],
            edgecolor=C["bg"],
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            min(v + xmax * 0.02, xmax * 0.96),
            i,
            f"{v:.2f}x",
            va="center",
            ha="left",
            color=C["fg"],
            fontsize=11,
            fontweight="bold" if label == "GPT-5.6 Sol" else "normal",
            zorder=4,
        )

    ax.axvline(1.0, color=C["bad"], lw=0.9, ls="--", alpha=0.75, zorder=2)
    ax.text(1.05, len(labels) - 0.45, "baseline", color=C["bad"], fontsize=8, va="top")

    ax.set_xlim(0, xmax)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("decode speedup over optimized PyTorch ↑", fontsize=10, color=C["fg_muted"])
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, axis="x", alpha=0.55, zorder=0)

    handles = [Patch(facecolor=COLORS[label], label=label) for label, _ in ROWS]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.52, 0.975),
        frameon=True,
        facecolor=C["surface"],
        edgecolor=C["border"],
        labelcolor=C["fg"],
        fontsize=9,
        framealpha=0.96,
        borderpad=0.5,
        handlelength=1.2,
        columnspacing=1.1,
    )
    fig.text(
        0.5,
        0.03,
        "KernelBench-Mega · Kimi-Linear W4A16 decode · RTX PRO 6000 · clean audited",
        ha="center",
        va="bottom",
        color=C["fg_muted"],
        fontsize=9,
    )

    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")
    for label, v in zip(labels, vals):
        print(f"{label:14s} {v:.3f}x")


if __name__ == "__main__":
    main()
