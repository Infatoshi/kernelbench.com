"""Grok 4.5 vs Fable 5 / Opus 4.8 / GLM-5.2 on KernelBench-Hard — 3 GPUs.

Visual-first engagement chart (see AGENTS.md + kbh_theme): square 1:1, three
stacked GPU panels, bars + axes + compact legend only — no title essay.

Models (subject set only — add GPT when the new Sol row is published):
  Grok 4.5, Claude Fable 5, Opus 4.8, GLM-5.2

Reads leaderboard.json / .h100.json / .b200.json. Missing cells omit the bar.
Fable RTX fp8: published allowlist is still 5/6; fill clean audited 0.4098
(20260703_001306) so the panel is complete.

  uv run python media/make_grok45_frontier.py
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
HARD = ROOT / "benchmarks/hard/results"
OUT = Path(__file__).resolve().parent / "grok45_frontier.png"

PROB = [
    "01_fp8_gemm",
    "02_kda_cutlass",
    "03_paged_attention",
    "05_topk_bitonic",
    "06_sonic_moe_swiglu",
    "07_w4a16_gemm",
]
PROB_LBL = ["fp8", "kda", "paged", "topk", "sonic", "w4a16"]

# Subject set for the Grok engagement post. GPT slot reserved for post-Sol release.
ROWS = [
    ("Grok 4.5", "grok", "grok-4.5"),
    ("Fable 5", "claude", "claude-fable-5"),
    ("Opus 4.8", "claude", "claude-opus-4-8"),
    ("GLM-5.2", "zai-claude", "glm-5.2"),
    # ("GPT-5.x", "codex", "gpt-…"),  # add when published
]

MCOL = {
    "Grok 4.5": SERIES[0],
    "Fable 5": SERIES[1],
    "Opus 4.8": SERIES[2],
    "GLM-5.2": SERIES[3],
}

PANELS = [
    ("RTX PRO 6000", "leaderboard.json"),
    ("H100", "leaderboard.h100.json"),
    ("B200", "leaderboard.b200.json"),
]

# Clean Fable RTX fp8 not yet on published allowlist row.
FABLE_FP8_RTX = 0.4098


def load_board(name: str) -> dict[tuple[str, str], dict]:
    path = HARD / name
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {(m["harness"], m["model"]): (m.get("results") or {}) for m in d["models"]}


def peak(res: dict | None, prob: str) -> float | None:
    if not res or prob not in res or res[prob] is None:
        return None
    c = res[prob]
    if not isinstance(c, dict):
        return float(c) if c else None
    if c.get("correct") is False:
        return None
    pf = c.get("peak_fraction")
    return float(pf) if pf is not None else None


def series_for(board: dict, gpu_key: str) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    for lbl, harness, model in ROWS:
        res = board.get((harness, model))
        vals: list[float | None] = []
        for p in PROB:
            pf = peak(res, p)
            if (
                pf is None
                and lbl == "Fable 5"
                and p == "01_fp8_gemm"
                and gpu_key == "leaderboard.json"
            ):
                pf = FABLE_FP8_RTX
            vals.append(pf)
        out[lbl] = vals
    return out


def draw_panel(ax, data: dict[str, list[float | None]], gpu_label: str):
    models = [r[0] for r in ROWS]
    n = len(models)
    x = np.arange(len(PROB))
    w = 0.8 / n
    ymax = 0.05

    for mi, m in enumerate(models):
        off = (mi - (n - 1) / 2) * w
        for j, s in enumerate(data[m]):
            if s is None:
                continue
            h = max(float(s), 0.002)
            ymax = max(ymax, float(s))
            ax.bar(
                x[j] + off,
                h,
                w * 0.92,
                color=MCOL[m],
                edgecolor=C["bg"],
                linewidth=0.4,
                zorder=3,
            )

    # Full X/Y axes on every panel — feed chrome stays off, axes stay on.
    ax.set_xticks(x)
    ax.set_xticklabels(PROB_LBL, fontsize=9)
    ax.set_xlabel("problem", fontsize=9, color=C["fg_muted"], labelpad=2)
    ax.set_ylabel("peak_fraction (roofline)", fontsize=9, color=C["fg_muted"], labelpad=4)
    ax.set_ylim(0, min(0.85, ymax * 1.12 + 0.02))
    ax.set_xlim(-0.55, len(PROB) - 0.45)
    ax.grid(True, axis="y", alpha=0.45, zorder=0)
    ax.tick_params(axis="both", labelsize=8)
    # tiny in-axes GPU tag — not a figure title band
    ax.text(
        0.01,
        0.97,
        gpu_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=C["fg"],
        fontsize=10,
        fontweight="bold",
        zorder=5,
    )


def main() -> None:
    fig, axes = tight_square(nrows=3, size=10.5)
    # a bit more room so per-panel x/y labels don't collide
    fig.subplots_adjust(left=0.11, right=0.985, top=0.985, bottom=0.06, hspace=0.32)
    for ax, (gpu_label, board_name) in zip(axes, PANELS):
        board = load_board(board_name)
        data = series_for(board, board_name)
        draw_panel(ax, data, gpu_label)

    leg = [Patch(facecolor=MCOL[m], label=m) for m, _, _ in ROWS]
    axes[0].legend(
        handles=leg,
        loc="upper right",
        ncol=2,
        fontsize=8,
        frameon=True,
        facecolor=C["surface"],
        edgecolor=C["border"],
        labelcolor=C["fg"],
        framealpha=0.95,
        borderpad=0.4,
        handlelength=1.2,
        columnspacing=0.8,
        labelspacing=0.3,
    )

    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")
    for board_name in [p[1] for p in PANELS]:
        data = series_for(load_board(board_name), board_name)
        print(f"--- {board_name}")
        for m, _, _ in ROWS:
            cells = " ".join(
                f"{v:.3f}" if v is not None else "  -  " for v in data[m]
            )
            print(f"  {m:10s} {cells}")


if __name__ == "__main__":
    main()
