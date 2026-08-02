"""DeepSeek V4 Flash 0731 vs Fable 5 / Opus 5 / GLM-5.2 / GPT-5.6 Sol on
KernelBench-Hard — RTX PRO 6000 + H100 SXM panels.

Visual-first engagement chart (see AGENTS.md + kbh_theme): square 1:1, two
stacked GPU panels, bars + axes + compact legend only — no title essay.

Models with two board rows (Fable via native claude + or-fable) merge to the
best correct cell per problem, matching how the site ranks models.

  uv run python media/make_v4flash_frontier.py
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
OUT = Path(__file__).resolve().parent / "v4flash_frontier.png"

PROB = [
    "01_fp8_gemm",
    "02_kda_cutlass",
    "03_paged_attention",
    "05_topk_bitonic",
    "06_sonic_moe_swiglu",
    "07_w4a16_gemm",
]
PROB_LBL = ["fp8", "kda", "paged", "topk", "sonic", "w4a16"]

# Subject set: V4 Flash leads in accent green; each entry lists every board
# row that counts toward the model (best correct cell wins).
ROWS = [
    ("DeepSeek V4 Flash", [("or-fable", "deepseek/deepseek-v4-flash-0731")]),
    ("Fable 5", [("claude", "claude-fable-5"), ("or-fable", "anthropic/claude-fable-5")]),
    ("Opus 5", [("or-opus", "anthropic/claude-opus-5")]),
    ("GLM-5.2", [("zai-claude", "glm-5.2")]),
    ("GPT-5.6 Sol", [("codex", "gpt-5.6-sol")]),
]

MCOL = {
    "DeepSeek V4 Flash": C["accent"],
    "Fable 5": SERIES[1],
    "Opus 5": SERIES[2],
    "GLM-5.2": SERIES[3],
    "GPT-5.6 Sol": SERIES[4] if len(SERIES) > 4 else C["fg_muted"],
}

PANELS = [
    ("RTX PRO 6000", "leaderboard.json"),
    ("H100 SXM", "leaderboard.h100.json"),
]


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


def series_for(board: dict) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    for lbl, keys in ROWS:
        vals: list[float | None] = []
        for p in PROB:
            best = None
            for key in keys:
                pf = peak(board.get(key), p)
                if pf is not None and (best is None or pf > best):
                    best = pf
            vals.append(best)
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

    ax.set_xticks(x)
    ax.set_xticklabels(PROB_LBL, fontsize=9)
    ax.set_xlabel("problem", fontsize=9, color=C["fg_muted"], labelpad=2)
    ax.set_ylabel("peak_fraction (roofline)", fontsize=9, color=C["fg_muted"], labelpad=4)
    ax.set_ylim(0, min(0.85, ymax * 1.12 + 0.02))
    ax.set_xlim(-0.55, len(PROB) - 0.45)
    ax.grid(True, axis="y", alpha=0.45, zorder=0)
    ax.tick_params(axis="both", labelsize=8)
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
    fig, axes = tight_square(nrows=2, size=10.5)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.985, bottom=0.06, hspace=0.24)
    for ax, (gpu_label, board_name) in zip(axes, PANELS):
        board = load_board(board_name)
        draw_panel(ax, series_for(board), gpu_label)

    leg = [Patch(facecolor=MCOL[m], label=m) for m, _ in ROWS]
    axes[0].legend(
        handles=leg,
        loc="upper right",
        fontsize=9,
        framealpha=0.0,
        ncols=2,
    )
    fig.savefig(OUT, dpi=200)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
