"""Claude Opus 5 vs the frontier on KernelBench-Hard — RTX PRO 6000 + H100.

Visual-first engagement chart (AGENTS.md + kbh_theme): square 1:1, two stacked
GPU panels, bars + axes + compact legend only — no title essay.

Subject set: Opus 5 (green accent, the subject), Fable 5, Opus 4.8,
GPT-5.6 Sol, Grok 4.5. Missing cells omit the bar (Opus 5 RTX ran 2/6 —
the other four hard cells went to H100 boxes; honesty over symmetry).

  uv run python media/make_opus5_frontier.py
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
OUT = Path(__file__).resolve().parent / "opus5_frontier.png"

PROB = [
    "01_fp8_gemm",
    "02_kda_cutlass",
    "03_paged_attention",
    "05_topk_bitonic",
    "06_sonic_moe_swiglu",
    "07_w4a16_gemm",
]
PROB_LBL = ["fp8", "kda", "paged", "topk", "sonic", "w4a16"]

ROWS = [
    ("Opus 5", "or-opus", "anthropic/claude-opus-5"),
    ("Fable 5", "or-fable", "anthropic/claude-fable-5"),
    ("Opus 4.8", "claude", "claude-opus-4-8"),
    ("GPT-5.6 Sol", "codex", "gpt-5.6-sol"),
    ("Grok 4.5", "grok", "grok-4.5"),
]

MCOL = {
    "Opus 5": C["accent"],
    "Fable 5": SERIES[1],
    "Opus 4.8": SERIES[2],
    "GPT-5.6 Sol": SERIES[3],
    "Grok 4.5": SERIES[4] if len(SERIES) > 4 else SERIES[0],
}

PANELS = [
    ("RTX PRO 6000", "leaderboard.json"),
    ("H100", "leaderboard.h100.json"),
]


def load_board(name: str) -> dict[tuple[str, str], dict]:
    path = HARD / name
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    # A model can appear as several board rows (effort tiers, e.g. base and
    # [max]). Merge them per problem, keeping the best clean peak, so a
    # 1-cell [max] row cannot clobber the full base column.
    out: dict[tuple[str, str], dict] = {}
    for m in d["models"]:
        key = (m["harness"], m["model"])
        merged = out.setdefault(key, {})
        for p, cell in (m.get("results") or {}).items():
            old = merged.get(p)
            new_pf = (cell or {}).get("peak_fraction") or 0
            old_pf = (old or {}).get("peak_fraction") or 0
            if old is None or new_pf > old_pf:
                merged[p] = cell
    return out


def peak(res: dict | None, prob: str) -> float | None:
    if not res or prob not in res or res[prob] is None:
        return None
    c = res[prob]
    if not isinstance(c, dict):
        return float(c) if c else None
    if c.get("correct") is False:
        return None
    if c.get("annotation_verdict") not in (None, "clean"):
        return None
    pf = c.get("peak_fraction")
    return float(pf) if pf is not None else None


def draw_panel(ax, board: dict, gpu_label: str):
    n = len(ROWS)
    x = np.arange(len(PROB))
    w = 0.8 / n
    ymax = 0.05
    for mi, (lbl, harness, model) in enumerate(ROWS):
        res = board.get((harness, model))
        off = (mi - (n - 1) / 2) * w
        for j, p in enumerate(PROB):
            pf = peak(res, p)
            if pf is None:
                continue
            ymax = max(ymax, pf)
            ax.bar(x[j] + off, max(pf, 0.002), w * 0.92,
                   color=MCOL[lbl], edgecolor=C["bg"], linewidth=0.4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(PROB_LBL, fontsize=9)
    ax.set_ylim(0, ymax * 1.14)
    ax.set_ylabel("peak fraction", fontsize=9)
    ax.text(0.99, 0.94, gpu_label, transform=ax.transAxes, ha="right",
            va="top", fontsize=10, color=C["fg_muted"])
    ax.grid(axis="y", color=C["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


fig, axes = tight_square(nrows=len(PANELS))
for ax, (gpu_label, board_file) in zip(np.atleast_1d(axes), PANELS):
    draw_panel(ax, load_board(board_file), gpu_label)

handles = [Patch(facecolor=MCOL[r[0]], label=r[0]) for r in ROWS]
np.atleast_1d(axes)[0].legend(handles=handles, loc="upper left", fontsize=8,
                              ncols=3, frameon=False, labelcolor=C["fg"])
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
