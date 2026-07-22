"""Fable rerun cells + KernelBench-CUDA board snapshot — engagement chart.

Square 1:1, three stacked panels (see AGENTS.md + kbh_theme):
  1. The three api_error-truncated fable cells, rerun with full sessions:
     faded bar = old truncated-session score, green bar = isolated re-grade,
     amber tick = the board ceiling BEFORE the rerun.
  2. KernelBench-CUDA, RTX PRO 6000 — best cell per model per problem.
  3. KernelBench-CUDA, B200 — same.

Reads the live cuda leaderboards; rerun panel values are frozen in-code
(the truncated numbers are no longer on any board).

  uv run python media/make_fable_cuda_update.py
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
CUDA = ROOT / "benchmarks/cuda/results"
OUT = Path(__file__).resolve().parent / "fable_cuda_update.png"

# ---- panel 1: fable rerun cells (values frozen at publish time) ----------
# (label, truncated-session score, full-session isolated re-grade, prior ceiling)
RERUNS = [
    ("nsa\n(cuda PRO)", 0.2934, 0.7266, 0.4246),
    ("kda\n(hard H100)", 0.0180, 0.0527, 0.0255),
    ("fp8\n(hard PRO)", 0.3480, 0.3911, 0.4059),
]

# ---- panels 2-3: cuda boards ---------------------------------------------
PROB = ["01_glm52_fused_moe", "02_deepseek_nsa", "03_megaqwen_decode", "04_grid_mingru_sps"]
PROB_LBL = ["glm52-moe", "nsa", "megaqwen", "mingru-sps"]

ROWS = [
    ("Fable 5", ["anthropic/claude-fable-5", "claude-fable-5"]),
    ("Opus 4.8", ["claude-opus-4-8"]),
    ("Grok 4.5", ["grok-4.5"]),
    ("Kinetic", ["kinetic-0715", "kinetic-0715[1m]"]),
]
MCOL = {
    "Fable 5": SERIES[0],
    "Opus 4.8": SERIES[2],
    "Grok 4.5": SERIES[3],
    "Kinetic": SERIES[1],
}
PANELS = [("RTX PRO 6000", "leaderboard.json"), ("B200", "leaderboard.b200.json")]


def load(name: str) -> dict[tuple[str, str], float]:
    d = json.loads((CUDA / name).read_text())
    out: dict[tuple[str, str], float] = {}
    for mm in d["models"]:
        model = mm.get("model") or ""
        for p, c in (mm.get("results") or {}).items():
            if not c or not c.get("correct") or c.get("invalid_reason"):
                continue
            pf = c.get("peak_fraction")
            if pf is None:
                continue
            key = (model, p)
            out[key] = max(out.get(key, 0.0), pf)
    return out


fig, axes = tight_square(nrows=3, size=10.0)
ax1, ax2, ax3 = np.ravel(axes)

# ---- panel 1 ---------------------------------------------------------------
x = np.arange(len(RERUNS))
w = 0.32
old = [r[1] for r in RERUNS]
new = [r[2] for r in RERUNS]
ax1.bar(x - w / 2, old, w, color=SERIES[0], alpha=0.25,
        edgecolor=C["fg_dim"], linewidth=0.8, linestyle=(0, (2, 2)))
ax1.bar(x + w / 2, new, w, color=C["accent"])
for i, (_, o, n, ceil) in enumerate(RERUNS):
    ax1.hlines(ceil, i - 0.45, i + 0.45, color=C["warn"], linewidth=1.6)
    ax1.text(i + w / 2, n + 0.015, f"{n:.3f}", ha="center",
             color=C["fg_bright"], fontsize=10)
    ax1.text(i - w / 2, o + 0.015, f"{o:.3f}", ha="center",
             color=C["fg_muted"], fontsize=9)
ax1.set_xticks(x, [r[0] for r in RERUNS])
ax1.set_ylabel("peak fraction")
ax1.set_ylim(0, 0.97)
ax1.grid(axis="y", linewidth=0.5)
ax1.set_axisbelow(True)
ax1.legend(handles=[
    Patch(facecolor=SERIES[0], alpha=0.25, label="session cut by api_error"),
    Patch(facecolor=C["accent"], label="full session (isolated re-grade)"),
    Patch(facecolor=C["warn"], label="board ceiling before rerun"),
], loc="upper right", frameon=False, fontsize=9)
ax1.text(0.012, 0.94, "Fable 5 — rerun of truncated cells", transform=ax1.transAxes,
         color=C["fg_muted"], fontsize=9, va="top")

# ---- panels 2-3 ------------------------------------------------------------
for ax, (gpu, board) in zip((ax2, ax3), PANELS):
    data = load(board)
    xp = np.arange(len(PROB))
    nb = len(ROWS)
    bw = 0.8 / nb
    for j, (label, ids) in enumerate(ROWS):
        vals = []
        for p in PROB:
            v = max((data.get((m, p), 0.0) for m in ids), default=0.0)
            vals.append(v)
        pos = xp - 0.4 + bw * (j + 0.5)
        ax.bar(pos, vals, bw * 0.92, color=MCOL[label], label=label)
    ax.set_xticks(xp, PROB_LBL)
    ax.set_ylabel("peak fraction")
    ax.grid(axis="y", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(0.012, 0.94, f"KernelBench-CUDA — {gpu}", transform=ax.transAxes,
            color=C["fg_muted"], fontsize=9, va="top")

ax2.legend(loc="upper right", frameon=False, fontsize=9, ncols=4)

fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")
