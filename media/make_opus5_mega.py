"""Opus 5 sets the H100 record on KernelBench-Mega (kimi_linear_decode).

Single square panel, horizontal speedup bars sorted descending, subject in
NVIDIA green. Visual-first: bars + axis + numbers, no chrome (AGENTS.md).

Reads public/data/mega/results.csv (H100 rows, correct only, best per model).

  uv run python media/make_opus5_mega.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply

apply()
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "public/data/mega/results.csv"
OUT = Path(__file__).resolve().parent / "opus5_mega_h100.png"

PRETTY = {
    "anthropic/claude-opus-5": "Opus 5",
    "anthropic/claude-fable-5": "Fable 5",
    "claude-opus-4-8": "Opus 4.8",
    "kinetic-0715": "Kinetic 0715",
    "kinetic-0715[1m]": "Kinetic 0715 1M",
    "gpt-5.5": "GPT-5.5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "glm-5.2": "GLM-5.2",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "grok-4.5": "Grok 4.5",
    "composer-2.5-fast": "Composer 2.5 Fast",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}

best: dict[str, float] = {}
for row in csv.DictReader(open(CSV)):
    if row.get("gpu") != "H100" or row.get("correct") != "true":
        continue
    m, s = row["model"], float(row["score"])
    if s > best.get(m, 0):
        best[m] = s

rows = sorted(best.items(), key=lambda kv: kv[1])
labels = [PRETTY.get(m, m) for m, _ in rows]
vals = [v for _, v in rows]
colors = [C["accent"] if m == "anthropic/claude-opus-5" else "#4d5d66"
          for m, _ in rows]

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
y = np.arange(len(rows))
ax.barh(y, vals, 0.72, color=colors, edgecolor=C["bg"], linewidth=0.4, zorder=3)
for yi, v in zip(y, vals):
    ax.text(v + 0.35, yi, f"{v:.1f}x", va="center", fontsize=9,
            color=C["fg"] if v == max(vals) else C["fg_muted"])
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10, color=C["fg"])
ax.set_xlabel("speedup vs eager reference (kimi_linear_decode, H100)",
              fontsize=9, color=C["fg_muted"])
ax.set_xlim(0, max(vals) * 1.13)
ax.grid(axis="x", color=C["grid"], linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(C["border"])
ax.tick_params(colors=C["fg_muted"])
fig.subplots_adjust(left=0.22, right=0.97, top=0.98, bottom=0.09)
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
