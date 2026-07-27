"""Megakernel-vs-launches diagram for the Opus 5 X article.

Left: a conventional decode step as a stack of kernel launches, each paying
launch overhead. Right: the Opus 5 megakernel — one launch, the same stages
separated by software grid barriers. House palette, square-ish, no chrome.

  uv run python media/make_opus5_megakernel_diagram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply

apply()

OUT = Path(__file__).resolve().parent / "opus5_megakernel_diagram.png"

STAGES = ["norm", "qkv proj", "attention", "o proj", "norm", "moe route",
          "expert gemm", "down proj"]

fig, ax = plt.subplots(figsize=(7.6, 7.0))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10.6)
ax.axis("off")


def box(x, y, w, h, color, label, lbl_color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.06",
                                facecolor=color, edgecolor=C["border"],
                                linewidth=0.8, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=lbl_color, zorder=4)


# ---- left column: conventional (kernel per op, gaps = launch overhead) ----
lx, lw = 0.7, 3.4
ax.text(lx + lw / 2, 10.25, "kernel per op", ha="center", fontsize=11,
        color=C["fg"])
ax.text(lx + lw / 2, 9.85, "(what models usually write)", ha="center",
        fontsize=8, color=C["fg_dim"])
y = 9.2
for s in STAGES:
    box(lx, y - 0.62, lw, 0.62, "#3c4a52", s, C["fg_muted"])
    y -= 0.62
    gap = 0.44
    ax.text(lx + lw / 2, y - gap / 2, "launch + sync", ha="center",
            va="center", fontsize=7, color=C["bad"])
    y -= gap
ax.text(lx + lw / 2, y - 0.25, "x 4 blocks x every token",
        ha="center", fontsize=8, color=C["fg_dim"])

# ---- right column: megakernel ----
rx, rw = 5.9, 3.4
ax.text(rx + rw / 2, 10.25, "Opus 5 megakernel", ha="center", fontsize=11,
        color=C["accent"])
ax.text(rx + rw / 2, 9.85, "one launch per decode step", ha="center",
        fontsize=8, color=C["fg_dim"])
# outer shell
ax.add_patch(FancyBboxPatch((rx - 0.18, 0.62), rw + 0.36, 8.9,
                            boxstyle="round,pad=0.05,rounding_size=0.1",
                            facecolor="none", edgecolor=C["accent"],
                            linewidth=1.6, zorder=2))
y = 9.2
for s in STAGES:
    box(rx, y - 0.72, rw, 0.72, "#243a10", s, C["fg"])
    y -= 0.72
    bar = 0.34
    ax.plot([rx + 0.15, rx + rw - 0.15], [y - bar / 2, y - bar / 2],
            color=C["accent"], linewidth=1.1, zorder=4)
    ax.text(rx + rw / 2, y - bar / 2 + 0.13, "grid barrier", ha="center",
            fontsize=6.5, color=C["accent"], zorder=5)
    y -= bar
ax.text(rx + rw / 2, y - 0.32,
        "33 barriers per step, zero relaunches",
        ha="center", fontsize=8, color=C["fg_muted"])

fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}")
