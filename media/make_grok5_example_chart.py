"""Example in-body chart for the Grok 5 article preview. Fake numbers.

CUDA deck first. One GPU. Do not add a second SKU column.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from kbh_theme import C, apply

apply()
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
labels = ["Fused MoE", "NSA", "Decode", "Craftax RL"]
vals = [0.38, 0.91, 0.22, 0.44]
colors = [C["accent"] if v == max(vals) else C["fg_dim"] for v in vals]
ax.bar(labels, vals, color=colors, width=0.62)
ax.set_ylim(0, 1.05)
ax.set_ylabel("isolated score (example)", color=C["fg_muted"])
for spine in ax.spines.values():
    spine.set_color(C["border"])
ax.tick_params(colors=C["fg_muted"])
ax.yaxis.grid(True, color=C["grid"], linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "grok5_example_cuda.png"
fig.savefig(out, dpi=160, facecolor=C["bg"])
print(out)
