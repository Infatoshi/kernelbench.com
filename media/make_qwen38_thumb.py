"""Qwen 3.8 Max article cover — mark + subject token.

Official Qwen hex pinwheel (qwen.ai / HF). Subject is 3.8.
The old C-dash chat icon is wrong. No "QWEN". No KernelBench signature.

Usage: uv run --with matplotlib,numpy,pillow python make_qwen38_thumb.py [out.png]
"""
from __future__ import annotations

import sys

import matplotlib.patheffects as pe

from kbh_theme import C
from thumb_card import launch_card, save


def glow(text, color, layers=((10, 0.10), (5, 0.20), (2, 0.34))):
    text.set_path_effects([
        *[pe.Stroke(linewidth=width, foreground=color, alpha=alpha)
          for width, alpha in layers],
        pe.Normal(),
    ])


fig, ax = launch_card(mark="qwen-logo-1024.png", mark_is_tile=False, seed=38)
t = ax.text(360, 200, "3.8", fontsize=160, fontweight="bold", family="monospace",
            ha="left", va="center", color=C["accent"], zorder=6)
glow(t, C["accent"])

out = sys.argv[1] if len(sys.argv) > 1 else "qwen38_thumb.png"
print(save(fig, out))
