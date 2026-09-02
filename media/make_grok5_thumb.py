"""Grok 5 example article cover — mark + huge version number.

Official Grok black-hole G (grok.com favicon / xai-logo-1024.png).
Subject is the digit 5, same visual size as the mark. No "GROK".
No KernelBench signature. Example-only. Do not publish.

Usage: uv run --with matplotlib,numpy,pillow python make_grok5_thumb.py [out.png]
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


fig, ax = launch_card(mark="xai-logo-1024.png", mark_is_tile=False, seed=5)
# Mark is 250px at (186, 208). Digit matches that height.
t = ax.text(360, 200, "5", fontsize=220, fontweight="bold", family="monospace",
            ha="left", va="center", color=C["accent"], zorder=6)
glow(t, C["accent"])

out = sys.argv[1] if len(sys.argv) > 1 else "grok5_thumb.png"
print(save(fig, out))
