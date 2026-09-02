"""Shared 5:2 KernelBench article launch card.

House style, locked 2026-08-13:

- 3000x1200 (15.0 x 6.0 in at 200 dpi). 5:2 only.
- Left mark is the official lab asset. Kimi ships a black App Store tile
  (`kimi-app-icon-1024.png`). DeepSeek / Qwen / Grok do not — never invent
  a black rounded app-icon tile for a transparent SVG. Grok is the Feb 2025
  black-hole G (`xai-logo-1024.png` / `public/logos/labs/xai.svg`), not the
  old xAI X-mark. Qwen is the hex pinwheel (`qwen-logo-1024.png` /
  `public/logos/labs/qwen.svg`), not the old C-dash chat icon.
- Identity is mark + one subject token, same visual size as the mark.
  Grok 5 is the digit `5`. DeepSeek subject is `0731`. Do not print the
  lab name next to the lab mark.
- No KernelBench signature. No second word row.
- Forbidden on the card: charts, outcome bars, pass counts, peak_fraction,
  "writes kernels on", GPU lists, audit tallies, taglines, "GROK"/"QWEN"/
  family words.
- Palette from kbh_theme only.

Usage:
    from thumb_card import launch_card
    fig, ax = launch_card(mark="deepseek-logo-1024.png", mark_is_tile=False)
    # draw the subject token, then:
    save(fig, "out.png")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from kbh_theme import C, apply

W, H = 1000, 400
FIGSIZE = (15.0, 6.0)
DPI = 200
MEDIA = Path(__file__).resolve().parent

# Official black App Store / marketing tiles. Everything else is a
# transparent mark and must stay transparent.
TILE_MARKS = frozenset({"kimi-app-icon-1024.png", "anthropic-tile-1024.png"})


def launch_card(*, mark: str | Path, mark_is_tile: bool | None = None, seed: int = 42):
    """Return (fig, ax) with starfield + bloom + optional official tile.

    `mark_is_tile` defaults to True only for names in TILE_MARKS.
    """
    apply()
    mark_path = Path(mark)
    if not mark_path.is_absolute():
        mark_path = MEDIA / mark_path
    if mark_is_tile is None:
        mark_is_tile = mark_path.name in TILE_MARKS

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])

    rng = np.random.default_rng(seed)
    sx = rng.uniform(0, W, 120)
    sy = rng.uniform(0, H, 120)
    ax.scatter(sx, sy, s=rng.uniform(0.3, 1.8, 120), c=C["fg_bright"],
               alpha=0.28, lw=0, zorder=1)

    yy, xx = np.mgrid[0:H:400j, 0:W:1000j]
    bloom = 0.18 * np.exp(-(((xx - 390) / 470) ** 2 + ((yy - 205) / 260) ** 2))
    rgba = np.zeros((400, 1000, 4))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 0x76 / 255, 0xB9 / 255, 0
    rgba[..., 3] = bloom
    ax.imshow(rgba, extent=(0, W, 0, H), origin="lower", zorder=0)

    segs, cols = [], []
    for _ in range(18):
        x_, y_ = rng.uniform(0, W), rng.uniform(250, 395)
        segs.append([(x_, y_), (x_ + rng.uniform(20, 70), y_)])
        cols.append((0x76 / 255, 0xB9 / 255, 0, rng.uniform(0.04, 0.14)))
    ax.add_collection(LineCollection(segs, colors=cols, lw=1.0, zorder=1))

    logo = np.asarray(Image.open(mark_path).convert("RGBA"))
    U = 250.0
    cx, cy = 186, 208
    ext = (cx - U / 2, cx + U / 2, cy - U / 2, cy + U / 2)
    if mark_is_tile:
        for pad, a in [(16, 0.28), (30, 0.12)]:
            ax.add_patch(FancyBboxPatch(
                (ext[0] - pad, ext[2] - pad), U + 2 * pad, U + 2 * pad,
                boxstyle="round,pad=0,rounding_size=28",
                facecolor="none", edgecolor=C["accent"], linewidth=1.1,
                alpha=a, zorder=3))
    ax.imshow(logo, extent=ext, zorder=5)

    scan = np.zeros((400, 4, 4))
    scan[::2, :, 3] = 0.04
    ax.imshow(scan, extent=(0, W, 0, H), origin="lower", zorder=9,
              interpolation="nearest", aspect="auto")
    return fig, ax


def signature(ax, text: str = "KernelBench") -> None:
    """Retired 2026-08-13. Do not call. Kept so old generators still import."""
    del ax, text


def save(fig, out: str | Path) -> Path:
    out = Path(out)
    fig.savefig(out, dpi=DPI, facecolor=C["bg"])
    return out
