"""Opus 5 article 5:2 thumbnail — launch card in the K3 house style.

Anthropic tile left (rasterized from public/logos/labs/anthropic.svg into
media/anthropic-tile-1024.png), OPUS 5 wordmark, "writes kernels on
KernelBench" tagline, stat lines, mega-speedup mini chart top right,
starfield + synthwave floor + scanlines.

Usage: uv run --with matplotlib,numpy,pillow python make_opus5_thumb.py [out.png]
"""
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

from kbh_theme import C, apply

apply()

W, H = 1000, 400
fig = plt.figure(figsize=(15.0, 6.0), dpi=200)  # 3000x1200
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(C["bg"])

rng = np.random.default_rng(7)

# --- starfield ---
sx = rng.uniform(0, W, 150)
sy = rng.uniform(130, H, 150)
ss = rng.uniform(0.4, 2.4, 150)
ax.scatter(sx, sy, s=ss, c=C["fg_bright"], alpha=0.5, lw=0, zorder=1)
ax.scatter(sx[::6], sy[::6], s=14, c=C["accent"], alpha=0.12, lw=0, zorder=1)

# --- soft green bloom ---
yy, xx = np.mgrid[0:H:400j, 0:W:1000j]
bloom = 0.15 * np.exp(-(((xx - 500) / 430) ** 2 + (((yy - 220) / 220) ** 2)))
bloom += 0.09 * np.exp(-(((xx - 205) / 240) ** 2 + (((yy - 225) / 200) ** 2)))
img = np.zeros((400, 1000, 4))
img[..., 0], img[..., 1], img[..., 2] = 0x76 / 255, 0xB9 / 255, 0
img[..., 3] = np.clip(bloom, 0, 1)
ax.imshow(img, extent=(0, W, 0, H), origin="lower", zorder=0)

# --- synthwave floor ---
vpx, vpy, horizon = 500, 118, 118
for x_ in np.linspace(-900, 1900, 29):
    ax.plot([x_, vpx], [0, vpy], color=C["accent"], lw=0.7, alpha=0.14, zorder=1)
for t in np.geomspace(1, 118, 9):
    y_ = horizon - t
    ax.plot([0, W], [y_, y_], color=C["accent"], lw=0.8,
            alpha=0.04 + 0.15 * (t / 118), zorder=1)
ax.plot([0, W], [horizon, horizon], color=C["accent"], lw=1.1, alpha=0.28, zorder=1)

# --- warp streaks ---
segs, cols = [], []
for _ in range(26):
    x_, y_ = rng.uniform(0, W), rng.uniform(250, 395)
    segs.append([(x_, y_), (x_ + rng.uniform(25, 90), y_)])
    cols.append((0x76 / 255, 0xB9 / 255, 0, rng.uniform(0.05, 0.20)))
ax.add_collection(LineCollection(segs, colors=cols, lw=1.1, zorder=1))

# --- Anthropic tile + green halo + outline rings ---
tile = np.asarray(Image.open(Path(__file__).parent / "anthropic-tile-1024.png"))
U = 235.0
tcx, tcy = 176, 215
ext = (tcx - U / 2, tcx + U / 2, tcy - U / 2, tcy + U / 2)
for grow, a in [(66, 0.05), (38, 0.10), (18, 0.17)]:
    hj, hi = np.mgrid[0:200, 0:200]
    dd = np.hypot(hi - 99.5, hj - 99.5)
    hg = np.zeros((200, 200, 4))
    hg[..., 0], hg[..., 1], hg[..., 2] = 0x76 / 255, 0xB9 / 255, 0
    hg[..., 3] = a * np.clip(1 - dd / 100, 0, 1) ** 1.6
    g = (U + 2 * grow) / 2
    ax.imshow(hg, extent=(tcx - g, tcx + g, tcy - g, tcy + g), zorder=3)
for pad, a in [(16, 0.35), (30, 0.15)]:
    ax.add_patch(FancyBboxPatch(
        (tcx - U / 2 - pad, tcy - U / 2 - pad), U + 2 * pad, U + 2 * pad,
        boxstyle=f"round,pad=0,rounding_size={28 + pad * 0.5}",
        facecolor="none", edgecolor=C["accent"], linewidth=1.2, alpha=a, zorder=3))
ax.imshow(tile, extent=ext, zorder=5)


def glow(t, color, layers=((10, 0.12), (5, 0.24), (2.5, 0.40))):
    t.set_path_effects([pe.Stroke(linewidth=lw, foreground=color, alpha=a)
                        for lw, a in layers] + [pe.Normal()])


# --- wordmark: OPUS white + 5 green ---
TXT_X = 336
t1 = ax.text(TXT_X, 288, "OPUS", fontsize=76, fontweight="bold",
             family="monospace", ha="left", va="center",
             color=C["fg_bright"], zorder=6)
glow(t1, C["fg_bright"], layers=((6, 0.10), (3, 0.18)))
fig.canvas.draw()
x2 = ax.transData.inverted().transform(
    t1.get_window_extent(fig.canvas.get_renderer()))[1, 0]
t2 = ax.text(x2 + 26, 288, "5", fontsize=76, fontweight="bold",
             family="monospace", ha="left", va="center",
             color=C["accent"], zorder=6)
glow(t2, C["accent"])
fig.canvas.draw()
x3 = ax.transData.inverted().transform(
    t2.get_window_extent(fig.canvas.get_renderer()))[1, 0]
ax.plot([TXT_X + 2, x3], [246, 246], color=C["accent"], lw=2.6,
        alpha=0.9, zorder=6, solid_capstyle="round")

# --- tagline + stats ---
t3 = ax.text(TXT_X, 205, "writes kernels on ", fontsize=31,
             family="monospace", ha="left", va="center",
             color=C["fg_muted"], zorder=6)
fig.canvas.draw()
x4 = ax.transData.inverted().transform(
    t3.get_window_extent(fig.canvas.get_renderer()))[1, 0]
t4 = ax.text(x4, 205, "KernelBench", fontsize=31, fontweight="bold",
             family="monospace", ha="left", va="center",
             color=C["fg_bright"], zorder=6)
glow(t4, C["fg_bright"], layers=((5, 0.10), (2.5, 0.18)))

ax.text(TXT_X + 2, 152, "24.3x megakernel record  ·  one launch, 33 grid barriers",
        fontsize=19.5, family="monospace", ha="left", va="center",
        color=C["fg_muted"], zorder=6)
ax.text(TXT_X + 2, 118, "RTX PRO 6000  ·  H100  —  17/17 cells hand-audited clean",
        fontsize=19.5, family="monospace", ha="left", va="center",
        color=C["fg_muted"], zorder=6)

# --- mini chart card top right: mega H100 speedups, subject green ---
cx0, cy0, cw, ch = 762, 282, 200, 96
ax.add_patch(FancyBboxPatch((cx0, cy0), cw, ch,
                            boxstyle="round,pad=0,rounding_size=9",
                            facecolor="#181c16", edgecolor="#2c332a",
                            linewidth=1.1, zorder=4))
vals = [1.4, 2.5, 4.5, 4.5, 15.5, 19.1, 24.3]
bw = cw / len(vals) * 0.62
gap = cw / len(vals)
for i, v in enumerate(vals):
    h_ = (v / 24.3) * (ch - 26)
    last = i == len(vals) - 1
    ax.add_patch(plt.Rectangle((cx0 + 12 + i * gap * 0.92, cy0 + 12), bw, h_,
                               facecolor=C["accent"] if last else "#41503b",
                               edgecolor="none", zorder=5))
    if last:
        dot = ax.scatter([cx0 + 12 + i * gap * 0.92 + bw / 2], [cy0 + 12 + h_ + 6],
                         s=52, c=C["accent"], zorder=6)
        dot.set_path_effects([pe.Stroke(linewidth=7, foreground=C["accent"],
                                        alpha=0.35), pe.Normal()])

# --- CRT scanlines ---
scan = np.zeros((400, 4, 4))
scan[::2, :, 3] = 0.055
ax.imshow(scan, extent=(0, W, 0, H), origin="lower", zorder=9,
          interpolation="nearest", aspect="auto")

out = sys.argv[1] if len(sys.argv) > 1 else "opus5_thumb.png"
fig.savefig(out, dpi=200, facecolor=C["bg"])
print(out)
