"""K3 article 5:2 thumbnail — launch card.

Official Kimi app-icon tile (media/kimi-app-icon-1024.png, App Store artwork,
Beijing Moonshot Technology) + a huge KernelBench wordmark. No model-name
text: the mark carries the identity, the wordmark carries the bench.

Usage: uv run --with matplotlib,numpy,pillow python make_k3_thumb.py [out.png]
Palette from kbh_theme (site tokens).
"""
import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

from kbh_theme import C, apply

apply()

W, H = 1000, 400
fig = plt.figure(figsize=(15.0, 6.0), dpi=200)  # 5:2 -> 3000x1200
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(C["bg"])

rng = np.random.default_rng(42)

# --- starfield ---
sx = rng.uniform(0, W, 150)
sy = rng.uniform(130, H, 150)
ss = rng.uniform(0.4, 2.4, 150)
ax.scatter(sx, sy, s=ss, c=C["fg_bright"], alpha=0.5, lw=0, zorder=1)
ax.scatter(sx[::6], sy[::6], s=14, c=C["accent"], alpha=0.12, lw=0, zorder=1)

# --- soft green bloom behind the lockup ---
yy, xx = np.mgrid[0:H:400j, 0:W:1000j]
bloom = 0.15 * np.exp(-(((xx - 500) / 430) ** 2 + (((yy - 220) / 220) ** 2)))
bloom += 0.09 * np.exp(-(((xx - 205) / 240) ** 2 + (((yy - 225) / 200) ** 2)))
img = np.zeros((400, 1000, 4))
img[..., 0], img[..., 1], img[..., 2] = 0x76 / 255, 0xB9 / 255, 0
img[..., 3] = np.clip(bloom, 0, 1)
ax.imshow(img, extent=(0, W, 0, H), origin="lower", zorder=0)

# --- synthwave floor: perspective grid to a vanishing point ---
vpx, vpy, horizon = 500, 118, 118
for x_ in np.linspace(-900, 1900, 29):
    ax.plot([x_, vpx], [0, vpy], color=C["accent"], lw=0.7, alpha=0.14,
            zorder=1)
for t in np.geomspace(1, 118, 9):
    y_ = horizon - t
    ax.plot([0, W], [y_, y_], color=C["accent"], lw=0.8,
            alpha=0.04 + 0.15 * (t / 118), zorder=1)
ax.plot([0, W], [horizon, horizon], color=C["accent"], lw=1.1, alpha=0.28,
        zorder=1)

# --- warp streaks ---
segs, cols = [], []
for _ in range(26):
    x_, y_ = rng.uniform(0, W), rng.uniform(250, 395)
    segs.append([(x_, y_), (x_ + rng.uniform(25, 90), y_)])
    cols.append((0x76 / 255, 0xB9 / 255, 0, rng.uniform(0.05, 0.20)))
ax.add_collection(LineCollection(segs, colors=cols, lw=1.1, zorder=1))

# --- official Kimi tile: rounded-corner mask + green halo ---
icon_path = Path(__file__).parent / "kimi-app-icon-1024.png"
icon = np.asarray(Image.open(icon_path).convert("RGB"), dtype=float) / 255.0
n = icon.shape[0]
r = 0.20 * n
jj, ii = np.mgrid[0:n, 0:n].astype(float)
d2 = (ii - np.clip(ii, r, n - 1 - r)) ** 2 + \
     (jj - np.clip(jj, r, n - 1 - r)) ** 2
rgba = np.dstack([icon, (d2 <= r * r).astype(float)])

U = 240.0
tcx, tcy = 176, 210
ext = (tcx - U / 2, tcx + U / 2, tcy - U / 2, tcy + U / 2)

for grow, a in [(66, 0.05), (38, 0.10), (18, 0.17)]:
    hj, hi = np.mgrid[0:200, 0:200]
    dd = np.hypot(hi - 99.5, hj - 99.5)
    hg = np.zeros((200, 200, 4))
    hg[..., 0], hg[..., 1], hg[..., 2] = 0x76 / 255, 0xB9 / 255, 0
    hg[..., 3] = a * np.clip(1 - dd / 100, 0, 1) ** 1.6
    g = (U + 2 * grow) / 2
    ax.imshow(hg, extent=(tcx - g, tcx + g, tcy - g, tcy + g), zorder=3)

ax.imshow(rgba, extent=ext, zorder=5)

# --- wordmark: huge, no model name ---
TXT_X = 336


def glow(t, color, layers=((10, 0.12), (5, 0.24), (2.5, 0.40))):
    t.set_path_effects([pe.Stroke(linewidth=lw, foreground=color, alpha=a)
                        for lw, a in layers] + [pe.Normal()])


t1 = ax.text(TXT_X, 248, "Kernel", fontsize=86, fontweight="bold",
             ha="left", va="center", color=C["fg_bright"], zorder=6)
glow(t1, C["fg_bright"], layers=((6, 0.10), (3, 0.18)))
fig.canvas.draw()
x2 = ax.transData.inverted().transform(
    t1.get_window_extent(fig.canvas.get_renderer()))[1, 0]
t2 = ax.text(x2, 248, "Bench", fontsize=86, fontweight="bold",
             ha="left", va="center", color=C["accent"], zorder=6)
glow(t2, C["accent"])

ax.text(TXT_X + 6, 148, "RTX PRO 6000  ·  H100  ·  B200",
        fontsize=27, ha="left", va="center", color=C["fg_muted"], zorder=6)

# --- CRT scanlines over everything ---
scan = np.zeros((400, 4, 4))
scan[::2, :, 3] = 0.055
ax.imshow(scan, extent=(0, W, 0, H), origin="lower", zorder=9,
          interpolation="nearest", aspect="auto")

out = sys.argv[1] if len(sys.argv) > 1 else "k3_thumb.png"
fig.savefig(out, dpi=200, facecolor=C["bg"])
print(out)
