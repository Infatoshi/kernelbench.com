"""K3 article — 3:1 X-article banner. Title card, not a chart: name left,
faded real mega-speedup bar motif right (K3 vs field, kimi_linear_decode RTX).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from kbh_theme import C, apply

apply()

fig = plt.figure(figsize=(15, 5), dpi=100)
fig.patch.set_facecolor(C["bg"])
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 3)
ax.set_ylim(0, 1)
ax.axis("off")

# left: title block
ax.text(0.14, 0.60, "KIMI K3", color=C["accent"], fontsize=64,
        fontweight="bold", va="center", ha="left")
ax.text(0.145, 0.34, "on KernelBench", color=C["fg"], fontsize=28,
        va="center", ha="left")
ax.text(0.148, 0.18, "RTX PRO 6000  ·  H100  ·  B200",
        color=C["fg_muted"], fontsize=15, va="center", ha="left")

# right: faded bar motif — real kimi_linear_decode RTX speedups
bars = [("Fable 5", 18.72), ("Kimi K3 (256k)", 18.09), ("Opus 4.8", 14.40),
        ("GLM 5.2", 11.14), ("Sol", 2.64)]
x0, x1 = 1.55, 2.86
ymax = 20.5
bw = (x1 - x0) / len(bars) * 0.62
for i, (name, v) in enumerate(bars):
    x = x0 + (i + 0.5) * (x1 - x0) / len(bars)
    h = 0.72 * v / ymax
    color = C["accent"] if name == "Kimi K3 (256k)" else C["surface_muted"]
    ax.add_patch(plt.Rectangle((x - bw / 2, 0.14), bw, h,
                               color=color, zorder=3))
    ax.text(x, 0.14 + h + 0.03, f"{v:.1f}x", color=C["fg_muted"] if
            name != "Kimi K3 (256k)" else C["accent"], fontsize=12,
            ha="center", va="bottom")
    ax.text(x, 0.085, name, color=C["fg_dim"], fontsize=11,
            ha="center", va="center")

out = sys.argv[1] if len(sys.argv) > 1 else "k3_banner.png"
fig.savefig(out, dpi=100)
print(out)
