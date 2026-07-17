"""K3 (kinetic-0715) article — Mega kimi_linear_decode speedup across GPUs.

Geomean speedup vs eager reference (ctx 2048/8192/16384). One group per GPU;
missing cells (still in flight / instance gap) are simply omitted.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from kbh_theme import C, SERIES, apply, tight_square

apply()

MODELS = ["Kimi K3 (256k)", "Fable 5", "Opus 4.8", "GLM 5.2", "GPT-5.6 Sol"]
GPUS = ["RTX PRO 6000", "H100", "B200"]
DATA = {
    "Kimi K3 (256k)":     [18.09, 14.82, None],
    "Fable 5":     [18.72, None,  None],
    "Opus 4.8":    [14.40, 15.50, 19.35],
    "GLM 5.2":     [11.14, None,  7.30],
    "GPT-5.6 Sol": [2.64,  None,  None],
}
# Model->color mapping shared across all article charts (matches make_k3_hard).
MODEL_COLORS = {"Kimi K3 (256k)": SERIES[0], "Fable 5": SERIES[2], "Opus 4.8": SERIES[3],
                "GLM 5.2": SERIES[4], "GPT-5.6 Sol": SERIES[5]}

fig, ax = tight_square(size=10.0)
fig.subplots_adjust(left=0.16)
nm = len(MODELS)
bar_h = 0.8 / nm
ys = np.arange(len(GPUS))[::-1]

for mi, model in enumerate(MODELS):
    color = MODEL_COLORS[model]
    for gi, val in enumerate(DATA[model]):
        if val is None:
            continue
        y = ys[gi] + (nm / 2 - mi - 0.5) * bar_h
        ax.barh(y, val, height=bar_h * 0.9, color=color,
                zorder=3, label=model if gi == 0 else None)
        ax.text(val + 0.2, y, f"{val:.1f}x", va="center", ha="left",
                color=C["fg_muted"], fontsize=8)

ax.set_yticks(ys)
ax.set_yticklabels(GPUS, fontsize=11)
ax.set_xlabel("geomean speedup vs eager (kimi_linear_decode)", fontsize=10)
ax.set_xlim(0, 22)
ax.grid(axis="x", zorder=0)
ax.legend(loc="center right", fontsize=9, framealpha=0.15,
          facecolor=C["surface"], edgecolor=C["border"])

out = sys.argv[1] if len(sys.argv) > 1 else "k3_mega.png"
fig.savefig(out, dpi=160)
print(out)
