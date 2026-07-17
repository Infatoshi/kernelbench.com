"""K3 article — KernelBench-CUDA deck (RTX PRO 6000), horizontal bars per problem.

The [1M] megaqwen cell is omitted (audit verdict: suspect — tainted provenance).
K3 mingru is round-1; round-2 session still in flight at draft time.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from kbh_theme import C, SERIES, apply, tight_square

apply()

PROBLEMS = ["glm52_fused_moe", "deepseek_nsa", "megaqwen_decode", "grid_mingru"]
# grid_mingru values are the quiet-GPU re-benchmark (anvil RTX PRO 6000,
# 2026-07-17, official benchmark.py protocol; scheduling-sensitive kernels
# need an idle box for stable timing).
DATA = {
    "Kimi K3":      [0.0595, 0.4246, 0.0470, 0.174],
    "Kimi K3 [1M]": [0.0810, 0.0584, None,   0.224],
    "Opus 4.8":     [0.0653, 0.1784, 0.0097, 0.327],
    "Grok 4.5":     [0.0844, 0.0177, 0.0345, 0.0020],
}
MODELS = list(DATA)
# Model->color mapping shared across all article charts (matches make_k3_hard).
MODEL_COLORS = {"Kimi K3": SERIES[0], "Kimi K3 [1M]": SERIES[1],
                "Opus 4.8": SERIES[3], "Grok 4.5": C["warn"]}

fig, ax = tight_square(size=10.0)
fig.subplots_adjust(left=0.185)
nm = len(MODELS)
bar_h = 0.8 / nm
ys = np.arange(len(PROBLEMS))[::-1]
XLIM = 0.5

for mi, model in enumerate(MODELS):
    color = MODEL_COLORS[model]
    for pi, val in enumerate(DATA[model]):
        y = ys[pi] + (nm / 2 - mi - 0.5) * bar_h
        if val is None:
            continue
        ax.barh(y, val, height=bar_h * 0.9, color=color,
                zorder=3, label=model if pi == 0 else None)
        ax.text(val + XLIM * 0.008, y, f"{val:.3f}", va="center", ha="left",
                color=C["fg_muted"], fontsize=8)

ax.set_yticks(ys)
ax.set_yticklabels(PROBLEMS, fontsize=10)
ax.set_xlabel("fraction of roofline peak (CUDA-only deck)", fontsize=10)
ax.set_xlim(0, XLIM)
ax.grid(axis="x", zorder=0)
ax.legend(loc="lower right", fontsize=9, framealpha=0.15,
          facecolor=C["surface"], edgecolor=C["border"])
ax.text(0.985, 0.985, "RTX PRO 6000 Blackwell", transform=ax.transAxes,
        ha="right", va="top", color=C["fg_dim"], fontsize=9)

out = sys.argv[1] if len(sys.argv) > 1 else "k3_cuda.png"
fig.savefig(out, dpi=160)
print(out)
