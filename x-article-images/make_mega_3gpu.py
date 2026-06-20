"""Mega megakernel speedup, 8 frontier models across Blackwell / H100 / B200.

Hero chart for the trace-release post. Speedup over the reference megakernel
(W4A16 linear decode), clean post-contamination numbers from
public/data/mega/results.csv. opus = NVIDIA green (the ceiling/subject).
"""
import sys; sys.path.insert(0, "..")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from kbh_theme import C, apply
apply()

GPUS = ["Blackwell", "H100", "B200"]
# model -> [blackwell, h100, b200]  (None = no clean run on that GPU)
DATA = {
    "Claude Opus 4.8":   [14.4, 15.5, 19.4],
    "GLM-5.2":           [11.1, None, 7.3],
    "GPT-5.5":           [4.3, 5.6, 9.4],
    "MiniMax-M3":        [2.6, None, 4.2],
    "Gemini 3.5 Flash":  [2.3, 2.7, 2.5],
    "Kimi K2.7-Code":    [2.6, None, None],
    "Composer 2.5 Fast": [2.5, 1.8, 1.2],
    "DeepSeek V4 Pro":   [2.1, 1.4, 1.6],
}
# GPU bar shades: B200 = green accent (top hardware), others legible companions.
GPU_COL = {"Blackwell": "#4d9fff", "H100": "#b07cff", "B200": C["accent"]}

models = list(DATA)
x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(14.5, 7.4))
fig.subplots_adjust(top=0.80, left=0.06, right=0.985, bottom=0.16)
ax.set_facecolor(C["bg"])
for sp in ax.spines.values(): sp.set_color(C["border"])

for gi, g in enumerate(GPUS):
    off = (gi - 1) * w
    for mi, m in enumerate(models):
        v = DATA[m][gi]
        if v is None:
            ax.text(x[mi] + off, 0.3, "DNF", ha="center", va="bottom",
                    color=C["fg_dim"], fontsize=7, rotation=90)
            continue
        ax.bar(x[mi] + off, v, w, color=GPU_COL[g], edgecolor=C["bg"], zorder=3)
        ax.text(x[mi] + off, v + 0.25, f"{v:.1f}", ha="center", va="bottom",
                color=C["fg"], fontsize=7.6,
                fontweight="bold" if m == "Claude Opus 4.8" else "normal")

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=22, ha="right", fontsize=10, color=C["fg"])
ax.set_ylabel("speedup over reference megakernel  (higher is better)", fontsize=10.5)
ax.set_ylim(0, 21)
ax.grid(axis="y", lw=0.6, alpha=0.5); ax.set_axisbelow(True)

fig.text(0.06, 0.945, "KernelBench-Mega: frontier coding agents building a GPU megakernel",
         color=C["accent"], fontsize=17, fontweight="bold", ha="left")
fig.text(0.06, 0.895, "Speedup over the reference W4A16 linear-decode megakernel. One sandboxed autonomous run per (model, GPU). Contamination-audited: 0 hacked cells.",
         color=C["fg_muted"], fontsize=10.5, ha="left")
fig.text(0.06, 0.862, "Claude Opus 4.8 wins on every GPU (14.4x -> 15.5x -> 19.4x). GLM-5.2 is the real open-weight challenger at 11.1x on Blackwell.",
         color=C["fg_muted"], fontsize=10, ha="left")

ax.legend(handles=[Patch(facecolor=GPU_COL[g], label=g) for g in GPUS],
          loc="upper right", frameon=False, fontsize=10.5, labelcolor=C["fg"])

fig.savefig("mega_3gpu.png", dpi=150)
print("wrote mega_3gpu.png")
