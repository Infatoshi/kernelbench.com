"""KernelBench-Mega grouped bars: per-model speedup across the 3 GPUs.

Clean/minimal: title = problem name, grouped bars colored by GPU, value labels,
GPU legend, no prose. Mega = the Kimi linear-decode megakernel (03).
"""
import sys; sys.path.insert(0, "..")
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from kbh_theme import C, apply
apply()

TITLE = "Kimi Linear Decode Megakernel"
GPUS = ["RTX PRO 6000 Blackwell", "H100", "B200"]
GPU_LABEL = {"RTX PRO 6000 Blackwell": "RTX PRO 6000", "H100": "H100", "B200": "B200"}
GPU_COL = {"RTX PRO 6000 Blackwell": "#4d9fff", "H100": "#b07cff", "B200": C["accent"]}
DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8", "glm-5.2": "GLM-5.2", "gpt-5.5": "GPT-5.5",
    "MiniMax-M3": "MiniMax-M3", "kimi-k2.7-code": "Kimi K2.7-Code",
    "composer-2.5-fast": "Composer 2.5 Fast", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}

# cell[model][gpu] = score (or None)
cell = {}
for r in csv.DictReader(open("../public/data/mega/results.csv")):
    if r["correct"] == "true":
        cell.setdefault(r["model"], {})[r["gpu"]] = float(r["score"])

models = [m for m in DISPLAY if m in cell]
models.sort(key=lambda m: -max([cell[m].get(g, 0) for g in GPUS]))

x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(14, 7))
fig.subplots_adjust(top=0.86, left=0.05, right=0.985, bottom=0.14)
ax.set_facecolor(C["bg"])
for sp in ax.spines.values(): sp.set_color(C["border"])

ymax = max(v for m in models for v in cell[m].values()) * 1.12
for gi, g in enumerate(GPUS):
    off = (gi - 1) * w
    for mi, m in enumerate(models):
        v = cell[m].get(g)
        if v is None:
            ax.text(x[mi] + off, ymax * 0.02, "DNF", ha="center", va="bottom",
                    color=C["fg_dim"], fontsize=7, rotation=90)
            continue
        ax.bar(x[mi] + off, v, w, color=GPU_COL[g], edgecolor=C["bg"], zorder=3)
        ax.text(x[mi] + off, v + ymax * 0.012, f"{v:.1f}", ha="center", va="bottom",
                color=C["fg"], fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels([DISPLAY[m] for m in models], rotation=20, ha="right",
                   fontsize=11, color=C["fg"])
ax.set_ylim(0, ymax)
ax.set_yticks([])
ax.tick_params(length=0)

fig.text(0.05, 0.93, TITLE, color=C["accent"], fontsize=20, fontweight="bold", ha="left")
ax.legend(handles=[Patch(facecolor=GPU_COL[g], label=GPU_LABEL[g]) for g in GPUS],
          loc="upper right", frameon=False, fontsize=12, labelcolor=C["fg"])

fig.savefig("mega_3gpu.png", dpi=150)
print("wrote mega_3gpu.png")
