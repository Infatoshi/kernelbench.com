"""KernelBench-Hard: problems solved per model across RTX PRO 6000 / H100 / B200.

Pass count (correct kernels out of 6) is the only honestly cross-GPU-comparable
metric here: peak_fraction is measured against each GPU's own (different)
roofline, so raw fractions are not comparable across GPUs. Budget caveat is
stated in the subtitle (RTX = unlimited time, H100/B200 = 45-min).

Reads the live leaderboards directly so it stays in sync with the site.
"""
import sys; sys.path.insert(0, "..")
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from kbh_theme import C, apply
apply()

FILES = [
    ("RTX PRO 6000", "../benchmarks/hard/results/leaderboard.json"),
    ("H100", "../benchmarks/hard/results/leaderboard.h100.json"),
    ("B200", "../benchmarks/hard/results/leaderboard.b200.json"),
]
GPUS = [g for g, _ in FILES]
GPU_COL = {"RTX PRO 6000": "#4d9fff", "H100": "#b07cff", "B200": C["accent"]}

def short(label):
    return label.split(" [")[0].split("/")[-1]

pc = {}
for g, f in FILES:
    for m in json.load(open(f))["models"]:
        pc.setdefault(short(m["label"]), {})[g] = m["pass_count"]
models = [m for m in pc if all(g in pc[m] for g in GPUS)]
models.sort(key=lambda m: -sum(pc[m][g] for g in GPUS))

DISPLAY = {
    "gpt-5.5": "GPT-5.5", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "claude-opus-4-8": "Claude Opus 4.8", "composer-2.5-fast": "Composer 2.5 Fast",
    "deepseek-v4-pro": "DeepSeek V4 Pro", "kimi-k2.7-code": "Kimi K2.7-Code",
    "glm-5.2": "GLM-5.2", "MiniMax-M3": "MiniMax-M3",
}

x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(14.5, 7.4))
fig.subplots_adjust(top=0.80, left=0.055, right=0.985, bottom=0.16)
ax.set_facecolor(C["bg"])
for sp in ax.spines.values(): sp.set_color(C["border"])

for gi, g in enumerate(GPUS):
    off = (gi - 1) * w
    for mi, m in enumerate(models):
        v = pc[m][g]
        ax.bar(x[mi] + off, v, w, color=GPU_COL[g], edgecolor=C["bg"], zorder=3)
        ax.text(x[mi] + off, v + 0.06, str(v), ha="center", va="bottom",
                color=C["fg"], fontsize=8.5,
                fontweight="bold" if m == "claude-opus-4-8" else "normal")

ax.set_xticks(x)
ax.set_xticklabels([DISPLAY.get(m, m) for m in models], rotation=22, ha="right",
                   fontsize=10, color=C["fg"])
ax.set_ylabel("problems solved  (out of 6, higher is better)", fontsize=10.5)
ax.set_ylim(0, 6.6)
ax.set_yticks(range(0, 7))
ax.grid(axis="y", lw=0.6, alpha=0.5); ax.set_axisbelow(True)

fig.text(0.055, 0.945, "KernelBench-Hard: a curated hard CUDA/Triton deck across three NVIDIA GPUs",
         color=C["accent"], fontsize=17, fontweight="bold", ha="left")
fig.text(0.055, 0.895, "Problems solved (correct kernel out of 6) per model on RTX PRO 6000 / H100 / B200. Roofline-graded; one autonomous agent run per (model, problem, GPU).",
         color=C["fg_muted"], fontsize=10.5, ha="left")
fig.text(0.055, 0.862, "Note: RTX PRO 6000 runs are unlimited-time; H100 and B200 use a 45-min budget, so lower H100/B200 bars partly reflect less agent time.",
         color=C["fg_muted"], fontsize=10, ha="left")

ax.legend(handles=[Patch(facecolor=GPU_COL[g], label=g) for g in GPUS],
          loc="upper right", frameon=False, fontsize=10.5, labelcolor=C["fg"])

fig.savefig("hard_3gpu.png", dpi=150)
print("wrote hard_3gpu.png")
