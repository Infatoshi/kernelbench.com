"""KernelBench-Mega: single sorted bar, one model per bar (glanceable).

Title = problem name, bars = per-model speedup over the reference megakernel.
One GPU (RTX PRO 6000 Blackwell). Mega currently = the Kimi linear-decode
megakernel only (02_kimi_linear_decode).
"""
import sys; sys.path.insert(0, "..")
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kbh_theme import C, apply
apply()

GPU = "RTX PRO 6000 Blackwell"
TITLE = "Kimi Linear Decode Megakernel"
DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8", "glm-5.2": "GLM-5.2", "gpt-5.5": "GPT-5.5",
    "MiniMax-M3": "MiniMax-M3", "kimi-k2.7-code": "Kimi K2.7-Code",
    "composer-2.5-fast": "Composer 2.5 Fast", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}

rows = [r for r in csv.DictReader(open("../public/data/mega/results.csv"))
        if r["gpu"] == GPU and r["correct"] == "true"]
rows.sort(key=lambda r: float(r["score"]))  # ascending -> largest at top of barh
labels = [DISPLAY.get(r["model"], r["model"]) for r in rows]
vals = [float(r["score"]) for r in rows]

fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(top=0.86, left=0.20, right=0.965, bottom=0.07)
ax.set_facecolor(C["bg"])
for sp in ax.spines.values(): sp.set_color(C["border"])

y = range(len(rows))
ax.barh(list(y), vals, color=C["accent"], edgecolor=C["bg"], zorder=3, height=0.74)
for yi, v in zip(y, vals):
    ax.text(v + max(vals) * 0.012, yi, f"{v:.1f}x", va="center", ha="left",
            color=C["fg"], fontsize=13, fontweight="bold")
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=13, color=C["fg"])
ax.set_xlim(0, max(vals) * 1.12)
ax.set_xticks([])
ax.grid(axis="x", lw=0)
ax.tick_params(length=0)

fig.text(0.20, 0.93, TITLE, color=C["accent"], fontsize=20, fontweight="bold", ha="left")

fig.savefig("mega_bar.png", dpi=150)
print("wrote mega_bar.png")
