"""The contamination correction — the methodology story.

Before: agents could read prior winning solutions from the shared run archive
(no filesystem sandbox), producing fake "open models crush opus" headline cells.
After: a bwrap-sandboxed harness + a contamination tripwire. The retracted cells
are shown hatched rose (reward-hack convention); the verified-clean numbers in
green. Mega, Blackwell GPU.
"""
import sys; sys.path.insert(0, "..")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from kbh_theme import C, apply
apply()

# model -> (contaminated_published, verified_clean)
DATA = {
    "GLM-5.2":         (17.4, 11.1),
    "MiniMax-M3":      (16.5, 2.6),
    "Claude Opus 4.8": (14.4, 14.4),  # always clean — opus never needed the archive
}
models = list(DATA)
x = np.arange(len(models)); w = 0.36
fig, ax = plt.subplots(figsize=(12.5, 7.2))
fig.subplots_adjust(top=0.79, left=0.07, right=0.975, bottom=0.10)
ax.set_facecolor(C["bg"])
for sp in ax.spines.values(): sp.set_color(C["border"])

for mi, m in enumerate(models):
    fake, real = DATA[m]
    # retracted / contaminated bar (rose, hatched)
    if fake != real:
        ax.bar(x[mi] - w / 2, fake, w, color=C["bad"], hatch="////",
               edgecolor=C["bg"], zorder=3, alpha=0.85)
        ax.text(x[mi] - w / 2, fake + 0.2, f"{fake:.1f}x", ha="center", va="bottom",
                color=C["bad"], fontsize=9, fontweight="bold")
        ax.text(x[mi] - w / 2, fake / 2, "RETRACTED\ncontaminated", ha="center",
                va="center", color=C["fg_bright"], fontsize=8, fontweight="bold")
    # verified clean bar (green)
    ax.bar(x[mi] + w / 2, real, w, color=C["accent"], edgecolor=C["bg"], zorder=3)
    ax.text(x[mi] + w / 2, real + 0.2, f"{real:.1f}x", ha="center", va="bottom",
            color=C["fg"], fontsize=9, fontweight="bold")
    if fake == real:
        ax.text(x[mi] + w / 2, real / 2, "clean\nall along", ha="center", va="center",
                color=C["bg_depth"], fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11.5, color=C["fg"])
ax.set_ylabel("megakernel speedup over reference", fontsize=10.5)
ax.set_ylim(0, 19.5)
ax.grid(axis="y", lw=0.6, alpha=0.5); ax.set_axisbelow(True)

fig.text(0.07, 0.945, "The \"open models beat Opus\" headline was contamination",
         color=C["accent"], fontsize=17, fontweight="bold", ha="left")
fig.text(0.07, 0.895, "KernelBench-Mega, Blackwell. No filesystem sandbox: agents could read prior winning solutions from the shared run archive.",
         color=C["fg_muted"], fontsize=10.5, ha="left")
fig.text(0.07, 0.862, "A bwrap sandbox + contamination tripwire removed 8 fake cells. Verified-clean: Opus still wins; GLM-5.2 is the genuine challenger.",
         color=C["fg_muted"], fontsize=10, ha="left")

ax.legend(handles=[Patch(facecolor=C["bad"], hatch="////", label="retracted (contaminated)"),
                   Patch(facecolor=C["accent"], label="verified clean (sandboxed)")],
          loc="upper right", frameon=False, fontsize=10.5, labelcolor=C["fg"])

fig.savefig("contamination.png", dpi=150)
print("wrote contamination.png")
