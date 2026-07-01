import sys; sys.path.insert(0, "..")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from kbh_theme import C, apply
apply()

PROB = ["01 fp8_gemm", "02 kda_cutlass", "03 paged_attn", "05 topk", "06 sonic_moe", "07 w4a16"]
# all cells correct + clean/interesting (no reward hacks in this set)
DATA = {
    "Claude Sonnet 5":  [0.3412, 0.0437, 0.6482, 0.0489, 0.0670, 0.2184],
    "Claude Opus 4.8":  [0.3855, 0.0552, 0.6706, 0.0335, 0.0864, 0.2355],
    "GPT-5.5":          [0.3638, 0.0373, 0.5560, 0.0457, 0.0989, 0.2025],
    "GLM-5.2":          [0.4059, 0.0323, 0.6771, 0.0341, 0.0980, 0.3207],
}
# Sonnet 5 = NVIDIA green accent (the subject); others legible companions.
MCOL = {"Claude Sonnet 5": C["accent"], "Claude Opus 4.8": "#4d9fff",
        "GPT-5.5": "#b07cff", "GLM-5.2": "#f0883e"}
GREEN_HI = C["accent"]; GREY = C["fg_muted"]
models = list(DATA); x = np.arange(len(PROB)); w = 0.2
fig, ax = plt.subplots(figsize=(14.5, 7.6))
fig.subplots_adjust(top=0.80, left=0.065, right=0.975, bottom=0.10)
ax.set_facecolor(C["bg"])
for spine in ax.spines.values(): spine.set_color(C["border"])
fig.text(0.065, 0.945, "Claude Sonnet 5 on KernelBench-Hard  vs  Opus 4.8 / GPT-5.5 / GLM-5.2",
         color=GREEN_HI, fontsize=16, fontweight="bold", ha="left")
fig.text(0.065, 0.895, "KernelBench-Hard, one unlimited autonomous run per problem, effort=max. bar = peak_fraction of the SM120 (RTX PRO 6000) roofline. all cells correct + audited clean.",
         color=GREY, fontsize=10.5, ha="left")
fig.text(0.065, 0.862, "Sonnet 5 goes 6/6 clean with ZERO reward hacks and tracks the frontier pack - just behind its Opus 4.8 sibling on 5/6, near-ceiling on topk.",
         color=GREY, fontsize=10, ha="left")

for mi, m in enumerate(models):
    off = (mi - 1.5) * w
    for j, s in enumerate(DATA[m]):
        ax.bar(x[j] + off, max(s, 0.004), w, color=MCOL[m], edgecolor=C["bg"], zorder=3)
        ax.text(x[j] + off, max(s, 0.004) + 0.008, f"{s:.3f}", ha="center", va="bottom",
                color=C["fg"], fontsize=7.2, fontweight="bold" if m == "Claude Sonnet 5" else "normal")

ax.set_xticks(x); ax.set_xticklabels(PROB, fontsize=11)
ax.set_ylabel("peak_fraction (fraction of roofline)")
ax.set_ylim(0, 0.74); ax.set_xlim(-0.55, len(PROB) - 0.45)
ax.grid(True, axis="y", alpha=0.5)
leg = [Patch(facecolor=MCOL[m], label=m) for m in models]
ax.legend(handles=leg, loc="upper right", ncol=1, facecolor=C["surface"], edgecolor=C["border"],
          labelcolor=C["fg"], fontsize=9.5, framealpha=0.97)
fig.text(0.065, 0.832, "higher = closer to hardware roofline   |   05 topk is launch-overhead-bound (~0.02 ceiling for every model), so ~0.05 is effectively maxed.",
         color=GREY, fontsize=9, ha="left")
fig.savefig("sonnet5_frontier.png", dpi=140)
print("wrote sonnet5_frontier.png")
