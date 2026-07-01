import sys; sys.path.insert(0, "..")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kbh_theme import C, apply
apply()

# (label, peak_fraction, rank_string, field_size)
CELLS = [
    ("01 fp8_gemm",   0.3412, "#8/10"),
    ("02 kda_cutlass", 0.0437, "#2/7"),
    ("03 paged_attn", 0.6482, "#3/10"),
    ("05 topk",       0.0489, "#2/9"),
    ("06 sonic_moe",  0.0670, "#9/10"),
    ("07 w4a16",      0.2184, "#4/10"),
]
labels = [c[0] for c in CELLS]
vals = [c[1] for c in CELLS]
ranks = [c[2] for c in CELLS]
# green when top-half rank, dimmer green when bottom-pack, to show strong vs weak.
strong = {1, 1, 1}  # noqa
col = []
for r in ranks:
    n = int(r.split("/")[0][1:]); tot = int(r.split("/")[1])
    col.append(C["accent"] if n <= max(2, tot // 2) else "#3f5a12")

x = np.arange(len(CELLS))
fig, ax = plt.subplots(figsize=(12.5, 7.0))
fig.subplots_adjust(top=0.80, left=0.075, right=0.965, bottom=0.10)
ax.set_facecolor(C["bg"])
for spine in ax.spines.values(): spine.set_color(C["border"])
fig.text(0.075, 0.945, "Claude Sonnet 5 - KernelBench-Hard report card (6/6 correct, 6/6 clean)",
         color=C["accent"], fontsize=16, fontweight="bold", ha="left")
fig.text(0.075, 0.895, "bar = peak_fraction of the SM120 (RTX PRO 6000) roofline; label = Sonnet 5's rank vs every other model on that problem.",
         color=C["fg_muted"], fontsize=10.5, ha="left")
fig.text(0.075, 0.862, "Strong where correctness is the wall (kda #2, paged #3, topk #2/near-ceiling). Bottom-pack on the throughput-tuning GEMMs (fp8 #8, sonic_moe #9).",
         color=C["fg_muted"], fontsize=10, ha="left")

bars = ax.bar(x, vals, 0.62, color=col, edgecolor=C["bg"], zorder=3)
for j, (v, r) in enumerate(zip(vals, ranks)):
    ax.text(x[j], v + 0.009, f"{v:.3f}", ha="center", va="bottom", color=C["fg"], fontsize=10, fontweight="bold")
    ax.text(x[j], v + 0.045, r, ha="center", va="bottom", color=C["accent"] if r.startswith(("#1", "#2", "#3")) else C["fg_muted"], fontsize=10.5, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("peak_fraction (fraction of roofline)")
ax.set_ylim(0, 0.74); ax.set_xlim(-0.6, len(CELLS) - 0.4)
ax.grid(True, axis="y", alpha=0.5)
ax.text(0.985, 0.74, "green = top-half finish   |   dark = bottom-pack",
        transform=ax.transAxes, ha="right", va="top", color=C["fg_muted"], fontsize=9)
fig.savefig("sonnet5_report.png", dpi=140)
print("wrote sonnet5_report.png")
