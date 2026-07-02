"""Hard deck, four-model comparison — one long grouped bar chart.

Fable 5 vs the three strongest rivals (GLM-5.2, Opus 4.8, GPT-5.5) on every
hard problem (RTX PRO 6000). Bars are normalized to the per-problem best so
relative standing is readable across problems whose roofline fractions differ
by 20x; each bar is labeled with its absolute roofline fraction.

Data: benchmarks/hard/results/leaderboard.json (published board).
Regenerate: uv run --with matplotlib python media/make_fable5_hard_4model.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kbh_theme import C, apply

apply()

MODELS = [
    ("Fable 5 [max]", C["accent"]),
    ("GLM-5.2",       "#4d9fff"),
    ("Opus 4.8",      "#b07cff"),
    ("GPT-5.5 [xhigh]", "#cfcfcf"),
]

# peak fractions from results/leaderboard.json (None = no run)
DATA = {  # problem: (fable, glm, opus, gpt)
    "01_fp8_gemm":        (None,   0.4059, 0.3855, 0.3638),
    "02_kda_cutlass":     (0.0358, 0.0323, 0.0552, 0.0373),
    "03_paged_attention": (0.6299, 0.6771, 0.6706, 0.5560),
    "05_topk_bitonic":    (0.0494, 0.0341, 0.0335, 0.0457),
    "06_sonic_moe_swiglu":(0.1075, 0.0980, 0.0864, 0.0989),
    "07_w4a16_gemm":      (0.3477, 0.3207, 0.2355, 0.2025),
}

probs = list(DATA)
n_m = len(MODELS)
width = 0.19
x = np.arange(len(probs))

fig, ax = plt.subplots(figsize=(14, 5.2))
fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.12)

for i, (name, color) in enumerate(MODELS):
    xs = x + (i - (n_m - 1) / 2) * width
    for xi, p in zip(xs, probs):
        vals = DATA[p]
        v = vals[i]
        best = max(v_ for v_ in vals if v_ is not None)
        if v is None:
            ax.text(xi, 0.03, "no run\n(quota)", ha="center", va="bottom",
                    color=C["fg_dim"], fontsize=7.5, rotation=90)
            continue
        rel = v / best
        ax.bar(xi, rel, width * 0.92, color=color,
               edgecolor=C["accent"] if rel == 1.0 else "none",
               linewidth=1.2 if rel == 1.0 else 0)
        ax.text(xi, rel + 0.015, f"{v:.3f}", ha="center", va="bottom",
                color=C["fg"] if rel == 1.0 else C["fg_muted"], fontsize=8,
                rotation=0)

ax.set_xticks(x)
ax.set_xticklabels(probs, fontsize=10)
ax.set_ylim(0, 1.14)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("share of problem best\n(labels: roofline fraction)")
ax.grid(axis="y", color=C["grid"], linewidth=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS]
ax.legend(handles, [n for n, _ in MODELS], loc="upper left",
          bbox_to_anchor=(0.0, 1.14), ncol=4, frameon=False,
          fontsize=10, handlelength=1.2, columnspacing=1.6)

fig.suptitle("KernelBench-Hard — fraction of roofline per problem, top four models",
             color=C["fg_bright"], fontsize=13, x=0.05, ha="left", y=0.97)
fig.text(0.05, 0.90, "RTX PRO 6000 Blackwell · unlimited-time autonomous sessions · "
         "tallest bar in each group = problem leader (absolute values on bars)",
         color=C["fg_muted"], fontsize=9)

out = pathlib.Path(__file__).parent / "fable5_hard_4model.png"
fig.savefig(out, dpi=170)
print(f"wrote {out}")
