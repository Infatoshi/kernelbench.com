"""Hard-deck four-model bar charts, per GPU — Fable 5 vs GLM-5.2 / Opus 4.8 / GPT-5.5.

Emits four candidate layouts so we can pick what reads best on X:
  fable5_bars_rtx.png        RTX PRO 6000 alone
  fable5_bars_b200.png       B200 alone
  fable5_bars_side.png       RTX | B200 side by side (wide)
  fable5_bars_stacked.png    RTX over B200 (tall)

Bars are normalized to the per-problem best (roofline fractions differ ~20x
across problems), each labelled with its absolute fraction of roofline.
Data: benchmarks/hard/results/leaderboard.json + leaderboard.b200.json.

Regenerate: uv run --with matplotlib python media/make_fable5_hard_bars.py
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
    ("Fable 5 [max]",   C["accent"]),
    ("GLM-5.2",         "#4d9fff"),
    ("Opus 4.8",        "#b07cff"),
    ("GPT-5.5 [xhigh]", "#cfcfcf"),
]

PROBS = ["FP8 GEMM", "Kimi Delta\nAttention", "Paged\nAttention",
         "Top-K", "MoE\nSwiGLU", "W4A16 GEMM"]

# (fable, glm, opus, gpt); None = no valid result
RTX = {
    "FP8 GEMM":              (None,   0.4059, 0.3855, 0.3638),
    "Kimi Delta\nAttention": (0.0358, 0.0323, 0.0552, 0.0373),
    "Paged\nAttention":      (0.6299, 0.6771, 0.6706, 0.5560),
    "Top-K":                 (0.0494, 0.0341, 0.0335, 0.0457),
    "MoE\nSwiGLU":           (0.1075, 0.0980, 0.0864, 0.0989),
    "W4A16 GEMM":            (0.3477, 0.3207, 0.2355, 0.2025),
}
B200 = {
    "FP8 GEMM":              (0.2535, 0.1997, 0.1962, 0.1459),
    "Kimi Delta\nAttention": (None,   0.0110, 0.0133, 0.0006),
    "Paged\nAttention":      (0.1696, 0.2639, 0.2560, 0.1627),
    "Top-K":                 (None,   0.0018, 0.0027, 0.0057),
    "MoE\nSwiGLU":           (0.0762, 0.0584, 0.0790, 0.1063),
    "W4A16 GEMM":            (0.0434, 0.0483, 0.0270, None),
}
# short reason for Fable's missing cells (index 0)
MISS = {
    ("RTX", "FP8 GEMM"): "rate\nlimit",
    ("B200", "Kimi Delta\nAttention"): "fail",
    ("B200", "Top-K"): "rate\nlimit",
}

width = 0.19
n_m = len(MODELS)


def draw_panel(ax, data, tag, title, show_ylabel=True):
    x = np.arange(len(PROBS))
    for i, (name, color) in enumerate(MODELS):
        xs = x + (i - (n_m - 1) / 2) * width
        for xi, p in zip(xs, PROBS):
            vals = data[p]
            v = vals[i]
            best = max(v_ for v_ in vals if v_ is not None)
            if v is None:
                if i == 0 and (tag, p) in MISS:
                    ax.text(xi, 0.03, MISS[(tag, p)], ha="center", va="bottom",
                            color=C["fg_dim"], fontsize=7, rotation=90)
                continue
            rel = v / best
            ax.bar(xi, rel, width * 0.92, color=color,
                   edgecolor=C["fg_bright"] if rel == 1.0 else "none",
                   linewidth=1.1 if rel == 1.0 else 0)
            ax.text(xi, rel + 0.015, f"{v:.3f}", ha="center", va="bottom",
                    color=C["fg"] if rel == 1.0 else C["fg_muted"], fontsize=7.4)
    ax.set_xticks(x)
    ax.set_xticklabels(PROBS, fontsize=9.5)
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if show_ylabel:
        ax.set_ylabel("share of problem best\n(labels: fraction of roofline)", fontsize=9.5)
    ax.grid(axis="y", color=C["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, color=C["fg_bright"], fontsize=12, loc="left", pad=8)


def legend_on(fig, y):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in MODELS]
    fig.legend(handles, [n for n, _ in MODELS], loc="upper left",
               bbox_to_anchor=(0.008, y), ncol=4, frameon=False,
               fontsize=10, handlelength=1.2, columnspacing=1.8)


here = pathlib.Path(__file__).parent

# --- RTX alone ---
fig, ax = plt.subplots(figsize=(13, 5.0))
fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.11)
draw_panel(ax, RTX, "RTX", "KernelBench-Hard · RTX PRO 6000 Blackwell")
legend_on(fig, 0.99)
fig.savefig(here / "fable5_bars_rtx.png", dpi=170)

# --- B200 alone ---
fig, ax = plt.subplots(figsize=(13, 5.0))
fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.11)
draw_panel(ax, B200, "B200", "KernelBench-Hard · NVIDIA B200 (SM100)")
legend_on(fig, 0.99)
fig.savefig(here / "fable5_bars_b200.png", dpi=170)

# --- side by side ---
fig, axes = plt.subplots(1, 2, figsize=(19, 5.6))
fig.subplots_adjust(left=0.05, right=0.995, top=0.78, bottom=0.10, wspace=0.13)
draw_panel(axes[0], RTX, "RTX", "RTX PRO 6000 Blackwell")
draw_panel(axes[1], B200, "B200", "NVIDIA B200 (SM100)", show_ylabel=False)
fig.suptitle("KernelBench-Hard — Fable 5 vs top rivals, fraction of roofline per problem",
             color=C["fg_bright"], fontsize=13, x=0.05, ha="left", y=0.965)
legend_on(fig, 0.905)
fig.savefig(here / "fable5_bars_side.png", dpi=160)

# --- stacked ---
fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.6))
fig.subplots_adjust(left=0.10, right=0.99, top=0.89, bottom=0.06, hspace=0.34)
draw_panel(axes[0], RTX, "RTX", "RTX PRO 6000 Blackwell")
draw_panel(axes[1], B200, "B200", "NVIDIA B200 (SM100)")
fig.suptitle("KernelBench-Hard — Fable 5 vs top rivals, fraction of roofline per problem",
             color=C["fg_bright"], fontsize=13, x=0.10, ha="left", y=0.975)
legend_on(fig, 0.945)
fig.savefig(here / "fable5_bars_stacked.png", dpi=160)

print("wrote fable5_bars_{rtx,b200,side,stacked}.png")
