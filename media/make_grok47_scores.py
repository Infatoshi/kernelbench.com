"""Grok 4.7 [xhigh] thread, post 1: all five cells in one picture.
Left: Mega Kimi-Linear Decode (speedup). Right: the four CUDA problems (peak fraction).
Field = published boards 2026-09-17 (benchmarks/cuda/results/leaderboard.json, public/data/mega/results.csv),
RTX PRO 6000, best per model. Grok 4.7 = isolated regrade on tetra GPU 1.

  uv run --no-project --with matplotlib,numpy python media/make_grok47_scores.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply
apply()
OUT = Path(__file__).resolve().parent / "grok47_scores.png"
MODELS = ["Fable 5.1", "GPT-6 Astra", "GLM-5.3", "Grok 4.6", "Grok 4.7"]
COL = {"Fable 5.1": "#8a9aa5", "GPT-6 Astra": "#5f7380", "GLM-5.3": "#3f4f58", "Grok 4.6": "#4d9fff", "Grok 4.7": C["accent"]}
MEGA = {"Fable 5.1": 22.946, "GPT-6 Astra": 24.804, "GLM-5.3": 19.435, "Grok 4.6": None, "Grok 4.7": 6.379}
MEGA_TAG = {"Grok 4.6": "excluded: contamination"}
CUDA_LABELS = ["Fused MoE", "NSA", "MegaQwen", "MinGRU"]
CUDA = {
    "Fable 5.1":   [0.1017, 1.0627, 0.0643, 0.7093],
    "GPT-6 Astra": [0.0945, 0.3946, 0.0490, 0.6837],
    "GLM-5.3":     [0.0997, None,   None,   None],
    "Grok 4.6":    [0.0939, 0.0796, 0.0542, None],
    "Grok 4.7":    [0.0948, 0.1002, 0.0455, 0.6398],
}

fig, (axm, axc) = plt.subplots(1, 2, figsize=(16, 8), dpi=150, gridspec_kw={"width_ratios": [1, 1.35]})
fig.patch.set_facecolor(C["bg"])
for ax in (axm, axc):
    ax.set_facecolor(C["bg"]); ax.grid(axis="x", color=C["grid"], linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=C["fg_muted"], labelsize=11)

# left: mega rank, sorted, one bar per model
order = sorted(MODELS, key=lambda m: -(MEGA[m] or 0))
y = np.arange(len(order))[::-1]
span = max(v for v in MEGA.values() if v)
for yi, m in zip(y, order):
    v = MEGA[m]
    if v is None:
        axm.barh(yi, span * 0.06, 0.62, color=C["bad"], zorder=3)
        axm.text(span * 0.06 + span * 0.02, yi, MEGA_TAG[m], va="center", fontsize=11, color=C["bad"], fontweight="bold")
        continue
    hi = m == "Grok 4.7"
    axm.barh(yi, v, 0.62, color=COL[m], zorder=3)
    axm.text(v + span * 0.02, yi, f"{v:.1f}x", va="center", fontsize=13 if hi else 12,
             color=C["fg_bright"] if hi else C["fg_muted"], fontweight="bold" if hi else "regular")
axm.set_yticks(y); axm.set_yticklabels(order, fontsize=14, color=C["fg"])
axm.set_xlim(0, span * 1.35)
axm.set_title("Mega · Kimi-Linear Decode\nspeedup vs optimized PyTorch", fontsize=13, color=C["fg"], loc="left", pad=12)

# right: cuda grouped
n, k = len(CUDA_LABELS), len(MODELS)
yy = np.arange(n)[::-1]
h = 0.72 / k
present = [v for m in MODELS for v in CUDA[m] if v is not None]
spanc = max(present)
for si, m in enumerate(MODELS):
    offs = ((k - 1) / 2 - si) * h
    for i, v in enumerate(CUDA[m]):
        if v is None:
            axc.barh(yy[i] + offs, spanc * 0.05, h * 0.9, color=C["bad"], zorder=3)
            axc.text(spanc * 0.05 + spanc * 0.012, yy[i] + offs, "no cell", va="center", fontsize=9, color=C["bad"], fontweight="bold")
            continue
        hi = m == "Grok 4.7"
        axc.barh(yy[i] + offs, v, h * 0.9, color=COL[m], zorder=3)
        axc.text(v + spanc * 0.012, yy[i] + offs, f"{v:.3f}", va="center", fontsize=10 if hi else 9,
                 color=C["fg_bright"] if hi else C["fg_muted"], fontweight="bold" if hi else "regular")
axc.set_yticks(yy); axc.set_yticklabels(CUDA_LABELS, fontsize=14, color=C["fg"])
axc.set_xlim(0, spanc * 1.18)
axc.set_title("CUDA · four problems\npeak fraction of roofline", fontsize=13, color=C["fg"], loc="left", pad=12)
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[m], label=m) for m in MODELS]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=12, labelcolor=C["fg"], bbox_to_anchor=(0.5, 0.01))
fig.suptitle("Grok 4.7 [xhigh] on KernelBench · RTX PRO 6000 · isolated regrade, audited", fontsize=16, color=C["fg_bright"], x=0.03, ha="left", y=0.97)
fig.text(0.03, 0.925, "kernelbench.com · field = best published cell per model on the same GPU", fontsize=11, color=C["fg_muted"])
fig.subplots_adjust(left=0.10, right=0.97, top=0.84, bottom=0.12, wspace=0.42)
fig.savefig(OUT, dpi=150, facecolor=C["bg"])
print(OUT)
