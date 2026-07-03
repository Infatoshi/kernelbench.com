"""Hard-deck status matrix — what came back per (problem x GPU), Fable 5.

Minimal: just the grid, colours, numbers, GPU headers, English problem names.
Green cell = pass (labelled with fraction of roofline). Red = provider
rate-limit / quota. Grey = genuine correctness/compile failure. Two GPUs only
(RTX PRO 6000 + B200); H100 is held out (infra gap, resweep pending).

Regenerate: uv run --with matplotlib python media/make_fable5_hard_status.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kbh_theme import C, apply

apply()

# English problem names, in deck order.
PROBS = ["FP8 GEMM", "Kimi Delta Attention", "Paged Attention",
         "Top-K", "MoE SwiGLU", "W4A16 GEMM"]
GPUS = ["RTX PRO 6000", "B200"]

# (gpu, problem) -> (category, text). pass / rate / fail.
CELLS = {
    ("RTX PRO 6000", "FP8 GEMM"):             ("rate", "rate\nlimited"),
    ("RTX PRO 6000", "Kimi Delta Attention"): ("pass", "0.036"),
    ("RTX PRO 6000", "Paged Attention"):      ("pass", "0.630"),
    ("RTX PRO 6000", "Top-K"):                ("pass", "0.049"),
    ("RTX PRO 6000", "MoE SwiGLU"):           ("pass", "0.108"),
    ("RTX PRO 6000", "W4A16 GEMM"):           ("pass", "0.348"),

    ("B200", "FP8 GEMM"):             ("pass", "0.254"),
    ("B200", "Kimi Delta Attention"): ("fail", "failed\ncheck"),
    ("B200", "Paged Attention"):      ("pass", "0.170"),
    ("B200", "Top-K"):                ("rate", "rate\nlimited"),
    ("B200", "MoE SwiGLU"):           ("pass", "0.076"),
    ("B200", "W4A16 GEMM"):           ("pass", "0.043"),
}

FILL = {"pass": C["accent"], "fail": C["fg_dim"], "rate": C["bad"]}
TXTCOL = {"pass": "#0a0a0a", "fail": C["fg"], "rate": "#0a0a0a"}

nrows, ncols = len(PROBS), len(GPUS)
fig, ax = plt.subplots(figsize=(7.6, 7.2))
fig.subplots_adjust(left=0.31, right=0.97, top=0.90, bottom=0.03)

for j, gpu in enumerate(GPUS):
    for i, prob in enumerate(PROBS):
        y = nrows - 1 - i
        cat, txt = CELLS[(gpu, prob)]
        ax.add_patch(plt.Rectangle((j, y), 0.94, 0.94, facecolor=FILL[cat],
                                   edgecolor=C["bg"], linewidth=2.5))
        ax.text(j + 0.47, y + 0.47, txt, ha="center", va="center",
                color=TXTCOL[cat], fontsize=13 if cat == "pass" else 10.5,
                linespacing=1.15, fontweight="bold" if cat == "pass" else "normal")

ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
ax.set_xticks([j + 0.47 for j in range(ncols)])
ax.set_xticklabels(GPUS, fontsize=12, fontweight="bold")
ax.xaxis.tick_top()
ax.set_yticks([nrows - 1 - i + 0.47 for i in range(nrows)])
ax.set_yticklabels(PROBS, fontsize=11)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)

out = pathlib.Path(__file__).parent / "fable5_hard_status.png"
fig.savefig(out, dpi=170)
print(f"wrote {out}")
