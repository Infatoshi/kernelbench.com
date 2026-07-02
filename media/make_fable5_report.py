"""Fable 5 launch report chart — mega decode headline + hard deck vs field.

Two panels:
  left  — KernelBench-Mega 02_kimi_linear_decode speedup over reference decode
          (RTX PRO 6000). Fable 5 is the first entry whose submission is a
          judged-authentic single fused megakernel; every other bar is a
          multi-kernel pipeline that failed the authenticity gate.
  right — KernelBench-Hard (RTX PRO 6000): Fable 5's peak fraction relative to
          the best non-Fable model per problem (1.0 line = field best).

Regenerate: uv run --with matplotlib python media/make_fable5_report.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kbh_theme import C, apply

apply()

# ---- data (from results/leaderboard.json + public/data/mega/results.csv) ----
mega = [  # (label, speedup, is_subject, single_fused_kernel)
    ("Fable 5 [max]",    18.71, True,  True),
    ("Opus 4.8",         14.40, False, False),
    ("GLM-5.2",          11.14, False, False),
    ("GPT-5.5 [xhigh]",   4.34, False, False),
    ("Sonnet 5 [max]",    4.03, False, False),
    ("MiniMax-M3",        2.63, False, False),
    ("Kimi K2.7",         2.59, False, False),
]

hard = [  # (problem, fable_pf, best_other_pf, best_other_model)
    ("07_w4a16_gemm",     0.3477, 0.3207, "GLM-5.2"),
    ("06_sonic_moe",      0.1075, 0.1032, "Kimi K2.7"),
    ("05_topk_bitonic",   0.0494, 0.0489, "Sonnet 5"),
    ("03_paged_attn",     0.6299, 0.6771, "GLM-5.2"),
    ("02_kda_cutlass",    0.0358, 0.0552, "Opus 4.8"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), width_ratios=[1.15, 1])
fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.14, wspace=0.28)

# ---- left: mega decode ----
labels = [m[0] for m in mega][::-1]
vals   = [m[1] for m in mega][::-1]
colors = [C["accent"] if m[2] else C["fg_dim"] for m in mega][::-1]
bars = ax1.barh(labels, vals, color=colors, height=0.62)
for b, v, m in zip(bars, vals, mega[::-1]):
    ax1.text(v + 0.25, b.get_y() + b.get_height() / 2, f"{v:.1f}x",
             va="center", color=C["fg"], fontsize=10)
ax1.text(18.71, len(mega) - 1 + 0.42, "single fused kernel — judged authentic",
         va="bottom", ha="right", color=C["accent"], fontsize=8.5)
ax1.set_xlim(0, 22.5)
ax1.set_xlabel("speedup over reference decode step")
ax1.set_title("Mega: Kimi-Linear W4A16 decode megakernel\nRTX PRO 6000 Blackwell",
              color=C["fg"], fontsize=11, loc="left")
ax1.grid(axis="x", color=C["grid"], linewidth=0.7)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# ---- right: hard deck vs field best ----
probs  = [h[0] for h in hard][::-1]
ratios = [h[1] / h[2] for h in hard][::-1]
whos   = [h[3] for h in hard][::-1]
cols   = [C["accent"] if r >= 1 else C["fg_dim"] for r in ratios]
bars2 = ax2.barh(probs, ratios, color=cols, height=0.55)
ax2.axvline(1.0, color=C["warn"], linewidth=1.1, linestyle="--")
ax2.text(0.985, -0.42, "best rival model ", color=C["warn"], fontsize=8.5,
         va="top", ha="right")
for b, r, w, h_ in zip(bars2, ratios, whos, hard[::-1]):
    ax2.text(r + 0.02, b.get_y() + b.get_height() / 2,
             f"{h_[1]:.3f} vs {h_[2]:.3f}\n{w}",
             va="center", color=C["fg_muted"], fontsize=8, linespacing=1.3)
ax2.set_xlim(0, 1.6)
ax2.set_xlabel("relative to best rival (labels: absolute roofline fractions)")
ax2.set_title("Hard: Fable 5 vs best rival, per problem\nRTX PRO 6000 — 1.0 line = tied with field best",
              color=C["fg"], fontsize=11, loc="left")
ax2.grid(axis="x", color=C["grid"], linewidth=0.7)
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.suptitle("Claude Fable 5 on KernelBench — unlimited-time autonomous kernel engineering",
             color=C["fg_bright"], fontsize=13, x=0.13, ha="left")
fig.text(0.13, 0.895, "kernelbench.com  ·  hard: 3 of 6 problem ceilings taken  ·  "
         "mega: first judged-authentic single-kernel submission, 18.7x",
         color=C["fg_muted"], fontsize=9)

out = pathlib.Path(__file__).parent / "fable5_report.png"
fig.savefig(out, dpi=170)
print(f"wrote {out}")
