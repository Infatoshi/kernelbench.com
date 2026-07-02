"""Annotated optimization trajectory — Fable 5's 18.7x Kimi-decode megakernel.

Single run: mega/02_kimi_linear_decode on RTX PRO 6000, 2h33m autonomous
session, 548k output tokens. Y = speedup over reference decode; top axis =
wall clock, bottom axis = cumulative output tokens (nonlinear, marked at
checkpoints). Checkpoints mined from the agent transcript
(20260701_172615_claude_claude-fable-5_02_kimi_linear_decode).

Regenerate: uv run --with matplotlib python media/make_fable5_trajectory.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kbh_theme import C, apply

apply()

SESSION_S = 9208
TOTAL_TOK = 548_275

# (elapsed_s, cum_tokens, speedup, label, (label_x_min, label_y), va)
PTS = [
    (2615, 224_139,  1.00, "baseline timed:\n5.47 ms/tok floor",              (30.0,  2.6), "bottom"),
    (5926, 400_987, 14.38, "single cooperative megakernel v1\npasses check, first benchmark", (63.0, 15.3), "bottom"),
    (6344, 426_619, 13.00, "",                                                None, None),
    (6784, 451_615, 13.65, "MLA softmax +\ntranspose rework",                 (108.0, 10.6), "top"),
    (7267, 470_965, 16.08, "bf16x2 SIMD int4 dequant\n(LOP3 + HSUB2/HMUL2)",  (100.0, 18.3), "bottom"),
    (7572, 483_929, 16.92, "padded smem transpose;\nrouter merged into MoE",  (129.5, 13.9), "top"),
    (7772, 497_416, 17.63, "",                                                None, None),
    (7826, 497_699, 17.13, "KDA state update\nfolded into A1",                (125.0, 19.8), "bottom"),
    (8653, 531_455, 15.94, "finer split-K regresses\n-> measured, reverted",  (141.5, 11.6), "top"),
    (8821, 535_692, 17.97, "MoE gate/up + down fused\nvia spin counters",     (139.5, 20.3), "bottom"),
    (9081, 546_843, 18.03, "",                                                None, None),
    (9159, 547_851, 18.70, "final: MLA barrier folds,\n14 barriers/step -> 18.7x", (154.0, 14.6), "top"),
]

xs = [p[0] / 60 for p in PTS]           # minutes
ys = [p[2] for p in PTS]

fig, ax = plt.subplots(figsize=(13.5, 6.2))
fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.15)

# design phase shading (0 -> first kernel pass)
ax.axvspan(0, 5926 / 60, color=C["surface"], zorder=0)
ax.text(5926 / 120, 20.6,
        "measurement + design: 64% of session\nbaseline timing · grid-barrier microbench · bytes/token roofline (~29x ceiling)",
        ha="center", va="top", color=C["fg_muted"], fontsize=8.5, linespacing=1.4)

ax.axhline(1.0, color=C["fg_dim"], linewidth=0.9, linestyle="--")
ax.text(1.5, 1.35, "reference decode = 1x", color=C["fg_dim"], fontsize=8)

# baseline -> v1: no intermediate kernel existed; draw dashed to avoid
# implying a gradual climb
ax.plot(xs[:2], ys[:2], color=C["accent"], linewidth=1.4, linestyle=(0, (4, 3)),
        alpha=0.55, zorder=3)
ax.plot(xs[1:], ys[1:], color=C["accent"], linewidth=1.8, zorder=3)
ax.scatter(xs, ys, s=26, color=C["accent"], zorder=4)
ax.scatter([xs[-1]], [ys[-1]], s=90, color=C["accent"], zorder=5,
           edgecolor=C["fg_bright"], linewidth=1.2)
# regression point in rose
ax.scatter([8653 / 60], [15.94], s=30, color=C["bad"], zorder=5)

for t, tok, sp, label, pos, va in PTS:
    if not label or pos is None:
        continue
    ax.annotate(label, xy=(t / 60, sp), xytext=pos,
                fontsize=8, color=C["fg"], ha="center", va=va or "center",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=C["fg_dim"], linewidth=0.7,
                                shrinkA=2, shrinkB=3))

ax.set_xlim(0, 168)
ax.set_ylim(0, 22)
ax.set_ylabel("speedup over reference decode step")
ax.grid(axis="y", color=C["grid"], linewidth=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# bottom axis: cumulative output tokens at checkpoint marks (nonlinear)
tok_marks = [(2615, "224k"), (5926, "401k"), (7267, "471k"), (8653, "531k"), (9159, "548k")]
ax.set_xticks([t / 60 for t, _ in tok_marks])
ax.set_xticklabels([lbl for _, lbl in tok_marks], fontsize=9)
ax.set_xlabel(f"cumulative output tokens (total {TOTAL_TOK:,})")

# top axis: wall clock
ax_top = ax.twiny()
ax_top.set_xlim(0, 168)
hour_marks = [0, 30, 60, 90, 120, 150]
ax_top.set_xticks(hour_marks)
ax_top.set_xticklabels([f"{m // 60}h{m % 60:02d}" for m in hour_marks], fontsize=9)
ax_top.set_xlabel("wall clock (session total 2h33m)", color=C["fg_muted"], fontsize=10)
ax_top.spines["top"].set_color(C["border"])
for s in ("right", "bottom", "left"):
    ax_top.spines[s].set_visible(False)

fig.suptitle("One run, annotated — Fable 5 builds an 18.7x Kimi-Linear decode megakernel",
             color=C["fg_bright"], fontsize=13, x=0.06, ha="left", y=0.975)
fig.text(0.06, 0.915, "kernelbench.com/mega · RTX PRO 6000 · every point = the agent benchmarking its own kernel · "
         "profiler-verified single kernel launch per token",
         color=C["fg_muted"], fontsize=9)

out = pathlib.Path(__file__).parent / "fable5_trajectory.png"
fig.savefig(out, dpi=170)
print(f"wrote {out}")
