"""Annotated optimization trajectory — Fable 5's hand-written tcgen05 fp8 GEMM on B200.

Single run: hard/01_fp8_gemm on a rented NVIDIA B200 (SM100), 2h44m autonomous
session, 434k output tokens. Y = fraction of B200 fp8 roofline (agent's own
in-session benchmark of its kernel); top axis = wall clock, bottom axis =
cumulative output tokens. Checkpoints mined from the agent transcript
(20260702_090059_claude_claude-fable-5_01_fp8_gemm), /tmp/fable_b200_fp8_trajectory.json.

The 6-shape average plateaus near 28% because it is dominated by the
memory-bound skinny decode shape (~3% of the fp8 ceiling, which no tensor-core
kernel can help). The hand-written tcgen05 path is what pushes the
compute-bound GEMMs to 44-59% of fp8 peak.

Regenerate: uv run --with matplotlib python media/make_fable5_b200_fp8.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kbh_theme import C, apply

apply()

SESSION_S = 9868
TOTAL_TOK = 433_871

# (elapsed_s, cum_tokens, pct_of_fp8_peak, label, (label_x_min, label_y), va)
PTS = [
    (654,  21_488,  15.67, "Triton fp8 kernel\npasses check",          (13.0,  9.0),  "top"),
    (1589, 92_186,  21.92, "CUDA graphs +\nTriton autotune",           (30.0, 15.0),  "top"),
    (1934, 110_145, 25.10, "128B-aligned pad\nkernel for odd K",       (44.0, 20.6),  "bottom"),
    (5635, 299_371, 24.97, "", None, None),
    (5775, 305_462, 27.34, "Triton epilogue tuning\n(wide-N shape 69% peak)", (78.0, 20.0), "top"),
    (6155, 340_931, 28.11, "split-K decode\npath for M=32",            (104.0, 21.4), "bottom"),
    (9606, 420_305, 28.12, "hand tcgen05 2-CTA\ncluster MMA goes live", (139.0, 21.6), "bottom"),
    (9833, 429_532, 28.31, "swizzled TMA epilogue\n+ fused scale",     (156.0, 25.0), "top"),
]

xs = [p[0] / 60 for p in PTS]
ys = [p[2] for p in PTS]

fig, ax = plt.subplots(figsize=(13.5, 6.2))
fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.15)

# tcgen05 hand-kernel development region (odd-K pad done -> hand MMA live)
ax.axvspan(1934 / 60, 9606 / 60, color=C["surface"], zorder=0)
ax.text((1934 + 9606) / 120, 4.0,
        "hand-writing raw SM100 tcgen05 PTX in the background: 2-CTA cluster tcgen05.mma,\n"
        "TMEM accumulators, TMA 3D loads + mbarrier pipeline, swizzled fused-scale store epilogue",
        ha="center", va="bottom", color=C["fg_muted"], fontsize=8.5, linespacing=1.4)

# per-shape reality band: compute-bound GEMMs hit 44-59% of fp8 peak
ax.axhspan(44, 59, color=C["accent"], alpha=0.09, zorder=0)
ax.text(166.5, 51.5, "compute-bound\nGEMM shapes:\n44-59% of\nfp8 peak", ha="right",
        va="center", color=C["accent"], fontsize=8.5, linespacing=1.35)

ax.plot(xs, ys, color=C["accent"], linewidth=1.8, zorder=3)
ax.scatter(xs, ys, s=26, color=C["accent"], zorder=4)

# final published regrade (session self-measure 28.3% -> official 25.4%)
ax.scatter([SESSION_S / 60], [25.35], s=95, color=C["accent"], zorder=5,
           edgecolor=C["fg_bright"], linewidth=1.2)
ax.annotate("published: 25.4% avg\n(6-shape mean; session\nself-measure 28.3%)",
            xy=(SESSION_S / 60, 25.35), xytext=(150.0, 33.5),
            fontsize=8, color=C["fg"], ha="center", va="bottom", linespacing=1.35,
            arrowprops=dict(arrowstyle="-", color=C["fg_dim"], linewidth=0.7,
                            shrinkA=2, shrinkB=4))

for t, tok, sp, label, pos, va in PTS:
    if not label or pos is None:
        continue
    ax.annotate(label, xy=(t / 60, sp), xytext=pos,
                fontsize=8, color=C["fg"], ha="center", va=va or "center",
                linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=C["fg_dim"], linewidth=0.7,
                                shrinkA=2, shrinkB=3))

ax.set_xlim(0, 170)
ax.set_ylim(0, 62)
ax.set_ylabel("fraction of B200 fp8 roofline (%)")
ax.grid(axis="y", color=C["grid"], linewidth=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# bottom axis: cumulative output tokens at checkpoint marks (nonlinear)
tok_marks = [(654, "21k"), (1934, "110k"), (6155, "341k"), (9833, "430k")]
ax.set_xticks([t / 60 for t, _ in tok_marks])
ax.set_xticklabels([lbl for _, lbl in tok_marks], fontsize=9)
ax.set_xlabel(f"cumulative output tokens (total {TOTAL_TOK:,})")

# top axis: wall clock
ax_top = ax.twiny()
ax_top.set_xlim(0, 170)
hour_marks = [0, 30, 60, 90, 120, 150]
ax_top.set_xticks(hour_marks)
ax_top.set_xticklabels([f"{m // 60}h{m % 60:02d}" for m in hour_marks], fontsize=9)
ax_top.set_xlabel("wall clock (session total 2h44m)", color=C["fg_muted"], fontsize=10)
ax_top.spines["top"].set_color(C["border"])
for s in ("right", "bottom", "left"):
    ax_top.spines[s].set_visible(False)

fig.suptitle("One run, annotated — Fable 5 hand-writes a tcgen05 fp8 GEMM on B200",
             color=C["fg_bright"], fontsize=13, x=0.065, ha="left", y=0.975)
fig.text(0.065, 0.915, "kernelbench.com/hard (B200) · every point = the agent benchmarking its own kernel · "
         "raw SM100 PTX, no CUTLASS, no cuBLASLt",
         color=C["fg_muted"], fontsize=9)

out = pathlib.Path(__file__).parent / "fable5_b200_fp8.png"
fig.savefig(out, dpi=170)
print(f"wrote {out}")
