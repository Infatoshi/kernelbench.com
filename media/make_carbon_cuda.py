"""Trajectory chart for Gemini 4 Carbon on KernelBench-CUDA (minimal text, 3 panels).

Sources:
- Audited isolated regrades: public/data/models.json & benchmarks/cuda/results/annotations/20261110_*
- In-session trajectory timestamps & scores: benchmarks/cuda/outputs/runs/20261110_*/{transcript.jsonl,gpu_lock.log}
  and agent_home/.../transcript_full.jsonl on the workstation.

  uv run --no-project --with matplotlib,numpy python media/make_carbon_cuda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply  # noqa: E402

apply()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media/posts/audited/carbon-cuda/01.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Verified in-session trajectories (minutes from session start, metric, kind)
# kind: 'bench' (benchmark.py), 'self' (agent 4-shape/end-to-end bench), 'regress', 'final' (isolated regrade)
NSA_PTS = [
    (48.1, 0.1265, "bench"),
    (69.0, 0.1097, "bench"),
    (80.0, 0.0969, "bench"),
    (95.8, 0.1323, "regress"),
    (96.5, 0.1097, "bench"),
    (116.5, 0.0992, "bench"),
    (161.7, 0.0965, "bench"),
    (171.8, 0.0898, "bench"),
    (178.9, 0.0936, "regress"),
    (181.7, 0.0885, "bench"),
    (189.8, 0.0880, "bench"),
    (195.8, 0.0876, "bench"),
    (215.0, 0.0879, "final"),
]

MOE_PTS = [
    (77.2, 10.23, "bench"),
    (100.6, 10.22, "bench"),
    (166.5, 10.22, "bench"),
    (259.4, 10.20, "final"),
]
# Unshipped scratch warp-specialized end-to-end build (9.25 ms @ T=4096, 13.88 ms @ T=8192 -> ~10.8% rough projection)
MOE_SCRATCH = [(100.6, 10.22), (187.8, 10.80)]

MEGAQWEN_PTS = [
    (80.0, 5.28, "self"),
    (95.2, 5.70, "self"),
    (115.8, 5.97, "self"),
    (130.5, 6.12, "self"),
    (161.1, 6.21, "self"),
    (177.0, 6.10, "regress"),
    (213.6, 6.13, "bench"),
    (221.3, 6.12, "final"),
]

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), dpi=180)
fig.patch.set_facecolor(C["bg"])


def style_ax(ax, title: str, ylabel: str):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["fg_bright"], fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, color=C["fg_muted"], fontsize=9.5)
    ax.set_xlim(0, 275)
    ax.set_xticks([0, 60, 120, 180, 240])
    ax.set_xticklabels(["0h", "1h", "2h", "3h", "4h"])
    ax.grid(color=C["grid"], lw=0.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=C["fg_muted"], labelsize=9.5)


def plot_curve(ax, pts, final_lbl: str, final_xy_off=(6, 0)):
    ts = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(ts, ys, color=C["accent"], lw=2.2, zorder=3)
    for t, y, kind in pts[:-1]:
        is_reg = kind == "regress"
        is_self = kind == "self"
        col = C["bad"] if is_reg else C["accent"]
        ax.scatter(
            t,
            y,
            s=32,
            zorder=4,
            facecolor=C["bg"] if is_self else col,
            edgecolor=col,
            linewidth=1.4,
        )
    tf, yf, _ = pts[-1]
    ax.scatter(tf, yf, s=115, facecolor=C["accent"], edgecolor=C["fg_bright"], linewidth=1.5, zorder=5)
    ax.annotate(
        final_lbl,
        xy=(tf, yf),
        xytext=final_xy_off,
        textcoords="offset points",
        color=C["accent"],
        fontsize=10.5,
        fontweight="bold",
        va="center",
    )


# Panel 1: Native Sparse Attention (inverted ms so upward = faster)
ax = axes[0]
style_ax(ax, "Native Sparse Attention", "latency (ms, up = faster)")
ax.set_ylim(0.142, 0.064)
ax.axhline(0.072, color=C["fg_dim"], lw=0.9, ls=(0, (4, 3)))
ax.text(6, 0.0705, "lead 0.072 ms", color=C["fg_muted"], fontsize=8.5, va="bottom")
plot_curve(ax, NSA_PTS, "0.088 ms", final_xy_off=(9, 0))
ax.annotate(
    "Cauchy-Schwarz skip",
    xy=(161.7, 0.0965),
    xytext=(-18, -28),
    textcoords="offset points",
    color=C["fg_muted"],
    fontsize=8.3,
    arrowprops=dict(arrowstyle="-", color=C["fg_dim"], lw=0.8),
)

# Panel 2: Fused MoE (% roofline)
ax = axes[1]
style_ax(ax, "Fused MoE", "% of roofline")
ax.set_ylim(8.8, 11.8)
ax.axhline(11.3, color=C["fg_dim"], lw=0.9, ls=(0, (4, 3)))
ax.text(6, 11.36, "lead 11.3%", color=C["fg_muted"], fontsize=8.5, va="bottom")
plot_curve(ax, MOE_PTS, "10.2%", final_xy_off=(-28, -16))
ax.plot(
    [p[0] for p in MOE_SCRATCH],
    [p[1] for p in MOE_SCRATCH],
    color=C["warn"],
    lw=1.7,
    ls=(0, (4, 3)),
    zorder=3,
)
ax.scatter(187.8, 10.80, s=48, facecolor=C["bg"], edgecolor=C["warn"], linewidth=1.5, zorder=4)
ax.annotate(
    "scratch ~10.8%\n(stream cut off)",
    xy=(187.8, 10.80),
    xytext=(7, 2),
    textcoords="offset points",
    color=C["warn"],
    fontsize=8.3,
    va="center",
)

# Panel 3: MegaQwen Decode (% roofline)
ax = axes[2]
style_ax(ax, "MegaQwen Decode", "% of roofline")
ax.set_ylim(4.6, 8.5)
ax.axhline(8.1, color=C["fg_dim"], lw=0.9, ls=(0, (4, 3)))
ax.text(6, 8.16, "lead 8.1%", color=C["fg_muted"], fontsize=8.5, va="bottom")
plot_curve(ax, MEGAQWEN_PTS, "6.1%", final_xy_off=(9, 0))
ax.scatter(46.4, 4.82, s=38, marker="x", color=C["bad"], linewidth=1.6, zorder=4)
ax.text(52, 4.82, "30m warp deadlock\n(fixed in 2m)", color=C["fg_muted"], fontsize=8.2, va="center")
ax.annotate(
    "PCIe host RNG",
    xy=(213.6, 6.13),
    xytext=(-58, 22),
    textcoords="offset points",
    color=C["fg_muted"],
    fontsize=8.3,
    arrowprops=dict(arrowstyle="-", color=C["fg_dim"], lw=0.8),
)

fig.suptitle("Gemini 4 Carbon trajectories on KernelBench-CUDA", color=C["fg_bright"], fontsize=16, fontweight="bold", x=0.04, ha="left", y=0.97)
fig.subplots_adjust(left=0.055, right=0.975, top=0.82, bottom=0.11, wspace=0.25)
fig.savefig(OUT, dpi=180, facecolor=C["bg"])
plt.close(fig)
print(OUT)
