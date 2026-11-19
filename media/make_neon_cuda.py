"""Trajectory chart for Gemini 4 Neon (XHigh) on KernelBench-CUDA (minimal text, 2 panels:
DeepSeek Native Sparse Attention + MegaQwen Decode; Fused MoE rerunning solo for follow-up).

Same look as make_gpt62astra_cuda.py and make_carbon_cuda.py.
Sources:
- Final published scores: benchmarks/cuda/results/annotations/20261118_*neon*.yaml & benchmark.log
- In-session wall-clock minutes & scores: benchmarks/cuda/outputs/runs/20261118_*neon*/{gpu_lock.log,transcript.jsonl}

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_neon_cuda.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "benchmarks/cuda/results/annotations"
RUNS = ROOT / "benchmarks/cuda/outputs/runs"

MODEL = "Gemini 4 Neon"
OUT = ROOT / "media/posts/audited/neon-cuda/01.png"
NSA_ID = "20261118_150515_agy_gemini-4-neon-xhigh_02_deepseek_nsa"
MQ_ID = "20261118_150545_agy_gemini-4-neon-xhigh_03_megaqwen_decode"

apply()
OUT.parent.mkdir(parents=True, exist_ok=True)
MS_RE = r"event=variant_end shape=(\d) variant=solution .*? ms=([0-9.]+)"


def load_yaml(run_id: str) -> dict:
    return yaml.safe_load((ANN / f"{run_id}.yaml").read_text())


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


nsa_ms_final = geomean(
    [
        dict((int(s), float(v)) for s, v in re.findall(MS_RE, (RUNS / NSA_ID / "benchmark.log").read_text()))[i]
        for i in range(6)
    ]
)
mq_final = 100.0 * float(load_yaml(MQ_ID)["peak_fraction"])  # 7.55%

# Verified in-session trajectories: (wall minutes from session start, metric, kind)
# kind: 'bench' (benchmark.py), 'self' (4-shape self-bench), 'regress', 'final' (isolated regrade)
NSA_PTS = [
    (190.3, 0.2262, "bench"),
    (204.7, 0.1305, "bench"),
    (212.3, 0.1217, "bench"),
    (219.2, 0.1292, "regress"),
    (225.1, 0.1059, "bench"),
    (233.5, 0.1044, "bench"),
    (250.2, 0.1063, "regress"),
    (274.8, 0.0862, "bench"),
    (284.4, 0.0848, "bench"),
    (294.7, 0.0820, "bench"),
    (303.0, 0.0792, "bench"),
    (343.4, 0.0810, "regress"),
    (351.5, 0.0751, "bench"),
    (358.6, 0.0733, "bench"),
    (364.3, 0.0747, "regress"),
    (377.6, 0.0734, "bench"),
    (391.4, nsa_ms_final, "final"),
]

MQ_PTS = [
    (191.2, 5.49, "self"),
    (206.1, 6.68, "self"),
    (218.7, 6.57, "regress"),
    (239.3, 6.77, "self"),
    (249.7, 6.15, "regress"),
    (278.7, 6.34, "self"),
    (281.6, 7.10, "self"),
    (290.0, 7.22, "self"),
    (334.7, 7.11, "regress"),
    (339.4, 7.41, "self"),
    (347.4, 7.52, "self"),
    (355.0, 7.585, "self"),
    (392.0, 7.68, "self"),
    (410.2, mq_final, "final"),
]

fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), dpi=180)
fig.patch.set_facecolor(C["bg"])


def style_ax(ax, title: str, ylabel: str):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["fg_bright"], fontsize=13.5, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, color=C["fg_muted"], fontsize=10)
    ax.set_xlim(0, 440)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360, 420])
    ax.set_xticklabels(["0h", "1h", "2h", "3h", "4h", "5h", "6h", "7h"])
    ax.grid(color=C["grid"], lw=0.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=C["fg_muted"], labelsize=10)


def plot_curve(ax, pts, final_lbl: str, final_xy_off=(6, 0)):
    ts = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(ts, ys, color=C["accent"], lw=2.3, zorder=3)
    for t, y, kind in pts[:-1]:
        is_reg = kind == "regress"
        is_self = kind == "self"
        col = C["bad"] if is_reg else C["accent"]
        ax.scatter(
            t,
            y,
            s=34,
            zorder=4,
            facecolor=C["bg"] if is_self else col,
            edgecolor=col,
            linewidth=1.4,
        )
    tf, yf, _ = pts[-1]
    ax.scatter(tf, yf, s=120, facecolor=C["accent"], edgecolor=C["fg_bright"], linewidth=1.6, zorder=5)
    ax.annotate(
        final_lbl,
        xy=(tf, yf),
        xytext=final_xy_off,
        textcoords="offset points",
        color=C["accent"],
        fontsize=11,
        fontweight="bold",
        va="center",
    )


# Panel 1: DeepSeek Native Sparse Attention (inverted ms so upward = faster)
ax = axes[0]
style_ax(ax, "DeepSeek Native Sparse Attention", "latency (ms, up = faster)")
ax.set_ylim(0.245, 0.055)
ax.axhline(0.0724, color=C["fg_dim"], lw=0.9, ls=(0, (4, 3)))
ax.text(8, 0.070, "lead 0.072 ms", color=C["fg_muted"], fontsize=9, va="bottom")
plot_curve(ax, NSA_PTS, f"{nsa_ms_final:.3f} ms", final_xy_off=(-24, -24))

# Panel 2: MegaQwen Decode (% roofline)
ax = axes[1]
style_ax(ax, "MegaQwen Decode", "% of roofline")
ax.set_ylim(5.0, 8.6)
ax.axhline(8.00, color=C["fg_dim"], lw=0.9, ls=(0, (4, 3)))
ax.text(8, 8.07, "clean lead 8.0%", color=C["fg_muted"], fontsize=9, va="bottom")
plot_curve(ax, MQ_PTS, f"{mq_final:.2f}%", final_xy_off=(-20, -24))

fig.suptitle(
    f"{MODEL} trajectories on KernelBench-CUDA",
    color=C["fg_bright"],
    fontsize=16,
    fontweight="bold",
    x=0.055,
    ha="left",
    y=0.97,
)
fig.subplots_adjust(left=0.075, right=0.97, top=0.82, bottom=0.11, wspace=0.22)
fig.savefig(OUT, dpi=180, facecolor=C["bg"])
plt.close(fig)
for name, pts in (("NSA ms", NSA_PTS), ("MegaQwen %", MQ_PTS)):
    print(name, [(t, round(y, 4), k) for t, y, k in pts])
print(OUT)
