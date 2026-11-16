"""Trajectory chart for Kimi K3.1 Pro on KernelBench-CUDA (minimal text, 3 panels).

Same look as make_carbon_cuda.py. Sources:
- Annotation YAMLs (trajectory: lists, peak_fraction): benchmarks/cuda/results/annotations/<run_id>.yaml
- NSA geomean ms per point: the session's own benchmark.py tool results in
  benchmarks/cuda/outputs/runs/<nsa run_id>/kimi_wire.jsonl (six `shape=N variant=solution ... ms=X`
  lines per result, matched to each YAML point by peak_fraction and time); final = isolated
  regrade benchmark.log in the same archive.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_k31pro_cuda.py
"""
from __future__ import annotations

import math
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

# ---------------------------------------------------------------------------
# Per-model data block (swap this block for the next model)
# ---------------------------------------------------------------------------
MODEL = "Kimi K3.1 Pro"
OUT = ROOT / "media/posts/audited/k31pro-cuda/01.png"
MOE_YAML = ANN / "20261116_012108_kimi_kimi-code_k3.1-pro_01_glm52_fused_moe.yaml"
NSA_YAML = ANN / "20261116_012138_kimi_kimi-code_k3.1-pro_02_deepseek_nsa.yaml"
MQ_YAML = ANN / "20261116_012208_kimi_kimi-code_k3.1-pro_03_megaqwen_decode.yaml"

# NSA: per-shape ms (shapes 0..5) of each in-session benchmark.py result, from kimi_wire.jsonl
# (line number in the wire file), keyed to the YAML trajectory point by t. Final = isolated
# regrade benchmark.log (YAML: 0.950 ms; in-run graded 0.953 ms).
NSA_MS = {
    30.0: [0.529, 1.311, 1.707, 2.966, 0.342, 0.878],  # wire 615, peak_fraction 0.0977
    39.3: [0.505, 1.265, 1.641, 2.982, 0.333, 0.838],  # wire 731, peak_fraction 0.1009
    46.2: [0.534, 1.272, 1.675, 2.974, 0.370, 0.860],  # wire 875, peak_fraction 0.0974 (regress)
    48.6: [0.492, 1.215, 1.555, 3.051, 0.331, 0.821],  # wire 954, peak_fraction 0.1030
    58.0: [0.492, 1.216, 1.555, 2.995, 0.332, 0.821],  # wire 1541, peak_fraction 0.1032
    66.1: [0.490, 1.212, 1.549, 2.976, 0.328, 0.820],  # benchmark.log isolated regrade, 0.1037
}
# Trajectory points whose kind is not regress/final but are not benchmark.py output (hollow markers)
MQ_SELF_TIMED = True  # MegaQwen points before the final are the session's own 4-context timings
MOE_FLAG = "flagged: routing cache"  # verdict rubric_leak
# ---------------------------------------------------------------------------

apply()
OUT.parent.mkdir(parents=True, exist_ok=True)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def kind_of(p: dict, self_timed: bool) -> str:
    k = p.get("kind", "")
    if k in ("regress", "final"):
        return k
    return "self" if self_timed else "bench"


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# NSA: geomean ms per YAML point (a point without six ms values would be dropped, never estimated)
nsa = load(NSA_YAML)
NSA_PTS = []
for p in nsa["trajectory"]:
    ms = NSA_MS.get(float(p["t"]))
    if ms is None or len(ms) != 6:
        print(f"NSA point t={p['t']} dropped: no six ms values", file=sys.stderr)
        continue
    NSA_PTS.append((float(p["t"]), geomean(ms), kind_of(p, False)))

moe = load(MOE_YAML)
MOE_PTS = [(float(p["t"]), 100 * float(p["score"]), kind_of(p, False)) for p in moe["trajectory"]]
MOE_PTS[-1] = (MOE_PTS[-1][0], 100 * float(moe["peak_fraction"]), "final")

mq = load(MQ_YAML)
mq_final = float(mq["peak_fraction"])  # isolated regrade
MQ_PTS = [(float(p["t"]), 100 * float(p["score"]), kind_of(p, MQ_SELF_TIMED)) for p in mq["trajectory"]]
MQ_PTS[-1] = (MQ_PTS[-1][0], 100 * float(mq_final), "final")

T_MAX = max(p[0] for pts in (NSA_PTS, MOE_PTS, MQ_PTS) for p in pts)
XLIM = math.ceil((T_MAX + 15) / 10) * 10
XTICKS = list(range(0, XLIM + 1, 30))

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), dpi=180)
fig.patch.set_facecolor(C["bg"])


def style_ax(ax, title: str, ylabel: str):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["fg_bright"], fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, color=C["fg_muted"], fontsize=9.5)
    ax.set_xlim(0, XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{m}m" for m in XTICKS])
    ax.grid(color=C["grid"], lw=0.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=C["fg_muted"], labelsize=9.5)


def plot_curve(ax, pts, final_lbl: str, final_xy_off=(6, 0), final_col=None):
    final_col = final_col or C["accent"]
    ts = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(ts, ys, color=C["accent"], lw=2.2, zorder=3)
    for t, y, kind in pts[:-1]:
        is_reg = kind == "regress"
        is_self = kind == "self"
        col = C["bad"] if is_reg else C["accent"]
        ax.scatter(t, y, s=32, zorder=4, facecolor=C["bg"] if is_self else col, edgecolor=col, linewidth=1.4)
    tf, yf, _ = pts[-1]
    ax.scatter(tf, yf, s=115, facecolor=final_col, edgecolor=C["fg_bright"], linewidth=1.5, zorder=5)
    ax.annotate(final_lbl, xy=(tf, yf), xytext=final_xy_off, textcoords="offset points",
                color=final_col, fontsize=10.5, fontweight="bold", va="center")


def pad(lo, hi, frac=0.18):
    span = hi - lo
    return lo - frac * span, hi + frac * span


# Panel 1: DeepSeek Native Sparse Attention (inverted ms so upward = faster)
ax = axes[0]
style_ax(ax, "DeepSeek Native Sparse Attention", "latency (ms, up = faster)")
lo, hi = pad(min(p[1] for p in NSA_PTS), max(p[1] for p in NSA_PTS))
ax.set_ylim(hi, lo)
plot_curve(ax, NSA_PTS, f"{NSA_PTS[-1][1]:.3f} ms", final_xy_off=(9, 0))

# Panel 2: Fused MoE (% roofline); flagged endpoint in warn amber
ax = axes[1]
style_ax(ax, "Fused MoE", "% of roofline")
lo, hi = pad(min(p[1] for p in MOE_PTS), max(p[1] for p in MOE_PTS), 0.35)
ax.set_ylim(lo, hi + 0.3 * (hi - lo))
plot_curve(ax, MOE_PTS, f"{MOE_PTS[-1][1]:.1f}%", final_xy_off=(-20, 16), final_col=C["warn"])
ax.annotate(MOE_FLAG, xy=MOE_PTS[-1][:2], xytext=(12, 33), textcoords="offset points",
            color=C["warn"], fontsize=8.3, va="center", ha="right")

# Panel 3: MegaQwen Decode (% roofline)
ax = axes[2]
style_ax(ax, "MegaQwen Decode", "% of roofline")
lo, hi = pad(min(p[1] for p in MQ_PTS), max(p[1] for p in MQ_PTS), 0.35)
ax.set_ylim(lo, hi)
plot_curve(ax, MQ_PTS, f"{MQ_PTS[-1][1]:.1f}%", final_xy_off=(9, 0))

fig.suptitle(f"{MODEL} trajectories on KernelBench-CUDA", color=C["fg_bright"], fontsize=16,
             fontweight="bold", x=0.04, ha="left", y=0.97)
fig.subplots_adjust(left=0.055, right=0.975, top=0.82, bottom=0.11, wspace=0.25)
fig.savefig(OUT, dpi=180, facecolor=C["bg"])
plt.close(fig)
for name, pts in (("NSA ms", NSA_PTS), ("MoE %", MOE_PTS), ("MegaQwen %", MQ_PTS)):
    print(name, [(t, round(y, 4), k) for t, y, k in pts])
print(OUT)
