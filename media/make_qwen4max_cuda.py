"""Trajectory chart for Qwen 4 Max on KernelBench-CUDA (minimal text, 3 panels).

Same look as make_carbon_cuda.py and make_k31pro_cuda.py. Sources:
- Annotation YAMLs (trajectory: lists, peak_fraction): benchmarks/cuda/results/annotations/<run_id>.yaml
- NSA geomean ms per point: the session's own benchmark.py results (six `shape=N variant=solution
  ... ms=X` values per run), from the Claude Code session store f08a555e-...jsonl (record number
  in the comment) and the session's saved scratch/bench_*.log files, matched to each YAML point by
  peak_fraction and time. Final = isolated regrade benchmark.log in the pulled archive
  (benchmarks/cuda/outputs/runs/<nsa run_id>/benchmark.log), parsed here, never typed in.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_qwen4max_cuda.py
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

# ---------------------------------------------------------------------------
# Per-model data block (swap this block for the next model)
# ---------------------------------------------------------------------------
MODEL = "Qwen 4 Max"
OUT = ROOT / "media/posts/audited/qwen4max-cuda/01.png"
MOE_ID = "20261116_020128_or-fable_qwen_qwen4-max_01_glm52_fused_moe"
NSA_ID = "20261116_020158_or-fable_qwen_qwen4-max_02_deepseek_nsa"
MQ_ID = "20261116_020228_or-fable_qwen_qwen4-max_03_megaqwen_decode"

# NSA: per-shape ms (shapes 0..5) of each in-session benchmark.py result, keyed to the YAML
# trajectory point by t. The final point (kind: final) is read from the regraded benchmark.log.
NSA_MS = {
    77.6:  [0.203, 0.387, 0.419, 1.030, 0.190, 0.269],  # session:686, pf 0.2846
    118.0: [0.103, 0.202, 0.236, 0.431, 0.093, 0.143],  # session:972, pf 0.5656
    124.0: [0.107, 0.218, 0.250, 0.445, 0.101, 0.153],  # session:1086, pf 0.5324
    137.0: [0.100, 0.195, 0.230, 0.485, 0.093, 0.143],  # session:1253, pf 0.5628
    184.6: [0.090, 0.174, 0.222, 0.445, 0.077, 0.125],  # session:1568, pf 0.6287
    189.4: [0.079, 0.149, 0.174, 0.360, 0.072, 0.112],  # session:1640, pf 0.7325
    191.4: [0.081, 0.153, 0.175, 0.355, 0.073, 0.112],  # session:1658, pf 0.7248
    206.5: [0.096, 0.188, 0.243, 0.415, 0.084, 0.127],  # session:1724, pf 0.6015
    210.4: [0.105, 0.239, 0.298, 0.478, 0.086, 0.137],  # session:1738, pf 0.5285
    212.5: [0.089, 0.165, 0.180, 0.360, 0.084, 0.118],  # session:1746, pf 0.6787
    216.0: [0.080, 0.155, 0.177, 0.357, 0.075, 0.108],  # session:1777, pf 0.7247
    218.6: [0.078, 0.153, 0.174, 0.350, 0.073, 0.106],  # session:1783, pf 0.7390
    221.1: [0.079, 0.153, 0.176, 0.352, 0.073, 0.106],  # session:1798, pf 0.7350
    294.1: [0.075712, 0.151328, 0.173888, 0.558656, 0.070912, 0.103968],  # scratch/bench_v14.log (session:2172), pf 0.6941
    296.3: [0.075840, 0.151328, 0.173568, 0.351872, 0.069824, 0.103968],  # scratch/bench_v15.log (session:2201), pf 0.7516
    298.0: [0.084960, 0.164544, 0.183040, 0.350176, 0.081120, 0.112416],  # scratch/bench_v16.log (session:2233), pf 0.6946
    298.7: [0.084736, 0.163392, 0.182272, 0.349728, 0.080672, 0.110848],  # scratch/bench_v16b.log (session:2247), pf 0.6986
    301.1: [0.084672, 0.162048, 0.180000, 0.348352, 0.079648, 0.111520],  # scratch/bench_final.log (session:2284), pf 0.7024
}
# Points that are not a quiet benchmark.py run of that kernel, drawn hollow (keyed by t):
# Fused MoE 77.5 and 142.0 are the session's own dev timings converted to roofline by the audit;
# MegaQwen 292.1 and 297.8 are benchmark.py runs that overlapped the session's own check and bench.
MOE_HOLLOW = {77.5, 142.0}
MQ_HOLLOW = {292.1, 297.8}
# ---------------------------------------------------------------------------

apply()
OUT.parent.mkdir(parents=True, exist_ok=True)


def load(run_id: str) -> dict:
    return yaml.safe_load((ANN / f"{run_id}.yaml").read_text())


def kind_of(p: dict, hollow: set) -> str:
    k = p.get("kind", "")
    if k == "final":
        return k
    if float(p["t"]) in hollow:
        return "self"
    return "regress" if k == "regress" else "bench"


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def regrade_ms(run_id: str) -> list[float]:
    """Six per-shape solution ms from the isolated regrade's benchmark.log (variant_end events)."""
    text = (RUNS / run_id / "benchmark.log").read_text()
    ms = {int(s): float(v) for s, v in re.findall(r"event=variant_end shape=(\d) variant=solution .*? ms=([0-9.]+)", text)}
    if sorted(ms) != [0, 1, 2, 3, 4, 5]:
        sys.exit(f"{run_id}: benchmark.log does not hold six solution ms values")
    return [ms[i] for i in range(6)]


# NSA: geomean ms per YAML point (a point without six ms values would be dropped, never estimated)
nsa = load(NSA_ID)
NSA_PTS = []
for p in nsa["trajectory"]:
    ms = regrade_ms(NSA_ID) if p.get("kind") == "final" else NSA_MS.get(float(p["t"]))
    if ms is None or len(ms) != 6:
        print(f"NSA point t={p['t']} dropped: no six ms values", file=sys.stderr)
        continue
    NSA_PTS.append((float(p["t"]), geomean(ms), kind_of(p, set())))

moe = load(MOE_ID)
MOE_PTS = [(float(p["t"]), 100 * float(p["score"]), kind_of(p, MOE_HOLLOW)) for p in moe["trajectory"]]
MOE_PTS[-1] = (MOE_PTS[-1][0], 100 * float(moe["peak_fraction"]), "final")  # isolated regrade

mq = load(MQ_ID)
MQ_PTS = [(float(p["t"]), 100 * float(p["score"]), kind_of(p, MQ_HOLLOW)) for p in mq["trajectory"]]
MQ_PTS[-1] = (MQ_PTS[-1][0], 100 * float(mq["peak_fraction"]), "final")  # isolated regrade

T_MAX = max(p[0] for pts in (NSA_PTS, MOE_PTS, MQ_PTS) for p in pts)
XLIM = math.ceil((T_MAX + 40) / 60) * 60
XTICKS = list(range(0, XLIM + 1, 60))

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), dpi=180)
fig.patch.set_facecolor(C["bg"])


def style_ax(ax, title: str, ylabel: str):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["fg_bright"], fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, color=C["fg_muted"], fontsize=9.5)
    ax.set_xlim(0, XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{m // 60}h" for m in XTICKS])
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
plot_curve(ax, NSA_PTS, f"{NSA_PTS[-1][1]:.3f} ms", final_xy_off=(-18, -17))

# Panel 2: Fused MoE (% roofline)
ax = axes[1]
style_ax(ax, "Fused MoE", "% of roofline")
lo, hi = pad(min(p[1] for p in MOE_PTS), max(p[1] for p in MOE_PTS), 0.35)
ax.set_ylim(lo, hi)
plot_curve(ax, MOE_PTS, f"{MOE_PTS[-1][1]:.1f}%", final_xy_off=(-14, 16))

# Panel 3: MegaQwen Decode (% roofline)
ax = axes[2]
style_ax(ax, "MegaQwen Decode", "% of roofline")
lo, hi = pad(min(p[1] for p in MQ_PTS), max(p[1] for p in MQ_PTS), 0.35)
ax.set_ylim(lo, hi)
plot_curve(ax, MQ_PTS, f"{MQ_PTS[-1][1]:.1f}%", final_xy_off=(-14, 16))

fig.suptitle(f"{MODEL} trajectories on KernelBench-CUDA", color=C["fg_bright"], fontsize=16,
             fontweight="bold", x=0.04, ha="left", y=0.97)
fig.subplots_adjust(left=0.055, right=0.975, top=0.82, bottom=0.11, wspace=0.25)
fig.savefig(OUT, dpi=180, facecolor=C["bg"])
plt.close(fig)
for name, pts in (("NSA ms", NSA_PTS), ("MoE %", MOE_PTS), ("MegaQwen %", MQ_PTS)):
    print(name, [(t, round(y, 4), k) for t, y, k in pts])
print(OUT)
