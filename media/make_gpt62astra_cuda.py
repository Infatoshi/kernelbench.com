"""Trajectory chart for GPT-6.2 Astra on KernelBench-CUDA (minimal text, 3 panels).

Same look as make_qwen4max_cuda.py. Sources:
- Annotation YAMLs (trajectory: lists, peak_fraction): benchmarks/cuda/results/annotations/<run_id>.yaml
- NSA geomean ms per point: the six `event=variant_end shape=N variant=solution ... ms=X` values of
  each in-session benchmark.py result, parsed from the run's codex_session.jsonl at the record named
  below and checked against the YAML point's peak_fraction. Final = isolated regrade benchmark.log in
  the pulled archive. Nothing is typed in.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_gpt62astra_cuda.py
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
MODEL = "GPT-6.2 Astra"
OUT = ROOT / "media/posts/audited/gpt62astra-cuda/01.png"
MOE_ID = "20261117_133436_codex_gpt-6.2-astra_01_glm52_fused_moe"
NSA_ID = "20261117_133506_codex_gpt-6.2-astra_02_deepseek_nsa"
MQ_ID = "20261117_133536_codex_gpt-6.2-astra_03_megaqwen_decode"

# NSA: YAML trajectory t -> codex_session.jsonl record (1-based line) holding that benchmark.py output.
# The final point (kind: final) is read from the regraded benchmark.log instead.
NSA_RECORD = {7.6: 114, 11.3: 158, 14.2: 194, 16.8: 223, 19.1: 249, 30.7: 312}
X_TICK_MIN = 15
# ---------------------------------------------------------------------------

apply()
OUT.parent.mkdir(parents=True, exist_ok=True)
MS_RE = r"event=variant_end shape=(\d) variant=solution .*? ms=([0-9.]+)"


def load(run_id: str) -> dict:
    return yaml.safe_load((ANN / f"{run_id}.yaml").read_text())


def kind_of(p: dict) -> str:
    k = p.get("kind", "")
    return k if k in ("final", "regress") else "bench"


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def six(text: str, where: str) -> list[float]:
    ms = {int(s): float(v) for s, v in re.findall(MS_RE, text)}
    if sorted(ms) != [0, 1, 2, 3, 4, 5]:
        sys.exit(f"{where}: does not hold six solution ms values")
    return [ms[i] for i in range(6)]


def session_ms(run_id: str, record: int, pf: float) -> list[float]:
    line = (RUNS / run_id / "codex_session.jsonl").read_text().splitlines()[record - 1]
    line = line.replace("\\\\n", "\n").replace("\\n", "\n")
    got = re.findall(r"peak_fraction: ([0-9.]+)", line)
    if not got or abs(float(got[0]) - pf) > 1e-4:
        sys.exit(f"{run_id} record {record}: peak_fraction {got[:1]} does not match YAML {pf}")
    return six(line, f"{run_id} record {record}")


nsa = load(NSA_ID)
NSA_PTS = []
for p in nsa["trajectory"]:
    t = float(p["t"])
    if p.get("kind") == "final":
        ms = six((RUNS / NSA_ID / "benchmark.log").read_text(), f"{NSA_ID} benchmark.log")
    elif t in NSA_RECORD:
        ms = session_ms(NSA_ID, NSA_RECORD[t], float(p["score"]))
    else:
        print(f"NSA point t={t} dropped: no record mapped", file=sys.stderr)
        continue
    NSA_PTS.append((t, geomean(ms), kind_of(p)))

moe = load(MOE_ID)
MOE_PTS = [(float(p["t"]), 100 * float(p["score"]), kind_of(p)) for p in moe["trajectory"]]
MOE_PTS[-1] = (MOE_PTS[-1][0], 100 * float(moe["peak_fraction"]), "final")  # isolated regrade

mq = load(MQ_ID)
MQ_PTS = [(float(p["t"]), 100 * float(p["score"]), kind_of(p)) for p in mq["trajectory"]]
MQ_PTS[-1] = (MQ_PTS[-1][0], 100 * float(mq["peak_fraction"]), "final")  # isolated regrade

T_MAX = max(p[0] for pts in (NSA_PTS, MOE_PTS, MQ_PTS) for p in pts)
XLIM = math.ceil((T_MAX + 5) / X_TICK_MIN) * X_TICK_MIN
XTICKS = list(range(0, XLIM + 1, X_TICK_MIN))

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), dpi=180)
fig.patch.set_facecolor(C["bg"])


def style_ax(ax, title: str, ylabel: str):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, color=C["fg_bright"], fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, color=C["fg_muted"], fontsize=9.5)
    ax.set_xlim(0, XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{m}m" if m else "0" for m in XTICKS])
    ax.grid(color=C["grid"], lw=0.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=C["fg_muted"], labelsize=9.5)


def plot_curve(ax, pts, final_lbl: str, final_xy_off=(6, 0)):
    ts = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(ts, ys, color=C["accent"], lw=2.2, zorder=3)
    for t, y, kind in pts[:-1]:
        col = C["bad"] if kind == "regress" else C["accent"]
        ax.scatter(t, y, s=32, zorder=4, facecolor=col, edgecolor=col, linewidth=1.4)
    tf, yf, _ = pts[-1]
    ax.scatter(tf, yf, s=115, facecolor=C["accent"], edgecolor=C["fg_bright"], linewidth=1.5, zorder=5)
    ax.annotate(final_lbl, xy=(tf, yf), xytext=final_xy_off, textcoords="offset points",
                color=C["accent"], fontsize=10.5, fontweight="bold", va="center")


def pad(lo, hi, frac=0.18):
    span = hi - lo
    return lo - frac * span, hi + frac * span


# Panel 1: Fused MoE (% roofline), the standout cell
ax = axes[0]
style_ax(ax, "Fused MoE", "% of roofline")
ax.set_ylim(*pad(min(p[1] for p in MOE_PTS), max(p[1] for p in MOE_PTS), 0.35))
plot_curve(ax, MOE_PTS, f"{MOE_PTS[-1][1]:.2f}%", final_xy_off=(-20, 16))

# Panel 2: DeepSeek Native Sparse Attention (inverted ms so upward = faster)
ax = axes[1]
style_ax(ax, "DeepSeek Native Sparse Attention", "latency (ms, up = faster)")
lo, hi = pad(min(p[1] for p in NSA_PTS), max(p[1] for p in NSA_PTS))
ax.set_ylim(hi, lo)
plot_curve(ax, NSA_PTS, f"{NSA_PTS[-1][1]:.3f} ms", final_xy_off=(-50, -17))

# Panel 3: MegaQwen Decode (% roofline)
ax = axes[2]
style_ax(ax, "MegaQwen Decode", "% of roofline")
ax.set_ylim(*pad(min(p[1] for p in MQ_PTS), max(p[1] for p in MQ_PTS), 3.0))
plot_curve(ax, MQ_PTS, f"{MQ_PTS[-1][1]:.2f}%", final_xy_off=(-20, 16))

fig.suptitle(f"{MODEL} trajectories on KernelBench-CUDA", color=C["fg_bright"], fontsize=16,
             fontweight="bold", x=0.04, ha="left", y=0.97)
fig.subplots_adjust(left=0.055, right=0.975, top=0.82, bottom=0.11, wspace=0.25)
fig.savefig(OUT, dpi=180, facecolor=C["bg"])
plt.close(fig)
for name, pts in (("MoE %", MOE_PTS), ("NSA ms", NSA_PTS), ("MegaQwen %", MQ_PTS)):
    print(name, [(t, round(y, 4), k) for t, y, k in pts])
print(OUT)
