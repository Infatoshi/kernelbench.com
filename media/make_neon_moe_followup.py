"""Single-panel trajectory chart for Gemini 4 Neon (XHigh) solo rerun on KernelBench-CUDA
01_glm52_fused_moe (follow-up to the initial lock-blocked V1 run).

Matches media/make_neon_cuda.py (kbh_theme dark style, minimal text).
Sources:
- Final published score: benchmarks/cuda/results/annotations/20261119_123612_agy_gemini-4-neon-xhigh_01_glm52_fused_moe.yaml
- Prior lock-blocked V1: benchmarks/cuda/results/annotations/20261118_150445_agy_gemini-4-neon-xhigh_01_glm52_fused_moe.yaml
- In-session wall-clock minutes & scores: benchmarks/cuda/outputs/runs/20261119_123612_agy_gemini-4-neon-xhigh_01_glm52_fused_moe/agy_transcript_full.jsonl

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_neon_moe_followup.py
"""
from __future__ import annotations

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
OUT = ROOT / "media/posts/audited/neon-cuda-moe-followup/01.png"

RUN_ID = "20261119_123612_agy_gemini-4-neon-xhigh_01_glm52_fused_moe"
V1_ID = "20261118_150445_agy_gemini-4-neon-xhigh_01_glm52_fused_moe"

apply()
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_pf(run_id: str) -> float:
    return 100.0 * float(yaml.safe_load((ANN / f"{run_id}.yaml").read_text())["peak_fraction"])


moe_final = load_pf(RUN_ID)   # 11.05%
moe_v1 = load_pf(V1_ID)       # 10.10%

# Verified in-session uncontended 6-shape milestones: (wall minutes from session start, % roofline, kind)
# kind: 'bench' (benchmark.py), 'self' (6-shape self-bench), 'regress', 'final' (isolated GPU 0 regrade)
# Contended GPU 2 windows (steps 251/343/369) are omitted because an external host job overlapped GPU 2.
MOE_PTS = [
    (24.9, 10.69, "bench"),     # step 55: v1 benchmark.py
    (43.3, 10.59, "regress"),   # step 97: v2 6-shape self-bench
    (105.5, 10.79, "self"),     # step 172: v3 6-shape self-bench (disjoint [128, 96])
    (174.5, 11.15, "self"),     # step 365: v4 6-shape uncontended self-bench (3-bucket + 2-bucket TMA)
    (180.3, moe_final, "final"),
]

fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=180)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])

ax.set_title(
    "KernelBench-CUDA solo rerun",
    color=C["fg_muted"],
    fontsize=11.5,
    fontweight="normal",
    loc="left",
    pad=10,
)
ax.set_ylabel("% of roofline", color=C["fg_muted"], fontsize=10.5)
ax.set_xlim(0, 224)
ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
ax.set_xticklabels(["0h", "0.5h", "1h", "1.5h", "2h", "2.5h", "3h"])
ax.set_ylim(9.8, 11.55)
ax.grid(color=C["grid"], lw=0.6)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(colors=C["fg_muted"], labelsize=10)

# Reference horizontal lines
ax.axhline(11.30, color=C["fg_dim"], lw=1.0, ls=(0, (4, 3)))
ax.text(5, 11.33, "board lead 11.30% (Sonnet 5.5)", color=C["fg_muted"], fontsize=9.2, va="bottom")

ax.axhline(10.85, color=C["grid"], lw=0.9, ls=(0, (2, 3)))
ax.text(5, 10.88, "prior #2 10.85% (Fable 5.2)", color=C["fg_dim"], fontsize=8.8, va="bottom")

ax.axhline(moe_v1, color=C["fg_dim"], lw=0.9, ls=(0, (4, 3)))
ax.text(5, moe_v1 + 0.03, f"prior lock-blocked V1 {moe_v1:.2f}%", color=C["fg_muted"], fontsize=9.0, va="bottom")

# Plot trajectory curve
ts = [p[0] for p in MOE_PTS]
ys = [p[1] for p in MOE_PTS]
ax.plot(ts, ys, color=C["accent"], lw=2.5, zorder=3)

for t, y, kind in MOE_PTS[:-1]:
    is_reg = kind == "regress"
    is_self = kind == "self"
    col = C["bad"] if is_reg else C["accent"]
    ax.scatter(
        t,
        y,
        s=48,
        zorder=4,
        facecolor=C["bg"] if is_self else col,
        edgecolor=col,
        linewidth=1.6,
    )

tf, yf, _ = MOE_PTS[-1]
ax.scatter(tf, yf, s=140, facecolor=C["accent"], edgecolor=C["fg_bright"], linewidth=1.8, zorder=5)
ax.annotate(
    f"{moe_final:.2f}%",
    xy=(tf, yf),
    xytext=(12, 0),
    textcoords="offset points",
    color=C["accent"],
    fontsize=12.5,
    fontweight="bold",
    ha="left",
    va="center",
)

# Subtle annotations for milestone versions
ax.annotate("v1", xy=(24.9, 10.69), xytext=(0, 9), textcoords="offset points", color=C["fg_muted"], fontsize=9, ha="center")
ax.annotate("v2", xy=(43.3, 10.59), xytext=(0, -15), textcoords="offset points", color=C["bad"], fontsize=9, ha="center")
ax.annotate("v3", xy=(105.5, 10.79), xytext=(0, -15), textcoords="offset points", color=C["fg_muted"], fontsize=9, ha="center")
ax.annotate("v4 self", xy=(174.5, 11.15), xytext=(-14, 9), textcoords="offset points", color=C["fg_muted"], fontsize=9, ha="center")

fig.suptitle(
    "Gemini 4 Neon on GLM-5.2 Fused MoE",
    color=C["fg_bright"],
    fontsize=16,
    fontweight="bold",
    x=0.11,
    ha="left",
    y=0.95,
)
fig.subplots_adjust(left=0.11, right=0.96, top=0.81, bottom=0.12)
fig.savefig(OUT, dpi=180, facecolor=C["bg"])
plt.close(fig)
print("MOE_PTS:", [(t, round(y, 4), k) for t, y, k in MOE_PTS])
print(OUT)
