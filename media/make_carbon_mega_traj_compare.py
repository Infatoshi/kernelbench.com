"""Gemini 4 Carbon mega post, image 1: Carbon and Opus 5.5 hill-climbs on Kimi-Linear Decode.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_carbon_mega_traj_compare.py
"""
import sys
import yaml
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply

apply()
ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "benchmarks/mega/results/annotations"

# Carbon intermediate full-step self-timings across 2k/8k/16k from workstation transcript_full.jsonl
# converted against Carbon's in-session baseline geomean (6.131 ms)
CARBON_SELF = [
    {"t": 58.8, "score": 16.89, "label": "microbench"},
    {"t": 91.0, "score": 19.15, "label": "microbench"},
    {"t": 109.1, "score": 21.66, "label": "microbench"},
    {"t": 121.4, "score": 23.73, "label": "microbench"},
    {"t": 139.3, "score": 24.52, "label": "microbench"},
    {"t": 145.3, "score": 24.14, "label": "microbench", "kind": "regress"},
]

RUNS = [
    ("Opus 5.5", ANN / "20260922_121505_claude_claude-opus-5-5_02_kimi_linear_decode.yaml", "#4d9fff", 35.46, []),
    ("Gemini 4 Carbon", ANN / "20261110_113616_agy_gemini-4-carbon-high_02_kimi_linear_decode.yaml", C["accent"], 23.76, CARBON_SELF),
]
OUT = ROOT / "media/posts/audited/carbon-mega/01.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(11.8, 5.8), dpi=180)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
for name, f, col, final, extra in RUNS:
    pts = [p for p in yaml.safe_load(open(f))["trajectory"] if p.get("kind") != "baseline"] + extra
    pts.sort(key=lambda p: float(p["t"]))
    t = [0.0] + [float(p["t"]) for p in pts]
    s = [1.0] + [float(p["score"]) for p in pts]
    s[-1] = final
    ax.plot(t, s, color=col, lw=2.2, zorder=3)
    for p in pts[:-1]:
        micro = str(p.get("label", "")).startswith("microbench")
        reg = p.get("kind") == "regress"
        ax.scatter(
            float(p["t"]),
            float(p["score"]),
            s=34,
            zorder=4,
            facecolor=C["bg"] if micro else (C["bad"] if reg else col),
            edgecolor=C["bad"] if reg else col,
            linewidth=1.4,
        )
    ax.scatter(t[-1], final, s=135, facecolor=col, edgecolor=C["fg_bright"], linewidth=1.6, zorder=5)
    ax.text(t[-1] + 4.5, final, f"{name}  {final:.1f}x", color=col, fontsize=11.5, va="center", fontweight="bold")
ax.axhline(1, color=C["fg_dim"], lw=0.8, ls="--")
ax.text(3, 1.7, "PyTorch baseline (1x)", color=C["fg_dim"], fontsize=9)
ax.set_xlim(0, 255)
ax.set_ylim(0, 40)
ax.set_xticks(range(0, 241, 60))
ax.set_xticklabels([f"{h}h" for h in range(0, 5)])
ax.set_xlabel("wall clock", color=C["fg_muted"], fontsize=10.5)
ax.set_ylabel("speedup over baseline", color=C["fg_muted"], fontsize=10.5)
ax.grid(color=C["grid"], lw=0.6)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(colors=C["fg_muted"], labelsize=10)
fig.suptitle("Kimi-Linear Decode megakernel trajectories", color=C["fg_bright"], fontsize=15.5, fontweight="bold", x=0.06, ha="left", y=0.95)
fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.11)
fig.savefig(OUT, dpi=180, facecolor=C["bg"])
print(OUT)
