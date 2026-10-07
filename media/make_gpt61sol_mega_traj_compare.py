"""GPT-6.1 Sol mega post, image 1: GPT-6.1 Sol and Opus 5.5 hill-climbs on Kimi-Linear Decode.
Points come from each run's annotation `trajectory:` (benchmark.py results filled, the agent's own
full-step timings hollow, slower versions red). Finals are the published isolated regrades.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_gpt61sol_mega_traj_compare.py
"""
import sys, yaml
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply
apply()
ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "benchmarks/mega/results/annotations"
RUNS = [
    ("Opus 5.5", ANN / "20260922_121505_claude_claude-opus-5-5_02_kimi_linear_decode.yaml", "#4d9fff", 35.46, "stopped itself"),
    ("GPT-6.1 Sol", ANN / "20260930_115311_codex_gpt-6.1-sol_02_kimi_linear_decode.yaml", C["accent"], 4.99, "stopped itself"),
]
OUT = ROOT / "media/posts/audited/gpt61sol-mega/01.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=180)
fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])
for name, f, col, final, how in RUNS:
    pts = [p for p in yaml.safe_load(open(f))["trajectory"] if p.get("kind") != "baseline"]
    t = [0.0] + [float(p["t"]) for p in pts]; s = [1.0] + [float(p["score"]) for p in pts]
    ax.plot(t, s, color=col, lw=2.2, zorder=3)
    for p in pts:
        micro = str(p.get("label", "")).startswith("microbench")
        reg = p.get("kind") == "regress"
        ax.scatter(float(p["t"]), float(p["score"]), s=34, zorder=4,
                   facecolor=C["bg"] if micro else (C["bad"] if reg else col), edgecolor=C["bad"] if reg else col, linewidth=1.4)
    ax.scatter(t[-1], final, s=140, facecolor=col, edgecolor=C["fg_bright"], linewidth=1.6, zorder=5)   # big dot = the published regrade
    dy = 0
    ax.text(t[-1] + 4, final + dy, (f"{name}  {final:.1f}x\n{t[-1]:.0f} min, {how}" if t[-1] < 60 else f"{name}  {final:.1f}x\n{t[-1]/60:.1f} h, {how}"), color=col, fontsize=11, va="center", fontweight="bold")
ax.axhline(1, color=C["fg_dim"], lw=0.8, ls="--"); ax.text(2, 1.6, "optimized PyTorch baseline = 1x", color=C["fg_dim"], fontsize=9)
ax.set_xlim(0, 270); ax.set_ylim(0, 40)
ax.set_xticks(range(0, 241, 60)); ax.set_xticklabels([f"{h}h" for h in range(0, 5)])
ax.set_xlabel("wall clock since the session started", color=C["fg_muted"], fontsize=11)
ax.set_ylabel("speedup over the optimized PyTorch baseline", color=C["fg_muted"], fontsize=11)
ax.grid(color=C["grid"], lw=0.6); [sp.set_visible(False) for sp in ax.spines.values()]
ax.tick_params(colors=C["fg_muted"], labelsize=10)
fig.suptitle("Kimi-Linear Decode megakernel, GPT-6.1 Sol vs Opus 5.5 on RTX PRO 6000", color=C["fg_bright"], fontsize=15, x=0.06, ha="left", y=0.97)
fig.text(0.06, 0.905, "filled = benchmark.py, hollow = the agent timing its own kernel, red = a slower version, big dot = final (isolated regrade)", color=C["fg_muted"], fontsize=10)
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.1)
fig.savefig(OUT, dpi=180, facecolor=C["bg"]); print(OUT)
