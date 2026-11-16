"""Qwen 4 Max mega post, image 1: Qwen 4 Max and Opus 5.5 hill-climbs on Kimi-Linear Decode.

Same look as make_carbon_mega_traj_compare.py. Both curves end on their isolated regrade score.
Qwen's red points are the zero-barrier redesign it measured and dropped in its last hour; hollow
points are the session's own us/step timings converted to speedup by the audit, not benchmark.py.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_qwen4max_mega_traj_compare.py
"""
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "benchmarks/mega/results/annotations"

# ---------------------------------------------------------------------------
# Per-model data block (swap this block for the next model)
# ---------------------------------------------------------------------------
OUT = ROOT / "media/posts/audited/qwen4max-mega/01.png"
QWEN_YAML = ANN / "20261116_020058_or-fable_qwen_qwen4-max_02_kimi_linear_decode.yaml"
QWEN_FINAL = float(yaml.safe_load(open(QWEN_YAML))["peak_fraction"])  # isolated regrade
# (name, yaml, colour, final score, end label, label offset in points)
RUNS = [
    ("Opus 5.5", ANN / "20260922_121505_claude_claude-opus-5-5_02_kimi_linear_decode.yaml",
     "#4d9fff", 35.46, "Opus 5.5  35.5x", (9, 0)),
    ("Qwen 4 Max", QWEN_YAML, C["accent"], QWEN_FINAL, f"Qwen 4 Max  {QWEN_FINAL:.1f}x", (-52, 17)),
]
X_MAX_MIN = 420
BASELINE_LBL_X = 3
# ---------------------------------------------------------------------------

apply()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(11.8, 5.8), dpi=180)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
for name, f, col, final, end_lbl, off in RUNS:
    pts = [p for p in yaml.safe_load(open(f))["trajectory"] if p.get("kind") != "baseline"]
    pts.sort(key=lambda p: float(p["t"]))
    t = [0.0] + [float(p["t"]) for p in pts]
    s = [1.0] + [float(p["score"]) for p in pts]
    s[-1] = final
    ax.plot(t, s, color=col, lw=2.2, zorder=3)
    for p in pts[:-1]:
        lbl = str(p.get("label", ""))
        hollow = lbl.startswith("microbench") or "(converted)" in lbl
        reg = p.get("kind") == "regress"
        edge = C["bad"] if reg else col
        ax.scatter(float(p["t"]), float(p["score"]), s=34, zorder=4,
                   facecolor=C["bg"] if hollow else edge, edgecolor=edge, linewidth=1.4)
    ax.scatter(t[-1], s[-1], s=135, facecolor=col, edgecolor=C["fg_bright"], linewidth=1.6, zorder=5)
    ax.annotate(end_lbl, xy=(t[-1], s[-1]), xytext=off, textcoords="offset points", color=col,
                fontsize=11.5, va="center", fontweight="bold")
    print(name, [(round(a, 1), round(b, 2)) for a, b in zip(t, s)])
ax.axhline(1, color=C["fg_dim"], lw=0.8, ls="--")
ax.text(BASELINE_LBL_X, 1.7, "PyTorch baseline (1x)", color=C["fg_dim"], fontsize=9)
ax.set_xlim(0, X_MAX_MIN)
ax.set_ylim(0, 40)
ax.set_xticks(range(0, X_MAX_MIN + 1, 60))
ax.set_xticklabels([f"{h}h" for h in range(0, X_MAX_MIN // 60 + 1)])
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
