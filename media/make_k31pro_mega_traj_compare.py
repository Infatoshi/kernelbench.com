"""Kimi K3.1 Pro mega post, image 1: Kimi K3.1 Pro and Opus 5.5 hill-climbs on Kimi-Linear Decode.

Same look as make_carbon_mega_traj_compare.py. Kimi's kernel is real but its MLA attention is wrong
(verdict bug), so it has no board score: its curve is faded and dotted.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_k31pro_mega_traj_compare.py
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
OUT = ROOT / "media/posts/audited/k31pro-mega/01.png"
# (name, yaml, colour, final, extra points, scored, end label)
# final=None keeps the YAML's last in-session point (no board score to override it with).
RUNS = [
    ("Opus 5.5", ANN / "20260922_121505_claude_claude-opus-5-5_02_kimi_linear_decode.yaml",
     "#4d9fff", 35.46, [], True, "Opus 5.5  35.5x"),
    ("Kimi K3.1 Pro", ANN / "20261116_012038_kimi_kimi-code_k3.1-pro_02_kimi_linear_decode.yaml",
     C["accent"], None, [], False, "Kimi K3.1 Pro  no score (8.6x, wrong attention)"),
]
FADED_ALPHA = 0.5
FADED_TEXT_ALPHA = 0.75  # end label stays readable while the curve reads as faded
BASELINE_LBL_X = 125  # moved right of Carbon's x=3 so the faded curve does not cross it
# ---------------------------------------------------------------------------

apply()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(11.8, 5.8), dpi=180)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
for name, f, col, final, extra, scored, end_lbl in RUNS:
    pts = [p for p in yaml.safe_load(open(f))["trajectory"] if p.get("kind") != "baseline"] + extra
    pts.sort(key=lambda p: float(p["t"]))
    t = [0.0] + [float(p["t"]) for p in pts]
    s = [1.0] + [float(p["score"]) for p in pts]
    if final is not None:
        s[-1] = final
    a = 1.0 if scored else FADED_ALPHA
    ax.plot(t, s, color=col, lw=2.2, zorder=3, alpha=a, ls="-" if scored else (0, (1.2, 2.2)))
    for p in pts[:-1]:
        micro = str(p.get("label", "")).startswith("microbench")
        reg = p.get("kind") == "regress"
        ax.scatter(float(p["t"]), float(p["score"]), s=34, zorder=4, alpha=a,
                   facecolor=C["bg"] if micro else (C["bad"] if reg else col),
                   edgecolor=C["bad"] if reg else col, linewidth=1.4)
    if scored:
        ax.scatter(t[-1], s[-1], s=135, facecolor=col, edgecolor=C["fg_bright"], linewidth=1.6, zorder=5)
    else:
        ax.scatter(t[-1], s[-1], s=110, facecolor=C["bg"], edgecolor=col, linewidth=1.6, zorder=5, alpha=a)
    ax.text(t[-1] + 4.5, s[-1], end_lbl, color=col, fontsize=11.5, va="center", fontweight="bold",
            alpha=1.0 if scored else FADED_TEXT_ALPHA)
ax.axhline(1, color=C["fg_dim"], lw=0.8, ls="--")
ax.text(BASELINE_LBL_X, 1.7, "PyTorch baseline (1x)", color=C["fg_dim"], fontsize=9)
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
