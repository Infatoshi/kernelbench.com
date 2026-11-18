"""GPT-6.2 Astra mega post, image 1: GPT-6.2 Astra (session 1, the board row) and Opus 5.5
hill-climbs on Kimi-Linear Decode.

Same look as make_qwen4max_mega_traj_compare.py. Both curves end on their published isolated
regrade score, read from public/data/mega/results.csv (grader v3), never typed in. Astra's points
are in-session benchmark.py results from its annotation's trajectory list; Opus's hollow points are
its session's us/step timings converted to speedup by its audit, not benchmark.py.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_gpt62astra_mega_traj_compare.py
"""
import csv
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
OUT = ROOT / "media/posts/audited/gpt62astra-mega/01.png"
ASTRA_ID = "20261117_133336_codex_gpt-6.2-astra_02_kimi_linear_decode"
OPUS_ID = "20260922_121505_claude_claude-opus-5-5_02_kimi_linear_decode"
BOARD = {r["run_id"]: float(r["score"]) for r in csv.DictReader(open(ROOT / "public/data/mega/results.csv")) if r["score"]}
ASTRA_FINAL = BOARD[ASTRA_ID]  # published isolated regrade (23.478)
OPUS_FINAL = BOARD[OPUS_ID]    # published isolated regrade, grader v3 (35.337)
# (name, yaml, colour, final score, end label, label offset in points)
RUNS = [
    ("Opus 5.5", ANN / f"{OPUS_ID}.yaml", "#4d9fff", OPUS_FINAL, f"Opus 5.5  {OPUS_FINAL:.1f}x", (9, 0)),
    ("GPT-6.2 Astra", ANN / f"{ASTRA_ID}.yaml", C["accent"], ASTRA_FINAL,
     f"GPT-6.2 Astra  {ASTRA_FINAL:.1f}x", (9, 0)),
]
X_MAX_MIN = 240
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
