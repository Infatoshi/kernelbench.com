"""Gemini 4 Neon (XHigh) mega post, image 1: Neon, Opus 5.5, and Gemini 4 Carbon
hill-climbs on Kimi-Linear Decode.

Same look as make_gpt62astra_mega_traj_compare.py. All curves end on their published
isolated regrade score read from public/data/mega/results.csv (grader v3). Neon's
in-session benchmark.py points come from gpu_lock.log + transcript.jsonl; Opus 5.5
and Carbon come from their annotation YAMLs.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_neon_mega_traj_compare.py
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

OUT = ROOT / "media/posts/audited/neon-mega/01.png"
NEON_ID = "20261118_150415_agy_gemini-4-neon-xhigh_02_kimi_linear_decode"
OPUS_ID = "20260922_121505_claude_claude-opus-5-5_02_kimi_linear_decode"
CARBON_ID = "20261110_113616_agy_gemini-4-carbon-high_02_kimi_linear_decode"

BOARD = {
    r["run_id"]: float(r["score"])
    for r in csv.DictReader(open(ROOT / "public/data/mega/results.csv"))
    if r["score"]
}
NEON_FINAL = BOARD[NEON_ID]      # 36.655
OPUS_FINAL = BOARD[OPUS_ID]      # 35.337
CARBON_FINAL = BOARD[CARBON_ID]  # 24.677

# Exact in-session benchmark.py wall-clock minutes (gpu_lock.log) & scores (transcript.jsonl)
NEON_PTS = [
    {"t": 13.1, "score": 14.1502},
    {"t": 179.1, "score": 21.1905},
    {"t": 224.5, "score": 30.1603},
    {"t": 233.7, "score": 29.7809, "kind": "regress"},
    {"t": 247.7, "score": 31.0328},
    {"t": 282.7, "score": 34.2154},
    {"t": 293.8, "score": 35.0882},
    {"t": 308.6, "score": 35.3104},
    {"t": 333.8, "score": 35.9231},
    {"t": 363.9, "score": 35.9983},
    {"t": 571.1, "score": NEON_FINAL, "kind": "final"},
]

CARBON_SELF = [
    {"t": 58.8, "score": 16.89, "label": "microbench"},
    {"t": 91.0, "score": 19.15, "label": "microbench"},
    {"t": 109.1, "score": 21.66, "label": "microbench"},
    {"t": 121.4, "score": 23.73, "label": "microbench"},
    {"t": 139.3, "score": 24.52, "label": "microbench"},
    {"t": 145.3, "score": 24.14, "label": "microbench", "kind": "regress"},
]

apply()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(11.8, 5.8), dpi=180)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])

RUNS = [
    ("Gemini 4 Carbon", ANN / f"{CARBON_ID}.yaml", "#94a3b8", CARBON_FINAL, CARBON_SELF,
     f"Gemini 4 Carbon  {CARBON_FINAL:.1f}x", (-108, 18)),
    ("Opus 5.5", ANN / f"{OPUS_ID}.yaml", "#4d9fff", OPUS_FINAL, [],
     f"Opus 5.5  {OPUS_FINAL:.1f}x", (10, 14)),
    ("Gemini 4 Neon", None, C["accent"], NEON_FINAL, NEON_PTS,
     f"Gemini 4 Neon  {NEON_FINAL:.1f}x", (-150, 14)),
]

for name, f, col, final, extra, end_lbl, off in RUNS:
    if f is not None:
        pts = [p for p in yaml.safe_load(open(f))["trajectory"] if p.get("kind") != "baseline"] + extra
    else:
        pts = list(extra)
    pts.sort(key=lambda p: float(p["t"]))
    t = [0.0] + [float(p["t"]) for p in pts]
    s = [1.0] + [float(p["score"]) for p in pts]
    s[-1] = final
    ax.plot(t, s, color=col, lw=2.3 if name == "Gemini 4 Neon" else 2.0, zorder=4 if name == "Gemini 4 Neon" else 3)
    for p in pts[:-1]:
        lbl = str(p.get("label", ""))
        hollow = lbl.startswith("microbench") or "(converted)" in lbl
        reg = p.get("kind") == "regress"
        edge = C["bad"] if reg else col
        ax.scatter(
            float(p["t"]),
            float(p["score"]),
            s=34,
            zorder=5 if name == "Gemini 4 Neon" else 4,
            facecolor=C["bg"] if hollow else edge,
            edgecolor=edge,
            linewidth=1.4,
        )
    ax.scatter(t[-1], s[-1], s=135, facecolor=col, edgecolor=C["fg_bright"], linewidth=1.6, zorder=6)
    ax.annotate(
        end_lbl,
        xy=(t[-1], s[-1]),
        xytext=off,
        textcoords="offset points",
        color=col,
        fontsize=11.5,
        va="center",
        fontweight="bold",
    )
    print(name, [(round(a, 1), round(b, 2)) for a, b in zip(t, s)])

ax.axhline(1, color=C["fg_dim"], lw=0.8, ls="--")
ax.text(8, 1.7, "PyTorch baseline (1x)", color=C["fg_dim"], fontsize=9)
ax.set_xlim(0, 600)
ax.set_ylim(0, 42)
ax.set_xticks(range(0, 601, 120))
ax.set_xticklabels([f"{h}h" for h in range(0, 11, 2)])
ax.set_xlabel("wall clock", color=C["fg_muted"], fontsize=10.5)
ax.set_ylabel("speedup over baseline", color=C["fg_muted"], fontsize=10.5)
ax.grid(color=C["grid"], lw=0.6)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(colors=C["fg_muted"], labelsize=10)
fig.suptitle(
    "Kimi-Linear Decode megakernel trajectories",
    color=C["fg_bright"],
    fontsize=15.5,
    fontweight="bold",
    x=0.06,
    ha="left",
    y=0.95,
)
fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.11)
fig.savefig(OUT, dpi=180, facecolor=C["bg"])
print(OUT)
