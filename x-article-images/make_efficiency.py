"""Performance vs compute (output tokens) for KernelBench mega + hard.

Answers "is the win just bought with tokens?" with a Pareto scatter (perf vs
output tokens) and an efficiency bar (perf per 100k tokens). RTX PRO 6000 only
(richest clean token telemetry). Models whose tokens aren't captured by the
harness (kimi/deepseek native, routed-provider gaps) are excluded, noted in
the subtitle. Output tokens = the compute the model chose to spend; it is
hardware- and price-agnostic.
"""
import sys; sys.path.insert(0, "..")
import csv
import json
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kbh_theme import C, apply
apply()

MUTED = "#5b6470"
NAMES = {
    "claude-opus-4-8": "Opus 4.8", "glm-5.2": "GLM-5.2", "gpt-5.5": "GPT-5.5",
    "MiniMax-M3": "MiniMax-M3", "kimi-k2.7-code": "Kimi K2.7", "claude-fable-5": "Fable 5",
    "composer-2.5-fast": "Composer 2.5", "gemini-3.5-flash": "Gemini 3.5", "deepseek-v4-pro": "DeepSeek V4",
}


def frontier(points):
    """points: list of (x_tokens, y_perf, name). Return set of frontier names
    (max perf for <= token budget)."""
    fr = set()
    best = -1
    for x, y, n in sorted(points, key=lambda p: p[0]):
        if y > best:
            fr.add(n); best = y
    return fr


def scatter(points, title, subtitle, ylabel, yfmt, out):
    fr = frontier(points)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(top=0.84, left=0.09, right=0.97, bottom=0.12)
    ax.set_facecolor(C["bg"])
    for sp in ax.spines.values(): sp.set_color(C["border"])

    fpts = sorted([p for p in points if p[2] in fr], key=lambda p: p[0])
    ax.plot([p[0] / 1000 for p in fpts], [p[1] for p in fpts],
            color=C["accent_dim"], lw=1.4, zorder=2)
    xmax = max(p[0] for p in points) / 1000
    for i, (x, y, n) in enumerate(sorted(points, key=lambda p: p[0])):
        on = n in fr
        ax.scatter(x / 1000, y, s=130, color=C["accent"] if on else MUTED,
                   edgecolor=C["bg"], zorder=4)
        right = (x / 1000) > 0.78 * xmax
        dx, ha = (-10, "right") if right else (9, "left")
        dy = 18 if (i % 2) else 8
        ax.annotate(f"{n}", (x / 1000, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha,
                    color=C["fg"] if on else C["fg_muted"], fontsize=10.5)

    ax.set_xlabel("output tokens spent  (thousands)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(lw=0.6, alpha=0.4); ax.set_axisbelow(True)
    ax.tick_params(colors=C["fg_muted"])
    ax.set_xlim(0, xmax * 1.12); ax.set_ylim(0, max(p[1] for p in points) * 1.16)
    fig.text(0.09, 0.94, title, color=C["accent"], fontsize=17, fontweight="bold", ha="left")
    fig.text(0.09, 0.895, subtitle, color=C["fg_muted"], fontsize=10, ha="left")
    fig.text(0.09, 0.865, "Green = on the efficiency frontier (most perf per token). Grey = dominated (spent more, delivered less).",
             color=C["fg_muted"], fontsize=9.5, ha="left")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def effbar(points, title, subtitle, unit, out):
    rows = sorted([(n, y / (x / 100000)) for x, y, n in points], key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(top=0.83, left=0.16, right=0.96, bottom=0.12)
    ax.set_facecolor(C["bg"])
    for sp in ax.spines.values(): sp.set_color(C["border"])
    y = range(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(list(y), vals, color=C["accent"], edgecolor=C["bg"], height=0.72, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.012, yi, f"{v:.2f}", va="center", color=C["fg"], fontsize=11)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=C["fg"])
    ax.set_xlabel(unit, fontsize=11)
    ax.set_xlim(0, max(vals) * 1.12); ax.tick_params(colors=C["fg_muted"])
    ax.grid(axis="x", lw=0.6, alpha=0.4); ax.set_axisbelow(True)
    fig.text(0.16, 0.93, title, color=C["accent"], fontsize=17, fontweight="bold", ha="left")
    fig.text(0.16, 0.885, subtitle, color=C["fg_muted"], fontsize=10, ha="left")
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


# --- MEGA (RTX PRO 6000, clean output_tokens) ---
mega = []
for r in csv.DictReader(open("../public/data/mega/results.csv")):
    if r["correct"] != "true" or r["gpu"] != "RTX PRO 6000 Blackwell":
        continue
    ot = r["output_tokens"]
    ot = int(ot) if ot.strip().isdigit() else 0
    if ot > 1000:
        mega.append((ot, float(r["score"]), NAMES.get(r["model"], r["model"])))

scatter(mega, "Mega: speedup vs tokens spent",
        "Kimi linear-decode megakernel, RTX PRO 6000. Speedup over reference vs output tokens.",
        "speedup over reference", None, "mega_eff_scatter.png")
effbar(mega, "Mega: speedup per 100k tokens",
       "Token efficiency on the megakernel, RTX PRO 6000. Higher = more speedup per unit of generation.",
       "speedup per 100k output tokens", "mega_eff_bar.png")

# --- HARD (RTX PRO 6000, output_tokens summed across solved problems) ---
lb = json.load(open("../benchmarks/hard/results/leaderboard.json"))
hard = []
for m in lb["models"]:
    nm = NAMES.get(m["model"])
    if not nm:
        continue
    pfs, toks = [], 0
    for c in m["results"].values():
        if c.get("correct") and c.get("peak_fraction"):
            pfs.append(c["peak_fraction"])
            p = f"../benchmarks/hard/outputs/runs/{c['run_id']}/result.json"
            if os.path.exists(p):
                toks += (json.load(open(p)).get("usage") or {}).get("output_tokens") or 0
    if pfs and toks > 1000:
        hard.append((toks, 100 * sum(pfs) / len(pfs), nm))

scatter(hard, "Hard: roofline % vs tokens spent",
        "6-problem CUDA/Triton deck, RTX PRO 6000. Avg % of roofline (solved problems) vs total output tokens.",
        "percent of hardware roofline", None, "hard_eff_scatter.png")
effbar(hard, "Hard: roofline % per 100k tokens",
       "Token efficiency on the hard deck, RTX PRO 6000. Higher = more roofline per unit of generation.",
       "roofline % per 100k output tokens", "hard_eff_bar.png")
