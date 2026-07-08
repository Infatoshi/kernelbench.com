"""Hy3 + LongCat-2.0 debut on KernelBench-Hard, vs the same-tier incumbents
(Fable 5, Opus 4.8, GLM-5.2), on BOTH published GPUs (RTX PRO 6000 + H100).

Reads results/leaderboard.json and results/leaderboard.h100.json fresh, so
re-run after `kb publish` regenerates the panels with the final cells.
"""
import sys; sys.path.insert(0, "..")
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from kbh_theme import C, apply
apply()

HARD = Path(__file__).resolve().parents[1] / "benchmarks/hard/results"
PROB = ["01_fp8_gemm", "02_kda_cutlass", "03_paged_attention",
        "05_topk_bitonic", "06_sonic_moe_swiglu", "07_w4a16_gemm"]
PROB_LBL = ["01 fp8_gemm", "02 kda_cutlass", "03 paged_attn",
            "05 topk", "06 sonic_moe", "07 w4a16"]

# (display label, harness, model) — debut models lead, then the tier peers.
ROWS = [
    ("Tencent Hy3",    "hy3-claude",     "tencent/hy3-preview"),
    ("LongCat-2.0",    "longcat-claude", "LongCat-2.0"),
    ("Claude Fable 5", "claude",         "claude-fable-5"),
    ("Opus 4.8",       "claude",         "claude-opus-4-8"),
    ("GLM-5.2",        "zai-claude",     "glm-5.2"),
]
MCOL = {"Tencent Hy3": C["accent"], "LongCat-2.0": "#2dd4bf",
        "Claude Fable 5": "#cfcfcf", "Opus 4.8": "#4d9fff", "GLM-5.2": "#b07cff"}
GREY = C["fg_muted"]; AMBER = C["warn"]; RED = C["bad"]; SLATE = "#3a3a3a"


def load(board):
    d = json.load(open(HARD / board))
    out = {}
    for m in d["models"]:
        out[(m["harness"], m["model"])] = m["results"]
    return out


def cell(res, prob):
    """-> (peak_fraction, verdict) with verdict in clean|hack|fail|missing."""
    if res is None or prob not in res or res[prob] is None:
        return (0.0, "missing")
    c = res[prob]
    if isinstance(c, dict):
        pf = c.get("peak_fraction")
        verdict = c.get("verdict") or ("clean" if c.get("correct") else "fail")
        if verdict == "reward_hack":
            return (pf or 0.0, "hack")
        if pf is None or not c.get("correct", True):
            return (0.0, "fail")
        return (pf, "clean")
    return (c, "clean") if c else (0.0, "fail")


panels = [("RTX PRO 6000 Blackwell (SM120)", "leaderboard.json"),
          ("H100 PCIe (SM90)", "leaderboard.h100.json")]
fig, axes = plt.subplots(2, 1, figsize=(14.5, 12.6))
fig.subplots_adjust(top=0.86, left=0.065, right=0.975, bottom=0.06, hspace=0.42)

fig.text(0.065, 0.965, "New on KernelBench-Hard:  Tencent Hy3  and  Meituan LongCat-2.0  vs the frontier tier",
         color=C["accent"], fontsize=15.5, fontweight="bold", ha="left")
fig.text(0.065, 0.935, "One unlimited-time autonomous session per problem (Claude Code harness -> each vendor's Anthropic-compatible endpoint). bar = peak_fraction of that GPU's roofline.",
         color=GREY, fontsize=10, ha="left")
fig.text(0.065, 0.912, "Every passing cell was manually reward-hack audited (solution + full agent trace).",
         color=GREY, fontsize=10, ha="left")

x = np.arange(len(PROB)); w = 0.16
for ax, (title, board) in zip(axes, panels):
    data = load(board)
    ax.set_facecolor(C["bg"])
    for spine in ax.spines.values():
        spine.set_color(C["border"])
    top = 0.0
    for mi, (lbl, harness, model) in enumerate(ROWS):
        res = data.get((harness, model))
        off = (mi - (len(ROWS) - 1) / 2) * w
        for j, p in enumerate(PROB):
            pf, v = cell(res, p)
            base = MCOL[lbl]
            if v == "hack":
                col, hatch, alpha = RED, "////", 1.0
            elif v == "fail":
                col, hatch, alpha = base, "..", 0.4
            elif v == "missing":
                col, hatch, alpha = SLATE, None, 1.0
            else:
                col, hatch, alpha = base, None, 1.0
            h = max(pf, 0.004)
            top = max(top, pf)
            ax.bar(x[j] + off, h, w, color=col, hatch=hatch, alpha=alpha,
                   edgecolor=C["bg"], zorder=3)
            t = {"hack": "HACK", "fail": "fail", "missing": "-"}.get(v, f"{pf:.3f}")
            tc = {"clean": C["fg"], "hack": RED, "fail": AMBER, "missing": GREY}[v]
            ax.text(x[j] + off, h + 0.006, t, ha="center", va="bottom", color=tc,
                    fontsize=6.6, fontweight="bold" if v == "clean" else "normal",
                    rotation=90 if v == "clean" else 0)
    ax.set_title(title, color=C["fg"], fontsize=12, loc="left", pad=10)
    ax.set_xticks(x); ax.set_xticklabels(PROB_LBL, fontsize=10.5)
    ax.set_ylabel("peak_fraction")
    ax.set_ylim(0, max(0.75, top * 1.2)); ax.set_xlim(-0.55, len(PROB) - 0.45)
    ax.grid(True, axis="y", alpha=0.5)

leg = [Patch(facecolor=MCOL[lbl], label=lbl) for lbl, _, _ in ROWS]
leg += [Patch(facecolor=RED, hatch="////", label="reward hack (invalid)"),
        Patch(facecolor="#888888", hatch="..", alpha=0.4, label="failed correctness"),
        Patch(facecolor=SLATE, label="not run")]
axes[0].legend(handles=leg, loc="upper center", ncol=4, facecolor=C["surface"],
               edgecolor=C["border"], labelcolor=C["fg"], fontsize=8.4, framealpha=0.97)

out = Path(__file__).with_name("hy3_longcat_debut.png")
fig.savefig(out, dpi=140)
print(f"wrote {out}")
