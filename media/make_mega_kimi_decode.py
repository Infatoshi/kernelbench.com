"""KernelBench-Mega 02_kimi_linear_decode: decode speedup over the optimized
PyTorch baseline, per model, one panel per GPU (B200 / H100 / RTX PRO 6000).

Reads public/data/mega/results.csv (the published /mega board), plus the two
July 2026 debut cells (Hy3 0.31x clean-audited, LongCat DNF) which are not on
the board yet — marked "(new)".
"""
import sys; sys.path.insert(0, "..")
import csv
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kbh_theme import C, apply
apply()

CSV = Path(__file__).resolve().parents[1] / "public/data/mega/results.csv"
GPUS = ["B200", "H100", "RTX PRO 6000 Blackwell"]
GPU_TITLES = {"B200": "B200 (SM100)", "H100": "H100 PCIe (SM90)",
              "RTX PRO 6000 Blackwell": "RTX PRO 6000 Blackwell (SM120)"}

NAME = {
    "claude-opus-4-8": "Opus 4.8", "claude-fable-5": "Fable 5",
    "claude-sonnet-5": "Sonnet 5", "gpt-5.5": "GPT-5.5",
    "glm-5.2": "GLM-5.2", "MiniMax-M3": "MiniMax M3",
    "gemini-3.5-flash": "Gemini 3.5 Flash", "deepseek-v4-pro": "DeepSeek V4 Pro",
    "composer-2.5-fast": "Composer 2.5 Fast", "kimi-k2.7-code": "Kimi K2.7",
    "tencent/hy3-preview": "Tencent Hy3 (new)", "LongCat-2.0": "LongCat 2.0 (new)",
}

rows = {g: [] for g in GPUS}
for r in csv.DictReader(open(CSV)):
    if r["problem"] != "02_kimi_linear_decode":
        continue
    score = float(r["score"]) if r["score"] else None
    rows[r["gpu"]].append((NAME.get(r["model"], r["model"]), score))
# July 2026 debut cells (audited, not yet on the published board)
rows["RTX PRO 6000 Blackwell"] += [("Tencent Hy3 (new)", 0.31), ("LongCat 2.0 (new)", None)]

fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), sharey=False)
fig.subplots_adjust(top=0.80, bottom=0.30, left=0.05, right=0.99, wspace=0.16)
fig.text(0.05, 0.945, "KernelBench-Mega:  Kimi-Linear W4A16 decode megakernel — speedup over optimized-PyTorch baseline",
         color=C["accent"], fontsize=15, fontweight="bold")
fig.text(0.05, 0.905, "One autonomous session per cell (3h cloud / 45min local cap). The agent rewrites the whole decode step as fused kernels; score = end-to-end decode speedup, geomean over 2k/8k/16k contexts.",
         color=C["fg_muted"], fontsize=9.5)
fig.text(0.05, 0.875, "Grey dotted = wrote code but failed correctness or timed out (DNF). \"(new)\" = July 2026 debut runs, audited, not yet on the published board.",
         color=C["fg_muted"], fontsize=9.5)

for ax, g in zip(axes, GPUS):
    data = sorted(rows[g], key=lambda t: -(t[1] if t[1] is not None else -1))
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    x = np.arange(len(data))
    ax.set_facecolor(C["bg"])
    for spine in ax.spines.values():
        spine.set_color(C["border"])
    best = max((v for v in vals if v is not None), default=0)
    for i, (lbl, v) in enumerate(data):
        if v is None:
            ax.bar(x[i], best * 0.02, color=C["fg_muted"], alpha=0.4,
                   hatch="..", edgecolor=C["bg"], zorder=3)
            ax.text(x[i], best * 0.03, "DNF", ha="center", va="bottom",
                    color=C["warn"], fontsize=7.5)
        else:
            col = C["accent"] if v == best else "#4d9fff"
            ax.bar(x[i], v, color=col, edgecolor=C["bg"], zorder=3)
            ax.text(x[i], v + best * 0.015, f"{v:.2f}x", ha="center", va="bottom",
                    color=C["fg"], fontsize=8, fontweight="bold" if v == best else "normal")
    ax.axhline(1.0, color=C["bad"], lw=0.8, ls="--", alpha=0.7, zorder=2)
    ax.text(len(data) - 0.4, 1.06, "baseline", color=C["bad"], fontsize=7,
            ha="right", va="bottom")
    ax.set_title(GPU_TITLES[g], color=C["fg"], fontsize=11.5, loc="left", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=8.5)
    ax.set_ylim(0, best * 1.18)
    ax.grid(True, axis="y", alpha=0.5)
    if g == GPUS[0]:
        ax.set_ylabel("decode speedup (x)")

out = Path(__file__).with_name("mega_kimi_decode.png")
fig.savefig(out, dpi=140)
print(f"wrote {out}")
