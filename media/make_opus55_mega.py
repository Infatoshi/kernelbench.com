"""Opus 5.5 [xhigh] mega post: 01 rank chart (Kimi-Linear Decode field), 03 design diagram.
Field = public/data/mega/results.csv 2026-09-22, best published cell per model, RTX PRO 6000.
Opus 5.5 = isolated regrade on tetra GPU 2, audited clean (annotation 20260922_121505_...).

  uv run --no-project --with matplotlib,numpy python media/make_opus55_mega.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "posts" / "unaudited"))
from kbh_theme import C, apply
from make_charts import save_rank
apply()
OUT = Path(__file__).resolve().parent / "posts" / "audited" / "opus55-mega"

save_rank(OUT / "01.png", [
    ("Opus 5.5", 35.460, True), ("GPT-6 Astra", 24.804, False), ("Fable 5", 24.609, False),
    ("Fable 5.1", 22.946, False), ("GLM-5.3", 19.435, False), ("K3 (256k)", 18.088, False),
    ("V4.1 Flash", 17.101, False), ("Opus 4.8", 14.399, False), ("5.3 Flash", 13.642, False),
    ("GLM-5.2", 11.142, False), ("Grok 4.7", 6.379, False),
], "Kimi-Linear Decode · speedup vs optimized PyTorch · RTX PRO 6000", "{:.1f}x")

# design diagram: one SM's block over time, compute warps vs the streamer warp, barriers
fig, ax = plt.subplots(figsize=(12, 6.2), dpi=180)
fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"]); ax.axis("off")
ax.set_xlim(0, 12); ax.set_ylim(0, 6.2)
phases = ["KDA proj", "short conv", "gated-delta S", "MLA absorbed", "router", "experts"]
x0, w, gap = 1.9, 1.55, 0.12
ax.text(0.2, 5.75, "one of 188 blocks, one decode step (simplified: 6 of its 25 grid-barrier phases)", color=C["fg_bright"], fontsize=13, fontweight="bold")
ax.text(0.2, 5.35, "16 compute warps", color=C["fg"], fontsize=11)
ax.text(0.2, 3.35, "17th warp: L2 streamer", color=C["accent"], fontsize=11)
for i, ph in enumerate(phases):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 4.1), w, 1.05, boxstyle="round,pad=0.02", fc=C["surface_muted"], ec=C["border"]))
    ax.text(x + w / 2, 4.62, ph, ha="center", va="center", color=C["fg"], fontsize=9.5, wrap=True)
    ax.plot([x + w + gap / 2] * 2, [2.3, 5.25], color=C["fg_dim"], lw=1, ls=(0, (3, 3)))
    if i + 2 < len(phases):
        tx = x0 + (i + 2) * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 2.7), w, 0.55, boxstyle="round,pad=0.02", fc=C["accent_dim"], ec=C["accent"]))
        ax.text(x + w / 2, 2.97, f"loads {phases[i+2]}".replace("gated-delta S","delta S").replace("MLA absorbed","MLA"), ha="center", va="center", color=C["fg_bright"], fontsize=8)
        ax.annotate("", xy=(tx + w / 2, 4.08), xytext=(x + w / 2, 3.27),
                    arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.1, alpha=0.8))
ax.text(x0 + 2.5 * (w + gap), 2.1, "grid barrier", color=C["fg_dim"], fontsize=9, ha="center")
lines = [
    "int4 weights + KDA state + latent cache: 243-259 MB read per token",
    "effective 1.39-1.53 TB/s against 1.79 TB/s peak: bandwidth-bound, not launch- or barrier-bound",
    "streamer uses real ld.global.cg.L2::128B loads two phases ahead (ncu showed prefetch hints are dropped on sm_120)",
    "routed experts prefetched from a predicted top-8; a wrong guess only costs bandwidth",
]
for j, l in enumerate(lines):
    ax.text(0.2, 1.45 - j * 0.38, "· " + l, color=C["fg_muted"] if j else C["fg"], fontsize=10)
fig.savefig(OUT / "03.png", dpi=180, facecolor=C["bg"])
print("written", OUT)
