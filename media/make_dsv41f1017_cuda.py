"""Charts for the audited DeepSeek V4.1 Flash (1017) CUDA post: 01 per-problem vs the September V4.1 Flash checkpoint and the board leader.

Reads the published model index, so field ranks and labels follow the board.
The four audited annotation YAMLs are the source for verdicts.

  uv run --no-project --with matplotlib,numpy python media/make_dsv41f1017_cuda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply  # noqa: E402

apply()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media/posts/scheduled/dsv41f1017-cuda"
OUT.mkdir(parents=True, exist_ok=True)
index = json.loads((ROOT / "public/data/models.json").read_text())
models = index["models"]
subject = next(m for m in models if m["slug"] == "deepseek-v4.1-flash-1017")

PROBLEMS = [
    ("01_glm52_fused_moe", "Fused MoE"),
    ("02_deepseek_nsa", "Native Sparse Attention"),
    ("03_megaqwen_decode", "MegaQwen Decode"),
    ("04_grid_mingru_sps", "Grid + MinGRU"),
]


def cell(m: dict, problem: str) -> dict | None:
    c = (m.get("benches", {}).get("cuda") or {}).get("cells", {}).get(problem)
    return c if c and c.get("valid") and c.get("score") is not None else None


def strength(c: dict, problem: str) -> float:
    if problem == "02_deepseek_nsa":
        ms = c.get("latency_ms")
        assert ms and ms > 0, "NSA latency missing from models.json"
        return 1.0 / ms
    return float(c["score"])


def metric(c: dict, problem: str) -> str:
    if problem == "02_deepseek_nsa":
        return f'{c["latency_ms"]:.3f} ms'
    return f'{100 * c["score"]:.2f}%'


rows = []
for problem, label in PROBLEMS:
    mine = cell(subject, problem)
    assert mine, f"missing audited DeepSeek V4.1 Flash (1017) cell: {problem}"
    peers = [(m, cell(m, problem)) for m in models if m["slug"] != subject["slug"]]
    peers = [(m, c) for m, c in peers if c]
    peer, best = max(peers, key=lambda item: strength(item[1], problem))
    prev_m = next(m for m in models if m["slug"] == "deepseek-flash")
    prev = cell(prev_m, problem)
    ranked = sorted([(subject, mine), *peers], key=lambda item: strength(item[1], problem), reverse=True)
    rank = next(i for i, (m, _) in enumerate(ranked, 1) if m["slug"] == subject["slug"])
    rows.append((problem, label, mine, peer, best, strength(mine, problem) / strength(best, problem), rank, len(ranked), prev, strength(prev, problem) / strength(best, problem) if prev else None))

# 01: one visual scale for the four cells, with the raw units printed beside each bar.
fig, ax = plt.subplots(figsize=(12.6, 6.3), dpi=180)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(-1.2, 2.45)
ax.set_ylim(-0.55, 5.45)
ax.axis("off")
ax.text(-1.15, 5.08, "DeepSeek V4.1 Flash (1017) on KernelBench-CUDA", color=C["fg_bright"], fontsize=21, weight="bold")
ax.text(-1.15, 4.66, "Final isolated regrades · RTX PRO 6000 · current audited field", color=C["fg_muted"], fontsize=11)
ax.plot([1, 1], [0.23, 4.35], color=C["fg_dim"], lw=1, ls=(0, (3, 4)), zorder=1)
ax.text(1.02, 4.42, "board leader", color=C["fg_muted"], fontsize=8.5)
for i, (problem, label, mine, peer, best, ratio, rank, n, prev, pratio) in enumerate(rows):
    y = 3.95 - i * 1.16
    ax.text(-1.15, y + 0.13, label, color=C["fg_bright"], fontsize=12, weight="bold")
    ax.text(-1.15, y - 0.21, f"#{rank} / {n}", color=C["accent"] if rank == 1 else C["fg_muted"], fontsize=10)
    ax.barh(y + 0.22, ratio, 0.2, color=C["accent"], zorder=3)
    if prev:
        ax.barh(y - 0.02, pratio, 0.16, color="#8a6d3b", zorder=2)
    ax.barh(y - 0.24, 1.0, 0.16, color="#4d5d66", zorder=2)
    ax.text(ratio + 0.035, y + 0.22, f"V4.1 Flash (1017)  {metric(mine, problem)}", va="center", color=C["fg_bright"], fontsize=9.3, weight="bold", zorder=6, bbox=dict(facecolor=C["bg"], edgecolor="none", pad=1.5))
    ax.text((pratio or 0) + 0.035, y - 0.02, f"V4.1 Flash (Sep)  {metric(prev, problem)}" if prev else "V4.1 Flash (Sep)  no valid kernel", va="center", color=C["fg_muted"], fontsize=8.8, zorder=6, bbox=dict(facecolor=C["bg"], edgecolor="none", pad=1.5))
    ax.text(1.035, y - 0.24, f'{peer["name"]}  {metric(best, problem)}', va="center", color=C["fg_muted"], fontsize=8.8, zorder=6, bbox=dict(facecolor=C["bg"], edgecolor="none", pad=1.5))
ax.text(-1.15, -0.24, "bar = performance / board leader · higher is better", color=C["fg_muted"], fontsize=9)
ax.text(-1.15, -0.48, "NSA uses inverse latency; Grid uses the deck's 150M SPS proxy.", color=C["warn"], fontsize=8.5)
fig.savefig(OUT / "01.png", dpi=180, facecolor=C["bg"], bbox_inches="tight", pad_inches=0.18)
plt.close(fig)

print("written", OUT / "01.png")
