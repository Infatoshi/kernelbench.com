"""Charts for the audited Opus 5.5 CUDA thread.

Reads the published model index, so field ranks and labels follow the board.
The four audited annotation YAMLs explain the overlapping operator resumes.

  uv run --no-project --with matplotlib,numpy python media/make_opus55_cuda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply  # noqa: E402

apply()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "media/posts/audited/opus55-cuda"
OUT.mkdir(parents=True, exist_ok=True)
index = json.loads((ROOT / "public/data/models.json").read_text())
models = index["models"]
subject = next(m for m in models if m["slug"] == "claude-opus-5-5")

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
    assert mine, f"missing audited Opus cell: {problem}"
    peers = [(m, cell(m, problem)) for m in models if m["slug"] != subject["slug"]]
    peers = [(m, c) for m, c in peers if c]
    peer, best = max(peers, key=lambda item: strength(item[1], problem))
    ranked = sorted([(subject, mine), *peers], key=lambda item: strength(item[1], problem), reverse=True)
    rank = next(i for i, (m, _) in enumerate(ranked, 1) if m["slug"] == subject["slug"])
    rows.append((problem, label, mine, peer, best, strength(mine, problem) / strength(best, problem), rank, len(ranked)))

# 01: one visual scale for the four cells, with the raw units printed beside each bar.
fig, ax = plt.subplots(figsize=(12.6, 6.3), dpi=180)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(-1.2, 2.45)
ax.set_ylim(-0.55, 5.45)
ax.axis("off")
ax.text(-1.15, 5.08, "Opus 5.5 on KernelBench-CUDA", color=C["fg_bright"], fontsize=21, weight="bold")
ax.text(-1.15, 4.66, "Final isolated regrades · RTX PRO 6000 · current audited field", color=C["fg_muted"], fontsize=11)
ax.plot([1, 1], [0.23, 4.35], color=C["fg_dim"], lw=1, ls=(0, (3, 4)))
ax.text(1.02, 4.42, "best other model", color=C["fg_muted"], fontsize=8.5)
for i, (problem, label, mine, peer, best, ratio, rank, n) in enumerate(rows):
    y = 3.95 - i * 1.16
    ax.text(-1.15, y + 0.13, label, color=C["fg_bright"], fontsize=12, weight="bold")
    ax.text(-1.15, y - 0.21, f"#{rank} / {n}", color=C["accent"] if rank == 1 else C["fg_muted"], fontsize=10)
    ax.barh(y + 0.14, ratio, 0.22, color=C["accent"], zorder=3)
    ax.barh(y - 0.19, 1.0, 0.18, color="#4d5d66", zorder=2)
    ax.text(ratio + 0.035, y + 0.14, f"Opus 5.5  {metric(mine, problem)}", va="center", color=C["fg_bright"], fontsize=9.3, weight="bold")
    ax.text(1.035, y - 0.19, f'{peer["name"]}  {metric(best, problem)}', va="center", color=C["fg_muted"], fontsize=8.8)
ax.text(-1.15, -0.24, "bar = Opus performance / best other published model · higher is better", color=C["fg_muted"], fontsize=9)
ax.text(-1.15, -0.48, "NSA uses inverse latency; Grid uses the deck's 150M SPS proxy. Three cells had overlapping operator resumes.", color=C["warn"], fontsize=8.5)
fig.savefig(OUT / "01.png", dpi=180, facecolor=C["bg"], bbox_inches="tight", pad_inches=0.18)
plt.close(fig)

# 02: raw latency on the structurally sparse problem; no dense-equivalent roofline.
problem = "02_deepseek_nsa"
field = sorted(((m, cell(m, problem)) for m in models), key=lambda item: strength(item[1], problem) if item[1] else -1, reverse=True)
field = [(m, c) for m, c in field if c][:3]
fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=180)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
for i, (m, c) in enumerate(field):
    y = 2 - i
    ms = c["latency_ms"]
    color = C["accent"] if m["slug"] == subject["slug"] else "#4d5d66"
    ax.plot([0, ms], [y, y], color=color, lw=5, solid_capstyle="round")
    ax.scatter([ms], [y], s=180, color=color, zorder=3)
    ax.text(ms + 0.005, y, f"{ms:.3f} ms", va="center", color=C["fg_bright"], fontsize=13, weight="bold")
ax.set_yticks([2, 1, 0], [m["name"] for m, _ in field], color=C["fg"], fontsize=13)
ax.set_xlim(0, max(c["latency_ms"] for _, c in field) * 1.43)
ax.set_ylim(-0.4, 2.5)
ax.set_xlabel("geomean milliseconds across six shapes · lower is better", color=C["fg_muted"], fontsize=10)
ax.set_title("Native Sparse Attention · RTX PRO 6000", loc="left", color=C["fg_bright"], fontsize=17, pad=22)
ax.grid(axis="x", color=C["grid"], lw=0.8)
for edge in ("top", "right", "left"):
    ax.spines[edge].set_visible(False)
ax.spines["bottom"].set_color(C["border"])
ax.tick_params(axis="y", length=0)
fig.subplots_adjust(left=0.28, right=0.96, top=0.83, bottom=0.17)
fig.savefig(OUT / "02.png", dpi=180, facecolor=C["bg"])
plt.close(fig)

# 03: the four submitted designs, simplified from their audited source files.
fig, ax = plt.subplots(figsize=(14, 6.8), dpi=180)
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.axis("off")
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.8)
ax.text(0.25, 6.35, "Four CUDA designs in the final submissions", color=C["fg_bright"], fontsize=18, weight="bold")
ax.text(0.25, 5.99, "Simplified from the audited solution.py files; arrows are data flow, not separate launches", color=C["fg_muted"], fontsize=9.5)
pipelines = [
    ("Fused MoE", ["count + scatter", "bf16 gate/up GEMM", "SiLU × up", "bf16 down GEMM", "fp32 combine"]),
    ("Sparse attention", ["block means + top-8", "key-major sparse MMA", "local window", "softmax merge"]),
    ("MegaQwen", ["stream step input", "QKV + RoPE", "full-prefix GQA", "O projection", "SwiGLU MLP"]),
    ("Grid + MinGRU", ["fold encoder once", "3× MinGRU", "greedy action", "grid + exact RNG"]),
]
for i, (name, steps) in enumerate(pipelines):
    y = 5.14 - i * 1.2
    ax.text(0.25, y + 0.23, name, va="center", color=C["accent"], fontsize=12, weight="bold")
    x0, span, gap = 2.45, 11.15, 0.12
    w = (span - gap * (len(steps) - 1)) / len(steps)
    for j, step in enumerate(steps):
        x = x0 + j * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y - 0.23), w, 0.62, boxstyle="round,pad=0.02", fc=C["surface_muted"], ec=C["border"]))
        ax.text(x + w / 2, y + 0.08, step, ha="center", va="center", color=C["fg"], fontsize=9.1)
        if j + 1 < len(steps):
            ax.annotate("", xy=(x + w + gap - 0.01, y + 0.08), xytext=(x + w + 0.01, y + 0.08),
                        arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.4))
ax.text(2.45, 0.38, "MegaQwen loops over steps and four layers in one cooperative launch; Grid loops over the rollout horizon in one.", color=C["fg_muted"], fontsize=9.5)
fig.savefig(OUT / "03.png", dpi=180, facecolor=C["bg"], bbox_inches="tight", pad_inches=0.18)
plt.close(fig)
print("written", *(OUT / f"0{i}.png" for i in range(1, 4)))
