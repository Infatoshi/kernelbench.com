"""Qwen 3.8 Max in-body charts — one GPU, green on black.

01: Sonic ranking (subject green, field muted). 02: Qwen Hard deck.
Reads the published RTX Hard leaderboard.

  uv run --no-project --with matplotlib,numpy python make_qwen38_article.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kbh_theme import C, apply

apply()

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "X-article-qwen38"
HARD = ROOT / "benchmarks/hard/results/leaderboard.json"
SONIC = "06_sonic_moe_swiglu"
SUBJECT = ("or-fable", "qwen/qwen3.8-max")
FIELD = "#4d5d66"
PRETTY = {
    "qwen/qwen3.8-max": "Qwen 3.8",
    "anthropic/claude-fable-5": "Fable 5",
    "claude-fable-5": "Fable 5",
    "anthropic/claude-opus-5": "Opus 5",
    "claude-opus-5": "Opus 5",
    "grok-4.5": "Grok 4.5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "glm-5.2": "GLM-5.2",
    "claude-opus-4-8": "Opus 4.8",
    "deepseek/deepseek-v4-flash-0731": "V4 Flash",
}


def peak(cell: dict | None) -> float | None:
    if not isinstance(cell, dict) or cell.get("correct") is False:
        return None
    pf = cell.get("peak_fraction")
    return float(pf) if pf is not None else None


def save_rank(path: Path, rows: list[tuple[str, float, bool]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    rows = list(reversed(rows))
    y = np.arange(len(rows))
    colors = [C["accent"] if subj else FIELD for _, _, subj in rows]
    vals = [v for _, v, _ in rows]
    ax.barh(y, vals, 0.68, color=colors, edgecolor=C["bg"], linewidth=0.4, zorder=3)
    for yi, (lbl, v, subj) in zip(y, rows):
        ax.text(v + max(vals) * 0.02, yi, f"{v:.3f}", va="center", fontsize=12,
                color=C["fg_bright"] if subj else C["fg_muted"],
                fontweight="bold" if subj else "regular")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=14, color=C["fg"])
    ax.set_xlabel("Sonic MoE · peak fraction · RTX PRO 6000", fontsize=10, color=C["fg_muted"])
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["border"])
    ax.tick_params(colors=C["fg_muted"], length=0)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.97, bottom=0.10)
    fig.savefig(path, dpi=180, facecolor=C["bg"])
    print(path)


def save_deck(path: Path, items: list[tuple[str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    labels = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    y = np.arange(len(vals))
    hi = max(vals)
    colors = [C["accent"] if v == hi else FIELD for v in vals]
    ax.barh(y, vals, 0.68, color=colors, edgecolor=C["bg"], linewidth=0.4, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + hi * 0.03, yi, f"{v:.3f}", va="center", fontsize=12,
                color=C["fg_bright"] if v == hi else C["fg_muted"],
                fontweight="bold" if v == hi else "regular")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14, color=C["fg"])
    ax.set_xlabel("peak fraction · RTX PRO 6000", fontsize=10, color=C["fg_muted"])
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="x", color=C["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["border"])
    ax.tick_params(colors=C["fg_muted"], length=0)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.97, bottom=0.10)
    fig.savefig(path, dpi=180, facecolor=C["bg"])
    print(path)


def main() -> None:
    board = json.loads(HARD.read_text())
    sonic_rows: list[tuple[str, float, bool]] = []
    hard_lbl = {
        "01_fp8_gemm": "FP8",
        "02_kda_cutlass": "KDA",
        "03_paged_attention": "Paged",
        "05_topk_bitonic": "TopK",
        "06_sonic_moe_swiglu": "Sonic",
    }
    best_sonic: dict[str, float] = {}
    for m in board.get("models", []):
        mid = m.get("model")
        if mid not in PRETTY:
            continue
        s = peak((m.get("results") or {}).get(SONIC))
        if s is None:
            continue
        name = PRETTY[mid]
        if s > best_sonic.get(name, -1):
            best_sonic[name] = s
    sonic_rows = sorted(
        [(n, v, n == "Qwen 3.8") for n, v in best_sonic.items()],
        key=lambda r: r[1],
        reverse=True,
    )[:6]
    # Publish-grade RTX cells from the model record. Catalog can lag a
    # qwen-claude / later or-fable cell that models.json already carries.
    # Skip reward-hack / invalid rows even if the leaderboard still has them.
    models = json.loads((ROOT / "public/data/models.json").read_text())
    qwen_hard: list[tuple[str, float]] = []
    for rec in models if isinstance(models, list) else models.get("models") or []:
        if rec.get("slug") != "qwen3.8-max":
            continue
        cells = ((rec.get("benches") or {}).get("hard") or {}).get("cells") or {}
        for prob, lbl in hard_lbl.items():
            cell = cells.get(prob) or {}
            if cell.get("verdict") != "clean" or cell.get("valid") is False:
                continue
            if cell.get("score") is None:
                continue
            qwen_hard.append((lbl, float(cell["score"])))
        break
    save_rank(OUT_DIR / "01_sonic.png", sonic_rows)
    save_deck(OUT_DIR / "02_hard.png", qwen_hard)


if __name__ == "__main__":
    sys.exit(main() or 0)
