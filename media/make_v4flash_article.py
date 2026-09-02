"""DeepSeek V4 Flash (0731) in-body charts — one GPU, green on black.

01: CUDA deck, subject only. 02: Hard FP8 ranking (subject green, field slate).
Reads the published RTX leaderboards. Minimal labels. No title essay.

  uv run --no-project --with matplotlib,numpy python make_v4flash_article.py
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
OUT_DIR = Path(__file__).resolve().parent / "X-article-v4flash"
MODEL = "deepseek/deepseek-v4-flash-0731"
HARNESS = "or-fable"
FIELD = "#4d5d66"
SHORT = {
    "deepseek/deepseek-v4-flash-0731": "V4 Flash",
    "anthropic/claude-fable-5": "Fable 5",
    "claude-fable-5": "Fable 5",
    "anthropic/claude-opus-5": "Opus 5",
    "claude-opus-5": "Opus 5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "glm-5.2": "GLM-5.2",
    "grok-4.5": "Grok 4.5",
    "qwen/qwen3.8-max": "Qwen 3.8",
}


def peak_cell(cell) -> float | None:
    if not isinstance(cell, dict) or cell.get("correct") is False:
        return None
    pf = cell.get("peak_fraction")
    return float(pf) if pf is not None else None


def load_row(path: Path, harness: str, model: str) -> dict:
    d = json.loads(path.read_text())
    for m in d.get("models", []):
        if m.get("harness") == harness and m.get("model") == model:
            return m.get("results") or {}
    return {}


def deck(path: Path, items: list[tuple[str, float]], xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    items = list(reversed(items))
    y = np.arange(len(items))
    vals = [v for _, v in items]
    hi = max(vals)
    colors = [C["accent"] if v == hi else FIELD for v in vals]
    ax.barh(y, vals, 0.68, color=colors, edgecolor=C["bg"], linewidth=0.4, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(
            v + hi * 0.03,
            yi,
            f"{v:.3f}",
            va="center",
            fontsize=12,
            color=C["fg_bright"] if v == hi else C["fg_muted"],
            fontweight="bold" if v == hi else "regular",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _ in items], fontsize=14, color=C["fg"])
    ax.set_xlabel(xlabel, fontsize=10, color=C["fg_muted"])
    ax.set_xlim(0, hi * 1.22)
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


def rank(path: Path, rows: list[tuple[str, float, bool]], xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=180)
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    rows = list(reversed(rows))
    y = np.arange(len(rows))
    vals = [v for _, v, _ in rows]
    colors = [C["accent"] if subj else FIELD for _, _, subj in rows]
    ax.barh(y, vals, 0.68, color=colors, edgecolor=C["bg"], linewidth=0.4, zorder=3)
    hi = max(vals)
    for yi, (_, v, subj) in zip(y, rows):
        ax.text(
            v + hi * 0.02,
            yi,
            f"{v:.3f}",
            va="center",
            fontsize=12,
            color=C["fg_bright"] if subj else C["fg_muted"],
            fontweight="bold" if subj else "regular",
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=14, color=C["fg"])
    ax.set_xlabel(xlabel, fontsize=10, color=C["fg_muted"])
    ax.set_xlim(0, hi * 1.18)
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


def main() -> None:
    cuda = load_row(ROOT / "benchmarks/cuda/results/leaderboard.json", HARNESS, MODEL)
    cuda_items = []
    for prob, lbl in [
        ("01_glm52_fused_moe", "MoE"),
        ("02_deepseek_nsa", "NSA"),
        ("03_megaqwen_decode", "Decode"),
        ("04_grid_mingru_sps", "Craftax"),
    ]:
        v = peak_cell(cuda.get(prob))
        if v is not None:
            cuda_items.append((lbl, v))
    hard = json.loads((ROOT / "benchmarks/hard/results/leaderboard.json").read_text())
    best: dict[str, float] = {}
    for m in hard.get("models", []):
        mid = m.get("model")
        if mid not in SHORT:
            continue
        v = peak_cell((m.get("results") or {}).get("01_fp8_gemm"))
        if v is None:
            continue
        name = SHORT[mid]
        if v > best.get(name, -1):
            best[name] = v
    rows = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:6]
    rank_rows = [(n, v, n == "V4 Flash") for n, v in rows]
    deck(OUT_DIR / "01_cuda.png", cuda_items, "CUDA · peak fraction · RTX PRO 6000")
    rank(OUT_DIR / "02_hard.png", rank_rows, "FP8 · peak fraction · RTX PRO 6000")


if __name__ == "__main__":
    sys.exit(main() or 0)
