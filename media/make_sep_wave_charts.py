"""Rank charts for the 2026-09 wave short posts (Fable 5.1, Gemini 3.8 Flash, Muse Spark 1.3).

  uv run --no-project --with matplotlib,numpy python media/make_sep_wave_charts.py

Field = published RTX PRO 6000 board on the day of drafting (models.json / leaderboard_v2.json).
Subject green, field grey. Re-read the board before regenerating.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "posts" / "unaudited"))
from make_charts import save_rank  # noqa: E402  (applies kbh_theme on import)

A = HERE / "posts" / "audited"

MEGA = [  # Kimi-Linear Decode, RTX PRO 6000, valid cells, 2026-09-05
    ("Fable 5", 24.609), ("Fable 5.1", 22.946), ("GLM-5.3", 19.435), ("K3 (256k)", 18.088),
    ("Opus 4.8", 14.399), ("5.3 Flash", 13.642), ("GLM-5.2", 11.142), ("K3 (1M)", 9.785),
    ("DeepSeek", 4.645), ("GPT-5.5", 4.338), ("Sonnet 5", 4.030), ("Gemini 3.8", 2.741),
    ("GPT-5.6 Sol", 2.637), ("Muse 1.3", 2.180), ("Grok 4.5", 0.816),
]


def mega(subject: str, out: Path, keep: set[str]) -> None:
    rows = [(n, s, n == subject) for n, s in MEGA if n in keep or n == subject]
    save_rank(out, rows, "Kimi-Linear Decode · x over PyTorch baseline · RTX PRO 6000", "{:.1f}x")


def main() -> None:
    for d in ("fable51-mega", "fable51-cuda", "gemini38flash-mega", "gemini38flash-cuda",
              "musespark13-mega", "musespark13-cuda"):
        (A / d).mkdir(parents=True, exist_ok=True)

    top = {"Fable 5", "GLM-5.3", "K3 (256k)", "Opus 4.8", "5.3 Flash", "GLM-5.2", "K3 (1M)", "DeepSeek"}
    mega("Fable 5.1", A / "fable51-mega" / "01.png", top)
    low = {"Fable 5", "GLM-5.3", "Opus 4.8", "GLM-5.2", "DeepSeek", "GPT-5.5", "Sonnet 5",
           "Gemini 3.8", "GPT-5.6 Sol", "Muse 1.3", "Grok 4.5"}
    mega("Gemini 3.8", A / "gemini38flash-mega" / "01.png", low)
    mega("Muse 1.3", A / "musespark13-mega" / "01.png", low)

    save_rank(  # 02_deepseek_nsa, valid passes (bug-verdict V4 Flash omitted)
        A / "fable51-cuda" / "01.png",
        [("Fable 5.1", 1.0627, True), ("Opus 5", 1.0367, False), ("Fable 5", 0.7266, False),
         ("Opus 4.8", 0.1784, False), ("Gemini 3.8", 0.0958, False), ("DeepSeek", 0.0945, False),
         ("Grok 4.6", 0.0796, False), ("K3 (1M)", 0.0584, False), ("5.3 Flash", 0.0445, False),
         ("Muse 1.3", 0.0251, False), ("Grok 4.5", 0.0177, False)],
        "DeepSeek NSA · dense-equivalent peak fraction · RTX PRO 6000", "{:.3f}",
    )
    save_rank(  # 04_grid_mingru_sps, valid passes
        A / "gemini38flash-cuda" / "01.png",
        [("Opus 5", 1.961, False), ("Fable 5.1", 0.7093, False), ("Gemini 3.8", 0.3637, True),
         ("Opus 4.8", 0.3269, False), ("Qwen 3.8", 0.2848, False), ("K3 (1M)", 0.2238, False),
         ("Muse 1.3", 0.2056, False), ("Fable 5", 0.1909, False), ("K3 (256k)", 0.1738, False),
         ("V4 Flash", 0.1453, False), ("Grok 4.5", 0.002, False)],
        "Grid + MinGRU · peak fraction · RTX PRO 6000", "{:.3f}",
    )
    save_rank(  # 01_glm52_fused_moe, valid passes
        A / "musespark13-cuda" / "01.png",
        [("Opus 5", 0.1072, False), ("Fable 5.1", 0.1017, False), ("Qwen 3.8", 0.1006, False),
         ("GLM-5.3", 0.0997, False), ("DeepSeek", 0.0968, False), ("Grok 4.6", 0.0939, False),
         ("Muse 1.3", 0.0879, True), ("Grok 4.5", 0.0844, False), ("K3 (1M)", 0.0810, False),
         ("Fable 5", 0.0804, False), ("Opus 4.8", 0.0653, False), ("K3 (256k)", 0.0595, False),
         ("V4 Flash", 0.0411, False)],
        "GLM-5.2 Fused MoE · peak fraction · RTX PRO 6000", "{:.3f}",
    )


if __name__ == "__main__":
    main()
