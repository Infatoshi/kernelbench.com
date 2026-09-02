"""Print figures for paper/main.tex. Light page, not the X-post dark theme."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)

ACCENT = "#3d6b00"
ACCENT_FILL = "#76b900"
BAD = "#b91c1c"
MUTED = "#6b6b6b"
INK = "#1a1a1a"
RULE = "#d4d4d4"
PAPER = "#ffffff"
CELL_NA = "#f3f3f3"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def hard_board() -> None:
    problems = ["FP8", "KDA", "Paged", "Top-k", "MoE", "W4A16"]
    models = [
        "GLM-5.2",
        "Opus 4.8",
        "Fable 5",
        "Grok 4.5",
        "Kimi K3 256k",
        "Kimi K3 1M",
        "V4 Flash",
        "GPT-5.6 Sol",
        "V4 Pro",
        "MiniMax-M3",
        "LongCat 2.0",
    ]
    # peak_fraction; nan = fail or excluded hack.
    # Source: benchmarks/hard/results/leaderboard.json (reward-hack cells dropped).
    data = np.array(
        [
            [0.406, 0.032, 0.677, 0.034, 0.098, 0.321],
            [0.386, 0.055, 0.671, 0.034, 0.086, 0.235],
            [0.391, 0.053, 0.430, 0.049, 0.105, 0.261],
            [0.337, 0.020, 0.654, 0.029, 0.102, 0.143],
            [0.320, 0.032, 0.485, 0.064, 0.088, 0.373],
            [0.353, 0.049, 0.581, 0.089, 0.033, 0.027],
            [0.409, 0.043, 0.486, 0.029, 0.097, 0.156],
            [0.387, 0.050, 0.566, 0.041, np.nan, 0.198],  # MoE = reward_hack
            [0.340, np.nan, 0.393, 0.014, 0.053, 0.154],
            [0.366, np.nan, 0.513, 0.006, 0.092, 0.145],
            [0.329, 0.001, 0.319, np.nan, 0.071, 0.101],
        ],
        dtype=float,
    )
    col_best = np.nanmax(data, axis=0)
    hacks = {(7, 4)}  # GPT-5.6 Sol × MoE

    cmap = LinearSegmentedColormap.from_list(
        "pf", ["#f4f4f4", "#c5e08a", "#76b900", "#3d6b00"], N=256
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.15))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=0.70, aspect="auto")
    ax.set_xticks(range(len(problems)))
    ax.set_xticklabels(problems)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.tick_params(length=0, colors=INK)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=CELL_NA,
                        edgecolor=RULE,
                        lw=0.4,
                    )
                )
                label = "hack" if (i, j) in hacks else "fail"
                ax.text(j, i, label, ha="center", va="center", color=BAD, fontsize=7.5)
            else:
                bold = abs(v - col_best[j]) < 1e-9
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="#102000" if v >= 0.28 else INK,
                    fontsize=7.5,
                    fontweight="bold" if bold else "normal",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("peak fraction", color=MUTED)
    cbar.ax.tick_params(labelsize=7, colors=MUTED, length=2)
    cbar.outline.set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "hard_board.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "hard_board.png", bbox_inches="tight", pad_inches=0.02, dpi=160)
    plt.close(fig)


def fp8_fix() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 2.55))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    cats = ["Before the spec fix", "After the spec fix"]
    genuine = np.array([0, 7])
    wrappers = np.array([5, 1])
    x = np.arange(len(cats))
    w = 0.36
    b1 = ax.bar(
        x - w / 2,
        genuine,
        w,
        label="Genuine FP8 MMA",
        color=ACCENT_FILL,
        edgecolor=ACCENT,
        lw=0.6,
    )
    b2 = ax.bar(
        x + w / 2,
        wrappers,
        w,
        label="Library wrapper / upcast",
        color="#f4c1c8",
        edgecolor=BAD,
        lw=0.6,
        hatch="///",
    )
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + 0.12,
                f"{int(h)}",
                ha="center",
                va="bottom",
                color=INK,
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Passing models")
    ax.set_ylim(0, 8.4)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(colors=INK, length=3)
    ax.yaxis.label.set_color(INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.yaxis.grid(True, color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "fp8_fix.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fp8_fix.png", bbox_inches="tight", pad_inches=0.02, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    hard_board()
    fp8_fix()
    print(f"wrote {OUT / 'hard_board.pdf'} and {OUT / 'fp8_fix.pdf'}")
