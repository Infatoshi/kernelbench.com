"""Grok 4.5 vs Fable 5 on KernelBench-Mega kimi-linear decode (RTX PRO 6000).

Two engagement charts (no essay chrome on-image):

  1) grok_vs_fable_mega_bars.png  — final geomean speedup bars
  2) grok_vs_fable_mega_traj.png  — both optimization trajectories on one axes

Clean independent Grok cell only (0.82x). Contaminated 18.9x Grok run is not
plotted — that was a byte-copy of Fable's solution via public/.

  uv run python media/make_grok_vs_fable_mega.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, SERIES, apply

apply()

OUT_DIR = Path(__file__).resolve().parent

# Final audited scores (geomean over 2k/8k/16k vs optimized baseline).
FABLE_FINAL = 18.7149  # 20260701_172615 clean
GROK_FINAL = 0.8157  # 20260709_221419 clean boxed rerun

# Fable trajectory: mined from transcript (same checkpoints as make_fable5_trajectory.py).
# (elapsed_min, speedup)
FABLE_TRAJ = [
    (2615 / 60, 1.00),  # baseline timed
    (5926 / 60, 14.38),  # first cooperative megakernel pass
    (6344 / 60, 13.00),
    (6784 / 60, 13.65),
    (7267 / 60, 16.08),
    (7572 / 60, 16.92),
    (7772 / 60, 17.63),
    (7826 / 60, 17.13),
    (8653 / 60, 15.94),  # split-K regression
    (8821 / 60, 17.97),
    (9081 / 60, 18.03),
    (9159 / 60, 18.70),
]

# Grok clean trajectory: wall minutes approximated from transcript position
# × 48.7m agent session; speedups from agent-reported ms/tok or geomean.
# Single-block experiment (~165 ms/tok) kept as a rose dip.
GROK_TRAJ = [
    (19.0, 0.25),  # first correct megakernel, ~4× slower than baseline
    (25.3, 0.034),  # single-block rewrite (regression)
    (28.2, 0.28),  # multi-block restored ~20 ms
    (30.7, 0.32),  # ~17 ms
    (33.1, 0.41),  # 13.45 ms
    (34.0, 0.44),  # 12.48 ms
    (36.9, 0.32),  # official sample still LOW
    (38.5, 0.54),  # 10.21 ms / 0.54x
    (39.4, 0.76),  # 7.24 ms short-ctx → climbing
    (41.2, 0.76),  # geomean ~0.76x
    (43.9, 0.82),  # best geomean
    (48.7, 0.82),  # final official 0.8157
]

# Sparse method tags only — (elapsed_min, speedup, short label, xytext offset in data)
# Labels are the *technique* at that checkpoint, not every numeric speedup.
FABLE_TAGS = [
    (2615 / 60, 1.00, "baseline timed", (58, 1.7)),
    (5926 / 60, 14.38, "coop megakernel v1\n14 barriers/step", (72, 7.5)),
    (7267 / 60, 16.08, "bf16x2 int4 dequant\n(LOP3/HMUL2)", (100, 24)),
    (8821 / 60, 17.97, "MoE gate+up fused\nspin counters", (128, 24)),
    (9159 / 60, 18.70, "MLA barrier fold\n→ 18.7×", (145, 10)),
]

GROK_TAGS = [
    (19.0, 0.25, "coop megakernel\nnaive GEMV + many grid.sync", (5, 0.65)),
    (25.3, 0.034, "single-block rewrite\n(bandwidth collapse)", (40, 0.055)),
    (33.1, 0.41, "batch MoE GEMVs\ncut barriers", (12, 1.1)),
    (39.4, 0.76, "still materialize\nMLA scores [H×L]", (58, 0.28)),
    (48.7, 0.82, "L2 GEMV, ~30 syncs\n→ 0.82×", (70, 0.11)),
]


def bars() -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.14)

    labels = ["Fable 5", "Grok 4.5"]
    vals = [FABLE_FINAL, GROK_FINAL]
    colors = [C["accent"], SERIES[1]]
    x = np.arange(len(labels))

    bars_ = ax.bar(x, vals, color=colors, width=0.62, edgecolor=C["bg"], zorder=3)
    ax.axhline(1.0, color=C["fg_dim"], linewidth=1.0, linestyle="--", zorder=2)
    ax.text(
        1.35,
        1.0,
        "baseline = 1×",
        color=C["fg_dim"],
        fontsize=9,
        va="bottom",
        ha="right",
    )

    for rect, v, col in zip(bars_, vals, colors):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            v + max(vals) * 0.02,
            f"{v:.2f}×",
            ha="center",
            va="bottom",
            color=C["fg"],
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("decode speedup (geomean 2k/8k/16k)", fontsize=11)
    ax.set_ylim(0, FABLE_FINAL * 1.18)
    ax.grid(axis="y", color=C["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # tiny in-axes GPU tag only
    ax.text(
        0.02,
        0.97,
        "RTX PRO 6000 · mega/kimi-linear",
        transform=ax.transAxes,
        color=C["fg_dim"],
        fontsize=9,
        va="top",
        ha="left",
    )

    out = OUT_DIR / "grok_vs_fable_mega_bars.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def _method_tags(ax, tags, color: str) -> None:
    """Sparse few-word method annotations at the important checkpoints only."""
    for x, y, label, (tx, ty) in tags:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(tx, ty),
            fontsize=8,
            color=color,
            ha="center",
            va="center",
            linespacing=1.25,
            zorder=6,
            arrowprops=dict(
                arrowstyle="-",
                color=C["fg_dim"],
                linewidth=0.7,
                shrinkA=2,
                shrinkB=3,
            ),
        )


def trajectories() -> Path:
    """Both sessions on one chart. Log y so Grok's sub-1× climb is visible;
    Fable's 18.7× still dominates the top of the frame.
    """
    fig, ax = plt.subplots(figsize=(10.0, 10.0))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.96, bottom=0.10)

    fx, fy = zip(*FABLE_TRAJ)
    gx, gy = zip(*GROK_TRAJ)

    # Fable: dashed until first real kernel, then solid (design phase)
    ax.plot(
        fx[:2],
        fy[:2],
        color=C["accent"],
        linewidth=1.6,
        linestyle=(0, (4, 3)),
        alpha=0.55,
        zorder=3,
    )
    ax.plot(fx[1:], fy[1:], color=C["accent"], linewidth=2.0, zorder=3, label="Fable 5")
    ax.scatter(fx, fy, s=28, color=C["accent"], zorder=4)
    ax.scatter(
        [fx[-1]],
        [fy[-1]],
        s=90,
        color=C["accent"],
        zorder=5,
        edgecolor=C["fg_bright"],
        linewidth=1.2,
    )
    # Fable regression
    ax.scatter([8653 / 60], [15.94], s=32, color=C["bad"], zorder=5)
    _method_tags(ax, FABLE_TAGS, C["accent"])

    # Grok
    ax.plot(gx, gy, color=SERIES[1], linewidth=2.0, zorder=3, label="Grok 4.5")
    ax.scatter(gx, gy, s=28, color=SERIES[1], zorder=4)
    # single-block regression dip
    ax.scatter([25.3], [0.034], s=32, color=C["bad"], zorder=5)
    ax.scatter(
        [gx[-1]],
        [gy[-1]],
        s=90,
        color=SERIES[1],
        zorder=5,
        edgecolor=C["fg_bright"],
        linewidth=1.2,
    )
    _method_tags(ax, GROK_TAGS, SERIES[1])

    ax.axhline(1.0, color=C["fg_dim"], linewidth=1.0, linestyle="--", zorder=2)
    ax.text(2.0, 1.15, "baseline = 1×", color=C["fg_dim"], fontsize=9, va="bottom")

    ax.set_yscale("log")
    ax.set_xlim(0, 160)
    ax.set_ylim(0.02, 30)
    ax.set_xlabel("wall clock (minutes)", fontsize=11)
    ax.set_ylabel("decode speedup (log)", fontsize=11)
    ax.grid(True, which="both", color=C["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    leg = ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=11,
        labelcolor=C["fg"],
    )

    ax.text(
        0.98,
        0.03,
        "RTX PRO 6000 · mega/kimi-linear · clean cells only",
        transform=ax.transAxes,
        color=C["fg_dim"],
        fontsize=8,
        ha="right",
        va="bottom",
    )

    out = OUT_DIR / "grok_vs_fable_mega_traj.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main() -> None:
    b = bars()
    t = trajectories()
    print(f"wrote {b}")
    print(f"wrote {t}")


if __name__ == "__main__":
    main()
