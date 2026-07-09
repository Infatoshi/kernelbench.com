"""Canonical KernelBench.com chart theme — MUST match the website (NVIDIA style).

Palette is copied verbatim from app/globals.css :root tokens. Every X-post /
article chart imports this so diagrams look native to kernelbench.com.

Usage:
    from kbh_theme import C, SERIES, apply, tight_square
    apply()
    fig, axes = tight_square(nrows=3)   # engagement multi-panel, 1:1, no header band
    ax.bar(..., color=C["accent"])

Engagement rules (also in AGENTS.md):
  - Bars + axes + compact legend only. No multi-line title / gray essay chrome.
  - Prefer square (1:1) multi-GPU panels. Context goes in the post copy.
"""
import matplotlib
import matplotlib.pyplot as plt

# website :root tokens (app/globals.css)
C = {
    "bg":            "#111111",
    "bg_depth":      "#000000",
    "surface":       "#1a1a1a",
    "surface_muted": "#222222",
    "fg":            "#eeeeee",
    "fg_bright":     "#ffffff",
    "fg_dim":        "#666666",
    "fg_muted":      "#999999",
    "accent":        "#76b900",  # NVIDIA green — THE accent
    "accent_dim":    "#004831",
    "warn":          "#fbbf24",
    "bad":           "#fb7185",
    "grid":          "#242424",
    "border":        "#333333",
    "border_strong": "#76b900",
}

# categorical bar colors for multi-model charts, in the NVIDIA dark aesthetic.
# accent green leads (use for the topper / subject); the rest are legible,
# desaturated companions that don't collide with warn(amber)/bad(rose).
SERIES = ["#76b900", "#4d9fff", "#b07cff", "#f0883e", "#cfcfcf", "#2dd4bf"]


def apply():
    matplotlib.rcParams.update({
        "figure.facecolor":  C["bg"],
        "axes.facecolor":    C["bg"],
        "savefig.facecolor": C["bg"],
        "axes.edgecolor":    C["border"],
        "axes.labelcolor":   C["fg_muted"],
        "text.color":        C["fg"],
        "xtick.color":       C["fg"],
        "ytick.color":       C["fg_muted"],
        "grid.color":        C["grid"],
        "font.family":       "monospace",
        "font.size":         11,
    })


def tight_square(nrows: int = 1, ncols: int = 1, size: float = 10.0, **subplot_kw):
    """1:1 figure with almost no outer padding — for feed / engagement charts.

    No reserved header band. Callers put GPU tags inside axes if needed.
    """
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(size, size),
        squeeze=False,
        **subplot_kw,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.985,
        bottom=0.07,
        hspace=0.18 if nrows > 1 else 0.0,
        wspace=0.12 if ncols > 1 else 0.0,
    )
    for ax in axes.flat:
        ax.set_facecolor(C["bg"])
        for spine in ax.spines.values():
            spine.set_color(C["border"])
    if nrows == 1 and ncols == 1:
        return fig, axes[0, 0]
    if ncols == 1:
        return fig, [axes[i, 0] for i in range(nrows)]
    if nrows == 1:
        return fig, [axes[0, j] for j in range(ncols)]
    return fig, axes
