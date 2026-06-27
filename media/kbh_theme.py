"""Canonical KernelBench.com chart theme — MUST match the website (NVIDIA style).

Palette is copied verbatim from app/globals.css :root tokens. Every X-post /
article chart imports this so diagrams look native to kernelbench.com.

Usage:
    import sys; sys.path.insert(0, "..")   # if in a subfolder
    from kbh_theme import C, apply
    apply()                                # sets matplotlib rcParams
    ax.bar(..., color=C["accent"])
    # model bars: use SERIES in order, or pick by role (ceiling/subject/other).
"""
import matplotlib

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
