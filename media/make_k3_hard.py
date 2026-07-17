"""K3 (kinetic-0715) article — Hard deck comparison, horizontal bars per problem.

Usage: python make_k3_hard.py [rtx|h100|b200] [out.png]
One panel per GPU; the RTX panel carries the full flagship roster, H100/B200
carry the models actually swept there (K3 variants + Fable). Values are peak
fraction; topk is additionally discussed in ms in the post copy (launch-bound).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from kbh_theme import C, SERIES, apply, tight_square

apply()

PROBLEMS = ["fp8_gemm", "kda", "paged_attn", "topk", "sonic_moe", "w4a16_gemm"]

# peak fraction; None = no cell yet / in flight, "FAIL" = correctness fail
PANELS = {
    "rtx": {
        "tag": "RTX PRO 6000 Blackwell",
        "models": {
            #               fp8     kda     paged   topk    sonic   w4a16
            "Kimi K3 (256k)":     [0.3196, 0.0315, 0.4855, 0.0640, 0.0885, 0.3733],
            "Kimi K3 (1M)": [0.3529, 0.0493, 0.5811, 0.0895, 0.0329, 0.027],
            "Fable 5":     [0.3480, 0.0358, 0.6299, 0.0494, 0.1075, 0.3477],
            "Opus 4.8":    [0.3855, 0.0552, 0.6706, 0.0335, 0.0864, 0.2355],
            "GLM 5.2":     [0.4059, 0.0323, 0.6771, 0.0341, 0.0980, 0.3207],
            "GPT-5.6 Sol": [0.3871, 0.0503, 0.5655, None,   None,   0.1980],
            "Grok 4.5":    [0.3367, 0.0198, 0.6538, 0.0293, 0.1019, 0.1435],
        },
        "xlim": 0.78,
    },
    "h100": {
        "tag": "H100 SXM5",
        "models": {
            "Kimi K3 (256k)":     [0.2819, 0.0255, 0.5143, 0.0494, 0.0793, 0.3063],
            "Kimi K3 (1M)": [0.2290, 0.0157, 0.4178, "FAIL", 0.0203, 0.2098],
            "Fable 5":     [0.3033, 0.0152, 0.4605, 0.0470, None,   0.3681],
        },
        "xlim": 0.58,
    },
    "b200": {
        "tag": "B200",
        "models": {
            "Kimi K3 (256k)":     [0.2222, 0.0057, 0.2117, 0.0101, 0.0760, 0.0425],
            "Fable 5":     [0.2535, "FAIL", 0.1696, None,   0.0762, 0.0434],
        },
        "xlim": 0.32,
    },
}

gpu = sys.argv[1] if len(sys.argv) > 1 else "rtx"
panel = PANELS[gpu]
MODELS = list(panel["models"])
DATA = panel["models"]
# Model->color mapping shared across all article charts (SERIES wraps at 6,
# and positional coloring would recolor models between per-GPU panels).
MODEL_COLORS = {"Kimi K3 (256k)": SERIES[0], "Kimi K3 (1M)": SERIES[1],
                "Fable 5": SERIES[2], "Opus 4.8": SERIES[3],
                "GLM 5.2": SERIES[4], "GPT-5.6 Sol": SERIES[5],
                "Grok 4.5": C["warn"]}

fig, ax = tight_square(size=10.0)
fig.subplots_adjust(left=0.145)
nm = len(MODELS)
bar_h = 0.8 / nm
ys = np.arange(len(PROBLEMS))[::-1]

for mi, model in enumerate(MODELS):
    color = MODEL_COLORS[model]
    for pi, val in enumerate(DATA[model]):
        y = ys[pi] + (nm / 2 - mi - 0.5) * bar_h
        if val is None:
            continue
        if val == "FAIL":
            ax.text(panel["xlim"] * 0.006, y, "fail", va="center", ha="left",
                    color=C["bad"], fontsize=8)
            continue
        ax.barh(y, val, height=bar_h * 0.9, color=color,
                zorder=3, label=model if pi == 0 else None)
        ax.text(val + panel["xlim"] * 0.008, y, f"{val:.3f}", va="center",
                ha="left", color=C["fg_muted"], fontsize=7)

ax.set_yticks(ys)
ax.set_yticklabels(PROBLEMS, fontsize=10)
ax.set_xlabel("fraction of roofline peak", fontsize=10)
ax.set_xlim(0, panel["xlim"])
ax.grid(axis="x", zorder=0)
ax.legend(loc="lower right", fontsize=9, framealpha=0.15,
          facecolor=C["surface"], edgecolor=C["border"])
ax.text(0.985, 0.985, panel["tag"], transform=ax.transAxes,
        ha="right", va="top", color=C["fg_dim"], fontsize=9)

out = sys.argv[2] if len(sys.argv) > 2 else f"k3_hard_{gpu}.png"
fig.savefig(out, dpi=160)
print(out)
