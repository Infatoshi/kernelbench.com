"""Hard-deck status matrix — what actually came back per (problem x GPU).

Honest outcome categories so a reader can tell a real correctness/portability
miss apart from an infra gap (missing ninja on the H100 box) or a provider
rate-limit wall. Capability lives on RTX PRO 6000 (full) + B200 (partial);
H100 is infra-invalidated pending a re-sweep (ninja now pinned in pyproject).

Regenerate: uv run --with matplotlib python media/make_fable5_hard_status.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from kbh_theme import C, apply

apply()

PROBS = ["01_fp8_gemm", "02_kda_cutlass", "03_paged_attention",
         "05_topk_bitonic", "06_sonic_moe_swiglu", "07_w4a16_gemm"]
GPUS = ["RTX PRO 6000", "H100", "B200"]

# cell -> (category, text). four categories:
#   pass  = green, labelled with fraction of roofline
#   rate  = red, provider rate-limit / quota wall (no real attempt)
#   infra = amber (hatched), missing-ninja box gap (rerun pending)
#   fail  = grey, genuine correctness / compilation / runtime failure
# per-cell reasons verified against each run's result.json + check.log.
CELLS = {
    ("RTX PRO 6000", "01_fp8_gemm"):        ("rate", "rate\nlimit"),
    ("RTX PRO 6000", "02_kda_cutlass"):     ("pass", "0.036"),
    ("RTX PRO 6000", "03_paged_attention"): ("pass", "0.630"),
    ("RTX PRO 6000", "05_topk_bitonic"):    ("pass", "0.049"),
    ("RTX PRO 6000", "06_sonic_moe_swiglu"):("pass", "0.108"),
    ("RTX PRO 6000", "07_w4a16_gemm"):      ("pass", "0.348"),

    ("H100", "01_fp8_gemm"):        ("pass",  "0.303"),
    ("H100", "02_kda_cutlass"):     ("fail",  "smem\noverflow"),
    ("H100", "03_paged_attention"): ("infra", "ninja"),
    ("H100", "05_topk_bitonic"):    ("infra", "ninja"),
    ("H100", "06_sonic_moe_swiglu"):("fail",  "check\ntimeout"),
    ("H100", "07_w4a16_gemm"):      ("infra", "ninja"),

    ("B200", "01_fp8_gemm"):        ("pass", "0.254"),
    ("B200", "02_kda_cutlass"):     ("fail", "correct-\nness"),
    ("B200", "03_paged_attention"): ("pass", "0.170"),
    ("B200", "05_topk_bitonic"):    ("rate", "rate\nlimit"),
    ("B200", "06_sonic_moe_swiglu"):("pass", "0.076"),
    ("B200", "07_w4a16_gemm"):      ("pass", "0.043"),
}

FILL = {
    "pass":  C["accent"],
    "fail":  C["fg_dim"],
    "infra": C["warn"],
    "rate":  C["bad"],
}
TXTCOL = {"pass": "#0a0a0a", "fail": C["fg"], "infra": "#0a0a0a",
          "rate": "#0a0a0a"}

fig, ax = plt.subplots(figsize=(9.2, 7.3))
fig.subplots_adjust(left=0.19, right=0.98, top=0.82, bottom=0.22)

nrows, ncols = len(PROBS), len(GPUS)
for j, gpu in enumerate(GPUS):
    for i, prob in enumerate(PROBS):
        y = nrows - 1 - i
        cat, txt = CELLS[(gpu, prob)]
        hatch = "///" if cat == "infra" else None
        ax.add_patch(plt.Rectangle((j, y), 0.94, 0.94, facecolor=FILL[cat],
                                   edgecolor=C["bg"], linewidth=2, hatch=hatch))
        ax.text(j + 0.47, y + 0.47, txt, ha="center", va="center",
                color=TXTCOL[cat], fontsize=9.5, linespacing=1.1,
                fontweight="bold" if cat in ("pass", "infra") else "normal")

ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
ax.set_xticks([j + 0.47 for j in range(ncols)])
ax.set_xticklabels(GPUS, fontsize=11)
ax.xaxis.tick_top()
ax.set_yticks([nrows - 1 - i + 0.47 for i in range(nrows)])
ax.set_yticklabels(PROBS, fontsize=10)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)

# column tallies
tally = {"RTX PRO 6000": "5 pass · 1 rate-limited",
         "H100": "1 pass · 3 infra · 2 fail",
         "B200": "4 pass · 1 fail · 1 rate"}
for j, gpu in enumerate(GPUS):
    ax.text(j + 0.47, -0.22, tally[gpu], ha="center", va="top",
            color=C["fg_muted"], fontsize=8.5)

legend = [
    Patch(facecolor=C["accent"], label="pass — fraction of roofline"),
    Patch(facecolor=C["bad"], label="provider rate limit / quota (no real attempt)"),
    Patch(facecolor=C["warn"], hatch="///", label="infra gap — ninja missing on box (rerun pending)"),
    Patch(facecolor=C["fg_dim"], label="correctness / compile / runtime failure"),
]
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.0, -0.14),
          frameon=False, fontsize=8.5, ncol=1, handlelength=1.4,
          labelcolor=C["fg"])

fig.suptitle("KernelBench-Hard — Claude Fable 5, what actually came back",
             color=C["fg_bright"], fontsize=13, x=0.19, ha="left", y=0.965)
fig.text(0.19, 0.895,
         "RTX PRO 6000 + B200 are the real numbers. H100 is held out (resweep pending): its box\n"
         "shipped without ninja, so hand-written CUDA cells couldn't compile at grading — infra, not capability.",
         color=C["fg_muted"], fontsize=8.5, linespacing=1.4)

out = pathlib.Path(__file__).parent / "fable5_hard_status.png"
fig.savefig(out, dpi=170)
print(f"wrote {out}")
