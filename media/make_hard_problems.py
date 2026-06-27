"""KernelBench-Hard: one clean grouped bar per problem (6 charts).

Matches the mega style: title = problem name, grouped bars by model with
RTX PRO 6000 / H100 / B200 as the colored series, value labels, GPU legend,
no prose. Metric = speedup of the agent kernel over the torch.compile baseline
(peak_fraction / compiled-baseline peak_fraction, same roofline). DNF cells
marked. Writes hard_<problem>.png for each problem.
"""
import sys; sys.path.insert(0, "..")
import json
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from kbh_theme import C, apply
apply()

HARD = "../benchmarks/hard"
sys.path.insert(0, HARD)
from src.hardware import get as get_hw  # noqa: E402

GPUS = [
    ("RTX PRO 6000", "results/leaderboard.json", "results/problem_baselines.json", "RTX_PRO_6000"),
    ("H100", "results/leaderboard.h100.json", "results/problem_baselines.h100.json", "H100"),
    ("B200", "results/leaderboard.b200.json", "results/problem_baselines.b200.json", "B200"),
]
GPU_COL = {"RTX PRO 6000": "#4d9fff", "H100": "#b07cff", "B200": C["accent"]}
PROBS = {
    "01_fp8_gemm": "FP8 GEMM",
    "02_kda_cutlass": "KimiDeltaAttention (CUTLASS)",
    "03_paged_attention": "Paged Attention",
    "05_topk_bitonic": "Top-K (Bitonic)",
    "06_sonic_moe_swiglu": "Sonic MoE SwiGLU",
    "07_w4a16_gemm": "W4A16 GEMM",
}
DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8", "glm-5.2": "GLM-5.2", "gpt-5.5": "GPT-5.5",
    "gemini-3.5-flash": "Gemini 3.5 Flash", "composer-2.5-fast": "Composer 2.5 Fast",
    "kimi-k2.7-code": "Kimi K2.7-Code", "deepseek-v4-pro": "DeepSeek V4 Pro",
    "MiniMax-M3": "MiniMax-M3",
}

def short(label):
    return label.split(" [")[0].split("/")[-1]

def compiled_pf(prob, base, hwname):
    meta = yaml.safe_load(open(f"{HARD}/problems/{prob}/problem.yaml"))
    hw = get_hw(hwname); b = base["problems"].get(prob, {}).get("compiled", {})
    if meta.get("regime") == "memory":
        peak, m = hw.peak_bandwidth_gb_s, b.get("gbps")
    else:
        peak, m = hw.peak_tflops_dense.get(meta["peak_tflops_key"], 0), b.get("tflops")
    return (m / peak) if (m and peak) else None

# data[prob][model][gpu] = speedup or None
data = {p: {} for p in PROBS}
for gname, lbf, basef, hwname in GPUS:
    lb = json.load(open(f"{HARD}/{lbf}")); base = json.load(open(f"{HARD}/{basef}"))
    for prob in PROBS:
        cpf = compiled_pf(prob, base, hwname)
        for m in lb["models"]:
            nm = short(m["label"])
            if nm not in DISPLAY:
                continue
            c = m["results"].get(prob, {})
            v = (c["peak_fraction"] / cpf) if (c.get("correct") and c.get("peak_fraction") and cpf) else None
            data[prob].setdefault(nm, {})[gname] = v

gpu_names = [g for g, *_ in GPUS]
for prob, title in PROBS.items():
    cell = data[prob]
    models = [m for m in DISPLAY if m in cell]
    models.sort(key=lambda m: -max([cell[m].get(g) or 0 for g in gpu_names]))
    ymax = max([cell[m].get(g) or 0 for m in models for g in gpu_names]) * 1.14 or 1

    x = np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(top=0.86, left=0.05, right=0.985, bottom=0.14)
    ax.set_facecolor(C["bg"])
    for sp in ax.spines.values(): sp.set_color(C["border"])

    for gi, g in enumerate(gpu_names):
        off = (gi - 1) * w
        for mi, m in enumerate(models):
            v = cell[m].get(g)
            if v is None:
                ax.text(x[mi] + off, ymax * 0.02, "x", ha="center", va="bottom",
                        color=C["fg_dim"], fontsize=8)
                continue
            ax.bar(x[mi] + off, v, w, color=GPU_COL[g], edgecolor=C["bg"], zorder=3)
            ax.text(x[mi] + off, v + ymax * 0.012, f"{v:.1f}", ha="center", va="bottom",
                    color=C["fg"], fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[m] for m in models], rotation=20, ha="right",
                       fontsize=11, color=C["fg"])
    ax.set_ylim(0, ymax); ax.set_yticks([]); ax.tick_params(length=0)
    fig.text(0.05, 0.93, title, color=C["accent"], fontsize=20, fontweight="bold", ha="left")
    ax.legend(handles=[Patch(facecolor=GPU_COL[g], label=g) for g in gpu_names],
              loc="upper right", frameon=False, fontsize=12, labelcolor=C["fg"])

    out = f"hard_{prob}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)
