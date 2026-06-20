"""KernelBench-Hard: speedup over torch.compile per model across RTX PRO 6000 / H100 / B200.

Metric: geometric-mean speedup of the agent's kernel over the torch.compile
baseline, across the problems the model solved. Derived from the published
per-cell peak_fraction and the per-GPU compiled baseline (both fractions of the
same roofline, so their ratio is the achieved speedup). torch.compile is a
strong baseline, so this avoids the inflated numbers a naive-eager reference
would give. Reads the live leaderboards + baselines so it stays in sync.
"""
import sys; sys.path.insert(0, "..")
import json
import math
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
PROBS = ["01_fp8_gemm", "02_kda_cutlass", "03_paged_attention",
         "05_topk_bitonic", "06_sonic_moe_swiglu", "07_w4a16_gemm"]
GPU_COL = {"RTX PRO 6000": "#4d9fff", "H100": "#b07cff", "B200": C["accent"]}

def compiled_pf(prob, base, hwname):
    meta = yaml.safe_load(open(f"{HARD}/problems/{prob}/problem.yaml"))
    hw = get_hw(hwname); b = base["problems"].get(prob, {}).get("compiled", {})
    if meta.get("regime") == "memory":
        peak, m = hw.peak_bandwidth_gb_s, b.get("gbps")
    else:
        peak, m = hw.peak_tflops_dense.get(meta["peak_tflops_key"], 0), b.get("tflops")
    return (m / peak) if (m and peak) else None

def short(label):
    return label.split(" [")[0].split("/")[-1]

# speedup[model][gpu] = geomean over solved problems
speed = {}
for gname, lbf, basef, hwname in GPUS:
    lb = json.load(open(f"{HARD}/{lbf}")); base = json.load(open(f"{HARD}/{basef}"))
    cpf = {p: compiled_pf(p, base, hwname) for p in PROBS}
    for m in lb["models"]:
        ratios = []
        for p in PROBS:
            c = m["results"].get(p, {})
            if c.get("correct") and c.get("peak_fraction") and cpf.get(p):
                ratios.append(c["peak_fraction"] / cpf[p])
        if ratios:
            g = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
            speed.setdefault(short(m["label"]), {})[gname] = g

gpu_names = [g for g, *_ in GPUS]
models = [m for m in speed if all(g in speed[m] for g in gpu_names)]
models.sort(key=lambda m: -max(speed[m][g] for g in gpu_names))

DISPLAY = {
    "gpt-5.5": "GPT-5.5", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "claude-opus-4-8": "Claude Opus 4.8", "composer-2.5-fast": "Composer 2.5 Fast",
    "deepseek-v4-pro": "DeepSeek V4 Pro", "kimi-k2.7-code": "Kimi K2.7-Code",
    "glm-5.2": "GLM-5.2", "MiniMax-M3": "MiniMax-M3",
}

x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(14.5, 7.4))
fig.subplots_adjust(top=0.80, left=0.055, right=0.985, bottom=0.16)
ax.set_facecolor(C["bg"])
for sp in ax.spines.values(): sp.set_color(C["border"])

ymax = max(speed[m][g] for m in models for g in gpu_names) * 1.15
for gi, g in enumerate(gpu_names):
    off = (gi - 1) * w
    for mi, m in enumerate(models):
        v = speed[m][g]
        ax.bar(x[mi] + off, v, w, color=GPU_COL[g], edgecolor=C["bg"], zorder=3)
        ax.text(x[mi] + off, v + ymax * 0.012, f"{v:.1f}", ha="center", va="bottom",
                color=C["fg"], fontsize=8,
                fontweight="bold" if m == "claude-opus-4-8" else "normal")

ax.set_xticks(x)
ax.set_xticklabels([DISPLAY.get(m, m) for m in models], rotation=22, ha="right",
                   fontsize=10, color=C["fg"])
ax.set_ylabel("speedup over torch.compile  (geomean, higher is better)", fontsize=10.5)
ax.set_ylim(0, ymax)
ax.grid(axis="y", lw=0.6, alpha=0.5); ax.set_axisbelow(True)

fig.text(0.055, 0.945, "KernelBench-Hard: agent kernels vs torch.compile across three NVIDIA GPUs",
         color=C["accent"], fontsize=17, fontweight="bold", ha="left")
fig.text(0.055, 0.895, "Geometric-mean speedup of the agent's kernel over the torch.compile baseline, across the problems each model solved (of 6).",
         color=C["fg_muted"], fontsize=10.5, ha="left")
fig.text(0.055, 0.862, "Claude Opus 4.8 tops the deck on every GPU. RTX PRO 6000 runs are unlimited-time; H100/B200 use a 45-min budget.",
         color=C["fg_muted"], fontsize=10, ha="left")

ax.legend(handles=[Patch(facecolor=GPU_COL[g], label=g) for g in gpu_names],
          loc="upper right", frameon=False, fontsize=10.5, labelcolor=C["fg"])

fig.savefig("hard_3gpu.png", dpi=150)
print("wrote hard_3gpu.png")
