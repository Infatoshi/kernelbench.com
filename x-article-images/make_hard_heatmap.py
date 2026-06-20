"""KernelBench-Hard heatmap: per-problem speedup over torch.compile, 3 GPUs.

One panel per GPU (RTX PRO 6000 / H100 / B200), rows = models, columns = the 6
problems. Each cell is the agent kernel's speedup over the torch.compile
baseline (peak_fraction / compiled-baseline peak_fraction, same roofline).
Failed / no-solution / timeout cells are grey. No aggregation, so a single
weak or failed cell can't distort a model's whole row.
"""
import sys; sys.path.insert(0, "..")
import json
import yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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
PCOL = ["fp8\ngemm", "kda\ncutlass", "paged\nattn", "topk\nbitonic", "sonic\nmoe", "w4a16\ngemm"]
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

# speed[gpu][model][prob] = float speedup or None (failed/dnf)
speed = {}
for gname, lbf, basef, hwname in GPUS:
    lb = json.load(open(f"{HARD}/{lbf}")); base = json.load(open(f"{HARD}/{basef}"))
    cpf = {p: compiled_pf(p, base, hwname) for p in PROBS}
    for m in lb["models"]:
        nm = short(m["label"])
        if nm not in DISPLAY:
            continue
        for p in PROBS:
            c = m["results"].get(p, {})
            v = (c["peak_fraction"] / cpf[p]) if (c.get("correct") and c.get("peak_fraction") and cpf.get(p)) else None
            speed.setdefault(gname, {}).setdefault(nm, {})[p] = v

# model order: by mean speedup on RTX (strongest on top)
def rowkey(nm):
    vals = [speed["RTX PRO 6000"].get(nm, {}).get(p) for p in PROBS]
    vals = [v for v in vals if v]
    return -(sum(vals) / len(vals)) if vals else 0
models = sorted(DISPLAY, key=rowkey)

CMAP = LinearSegmentedColormap.from_list("kbh", ["#15240a", "#3f6a12", C["accent"]])
VMAX = 8.0  # color saturates at 8x; true value still annotated

fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.2), sharey=True)
fig.subplots_adjust(top=0.78, left=0.105, right=0.99, bottom=0.13, wspace=0.08)

for ax, (gname, *_ ) in zip(axes, GPUS):
    grid = np.full((len(models), len(PROBS)), np.nan)
    for i, nm in enumerate(models):
        for j, p in enumerate(PROBS):
            v = speed.get(gname, {}).get(nm, {}).get(p)
            if v is not None:
                grid[i, j] = min(v, VMAX)
    ax.set_facecolor("#262626")
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=VMAX, aspect="auto")
    ax.set_title(gname, color=C["fg_bright"], fontsize=13, fontweight="bold", pad=8)
    ax.set_xticks(range(len(PROBS)))
    ax.set_xticklabels(PCOL, fontsize=8.5, color=C["fg_muted"])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([DISPLAY[m] for m in models], fontsize=9.5, color=C["fg"])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    # annotate
    for i, nm in enumerate(models):
        for j, p in enumerate(PROBS):
            v = speed.get(gname, {}).get(nm, {}).get(p)
            if v is None:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#2b2b2b", zorder=2))
                ax.text(j, i, "x", ha="center", va="center", color=C["fg_dim"], fontsize=9, zorder=3)
            else:
                tc = C["bg_depth"] if min(v, VMAX) > VMAX * 0.55 else C["fg"]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", color=tc,
                        fontsize=8.5, fontweight="bold" if v >= VMAX else "normal", zorder=3)

fig.text(0.105, 0.945, "KernelBench-Hard: per-problem speedup over torch.compile",
         color=C["accent"], fontsize=18, fontweight="bold", ha="left")
fig.text(0.105, 0.90, "Each cell = the agent kernel's speedup over the torch.compile baseline on that problem. Greener is faster; color saturates at 8x (true value shown). x = no correct kernel.",
         color=C["fg_muted"], fontsize=10.5, ha="left")
fig.text(0.105, 0.865, "RTX PRO 6000 runs are unlimited-time; H100/B200 use a 45-min budget, so some H100/B200 cells are early/unfinished kernels.",
         color=C["fg_muted"], fontsize=10, ha="left")

fig.savefig("hard_heatmap.png", dpi=150)
print("wrote hard_heatmap.png")
