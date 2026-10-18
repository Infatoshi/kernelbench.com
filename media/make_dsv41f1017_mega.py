"""DeepSeek V4.1 Flash (1017) mega post, image 1: the audit's MLA probe. The kernel matches the
reference on the graded random cache (where MLA attention is ~0.14% of the output) and falls apart on
caches where attention matters; baseline.py stays at 0.9999 on all of them. Numbers are the probe
quotes in the run's annotation YAML.

  uv run --no-project --with matplotlib,numpy,pyyaml python media/make_dsv41f1017_mega.py
"""
import math, re, sys, yaml
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kbh_theme import C, apply
apply()
ROOT = Path(__file__).resolve().parent.parent
RUN = "20261017_215137_or-fable_deepseek_deepseek-v4.1-flash-1017_02_kimi_linear_decode"
ann = yaml.safe_load(open(ROOT / f"benchmarks/mega/results/annotations/{RUN}.yaml"))
probe = next(q["text"] for q in ann["quotes"] if q["file"].startswith("audit (MLA probe"))
NAMES = {"graded random cache": "graded cache (check.py)", "identical rows x10": "repeated cache rows",
         "random cache x30": "cache scaled 30x"}
rows = []
for line in probe.strip().splitlines():
    m = re.match(r"\s*(.+?)\s+seed (\d+): \|MLA attn out\|/\|out\| ([\d.]+)\s+cos\(ref,sol\) (\S+)\s+cos\(ref,baseline\) (\S+)", line)
    rows.append((f"{NAMES[m.group(1)]}, seed {m.group(2)}", float(m.group(3)), float(m.group(4)), float(m.group(5))))

OUT = ROOT / "media/posts/scheduled/dsv41f1017-mega/01.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 6.4), dpi=180)
fig.patch.set_facecolor(C["bg"]); ax.set_facecolor(C["bg"])
lo = 0.70
for i, (name, share, sol, base) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(y + 0.17, base - lo, 0.3, left=lo, color="#4d5d66", zorder=2)
    ax.text(base + 0.004, y + 0.17, f"PyTorch baseline {base:.4f}", va="center", color=C["fg_muted"], fontsize=9)
    if math.isnan(sol):
        ax.text(lo + 0.004, y - 0.17, "DeepSeek kernel: NaN", va="center", color=C["bad"], fontsize=10, weight="bold")
    else:
        ok = sol >= 0.98
        ax.barh(y - 0.17, sol - lo, 0.3, left=lo, color=C["accent"] if ok else C["bad"], zorder=3)
        ax.text(sol + 0.004, y - 0.17, f"DeepSeek kernel {sol:.4f}", va="center", color=C["fg_bright"], fontsize=10, weight="bold")
    ax.text(lo - 0.006, y + 0.05, name, ha="right", va="center", color=C["fg_bright"], fontsize=10.5)
    ax.text(lo - 0.006, y - 0.25, f"attention = {100 * share:.1f}% of the output", ha="right", va="center", color=C["fg_muted"], fontsize=8.5)
ax.axvline(0.98, color=C["warn"], lw=1.2, ls="--", zorder=4)
ax.text(0.981, len(rows) - 0.45, "0.98 gate", color=C["warn"], fontsize=9)
ax.set_xlim(lo, 1.08); ax.set_ylim(-0.6, len(rows) - 0.3)
ax.set_yticks([]); ax.set_xticks([0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("cosine similarity to the reference output", color=C["fg_muted"], fontsize=11)
ax.grid(axis="x", color=C["grid"], lw=0.6); [sp.set_visible(False) for sp in ax.spines.values()]
ax.tick_params(colors=C["fg_muted"], labelsize=10)
fig.suptitle("Kimi-Linear Decode megakernel by DeepSeek V4.1 Flash (1017): MLA attention probe", color=C["fg_bright"], fontsize=15, x=0.03, ha="left", y=0.97)
fig.text(0.03, 0.905, "passes the graded check at 29.9x, but only 1/32 of the MLA attention is computed · RTX PRO 6000", color=C["fg_muted"], fontsize=10.5)
fig.subplots_adjust(left=0.29, right=0.97, top=0.86, bottom=0.1)
fig.savefig(OUT, dpi=180, facecolor=C["bg"]); print(OUT)
