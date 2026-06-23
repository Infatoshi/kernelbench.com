"""fugu-ultra FP8 GEMM optimization trajectory: peak_fraction (out of 1.0) at
each benchmark the agent ran during the session. Website NVIDIA theme.

Reads the run transcript, extracts the measured peak_fraction readings in order,
dedups consecutive repeats (each reading is echoed twice), and plots the climb
toward the hardware roofline ceiling (1.0). RTX PRO 6000, anvil.
"""
import sys; sys.path.insert(0, "..")
import json
import re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kbh_theme import C, apply
apply()

RUN = "../benchmarks/hard/outputs/runs/20260622_205948_opencode_sakana_fugu-ultra_01_fp8_gemm"

raw = []
for line in open(f"{RUN}/transcript.jsonl"):
    try:
        o = json.loads(line)
    except Exception:
        continue
    for m in re.finditer(r"peak_fraction:\s*([01]\.\d{2,})", json.dumps(o)):
        raw.append(float(m.group(1)))

# dedup consecutive repeats -> distinct benchmark readings in order
traj = []
for v in raw:
    if not traj or traj[-1] != v:
        traj.append(v)

x = list(range(1, len(traj) + 1))
final = traj[-1]
best = max(traj)

fig, ax = plt.subplots(figsize=(12, 7))
fig.subplots_adjust(top=0.82, left=0.085, right=0.97, bottom=0.12)
ax.set_facecolor(C["bg"])
for sp in ax.spines.values():
    sp.set_color(C["border"])

# roofline ceiling
ax.axhline(1.0, color=C["fg_dim"], lw=1, ls="--")
ax.text(len(traj), 1.005, "hardware roofline (1.0)", color=C["fg_muted"],
        fontsize=10, ha="right", va="bottom")

ax.plot(x, traj, color=C["accent"], lw=2.4, marker="o", markersize=7,
        markerfacecolor=C["accent"], markeredgecolor=C["bg"], zorder=3)

# annotate first + final
ax.annotate(f"{traj[0]:.2f}", (x[0], traj[0]), textcoords="offset points",
            xytext=(8, -14), color=C["fg_muted"], fontsize=10)
ax.annotate(f"{final:.3f}", (x[-1], final), textcoords="offset points",
            xytext=(8, 6), color=C["accent"], fontsize=13, fontweight="bold")

ax.set_xlim(0.5, len(traj) + 1.5)
ax.set_ylim(0, 1.06)
ax.set_ylabel("fraction of hardware roofline  (out of 1.0)", fontsize=11)
ax.set_xlabel("benchmark step during the agent session", fontsize=11)
ax.grid(axis="y", lw=0.6, alpha=0.4)
ax.set_axisbelow(True)
ax.tick_params(colors=C["fg_muted"])

fig.text(0.085, 0.94, "fugu-ultra optimizing an FP8 GEMM kernel",
         color=C["accent"], fontsize=19, fontweight="bold", ha="left")
fig.text(0.085, 0.895,
         f"Measured speed at each benchmark the agent ran, climbing from {traj[0]:.2f} to {final:.3f} of the RTX PRO 6000 roofline.",
         color=C["fg_muted"], fontsize=11, ha="left")
fig.text(0.085, 0.862,
         "FP8 tensor-core GEMM: fugu lands in the frontier pack here (contrast its 0.01 on the chunk-parallel KimiDeltaAttention).",
         color=C["fg_muted"], fontsize=10, ha="left")

fig.savefig("fugu_fp8_progress.png", dpi=150)
print(f"wrote fugu_fp8_progress.png ({len(traj)} steps, {traj[0]} -> {final})")
