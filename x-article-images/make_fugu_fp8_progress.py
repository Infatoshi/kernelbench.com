"""fugu-ultra FP8 GEMM optimization trajectory: peak_fraction (out of 1.0) at
each benchmark the agent ran during the session. Website NVIDIA theme.

Reads the run transcript, extracts the measured peak_fraction readings in order,
dedups consecutive repeats (each reading is echoed twice), and plots the climb
toward the hardware roofline ceiling (1.0). RTX PRO 6000, anvil.
"""
import sys; sys.path.insert(0, "..")
import json
import os
import re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kbh_theme import C, apply
apply()

RUNS = "../benchmarks/hard/outputs/runs"


def trajectory(run_id):
    """Distinct peak_fraction readings the agent measured, in order."""
    raw = []
    files = [f"{RUNS}/{run_id}/transcript.jsonl", f"{RUNS}/{run_id}/codex_session.jsonl"]
    for f in files:
        if not os.path.exists(f):
            continue
        for line in open(f):
            try:
                o = json.loads(line)
            except Exception:
                continue
            for m in re.finditer(r"peak_fraction:\s*([01]\.\d{2,})", json.dumps(o)):
                raw.append(float(m.group(1)))
    traj = []
    for v in raw:
        if not traj or traj[-1] != v:
            traj.append(v)
    return traj


# Comparison models (dotted), plus fugu (green solid). Colors per request.
COMPARE = [
    ("Claude Opus 4.8", "20260614_144216_claude_claude-opus-4-8_01_fp8_gemm", "#f59e0b"),  # orange
    ("GPT-5.5", "20260614_144224_codex_gpt-5.5_01_fp8_gemm", "#4d9fff"),                    # blue
    ("GLM-5.2", "20260614_145529_zai-claude_glm-5.2_01_fp8_gemm", "#b07cff"),               # purple
]
FUGU_RUN = "20260622_205948_opencode_sakana_fugu-ultra_01_fp8_gemm"

traj = trajectory(FUGU_RUN)
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

# comparison trajectories (dotted), each on its own step axis
for label, rid, color in COMPARE:
    t = trajectory(rid)
    if not t:
        continue
    ax.plot(range(1, len(t) + 1), t, color=color, lw=1.8, ls=":", marker="o",
            markersize=4, markerfacecolor=color, markeredgecolor=C["bg"],
            zorder=2, label=f"{label}  ({t[-1]:.3f})")

# fugu (the subject) — green solid on top
ax.plot(x, traj, color=C["accent"], lw=2.6, marker="o", markersize=6,
        markerfacecolor=C["accent"], markeredgecolor=C["bg"], zorder=3,
        label=f"fugu-ultra  ({final:.3f})")

ax.legend(loc="lower right", frameon=False, fontsize=10.5, labelcolor=C["fg"],
          handlelength=2.4)

# Major optimization milestones, keyed by the peak_fraction value the agent
# measured right after each change (from the run transcript). (value, label, label_y)
MILESTONES = [
    (0.2550, "Baseline Triton fp8 tl.dot kernel", 0.50),
    (0.3409, "128-aligned K padding\n(was padding K to 8192)", 0.66),
    (0.3737, "Per-shape autotuned\ntile / warp configs", 0.80),
    (0.3869, "Cached weight packing\n+ output-buffer reuse", 0.92),
    (0.3941, "CUDA graph on skinny shape\n(kills launch overhead)", 0.60),
]
for val, label, ly in MILESTONES:
    # first step that reached this value
    idx = next((i for i, v in enumerate(traj) if abs(v - val) < 1e-4), None)
    if idx is None:
        continue
    px, py = x[idx], traj[idx]
    lx = px + (1.2 if px < len(traj) * 0.7 else -1.2)
    ha = "left" if px < len(traj) * 0.7 else "right"
    ax.annotate(
        label, xy=(px, py), xytext=(lx, ly), ha=ha, va="center",
        color=C["fg_bright"], fontsize=10.5,
        arrowprops=dict(arrowstyle="-", color=C["fg_dim"], lw=0.9,
                        connectionstyle="arc3,rad=0.0"),
    )
    ax.scatter([px], [py], s=95, facecolor="none", edgecolor=C["accent"],
               linewidths=1.6, zorder=4)
    ax.annotate(f"{val:.3f}", (px, py), textcoords="offset points",
                xytext=(0, -15), ha="center", color=C["accent"], fontsize=9.5,
                fontweight="bold")

ax.set_xlim(0.5, len(traj) + 1.5)
ax.set_ylim(0, 1.06)
ax.set_ylabel("fraction of hardware roofline  (out of 1.0)", fontsize=11)
ax.set_xlabel("benchmark step during the agent session", fontsize=11)
ax.grid(axis="y", lw=0.6, alpha=0.4)
ax.set_axisbelow(True)
ax.tick_params(colors=C["fg_muted"])

fig.text(0.085, 0.94, "FP8 GEMM: optimization trajectories on RTX PRO 6000",
         color=C["accent"], fontsize=19, fontweight="bold", ha="left")
fig.text(0.085, 0.895,
         "Measured speed (fraction of roofline) at each benchmark the agent ran. fugu-ultra in green; frontier models dotted.",
         color=C["fg_muted"], fontsize=11, ha="left")
fig.text(0.085, 0.862,
         f"fugu climbs {traj[0]:.2f} -> {final:.3f}, landing in the frontier pack (GLM-5.2 0.407, Opus 0.384, GPT-5.5 0.364).",
         color=C["fg_muted"], fontsize=10, ha="left")

fig.savefig("fugu_fp8_progress.png", dpi=150)
print(f"wrote fugu_fp8_progress.png ({len(traj)} steps, {traj[0]} -> {final})")
