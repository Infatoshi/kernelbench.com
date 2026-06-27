import sys; sys.path.insert(0, "..")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from kbh_theme import C, apply
apply()

# verdict: clean|hack|realfail|timeout|fail
PROB=["01 fp8_gemm","02 kda_cutlass","03 paged_attn","05 topk","06 sonic_moe","07 w4a16"]
DATA={
 "Claude Fable 5": [(0.0,"realfail"),(0.0894,"clean"),(0.6299,"clean"),(0.0494,"clean"),(0.2688,"clean"),(0.3477,"clean")],
 "GLM-5.2":        [(0.0,"timeout"),(0.0,"timeout"),(0.4897,"clean"),(0.0163,"clean"),(0.19,"clean"),(0.096,"clean")],
 "MiniMax-M3":     [(0.0,"fail"),(0.0,"fail"),(0.4228,"clean"),(0.0,"fail"),(0.1206,"clean"),(0.1003,"clean")],
 "Kimi K2.7-Code": [(0.0,"hack"),(0.0214,"clean"),(0.0,"realfail"),(0.0,"realfail"),(0.1568,"clean"),(0.1012,"clean")],
}
# Fable = NVIDIA green accent (the ceiling); others legible companions in-theme.
MCOL={"Claude Fable 5":C["accent"],"GLM-5.2":"#4d9fff","MiniMax-M3":"#b07cff","Kimi K2.7-Code":"#f0883e"}
GREEN_HI=C["accent"]; GREY=C["fg_muted"]; AMBER=C["warn"]; RED=C["bad"]; SLATE="#3a3a3a"
models=list(DATA); x=np.arange(len(PROB)); w=0.2
fig,ax=plt.subplots(figsize=(14.5,7.6))
fig.subplots_adjust(top=0.80,left=0.065,right=0.975,bottom=0.10)
ax.set_facecolor(C["bg"])
for spine in ax.spines.values(): spine.set_color(C["border"])
fig.text(0.065,0.945,"Frontier coding models on KernelBench-Hard:  GLM-5.2  vs  MiniMax-M3  vs  Kimi K2.7-Code  vs  Fable 5",color=GREEN_HI,fontsize=15.5,fontweight="bold",ha="left")
fig.text(0.065,0.895,"KernelBench-Hard v2, one 45-min autonomous run per problem. bar = peak_fraction of the SM120 (RTX PRO 6000) roofline.",color=GREY,fontsize=10.5,ha="left")
fig.text(0.065,0.862,"GLM-5.2 (day-one via Z.ai) goes 4/6 clean - the most of any open-weight model here - with a real 0.49 paged-attention kernel. Fable still tops all 6.",color=GREY,fontsize=10,ha="left")

for mi,m in enumerate(models):
    off=(mi-1.5)*w
    for j,(s,v) in enumerate(DATA[m]):
        base=MCOL[m]
        if v=="hack": col,hatch=RED,"////"
        elif v in ("realfail","timeout"): col,hatch=base,".."
        elif v=="fail": col,hatch=SLATE,None
        else: col,hatch=base,None
        ax.bar(x[j]+off, max(s,0.004), w, color=col, hatch=hatch, edgecolor=C["bg"], zorder=3,
               alpha=0.4 if v in ("realfail","timeout") else 1.0)
        if v=="hack": t="HACK\n0.428*" if j==0 else "HACK"
        elif v=="timeout": t="time"
        elif v=="realfail": t="bug"
        elif v=="fail": t="fail"
        else: t=f"{s:.3f}"
        tc={"clean":C["fg"],"hack":RED,"realfail":AMBER,"timeout":AMBER,"fail":GREY}[v]
        ax.text(x[j]+off, max(s,0.004)+0.008, t, ha="center", va="bottom", color=tc, fontsize=7.2, fontweight="bold" if v=="clean" else "normal")

ax.set_xticks(x); ax.set_xticklabels(PROB,fontsize=11)
ax.set_ylabel("peak_fraction (fraction of roofline)")
ax.set_ylim(0,0.70); ax.set_xlim(-0.55,len(PROB)-0.45)
ax.grid(True,axis="y",alpha=0.5)
leg=[Patch(facecolor=C["accent"],label="Claude Fable 5 (ceiling)"),Patch(facecolor="#4d9fff",label="GLM-5.2"),
     Patch(facecolor="#b07cff",label="MiniMax-M3"),Patch(facecolor="#f0883e",label="Kimi K2.7-Code"),
     Patch(facecolor=RED,hatch="////",label="reward hack (invalid)")]
ax.legend(handles=leg,loc="upper center",ncol=5,facecolor=C["surface"],edgecolor=C["border"],labelcolor=C["fg"],fontsize=9,framealpha=0.97)
ax.text(0.985,0.70,"faded = real kernel that bugged/timed out   |   grey = failed   |   * hacked cells are invalid",transform=ax.transAxes,ha="right",va="top",color=GREY,fontsize=8.5)
fig.savefig("glm52_4way.png",dpi=140)
print("wrote glm52_4way.png")
