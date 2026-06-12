"""Build results/leaderboard_v2.json from the 2026-06-10/11 containerized sweep.

v2 environment: KBH_AGENT_CONTAINER=1, parallel sessions, per-command GPU lock,
6-problem deck. Applies audit verdicts from results/annotations/: reward_hack
cells are kept visible but marked invalid and excluded from ceiling ranking;
rubric_leak/interesting are valid but flagged.
"""
import json, glob, os, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ["01_fp8_gemm","02_kda_cutlass","03_paged_attention",
            "05_topk_bitonic","06_sonic_moe_swiglu","07_w4a16_gemm"]

# run_id -> (verdict, summary)
ann = {}
for f in glob.glob(str(ROOT/"results/annotations/*.yaml")):
    txt = open(f).read()
    rid = re.search(r"run_id:\s*(\S+)", txt)
    ver = re.search(r"verdict:\s*(\S+)", txt)
    summ = re.search(r'summary:\s*"(.*)"', txt, re.S)
    if rid and ver:
        ann[rid.group(1)] = (ver.group(1), (summ.group(1) if summ else "")[:400])

# collect v2 runs (2026-06-10 and 2026-06-11)
cells = defaultdict(dict)  # (harness,model,effort) -> problem -> list of result dicts
for rj in glob.glob(str(ROOT/"outputs/runs/20260610_*/result.json")) + \
          glob.glob(str(ROOT/"outputs/runs/20260611_*/result.json")) + \
          glob.glob(str(ROOT/"outputs/runs/20260612_*/result.json")):
    try: r = json.load(open(rj))
    except: continue
    rid = os.path.basename(os.path.dirname(rj))
    prob = None
    for p in PROBLEMS:
        if rid.endswith(p): prob = p
    if not prob: continue
    h, m, e = r.get("harness"), r.get("model"), r.get("reasoning_effort") or ""
    if not h or not m: continue
    run_dir = os.path.dirname(rj)
    has_sol_file = os.path.exists(os.path.join(run_dir, "solution.py"))
    has_check = os.path.exists(os.path.join(run_dir, "check.log"))
    cells[(h,m,e)].setdefault(prob, []).append({
        "run_id": rid, "correct": bool(r.get("correct")),
        "has_solution": has_sol_file or bool(r.get("has_solution")),
        "has_check": has_check,
        "peak_fraction": r.get("peak_fraction"),
        "elapsed_seconds": r.get("agent_wall_seconds") or r.get("total_elapsed_seconds"),
        "harness_exit_code": r.get("harness_exit_code"),
    })

def best_cell(runs):
    """Best correct cell. Prefer valid (non-reward_hack) by peak; track hacked best separately."""
    valid = []
    hacked = []
    for c in runs:
        if not c["correct"]: continue
        verdict = ann.get(c["run_id"], (None,None))[0]
        c = dict(c, verdict=verdict)
        if verdict == "reward_hack":
            c["invalid_reason"] = "reward_hack"
            hacked.append(c)
        else:
            valid.append(c)
    keyf = lambda c: (c["peak_fraction"] if c["peak_fraction"] is not None else -1)
    if valid:
        return max(valid, key=keyf)
    if hacked:
        return max(hacked, key=keyf)
    # no pass: surface the MOST INFORMATIVE failed attempt — one that actually
    # wrote a solution and ran check (a real attempt that failed correctness)
    # beats an empty context-overflow run. Rank: has_solution, has_check, peak.
    fails = [dict(c, verdict=None) for c in runs]
    if not fails:
        return None
    return max(fails, key=lambda c: (
        1 if c.get("has_solution") else 0,
        1 if c.get("has_check") else 0,
        c["peak_fraction"] if c["peak_fraction"] is not None else -1,
    ))

models = []
for (h,m,e), probs in sorted(cells.items()):
    label = f"{h}/{m}" + (f" [{e}]" if e else "")
    results = {}
    npass = 0
    for p in PROBLEMS:
        if p not in probs: continue
        bc = best_cell(probs[p])
        if bc is None: continue
        cell = {"run_id": bc["run_id"], "correct": bc["correct"],
                "has_solution": bc["has_solution"], "peak_fraction": bc["peak_fraction"],
                "elapsed_seconds": bc["elapsed_seconds"]}
        if bc.get("verdict"): cell["annotation_verdict"] = bc["verdict"]
        if bc.get("invalid_reason"): cell["invalid_reason"] = bc["invalid_reason"]
        results[p] = cell
        if bc["correct"] and bc.get("invalid_reason") is None and bc["peak_fraction"] is not None:
            npass += 1
    models.append({"label": label, "harness": h, "model": m, "effort": e,
                   "results": results, "valid_pass_count": npass,
                   "total_problems": len(results)})

# per-problem ceilings, valid cells only
per_problem = {}
for p in PROBLEMS:
    ranked = []
    for mm in models:
        c = mm["results"].get(p)
        if c and c["correct"] and c.get("invalid_reason") is None and c["peak_fraction"] is not None:
            ranked.append({"model": mm["label"], "peak_fraction": c["peak_fraction"],
                           "verdict": c.get("annotation_verdict")})
    ranked.sort(key=lambda x: -x["peak_fraction"])
    hacked_n = sum(1 for mm in models
                   for c in [mm["results"].get(p)] if c and c.get("invalid_reason")=="reward_hack")
    per_problem[p] = {
        "n_models": len([mm for mm in models if p in mm["results"]]),
        "n_valid_passes": len(ranked),
        "n_reward_hacks": hacked_n,
        "best_peak_fraction": ranked[0]["peak_fraction"] if ranked else None,
        "best_model": ranked[0]["model"] if ranked else None,
        "ranked_valid_passes": ranked,
    }

out = {
    "schema_version": 2,
    "environment": "v2_containerized",
    "environment_notes": "KBH_AGENT_CONTAINER=1, parallel sessions, per-command GPU lock, nvcc 13.2, torch 2.11+cu130; 6-problem deck",
    "hardware": {"name":"RTX PRO 6000 Blackwell Workstation","sm":"sm_120a","vram_gb":96,"peak_bandwidth_gb_s":1800},
    "problems": PROBLEMS,
    "models": sorted(models, key=lambda m: -m["valid_pass_count"]),
    "per_problem": per_problem,
    "reward_hack_count": sum(1 for mm in models for c in mm["results"].values() if c.get("invalid_reason")=="reward_hack"),
}
Path(ROOT/"results/leaderboard_v2.json").write_text(json.dumps(out, indent=2))
print("models:", len(models), "| total reward-hack cells:", out["reward_hack_count"])
for p in PROBLEMS:
    pp = per_problem[p]
    print(f"  {p}: ceiling {pp['best_peak_fraction']} ({pp['best_model']}) | valid {pp['n_valid_passes']} | hacked {pp['n_reward_hacks']}")
