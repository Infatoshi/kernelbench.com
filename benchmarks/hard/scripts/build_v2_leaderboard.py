"""Build results/leaderboard_v2.json from the 2026-06-10/11 containerized sweep.

v2 environment: KBH_AGENT_CONTAINER=1, parallel sessions, per-command GPU lock,
6-problem deck. Applies audit verdicts from results/annotations/: reward_hack
cells are kept visible but marked invalid and excluded from ceiling ranking;
rubric_leak/interesting are valid but flagged.
"""
import json, glob, os, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Per-GPU builds point this at outputs/runs-<gpu> so we never move dirs around.
RUNS_DIR = Path(os.environ.get("KBH_RUNS_DIR") or (ROOT / "outputs/runs"))
from src.hardware import get as get_hw  # noqa: E402

# Per-GPU target selects the published hardware block. Default RTX_PRO_6000
# preserves the original Blackwell leaderboard; set KBH_HARDWARE=H100 on the
# H100 box so the generated block reports Hopper specs.
_hw = get_hw(os.environ.get("KBH_HARDWARE", "RTX_PRO_6000"))
PROBLEMS = ["01_fp8_gemm","02_kda_cutlass","03_paged_attention",
            "05_topk_bitonic","06_sonic_moe_swiglu","07_w4a16_gemm"]

# run_id -> (verdict, summary)
ann = {}
# run_ids whose manual audit explicitly cleared contamination (annotation
# `contamination: clean`) even though the overall verdict is not `clean`
# (e.g. a reward_hack cell published flagged). Overrides the regex tripwire.
ann_contam_clean: set[str] = set()
for f in glob.glob(str(ROOT/"results/annotations/*.yaml")):
    txt = open(f).read()
    rid = re.search(r"run_id:\s*(\S+)", txt)
    ver = re.search(r"^verdict:\s*(\S+)", txt, re.M)
    summ = re.search(r'summary:\s*"(.*)"', txt, re.S)
    if rid and ver:
        ann[rid.group(1)] = (ver.group(1), (summ.group(1) if summ else "")[:400])
        if re.search(r"^contamination:\s*clean\b", txt, re.M):
            ann_contam_clean.add(rid.group(1))

# collect v2 runs: every run dated 2026-06-10 or later (v2 containerized era).
# Date-gated instead of an enumerated list so new sweep dates are picked up
# automatically (was hardcoded to 10/11/12 and silently dropped later sweeps).
V2_EPOCH = "20260610"

# Curation allowlist. The date gate alone over-includes experimental/superseded
# sweeps, so the *published* board is an explicit set of run_ids. When the
# manifest exists (and KBH_PUBLISHED_MANIFEST is not set to empty), only those
# run_ids are considered -- this makes leaderboard.json reproducible from the
# archives. Per-GPU builds (build_all_gpus.sh) set KBH_PUBLISHED_MANIFEST="" to
# disable it and keep their date-gated behavior. Empty manifest => no filter.
_MANIFEST_PATH = os.environ.get("KBH_PUBLISHED_MANIFEST", str(ROOT / "results/published_runs.json"))
PUBLISHED: set[str] = set()
if _MANIFEST_PATH and os.path.exists(_MANIFEST_PATH):
    PUBLISHED = set(json.load(open(_MANIFEST_PATH)).get("run_ids", []))
    print(f"  curation manifest: {len(PUBLISHED)} run_ids ({_MANIFEST_PATH})", file=sys.stderr)
# Roofline was corrected 2026-06-14 (peak_tflops 2.5x too low). Compute-regime
# problems (graded on TFLOPS) scored before the fix need peak_fraction x0.4;
# memory-regime problems (graded on bandwidth, unchanged) and post-fix runs do not.
_ROOFLINE_FIX_EPOCH = "20260614_000000"
_COMPUTE_RESCALE = {"02_kda_cutlass": 0.4, "06_sonic_moe_swiglu": 0.4}
def _rescale_pf(pf, prob, rid):
    if pf is None:
        return None
    f = _COMPUTE_RESCALE.get(prob)
    if f is not None and rid[:15] < _ROOFLINE_FIX_EPOCH:
        return pf * f
    return pf

cells = defaultdict(dict)  # (harness,model,effort) -> problem -> list of result dicts
_TS_RE = re.compile(r"outputs/runs/(\d{8}_\d{6})")
def _contaminated(run_dir, rid):
    # The harness does not sandbox the agent filesystem, so an agent can read the
    # shared outputs/runs/ archive (prior winning solutions). A run whose AGENT
    # transcript references ANOTHER run's archive is contaminated and excluded.
    self_ts = rid[:15]
    for fn in ("transcript.jsonl", "codex_session.jsonl"):
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            try: txt = open(p, errors="ignore").read()
            except Exception: continue
            if any(ts != self_ts for ts in _TS_RE.findall(txt)):
                return True
    return False

for rj in glob.glob(str(RUNS_DIR/"2026*/result.json")):
    rid = os.path.basename(os.path.dirname(rj))
    if rid[:8] < V2_EPOCH: continue
    if PUBLISHED and rid not in PUBLISHED: continue
    try: r = json.load(open(rj))
    except: continue
    prob = None
    for p in PROBLEMS:
        if rid.endswith(p): prob = p
    if not prob: continue
    h, m, e = r.get("harness"), r.get("model"), r.get("reasoning_effort") or ""
    if not h or not m: continue
    run_dir = os.path.dirname(rj)
    if _contaminated(run_dir, rid):
        # The regex tripwire over-fires on parallel sweeps: sibling run ids leak
        # into transcripts passively via the shared gpu.lock owner file and ps
        # output. A manual audit (results/annotations/, which includes a
        # contamination read of the transcript) marked `clean` overrides it;
        # anything unaudited or non-clean stays excluded.
        if ann.get(rid, (None, None))[0] == "clean" or rid in ann_contam_clean:
            print(f"  tripwire overridden (manual audit clean): {rid}", file=sys.stderr)
        else:
            print(f"  EXCLUDED (contaminated, read other archive): {rid}", file=sys.stderr)
            continue
    has_sol_file = os.path.exists(os.path.join(run_dir, "solution.py"))
    has_check = os.path.exists(os.path.join(run_dir, "check.log"))
    cells[(h,m,e)].setdefault(prob, []).append({
        "run_id": rid, "correct": bool(r.get("correct")),
        "has_solution": has_sol_file or bool(r.get("has_solution")),
        "has_check": has_check,
        "peak_fraction": _rescale_pf(r.get("peak_fraction"), prob, rid),
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
    fails = [dict(c, verdict=ann.get(c["run_id"], (None, None))[0]) for c in runs]
    if not fails:
        return None
    return max(fails, key=lambda c: (
        1 if c.get("has_solution") else 0,
        1 if c.get("has_check") else 0,
        c["peak_fraction"] if c["peak_fraction"] is not None else -1,
    ))

# Budget-generation hygiene: any model reswept under the uncapped campaign
# (run_id >= the campaign launch) must show ONLY its uncapped cells - never a
# best-of-both Frankenstein across the 45-min and unlimited-time generations.
# Models with no uncapped run keep all their (45-min) cells as legacy.
# When the curation manifest is active it supersedes this guard: every run_id
# in published_runs.json was hand-curated, so cross-generation rows there are
# deliberate (e.g. Fable 5's June cells + the 2026-07-16 uncapped fp8 rerun).
# Without this, publishing one uncapped cell silently nuked the rest of the row.
CAMPAIGN_EPOCH = "20260613_042249"
for _key, _probs in (cells.items() if not PUBLISHED else []):
    _has_uncapped = any(c["run_id"][:15] >= CAMPAIGN_EPOCH
                        for _lst in _probs.values() for c in _lst)
    if not _has_uncapped:
        continue
    for _p in list(_probs.keys()):
        _probs[_p] = [c for c in _probs[_p] if c["run_id"][:15] >= CAMPAIGN_EPOCH]
        if not _probs[_p]:
            del _probs[_p]

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
    "hardware": {"name":_hw.name,"sm":_hw.sm,"vram_gb":_hw.vram_gb,"peak_bandwidth_gb_s":_hw.peak_bandwidth_gb_s},
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
