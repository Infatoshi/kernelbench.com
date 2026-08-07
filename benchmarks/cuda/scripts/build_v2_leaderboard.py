"""Build results/leaderboard_v2.json from run archives (KernelBench-CUDA).

v2 environment: KBH_AGENT_CONTAINER=1, parallel sessions, per-command GPU lock,
4-problem CUDA-only deck. Applies audit verdicts from results/annotations/:
reward_hack cells are kept visible but marked invalid and excluded from
ceiling ranking; rubric_leak/interesting are valid but flagged.
"""

import atexit
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))
# Per-GPU builds point this at outputs/runs-<gpu> so we never move dirs around.
RUNS_DIR = Path(os.environ.get("KBH_RUNS_DIR") or (ROOT / "outputs/runs"))
from scripts.published_submission import (  # noqa: E402
    atomic_write_text,
    load_required_published_run_ids,
    prepare_selected_solution_outputs_from_board,
    read_bounded_text,
    read_publication_annotation,
    require_publishable_annotations,
    selected_solution_run_ids,
    trusted_archive_lock,
    validate_run_id,
)
from scripts.submission_bundle import (  # noqa: E402
    BundleError,
    load_run_result,
    verify_run_provenance,
)

from src.hardware import get as get_hw  # noqa: E402

_ARCHIVE_LOCK_CONTEXT = trusted_archive_lock()
_ARCHIVE_LOCK_CONTEXT.__enter__()
atexit.register(_ARCHIVE_LOCK_CONTEXT.__exit__, None, None, None)

# Per-GPU target selects the published hardware block. Default RTX_PRO_6000
# preserves the original Blackwell leaderboard; set KBH_HARDWARE=H100 on the
# H100 box so the generated block reports Hopper specs.
try:  # instrument version, recorded truthfully at publish time
    from importlib.metadata import version as _pkg_version

    _TORCH_VERSION = _pkg_version("torch")
except Exception:
    _TORCH_VERSION = "unknown"

_hw = get_hw(os.environ.get("KBH_HARDWARE", "RTX_PRO_6000"))
PROBLEMS = ["01_glm52_fused_moe", "02_deepseek_nsa", "03_megaqwen_decode", "04_grid_mingru_sps"]

# run_id -> (verdict, summary)
ann = {}
publication_annotations = {}
# run_ids whose manual audit explicitly cleared contamination (annotation
# `contamination: clean`) even though the overall verdict is not `clean`
# (e.g. a reward_hack cell published flagged). Overrides the regex tripwire.
ann_contam_clean: set[str] = set()


def _load_annotation(run_id: str) -> None:
    if run_id in publication_annotations:
        return
    path = ROOT / "results" / "annotations" / f"{run_id}.yaml"
    try:
        annotation = read_publication_annotation(path)
    except FileNotFoundError:
        return
    publication_annotations[run_id] = annotation
    summ = re.search(r'summary:\s*"(.*)"', annotation.text, re.S)
    ann[run_id] = (annotation.verdict, (summ.group(1) if summ else "")[:400])
    if annotation.contamination_clean:
        ann_contam_clean.add(run_id)


# collect v2 runs: every run dated 2026-06-10 or later (v2 containerized era).
# Date-gated instead of an enumerated list so new sweep dates are picked up
# automatically (was hardcoded to 10/11/12 and silently dropped later sweeps).
V2_EPOCH = "20260610"

# Curation allowlist. Date gating alone over-includes experimental/superseded
# sweeps, so every board requires an explicit, nonempty manifest.
_MANIFEST_PATH = os.environ.get("KBH_PUBLISHED_MANIFEST", str(ROOT / "results/published_runs.json"))
PUBLISHED = load_required_published_run_ids(_MANIFEST_PATH)
print(f"  curation manifest: {len(PUBLISHED)} run_ids ({_MANIFEST_PATH})", file=sys.stderr)
cells = defaultdict(dict)  # (harness,model,effort) -> problem -> list of result dicts
bundle_aware_runs: set[str] = set()
_TS_RE = re.compile(r"outputs/runs/(\d{8}_\d{6})")


def _contaminated(run_dir: Path, rid: str) -> bool:
    # The harness does not sandbox the agent filesystem, so an agent can read the
    # shared outputs/runs/ archive (prior winning solutions). A run whose AGENT
    # transcript references ANOTHER run's archive is contaminated and excluded.
    self_ts = rid[:15]
    for fn in ("transcript.jsonl", "codex_session.jsonl"):
        path = run_dir / fn
        if path.exists() or path.is_symlink():
            try:
                txt = read_bounded_text(path, max_bytes=64 * 1024 * 1024, errors="ignore")
            except OSError:
                return True
            if any(ts != self_ts for ts in _TS_RE.findall(txt)):
                return True
    return False


for rj in RUNS_DIR.glob("2026*/result.json"):
    rid = rj.parent.name
    if rid[:8] < V2_EPOCH:
        continue
    try:
        rid = validate_run_id(rid)
    except BundleError as exc:
        print(f"  EXCLUDED (invalid run id): {rid}: {exc}", file=sys.stderr)
        continue
    if rid not in PUBLISHED:
        continue
    try:
        r = load_run_result(rj.parent)
        provenance = verify_run_provenance(rj.parent, r, allow_legacy=True)
    except BundleError as exc:
        print(f"  EXCLUDED (submission replay provenance): {rid}: {exc}", file=sys.stderr)
        continue
    if provenance is not None:
        bundle_aware_runs.add(rid)
    _load_annotation(rid)
    if r.get("agent_container") is not True:
        print(f"  EXCLUDED (not containerized): {rid}", file=sys.stderr)
        continue
    prob = None
    for p in PROBLEMS:
        if rid.endswith(p):
            prob = p
    if not prob:
        continue
    h, m, e = r.get("harness"), r.get("model"), r.get("reasoning_effort") or ""
    if not h or not m:
        continue
    run_dir = rj.parent
    # A manual audit verdict of `contaminated` excludes the run outright. The
    # transcript tripwire below cannot catch every case: grok streaming
    # transcripts carry no tool events or paths, so a run that copied another
    # archive's solution can pass the regex clean (2026-07-21 B200 incident).
    if ann.get(rid, (None, None))[0] in {"contaminated", "contamination"}:
        print(f"  EXCLUDED (manual audit verdict=contaminated): {rid}", file=sys.stderr)
        continue
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
    has_check = type(r.get("check_exit_code")) is int
    cells[(h, m, e)].setdefault(prob, []).append(
        {
            "run_id": rid,
            "correct": bool(r.get("correct")),
            "has_solution": r.get("has_solution") is True,
            "has_check": has_check,
            "peak_fraction": r.get("peak_fraction"),
            "elapsed_seconds": r.get("agent_wall_seconds") or r.get("total_elapsed_seconds"),
            "harness_exit_code": r.get("harness_exit_code"),
        }
    )


def best_cell(runs):
    """Best correct cell. Prefer valid (non-reward_hack) by peak; track hacked best separately."""
    valid = []
    hacked = []
    for c in runs:
        if not c["correct"]:
            continue
        verdict = ann.get(c["run_id"], (None, None))[0]
        c = dict(c, verdict=verdict)
        if verdict == "reward_hack":
            c["invalid_reason"] = "reward_hack"
            hacked.append(c)
        else:
            valid.append(c)

    def keyf(c):
        return c["peak_fraction"] if c["peak_fraction"] is not None else -1

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
    return max(
        fails,
        key=lambda c: (
            1 if c.get("has_solution") else 0,
            1 if c.get("has_check") else 0,
            c["peak_fraction"] if c["peak_fraction"] is not None else -1,
        ),
    )


# Budget-generation hygiene: any model reswept under the uncapped campaign
# (run_id >= the campaign launch) must show ONLY its uncapped cells - never a
# best-of-both Frankenstein across the 45-min and unlimited-time generations.
# Models with no uncapped run keep all their (45-min) cells as legacy.
CAMPAIGN_EPOCH = "20260613_042249"
for _key, _probs in cells.items():
    _has_uncapped = any(
        c["run_id"][:15] >= CAMPAIGN_EPOCH for _lst in _probs.values() for c in _lst
    )
    if not _has_uncapped:
        continue
    for _p in list(_probs.keys()):
        _probs[_p] = [c for c in _probs[_p] if c["run_id"][:15] >= CAMPAIGN_EPOCH]
        if not _probs[_p]:
            del _probs[_p]

models = []
for (h, m, e), probs in sorted(cells.items()):
    label = f"{h}/{m}" + (f" [{e}]" if e else "")
    results = {}
    npass = 0
    for p in PROBLEMS:
        if p not in probs:
            continue
        bc = best_cell(probs[p])
        if bc is None:
            continue
        cell = {
            "run_id": bc["run_id"],
            "correct": bc["correct"],
            "has_solution": bc["has_solution"],
            "peak_fraction": bc["peak_fraction"],
            "elapsed_seconds": bc["elapsed_seconds"],
        }
        if bc.get("verdict"):
            cell["annotation_verdict"] = bc["verdict"]
        if bc.get("invalid_reason"):
            cell["invalid_reason"] = bc["invalid_reason"]
        results[p] = cell
        if bc["correct"] and bc.get("invalid_reason") is None and bc["peak_fraction"] is not None:
            npass += 1
    models.append(
        {
            "label": label,
            "harness": h,
            "model": m,
            "effort": e,
            "results": results,
            "valid_pass_count": npass,
            "total_problems": len(results),
        }
    )

# per-problem ceilings, valid cells only
per_problem = {}
for p in PROBLEMS:
    ranked = []
    for mm in models:
        c = mm["results"].get(p)
        if (
            c
            and c["correct"]
            and c.get("invalid_reason") is None
            and c["peak_fraction"] is not None
        ):
            ranked.append(
                {
                    "model": mm["label"],
                    "peak_fraction": c["peak_fraction"],
                    "verdict": c.get("annotation_verdict"),
                }
            )
    ranked.sort(key=lambda x: -x["peak_fraction"])
    hacked_n = sum(
        1
        for mm in models
        for c in [mm["results"].get(p)]
        if c and c.get("invalid_reason") == "reward_hack"
    )
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
    "environment_notes": f"KBH_AGENT_CONTAINER=1, parallel sessions, per-command GPU lock, torch {_TORCH_VERSION}; 4-problem CUDA-only deck (language gate: Triton/DSL fail)",
    "hardware": {
        "name": _hw.name,
        "sm": _hw.sm,
        "vram_gb": _hw.vram_gb,
        "peak_bandwidth_gb_s": _hw.peak_bandwidth_gb_s,
    },
    "problems": PROBLEMS,
    "models": sorted(models, key=lambda m: -m["valid_pass_count"]),
    "per_problem": per_problem,
    "reward_hack_count": sum(
        1
        for mm in models
        for c in mm["results"].values()
        if c.get("invalid_reason") == "reward_hack"
    ),
}
require_publishable_annotations(
    selected_solution_run_ids(out),
    bundle_aware_runs,
    publication_annotations,
)
prepare_selected_solution_outputs_from_board(
    out,
    RUNS_DIR,
    ROOT.parents[1] / "public/runs",
)
atomic_write_text(ROOT / "results/leaderboard_v2.json", json.dumps(out, indent=2) + "\n")
print("models:", len(models), "| total reward-hack cells:", out["reward_hack_count"])
for p in PROBLEMS:
    pp = per_problem[p]
    print(
        f"  {p}: ceiling {pp['best_peak_fraction']} ({pp['best_model']}) | valid {pp['n_valid_passes']} | hacked {pp['n_reward_hacks']}"
    )
