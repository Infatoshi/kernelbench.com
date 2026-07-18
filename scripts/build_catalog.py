#!/usr/bin/env python3
"""Build public/data/catalog.json — one row per published (bench, gpu, model, problem).

This is the flywheel source of truth for site cells: score, outcome code, solution
path, HF trace URL, audit verdict. Re-run after sweeps / publish:

    uv run python scripts/build_catalog.py

Outcome codes are short and non-jargony. The harness's failure_reason is kept
raw for debugging; `outcome` is what the UI shows.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "public" / "data" / "catalog.json"

# Short UI labels + one-line plain-English blurb (legend).
LEGEND = [
    {"code": "pass", "label": "pass", "blurb": "correct on the tests"},
    {"code": "wrong", "label": "wrong", "blurb": "ran, but answers don't match"},
    {"code": "build", "label": "build", "blurb": "couldn't compile or import"},
    {"code": "timeout", "label": "slow", "blurb": "ran too long / timed out"},
    {"code": "memory", "label": "OOM", "blurb": "ran out of GPU memory"},
    {"code": "empty", "label": "empty", "blurb": "never wrote a kernel"},
    {"code": "cut", "label": "cut", "blurb": "session stopped early"},
    {"code": "infra", "label": "infra", "blurb": "provider / harness glitch"},
    {"code": "flagged", "label": "flag", "blurb": "audit rejected the run"},
    {"code": "other", "label": "fail", "blurb": "didn't pass (other)"},
]

FLAG_VERDICTS = {
    "reward_hack",
    "contamination",
    "rubric_leak",
    "megakernel_not_authentic",
}

COMPILE_RE = re.compile(
    r"error:|nvcc|CompileError|SyntaxError|IndentationError|ModuleNotFound|"
    r"cannot find|undefined reference|ImportError|Ninja is required|"
    r"triton\.compiler|Failed to compile|compilation failed|load C\+\+|load_inline",
    re.I,
)
NUMERICS_RE = re.compile(
    r"max_abs|max_rel|atol|rtol|mismatch|not equal|not close|tolerance exceeded|"
    r"bad=|Numeric|wrong|cosine|FAIL: shape|incorrect",
    re.I,
)
OOM_RE = re.compile(r"out of memory|CUDA out of memory|\bOOM\b|OutOfResources", re.I)

HARD_RUN_DIRS = [
    REPO / "benchmarks/hard/outputs/runs",
    REPO / "benchmarks/hard/outputs/runs-h100",
    REPO / "benchmarks/hard/outputs/runs-b200",
    REPO / "benchmarks/hard/outputs/runs-rtx3090",
]
MEGA_RUN_DIRS = [
    REPO / "benchmarks/mega/outputs/runs",
    REPO / "benchmarks/mega/outputs/runs-h100",
    REPO / "benchmarks/mega/outputs/runs-b200",
]
CUDA_RUN_DIRS = [REPO / "benchmarks/cuda/outputs/runs"]

GPU_MAP = {
    "RTX PRO 6000": "rtxpro6000",
    "RTX PRO 6000 Blackwell": "rtxpro6000",
    "H100": "h100",
    "H100 PCIe": "h100",
    "B200": "b200",
    "RTX 3090": "rtx3090",
}


def solution_url(run_id: str) -> str | None:
    if not run_id:
        return None
    # Site serves redacted solutions under public/runs when published.
    rel = f"runs/{run_id}_solution.py.txt"
    if (REPO / "public" / rel).exists() or (REPO / "public" / "runs" / f"{run_id}_solution.py.txt").exists():
        return f"/{rel}" if not rel.startswith("runs/") else f"/runs/{run_id}_solution.py.txt"
    p = REPO / "public" / "runs" / f"{run_id}_solution.py.txt"
    return f"/runs/{run_id}_solution.py.txt" if p.exists() else f"/runs/{run_id}_solution.py.txt"


def trace_url(bench: str, run_id: str) -> str | None:
    if not run_id:
        return None
    return (
        f"https://huggingface.co/datasets/Infatoshi/kernelbench-{bench}-traces"
        f"/blob/main/{run_id}.jsonl"
    )


def find_result(run_id: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        p = d / run_id / "result.json"
        if p.exists():
            return p
    return None


def log_blob(run_dir: Path) -> str:
    parts = []
    for name in ("check.log", "stderr.log", "benchmark.log"):
        f = run_dir / name
        if f.exists():
            try:
                parts.append(f.read_text(errors="replace")[-12000:])
            except OSError:
                pass
    return "\n".join(parts)


def outcome_from_archive(
    *,
    correct: bool,
    has_solution: bool,
    failure_reason: str | None,
    verdict: str | None,
    text: str,
) -> str:
    if verdict and verdict in FLAG_VERDICTS:
        return "flagged"
    if correct:
        return "pass"
    fr = failure_reason or ""
    if fr == "template_mutated":
        return "flagged"
    if fr in {
        "provider_rate_limited",
        "provider_insufficient_credits",
        "provider_early_stop",
        "harness_error",
    }:
        return "infra"
    if fr in {"timeout", "check_timeout", "benchmark_timeout"}:
        return "timeout"
    if fr in {"no_solution"} or not has_solution:
        if fr == "incomplete_session":
            return "cut"
        return "empty"
    if fr == "incomplete_session":
        return "cut"
    # Has a solution but didn't pass — look at logs.
    if OOM_RE.search(text):
        return "memory"
    if COMPILE_RE.search(text):
        return "build"
    if NUMERICS_RE.search(text) or fr == "incorrect":
        return "wrong"
    if fr in {"check_failed", "benchmark_failed"}:
        # check_failed without clear log signal
        if "import error" in text.lower() or "ninja" in text.lower():
            return "build"
        return "wrong"
    return "other"


def load_annotations(bench: str) -> dict[str, str]:
    ann_dir = REPO / "benchmarks" / bench / "results" / "annotations"
    out: dict[str, str] = {}
    if not ann_dir.exists():
        return out
    for p in ann_dir.glob("*.yaml"):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        rid = p.stem
        # lightweight parse — avoid yaml dep requirement
        m = re.search(r"^verdict:\s*(\S+)", text, re.M)
        if m:
            out[rid] = m.group(1).strip().strip("\"'")
    return out


def cells_from_hard_leaderboard(path: Path, gpu_key: str, ann: dict[str, str]) -> list[dict]:
    if not path.exists():
        return []
    lb = json.loads(path.read_text())
    rows = []
    for m in lb.get("models", []):
        model = m.get("model") or ""
        for prob, c in (m.get("results") or {}).items():
            rid = c.get("run_id") or ""
            rj = find_result(rid, HARD_RUN_DIRS) if rid else None
            result = {}
            text = ""
            if rj:
                try:
                    result = json.loads(rj.read_text())
                except Exception:
                    result = {}
                text = log_blob(rj.parent)
            correct = bool(c.get("correct"))
            has_sol = bool(c.get("has_solution", result.get("has_solution", bool(rid))))
            # Prefer archive has_solution when present
            if "has_solution" in result:
                has_sol = bool(result["has_solution"])
            fr = result.get("failure_reason")
            verdict = ann.get(rid) or c.get("annotation_verdict") or "unaudited"
            score = c.get("peak_fraction")
            if score is None:
                score = result.get("peak_fraction")
            outcome = outcome_from_archive(
                correct=correct,
                has_solution=has_sol,
                failure_reason=fr,
                verdict=verdict,
                text=text,
            )
            rows.append(
                {
                    "id": f"hard/{gpu_key}/{model}/{prob}",
                    "bench": "hard",
                    "gpu": gpu_key,
                    "model": model,
                    "problem": prob,
                    "run_id": rid or None,
                    "correct": correct,
                    "score": score,
                    "outcome": outcome,
                    "failure_reason": fr,
                    "has_solution": has_sol,
                    "verdict": verdict,
                    "solution_url": solution_url(rid) if rid else None,
                    "trace_url": trace_url("hard", rid) if rid else None,
                    "elapsed_seconds": c.get("elapsed_seconds")
                    or result.get("elapsed_seconds")
                    or result.get("total_elapsed_seconds"),
                }
            )
    return rows


def cells_from_cuda(ann: dict[str, str]) -> list[dict]:
    path = REPO / "benchmarks/cuda/results/leaderboard.json"
    if not path.exists():
        return []
    lb = json.loads(path.read_text())
    rows = []
    for m in lb.get("models", []):
        model = m.get("model") or ""
        for prob, c in (m.get("results") or {}).items():
            rid = c.get("run_id") or ""
            rj = find_result(rid, CUDA_RUN_DIRS) if rid else None
            result = {}
            text = ""
            if rj:
                try:
                    result = json.loads(rj.read_text())
                except Exception:
                    result = {}
                text = log_blob(rj.parent)
            correct = bool(c.get("correct"))
            has_sol = bool(result.get("has_solution", c.get("has_solution", True)))
            fr = result.get("failure_reason")
            verdict = ann.get(rid) or "unaudited"
            score = c.get("peak_fraction") or c.get("score") or result.get("peak_fraction")
            outcome = outcome_from_archive(
                correct=correct,
                has_solution=has_sol,
                failure_reason=fr,
                verdict=verdict,
                text=text,
            )
            rows.append(
                {
                    "id": f"cuda/rtxpro6000/{model}/{prob}",
                    "bench": "cuda",
                    "gpu": "rtxpro6000",
                    "model": model,
                    "problem": prob,
                    "run_id": rid or None,
                    "correct": correct,
                    "score": score,
                    "outcome": outcome,
                    "failure_reason": fr,
                    "has_solution": has_sol,
                    "verdict": verdict,
                    "solution_url": solution_url(rid) if rid else None,
                    "trace_url": trace_url("cuda", rid) if rid else None,
                    "elapsed_seconds": c.get("elapsed_seconds") or result.get("elapsed_seconds"),
                }
            )
    return rows


def cells_from_mega(ann: dict[str, str]) -> list[dict]:
    csv_path = REPO / "public/data/mega/results.csv"
    if not csv_path.exists():
        return []
    rows = []
    for r in csv.DictReader(csv_path.read_text().splitlines()):
        gpu_raw = r.get("gpu") or ""
        gpu = GPU_MAP.get(gpu_raw)
        if not gpu:
            # fuzzy
            if "H100" in gpu_raw:
                gpu = "h100"
            elif "B200" in gpu_raw:
                gpu = "b200"
            elif "RTX" in gpu_raw or "6000" in gpu_raw:
                gpu = "rtxpro6000"
            else:
                gpu = "unknown"
        model = r.get("model") or ""
        prob = r.get("problem") or ""
        rid = r.get("run_id") or ""
        correct = (r.get("correct") or "").lower() == "true"
        score = float(r["score"]) if r.get("score") not in (None, "") else None
        rj = find_result(rid, MEGA_RUN_DIRS) if rid else None
        result = {}
        text = ""
        if rj:
            try:
                result = json.loads(rj.read_text())
            except Exception:
                result = {}
            text = log_blob(rj.parent)
        has_sol = bool(result.get("has_solution", True if rid else False))
        fr = result.get("failure_reason")
        verdict = ann.get(rid) or "unaudited"
        # mega authenticity flag
        if (r.get("megakernel_judged") or "").lower() == "true" and (
            r.get("megakernel") or ""
        ).lower() in {"fail", "false", "0"}:
            verdict = "megakernel_not_authentic"
        outcome = outcome_from_archive(
            correct=correct,
            has_solution=has_sol,
            failure_reason=fr,
            verdict=verdict,
            text=text,
        )
        rows.append(
            {
                "id": f"mega/{gpu}/{model}/{prob}",
                "bench": "mega",
                "gpu": gpu,
                "model": model,
                "problem": prob,
                "run_id": rid or None,
                "correct": correct,
                "score": score,
                "outcome": outcome,
                "failure_reason": fr,
                "has_solution": has_sol,
                "verdict": verdict,
                "solution_url": solution_url(rid) if rid else None,
                "trace_url": trace_url("mega", rid) if rid else None,
                "elapsed_seconds": int(r["elapsed_s"]) if r.get("elapsed_s") else None,
                "tok_s": float(r["tok_s"]) if r.get("tok_s") else None,
            }
        )
    return rows


def main() -> int:
    cells: list[dict] = []
    hard_ann = load_annotations("hard")
    cells += cells_from_hard_leaderboard(
        REPO / "benchmarks/hard/results/leaderboard.json", "rtxpro6000", hard_ann
    )
    cells += cells_from_hard_leaderboard(
        REPO / "benchmarks/hard/results/leaderboard.h100.json", "h100", hard_ann
    )
    cells += cells_from_hard_leaderboard(
        REPO / "benchmarks/hard/results/leaderboard.b200.json", "b200", hard_ann
    )
    cells += cells_from_cuda(load_annotations("cuda"))
    cells += cells_from_mega(load_annotations("mega"))

    # index by id for fast lookup
    by_id = {c["id"]: c for c in cells}
    # also by run_id
    by_run = {c["run_id"]: c for c in cells if c.get("run_id")}

    catalog = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "legend": LEGEND,
        "n_cells": len(cells),
        "cells": cells,
        "by_id": by_id,  # convenient for tools; duplicates cells
    }
    # Don't double-store huge by_id in file — keep cells only + build index in consumers
    catalog.pop("by_id")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=2) + "\n")

    # summary
    from collections import Counter

    oc = Counter(c["outcome"] for c in cells)
    print(f"wrote {OUT} ({len(cells)} cells)")
    for code, n in oc.most_common():
        print(f"  {n:4d}  {code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
