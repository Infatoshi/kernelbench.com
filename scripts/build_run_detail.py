#!/usr/bin/env python3
"""Emit public/data/rundetail/<run_id>.json for every published leaderboard cell.

Each file carries what the site's run-detail overlay renders: per-shape
benchmark metrics graded against the shape's governing ceiling (compute peak
vs HBM bandwidth), extended session stats, and GPU-lock totals. Solution text
is NOT embedded — the overlay reads the already-redacted
public/runs/<run_id>_solution.py.txt.

Run from repo root (kb publish calls this; safe to run by hand):
    uv run python scripts/build_run_detail.py
"""

from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "public" / "data" / "rundetail"

# (bench, leaderboard file, gpu key, deck dir, runs dir)
# Canonical boards (RTX PRO 6000) keep flat rundetail/<rid>.json paths; every
# other GPU board namespaces as rundetail/<gpu>/<rid>.json — the same run_id
# can exist on two boards (two different sessions), so bare-rid keys would
# collide and misattribute one GPU's session to the other board.
BOARDS = [
    ("hard", "results/leaderboard.json", "rtxpro6000", "problems-rtxpro6000", "runs"),
    ("hard", "results/leaderboard.h100.json", "h100", "problems-h100", "runs-h100"),
    ("hard", "results/leaderboard.b200.json", "b200", "problems-b200", "runs-b200"),
    ("hard", "results/leaderboard.rtx3090.json", "rtx3090", "problems-3090", "runs-rtx3090"),
    ("cuda", "results/leaderboard.json", "rtxpro6000", "problems-rtxpro6000", "runs"),
    ("cuda", "results/leaderboard.b200.json", "b200", "problems-rtxpro6000", "runs-b200"),
]

CANONICAL_GPU = "rtxpro6000"

HW_FILES = {
    "rtxpro6000": "rtx_pro_6000.py",
    "h100": "h100.py",
    "b200": "b200.py",
    "rtx3090": "rtx_3090.py",
}

SHAPE_RE = re.compile(
    r"shape=(\d+) variant=solution tflops=([\d.]+) gbps=([\d.]+) ms=([\d.]+)"
)
FRAC_RE = re.compile(r"shape=(\d+) solution_peak_fraction=([\d.]+)")
LOCK_RE = re.compile(
    r"^(\S+) (wait|start|end) pid=(\d+) cmd=(\S+)(?:.* elapsed_s=(\d+))?"
)


def parse_hardware(bench: str, gpu: str) -> dict:
    """Pull peak_bandwidth_gb_s + peak_tflops_dense out of src/hardware/*.py."""
    path = ROOT / "benchmarks" / bench / "src" / "hardware" / HW_FILES[gpu]
    if not path.exists():
        return {}
    text = path.read_text()
    peaks: dict = {}
    m = re.search(r"peak_bandwidth_gb_s\s*=\s*([\d.]+)", text)
    if m:
        peaks["bw_gb_s"] = float(m.group(1))
    for dt, val in re.findall(r'"(\w+)":\s*([\d.]+)', text):
        if dt in ("fp8", "bf16", "fp32", "fp16", "int8", "fp4"):
            peaks.setdefault("tflops", {})[dt] = float(val)
    return peaks


def parse_problem_meta(deck_dir: Path, problem: str) -> dict:
    meta: dict = {}
    yaml_path = deck_dir / problem / "problem.yaml"
    if yaml_path.exists():
        text = yaml_path.read_text()
        m = re.search(r"regime:\s*(\S+)", text)
        if m:
            meta["regime"] = m.group(1)
        m = re.search(r"peak_tflops_key:\s*(\S+)", text)
        if m:
            meta["dtype"] = m.group(1)
    shapes_path = deck_dir / problem / "shapes.py"
    if shapes_path.exists():
        try:
            tree = ast.parse(shapes_path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "SHAPES" for t in node.targets
                ):
                    shapes = ast.literal_eval(node.value)
                    meta["shape_dims"] = [
                        s if isinstance(s, dict) else {"shape": s} for s in shapes
                    ]
        except (SyntaxError, ValueError):
            pass
    return meta


def parse_lock_log(path: Path) -> dict | None:
    """Total GPU-lock wait vs held time from gpu_lock.log wait/start/end lines."""
    if not path.exists():
        return None
    waits: dict[str, str] = {}
    wait_s = 0.0
    active_s = 0.0
    acquisitions = 0

    def ts(s: str) -> datetime | None:
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None

    for line in path.read_text(errors="replace").splitlines():
        m = LOCK_RE.match(line)
        if not m:
            continue
        stamp, event, pid, _cmd, elapsed = m.groups()
        if event == "wait":
            waits[pid] = stamp
        elif event == "start":
            acquisitions += 1
            if pid in waits:
                a, b = ts(waits.pop(pid)), ts(stamp)
                if a and b:
                    wait_s += max((b - a).total_seconds(), 0)
        elif event == "end" and elapsed is not None:
            active_s += float(elapsed)
    return {
        "wait_s": round(wait_s),
        "active_s": round(active_s),
        "acquisitions": acquisitions,
    }


def parse_shapes(bench_log: Path) -> list[dict]:
    if not bench_log.exists():
        return []
    shapes: dict[int, dict] = {}
    for line in bench_log.read_text(errors="replace").splitlines():
        m = SHAPE_RE.match(line)
        if m:
            i = int(m.group(1))
            shapes.setdefault(i, {}).update(
                tflops=float(m.group(2)),
                gbps=float(m.group(3)),
                ms=float(m.group(4)),
            )
        m = FRAC_RE.match(line)
        if m:
            shapes.setdefault(int(m.group(1)), {})["frac"] = float(m.group(2))
    return [dict(idx=i, **v) for i, v in sorted(shapes.items())]


def dims_label(dims: dict) -> str:
    nums = [str(v) for v in dims.values() if isinstance(v, (int, float))]
    return "×".join(nums) if nums else str(dims)


def build_cell(
    bench: str, gpu: str, deck: str, runs: str, run_id: str, cell: dict
) -> dict | None:
    run_dir = ROOT / "benchmarks" / bench / "outputs" / runs / run_id
    if not run_dir.exists() and runs != "runs":
        # Boards that predate the runs-<gpu> split (rtx3090) archive in plain
        # runs/. Only fires when the board copy is absent, so a colliding rid
        # present in both dirs always resolves to the board's own session.
        run_dir = ROOT / "benchmarks" / bench / "outputs" / "runs" / run_id
    if not run_dir.exists():
        return None
    try:
        result = json.loads((run_dir / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        result = {}
    problem = result.get("problem") or cell.get("problem") or run_id.split("_", 4)[-1]
    deck_dir = ROOT / "benchmarks" / bench / deck
    meta = parse_problem_meta(deck_dir, problem)
    peaks = parse_hardware(bench, gpu)
    dtype = meta.get("dtype") or "bf16"
    peak_tf = (peaks.get("tflops") or {}).get(dtype)
    bw = peaks.get("bw_gb_s")

    shapes = parse_shapes(run_dir / "benchmark.log")
    dims = meta.get("shape_dims") or []
    for s in shapes:
        if s["idx"] < len(dims):
            s["label"] = dims_label(dims[s["idx"]])
        if peak_tf and "tflops" in s:
            s["compute_util"] = round(s["tflops"] / peak_tf, 4)
        if bw and "gbps" in s:
            s["mem_util"] = round(s["gbps"] / bw, 4)
        cu, mu = s.get("compute_util", 0), s.get("mem_util", 0)
        if cu or mu:
            s["bound"] = "compute" if cu >= mu else "memory"
            s["util"] = max(cu, mu)

    usage = result.get("usage") or {}
    detail = {
        "run_id": run_id,
        "bench": bench,
        "gpu": gpu,
        "problem": problem,
        "harness": result.get("harness"),
        "model": result.get("model"),
        "regime": meta.get("regime"),
        "dtype": dtype,
        "peak_tflops": peak_tf,
        "peak_bw_gb_s": bw,
        "correct": result.get("correct", cell.get("correct")),
        "failure_reason": result.get("failure_reason"),
        "peak_fraction": result.get("peak_fraction", cell.get("peak_fraction")),
        "annotation_verdict": cell.get("annotation_verdict"),
        "stats": {
            "agent_s": result.get("elapsed_seconds"),
            "total_s": result.get("total_elapsed_seconds"),
            "check_s": result.get("check_elapsed_seconds"),
            "benchmark_s": result.get("benchmark_elapsed_seconds"),
            "output_tokens": usage.get("output_tokens") or cell.get("output_tokens"),
            "input_tokens": usage.get("input_tokens"),
            "cost_usd": usage.get("total_cost_usd"),
            "template_mutated": result.get("template_mutated"),
        },
        "gpu_lock": parse_lock_log(run_dir / "gpu_lock.log"),
        "shapes": shapes,
        # Canonical boards read the flat public/runs/<rid> file; per-GPU
        # boards read public/runs/<gpu>/<rid> (emit_board_solutions.py) — a
        # bare file with a colliding rid belongs to the canonical session.
        "has_solution_text": (
            ROOT
            / "public"
            / "runs"
            / ("" if gpu == CANONICAL_GPU else gpu)
            / f"{run_id}_solution.py.txt"
        ).exists(),
    }
    return detail


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for bench, lb_rel, gpu, deck, runs in BOARDS:
        lb_path = ROOT / "benchmarks" / bench / lb_rel
        if not lb_path.exists():
            continue
        lb = json.loads(lb_path.read_text())
        for model in lb.get("models", []):
            for _prob, cell in (model.get("results") or {}).items():
                run_id = cell.get("run_id")
                if not run_id:
                    continue
                detail = build_cell(bench, gpu, deck, runs, run_id, cell)
                if detail is None:
                    skipped += 1
                    continue
                out_dir = OUT_DIR if gpu == CANONICAL_GPU else OUT_DIR / gpu
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{run_id}.json"
                out.write_text(json.dumps(detail, indent=1) + "\n")
                written += 1
    print(f"rundetail: wrote {written}, skipped (no archive) {skipped}")


if __name__ == "__main__":
    main()
