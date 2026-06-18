#!/usr/bin/env python3
"""Build the per-GPU KernelBench-Mega leaderboard CSV from run archives.

Scans outputs/runs/*/result.json, reads a per-run GPU label (a one-line `gpu`
file written into each run dir; defaults to the Blackwell workstation for
untagged anvil runs), and emits public/data/mega/results.csv with a `gpu`
column -- the same shape the /v3 page already filters on.

Usage:
  uv run python scripts/build_mega_leaderboard.py [--runs outputs/runs] [--out ../../public/data/mega/results.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_GPU = "RTX PRO 6000 Blackwell"


def _tok_s(bench_log: Path) -> str:
    """Best tok/s across shapes from benchmark.log, if present."""
    if not bench_log.exists():
        return ""
    best = 0.0
    for m in re.finditer(r"\((\d+) tok/s\)", bench_log.read_text(errors="ignore")):
        best = max(best, float(m.group(1)))
    return str(int(best)) if best else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parents[1]
    ap.add_argument("--runs", default=str(here / "outputs" / "runs"))
    ap.add_argument("--out", default=str(here.parents[1] / "public" / "data" / "mega" / "results.csv"))
    args = ap.parse_args()

    rows = []
    for rj in sorted(Path(args.runs).glob("*/result.json")):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        if not d.get("has_solution"):
            continue
        run_dir = rj.parent
        # Require an explicit `gpu` marker: only the W4A16 sweep runs are tagged,
        # which cleanly excludes legacy bf16 runs that share the 03 problem name.
        if not (run_dir / "gpu").exists():
            continue
        gpu = (run_dir / "gpu").read_text().strip()
        rows.append({
            "gpu": gpu,
            "harness": d.get("harness", ""),
            "model": d.get("model", ""),
            "problem": d.get("problem", ""),
            "correct": "true" if d.get("correct") else "false",
            "score": f"{d.get('peak_fraction'):.3f}" if d.get("peak_fraction") is not None else "",
            "tok_s": _tok_s(run_dir / "benchmark.log"),
            "elapsed_s": d.get("elapsed_seconds", ""),
            "run_id": d.get("run_id", ""),
        })

    # de-dup: keep the best (highest score) per (gpu, model, problem)
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["gpu"], r["model"], r["problem"])
        cur = best.get(key)
        if cur is None or (r["score"] and float(r["score"] or 0) > float(cur["score"] or 0)):
            best[key] = r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["gpu", "harness", "model", "problem", "correct", "score", "tok_s", "elapsed_s", "run_id"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(best.values(), key=lambda x: (x["gpu"], x["problem"], -float(x["score"] or 0))):
            w.writerow(r)
    print(f"wrote {len(best)} rows -> {out}")
    for r in sorted(best.values(), key=lambda x: (x["gpu"], -float(x["score"] or 0))):
        print(f"  {r['gpu']:28s} {r['model']:20s} {r['problem']:22s} score={r['score']} tok/s={r['tok_s']}")


if __name__ == "__main__":
    main()
