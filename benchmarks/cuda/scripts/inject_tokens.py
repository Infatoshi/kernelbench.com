"""Bake per-cell output_tokens into the committed leaderboard JSONs.

The efficiency chart needs token counts on Vercel, but outputs/runs is gitignored
so it can't read result.json at deploy time. This injects each cell's
usage.output_tokens (looked up by run_id from the local runs archive) directly
into the tracked leaderboard.<gpu>.json, which IS deployed. Idempotent.

    uv run python scripts/inject_tokens.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# leaderboard file : runs dir
TARGETS = {
    "results/leaderboard.json": "outputs/runs",
    "results/leaderboard.h100.json": "outputs/runs-h100",
    "results/leaderboard.b200.json": "outputs/runs-b200",
    "results/leaderboard.rtx3090.json": "outputs/runs-rtx3090",
}


PROBLEMS = ["01_fp8_gemm", "02_kda_cutlass", "03_paged_attention",
            "05_topk_bitonic", "06_sonic_moe_swiglu", "07_w4a16_gemm"]


def best_tokens_by_cell(runs: Path) -> dict[tuple[str, str], int]:
    """Max output_tokens per (model, problem) across every run in the dir.

    Token capture is fragile (truncation/rate-limit kills zero it out), so the
    honest "compute this model spent on this problem" is the largest clean count
    we ever recorded, independent of which run won on peak_fraction.
    """
    best: dict[tuple[str, str], int] = {}
    if not runs.exists():
        return best
    for rj in runs.glob("2026*/result.json"):
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        model = r.get("model")
        rid = rj.parent.name
        prob = next((p for p in PROBLEMS if rid.endswith(p)), None)
        if not model or not prob:
            continue
        tok = (r.get("usage") or {}).get("output_tokens") or 0
        key = (model, prob)
        if tok > best.get(key, 0):
            best[key] = tok
    return best


def main() -> int:
    for lb_rel, runs_rel in TARGETS.items():
        lb_path = ROOT / lb_rel
        if not lb_path.exists():
            continue
        best = best_tokens_by_cell(ROOT / runs_rel)
        lb = json.load(open(lb_path))
        injected = 0
        for m in lb.get("models", []):
            model = m.get("model")
            for prob, cell in m.get("results", {}).items():
                tok = best.get((model, prob))
                if tok:
                    cell["output_tokens"] = tok
                    injected += 1
        json.dump(lb, open(lb_path, "w"), indent=2)
        print(f"{lb_rel}: injected output_tokens into {injected} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
