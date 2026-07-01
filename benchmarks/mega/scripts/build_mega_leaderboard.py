#!/usr/bin/env python3
"""Build the per-GPU KernelBench-Mega leaderboard CSV from run archives.

Scans outputs/runs/*/result.json, requires a per-run `gpu` marker file (so
legacy bf16 runs sharing the 03 problem name are excluded), and emits
public/data/mega/results.csv with a `gpu` column plus the richer per-run
metrics the /mega page surfaces: agent wall-clock, output tokens, per-context
speedups, the approach/framework label, and whether a transcript viewer exists.

Usage:
  uv run python scripts/build_mega_leaderboard.py [--runs DIR] [--out CSV] [--runs-html DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def _bench(bench_log: Path):
    """Parse benchmark.log -> (best tok/s, {ctx: speedup}). Empty if absent."""
    if not bench_log.exists():
        return "", {}
    txt = bench_log.read_text(errors="ignore")
    best = max((float(m) for m in re.findall(r"\((\d+) tok/s\)", txt)), default=0.0)
    ctx = {}
    for m in re.finditer(r"ctx=(\d+):.*?speedup ([0-9.]+)x", txt):
        ctx[int(m.group(1))] = float(m.group(2))
    return (str(int(best)) if best else ""), ctx


def _sidecar_code(run_dir: Path, sol_code: str) -> str:
    """Code of local modules that solution.py imports.

    Some harnesses (notably cursor) keep the actual kernel in a sidecar module
    that solution.py imports (e.g. `from w4_triton import ...` with the
    @triton.jit kernels living in scratch/w4_triton.py) instead of inlining it.
    Without resolving these imports, _framework only sees the thin solution.py
    wrapper and mislabels a real Triton/CUDA kernel as "eager". Resolve the
    imported module names to files under run_dir (top level + scratch/, never
    .venv/site-packages) and return their concatenated source.
    """
    mods = set(re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_]\w*)", sol_code, re.M))
    if not mods:
        return ""
    cands = list(run_dir.glob("*.py"))
    sc = run_dir / "scratch"
    if sc.is_dir():
        cands += list(sc.rglob("*.py"))
    out = []
    for p in cands:
        if p.name == "solution.py" or "site-packages" in p.parts or ".venv" in p.parts:
            continue
        if p.stem in mods:
            try:
                out.append(p.read_text(errors="ignore"))
            except Exception:
                pass
    return "\n".join(out)


def _framework(run_dir: Path) -> str:
    f = run_dir / "framework.txt"
    if f.exists():
        return f.read_text().strip()
    sol = run_dir / "solution.py"
    if sol.exists():
        code = sol.read_text(errors="ignore")
        code += "\n" + _sidecar_code(run_dir, code)
        for name, pat in [
            ("ptx", r"asm\s+volatile|mma\.sync|tcgen05\."),
            ("cuda", r"load_inline|__global__\s+void"),
            ("triton", r"@triton\.jit"),
            ("cudagraph", r"CUDAGraph|make_graphed"),
            ("compile", r"torch\.compile"),
        ]:
            if re.search(pat, code):
                return name
    return "eager"


_RUN_TS = re.compile(r"outputs/runs/(\d{8}_\d{6})")


def megakernel_authentic(run_id: str, annotations_dir: Path) -> bool | None:
    """Read the authenticity verdict recorded by the post-run judge audit.

    Returns False only when an annotation explicitly says the submission is not a
    genuine megakernel (CUDA-graph / torch.compile / per-op-loop escape hatch, or
    plain eager dressed up as fusion). None = not yet judged (do NOT exclude).
    See docs/megakernel_authenticity_judge.md.
    """
    f = annotations_dir / f"{run_id}.yaml"
    if not f.exists():
        return None
    try:
        import yaml
        d = yaml.safe_load(f.read_text()) or {}
    except Exception:
        return None
    return d.get("megakernel_authentic")


def contamination(run_dir: Path) -> int:
    """Count distinct OTHER run-archive timestamps referenced in the AGENT transcript.

    The robust, false-positive-free contamination signal: a reference to another
    run's archive (`outputs/runs/<other_ts>`) in the agent's own transcript. A
    specific timestamped archive path only appears if the agent read the shared
    outputs/runs/ archive -- directly, OR via the DEVLOG / recipe-memory, which
    embed those same archive paths (so reading the recipe is caught here too).

    Deliberately scans ONLY transcript.jsonl / codex_session.jsonl -- NOT
    stderr.log or scratch/ (harness orchestration noise), and does NOT string-
    match "DEVLOG.md" or "/memory/" by name: those appear benignly in the
    CLAUDE.md instructions ("Read SPEC.md and DEVLOG.md") and as an agent's own
    per-run scratchpad, neither of which is contamination. Returns 0 = clean.
    """
    self_ts_m = re.match(r"(\d{8}_\d{6})", run_dir.name)
    self_ts = self_ts_m.group(1) if self_ts_m else ""
    seen: set[str] = set()
    for fn in ("transcript.jsonl", "codex_session.jsonl"):
        src = run_dir / fn
        try:
            txt = src.read_text(errors="ignore")
        except Exception:
            continue
        for ts in _RUN_TS.findall(txt):
            if ts != self_ts:
                seen.add(ts)
    return len(seen)


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(here / "outputs" / "runs"))
    ap.add_argument("--out", default=str(here.parents[1] / "public" / "data" / "mega" / "results.csv"))
    ap.add_argument("--runs-html", default=str(here.parents[1] / "public" / "runs"),
                    help="dir of generated transcript viewers (run_id.html); gates the viewer flag")
    args = ap.parse_args()

    html_ids = {p.stem for p in Path(args.runs_html).glob("*.html")} if Path(args.runs_html).exists() else set()
    annotations_dir = here / "results" / "annotations"

    rows = []
    for rj in sorted(Path(args.runs).glob("*/result.json")):
        try:
            d = json.loads(rj.read_text())
        except Exception:
            continue
        if not d.get("has_solution"):
            continue
        run_dir = rj.parent
        if not (run_dir / "gpu").exists():
            continue
        # Contamination tripwire: drop runs that read other runs' archives.
        nc = contamination(run_dir)
        if nc >= 1:
            print(f"  EXCLUDED (contaminated, read {nc} other archive(s)): {run_dir.name}")
            continue
        rid = d.get("run_id", run_dir.name)
        # Megakernel authenticity gate: drop runs the judge audit marked as not a
        # genuine fused megakernel (graph/compile/per-op-loop). None = unjudged, kept.
        if megakernel_authentic(rid, annotations_dir) is False:
            print(f"  EXCLUDED (not a megakernel, judge verdict): {run_dir.name}")
            continue
        gpu = (run_dir / "gpu").read_text().strip()
        tok_s, ctx = _bench(run_dir / "benchmark.log")
        usage = d.get("usage") or {}
        rows.append({
            "gpu": gpu,
            "harness": d.get("harness", ""),
            "model": d.get("model", ""),
            "problem": d.get("problem", ""),
            "correct": "true" if d.get("correct") else "false",
            "score": f"{d.get('peak_fraction'):.3f}" if d.get("peak_fraction") is not None else "",
            "tok_s": tok_s,
            "elapsed_s": d.get("elapsed_seconds", ""),
            "output_tokens": usage.get("output_tokens", "") if usage.get("output_tokens") is not None else "",
            "ctx2048": f"{ctx.get(2048):.2f}" if ctx.get(2048) else "",
            "ctx8192": f"{ctx.get(8192):.2f}" if ctx.get(8192) else "",
            "ctx16384": f"{ctx.get(16384):.2f}" if ctx.get(16384) else "",
            "framework": _framework(run_dir),
            "has_viewer": "true" if rid in html_ids else "false",
            "run_id": rid,
        })

    # keep best (highest score) per (gpu, model, problem)
    best = {}
    for r in rows:
        key = (r["gpu"], r["model"], r["problem"])
        cur = best.get(key)
        if cur is None or (r["score"] and float(r["score"] or 0) > float(cur["score"] or 0)):
            best[key] = r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["gpu", "harness", "model", "problem", "correct", "score", "tok_s",
              "elapsed_s", "output_tokens", "ctx2048", "ctx8192", "ctx16384",
              "framework", "has_viewer", "run_id"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(best.values(), key=lambda x: (x["gpu"], -float(x["score"] or 0))):
            w.writerow(r)
    print(f"wrote {len(best)} rows -> {out}")
    for r in sorted(best.values(), key=lambda x: (x["gpu"], -float(x["score"] or 0))):
        print(f"  {r['gpu']:24s} {r['model']:20s} score={r['score']:>7} tok/s={r['tok_s']:>5} "
              f"fw={r['framework']:8s} viewer={r['has_viewer']}")


if __name__ == "__main__":
    main()
