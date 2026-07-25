"""Measure and freeze the production-baseline anchor times for speedup problems.

For each problem with `metric: speedup`, times sota.py (never solution.py) on the
canonical node through the exact benchmark path the harness uses, and prints the
per-shape ms to paste into `anchor_ms:` in problem.yaml.

MUST be run on a quiet 4xH100 SXM node with no other CUDA tenants — an anchor
measured under contention permanently inflates every future speedup on that
problem. Same discipline as the sequential isolated re-grade rule.

    uv run python scripts/measure_anchors.py                     # all speedup problems
    uv run python scripts/measure_anchors.py problems-h100x4/07_gemm_allreduce_overlap
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

BENCH_ROOT = Path(__file__).resolve().parents[1]
DECK = BENCH_ROOT / "problems-h100x4"


def preflight_quiet_node() -> None:
    """Refuse to measure on a node with another CUDA tenant.

    An anchor is FROZEN and divides every future speedup on that problem, so one
    contended measurement permanently biases the whole column — worse than a
    contended re-grade, which can at least be redone. Learned the hard way on
    2026-07-25: a first pass measured all three anchors while a co-tenant vLLM
    held 70 GB on GPU0, inflating them by up to ~4x. Override only if you know
    the residency is stale: KBM_ALLOW_BUSY=1.
    """
    if os.environ.get("KBM_ALLOW_BUSY") == "1":
        print("WARNING: KBM_ALLOW_BUSY=1 — anchors may be contended.", flush=True)
        return
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    busy = []
    for line in out.strip().splitlines():
        idx, used = (p.strip() for p in line.split(","))
        if int(used) > 2048:
            busy.append(f"GPU{idx}={used}MiB")
    if busy:
        raise SystemExit(
            "REFUSING to measure anchors: another CUDA tenant on " + ", ".join(busy) + ".\n"
            "Anchors are frozen and divide every future speedup — measure only on a "
            "quiet node. Wait for the GPUs to free, or set KBM_ALLOW_BUSY=1 if the "
            "residency is stale."
        )


def _speedup_problems() -> list[Path]:
    out = []
    for d in sorted(DECK.iterdir()):
        meta_path = d / "problem.yaml"
        if not meta_path.is_file():
            continue
        meta = yaml.safe_load(meta_path.read_text())
        if meta.get("metric") == "speedup":
            out.append(d)
    return out


def measure(problem_dir: Path) -> list[float]:
    print(f"\n=== {problem_dir.name}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "from src.eval.launcher import run_anchor;"
         "from pathlib import Path;"
         "sys.exit(run_anchor(Path(%r)))" % (str(BENCH_ROOT), str(problem_dir))],
        capture_output=True, text=True,
    )
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"anchor measurement failed for {problem_dir.name}")
    times = []
    for line in proc.stdout.splitlines():
        if "variant=anchor" in line:
            times.append(float(line.split("ms=")[1].split()[0]))
    return times


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repeats = int(os.environ.get("KBM_ANCHOR_REPEATS", "5"))
    preflight_quiet_node()
    targets = [Path(a).resolve() for a in args] or _speedup_problems()
    for d in targets:
        # Median of N passes, not a single shot. Latency-bound shapes are not
        # jittery per-iteration (100 timed iters already average that out) — they
        # vary run to RUN, from NCCL channel setup and CPU-side syncs, so a
        # bigger iter count does not help and only more processes do. Measured
        # 2026-07-25: 09's 128-token shape spans 28% across five passes while its
        # 8192-token shape spans 1.6%. A single shot there would bias every
        # future speedup on that shape by up to ~15%.
        passes = [measure(d) for _ in range(repeats)]
        n = len(passes[0])
        if any(len(p) != n for p in passes):
            raise SystemExit(f"inconsistent shape count across passes for {d.name}")
        med, spreads = [], []
        for i in range(n):
            col = sorted(p[i] for p in passes)
            mid = col[len(col) // 2] if len(col) % 2 else 0.5 * (col[len(col) // 2 - 1] + col[len(col) // 2])
            med.append(mid)
            spreads.append((col[-1] - col[0]) / col[0] if col[0] > 0 else 0.0)
        print(f"anchor_ms: [{', '.join(f'{t:.4f}' for t in med)}]   # {d.name} "
              f"(median of {repeats})")
        print(f"#   per-shape spread over {repeats} passes: "
              f"{', '.join(f'{s * 100:.1f}%' for s in spreads)}")
        for i, s in enumerate(spreads):
            if s > 0.10:
                print(f"#   NOTE shape {i} spread {s * 100:.0f}% — latency-bound; the "
                      "median is the honest anchor but expect that much noise in "
                      "solution timings on this shape too.")
    print("\nPaste each anchor_ms line into the matching problem.yaml, then commit. "
          "Anchors are FROZEN once published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
