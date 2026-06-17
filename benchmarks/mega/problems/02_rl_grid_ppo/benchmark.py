"""Throughput benchmark for the grid-foraging PPO training megakernel.

Score is training throughput: environment steps per second over a full
from-scratch PPO run, relative to a target ceiling (peak_sps). This is the
RL analog of the roofline problems' peak_fraction.

The solution is timed first and is the only thing that gates scoring. A
benchmark-side return floor makes a fast no-op loop fail: SPS is only credited
when the training actually learned (so a path that detects benchmark.py and
skips the work cannot score).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml


def _final_return(curve, window=5) -> float:
    tail = list(curve)[-window:]
    return sum(tail) / len(tail) if tail else 0.0


def _time_run(steps: int, seed: int):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    curve = __import__("solution").train(steps, seed)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return dt, curve


def main() -> None:
    import shapes
    import solution  # noqa: F401  (imported lazily in _time_run too)

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    peak_sps = float(meta["peak_sps"])
    floor = float(meta.get("benchmark_return_floor", 0.0))
    steps = int(shapes.TOTAL_ENV_STEPS)
    seed = int(shapes.BENCH_SEED)
    trials = int(meta.get("num_perf_trials", 3))

    # Warmup absorbs torch.compile / Triton autotune / cuBLAS workspace alloc.
    _time_run(max(1, steps // 5), seed)

    times = []
    last_curve = None
    for _ in range(trials):
        dt, last_curve = _time_run(steps, seed)
        times.append(dt)
        print(f"trial sps={steps / dt:,.0f} elapsed={dt:.3f}s", flush=True)

    final_ret = _final_return(last_curve)
    if final_ret < floor:
        print(f"final_return={final_ret:.3f} below floor {floor:.3f} -- training did not learn")
        print("peak_fraction: 0.0")
        print("RESULT: LOW")
        return

    best_dt = min(times)  # best (least-contended) throughput
    sps = steps / best_dt
    frac = max(0.0, sps / peak_sps) if peak_sps > 0 else 0.0
    print(f"steps_per_second={sps:,.0f} final_return={final_ret:.3f} peak_sps={peak_sps:,.0f}")
    print(f"peak_fraction: {frac:.4f}")
    print(f"RESULT: {'OK' if frac >= 0.01 else 'LOW'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"benchmark error: {type(e).__name__}: {e}")
        print("peak_fraction: 0.0")
        print("RESULT: LOW")
        sys.exit(1)
