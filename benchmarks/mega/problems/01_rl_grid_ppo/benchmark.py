"""Throughput benchmark for the grid-foraging PPO training megakernel.

Score is training throughput: environment steps per second over a full
from-scratch PPO run, relative to a target ceiling (peak_sps). This is the
RL analog of the roofline problems' peak_fraction.

The solution is timed first and is the only thing that gates scoring. Because a
return curve is unfalsifiable by itself (train() returns floats; nothing proves
env steps happened), three mechanical guards make the cheap cheats score 0
(red-teamed 2026-07-01 on the 3090: a fabricated-curve no-op scored 22,006x and
a (steps,seed)-memoized replay 26,849x before these):

  * every timed trial uses a fresh RANDOM seed -- a memoized (steps, seed)
    lookup misses every trial, so cached replays pay full training cost;
  * the return floor is enforced PER TRIAL, not just on the last curve;
  * SPS above max_credible_sps_multiple * peak_sps scores 0.0 outright -- a
    no-work fabricator lands ~4 orders of magnitude over the ceiling.

Residual (documented, judge-gated): a fabricator that also SLEEPS to fake a
plausible elapsed time defeats mechanical timing entirely. That is caught by
the mandatory authenticity audit -- the judge reads solution.py and a "trainer"
with no environment or policy update fails on sight.
"""
from __future__ import annotations

import random
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
    sps_cap = float(meta.get("max_credible_sps_multiple", 0.0)) * peak_sps
    steps = int(shapes.TOTAL_ENV_STEPS)
    trials = int(meta.get("num_perf_trials", 3))

    # Non-deterministic trial seeds: SPS does not depend on the seed, but a
    # memoized (steps, seed) lookup does -- fresh seeds force real training.
    rng = random.SystemRandom()
    warm_seed = rng.randrange(2**31)

    # Warmup absorbs torch.compile / Triton autotune / cuBLAS workspace alloc.
    _time_run(max(1, steps // 5), warm_seed)

    times = []
    for _ in range(trials):
        trial_seed = rng.randrange(2**31)
        dt, curve = _time_run(steps, trial_seed)
        ret = _final_return(curve)
        print(f"trial seed={trial_seed} sps={steps / dt:,.0f} elapsed={dt:.3f}s "
              f"final_return={ret:.3f}", flush=True)
        if ret < floor:
            print(f"final_return={ret:.3f} below floor {floor:.3f} -- training did not learn")
            print("peak_fraction: 0.0")
            print("RESULT: LOW")
            return
        times.append(dt)

    best_dt = min(times)  # best (least-contended) throughput
    sps = steps / best_dt
    if sps_cap > 0 and sps > sps_cap:
        print(f"steps_per_second={sps:,.0f} exceeds credible cap {sps_cap:,.0f} "
              f"({meta['max_credible_sps_multiple']}x peak_sps) -- not physically "
              "plausible for this task; refusing to score. If a real megakernel "
              "ever hits this, recalibrate peak_sps.")
        print("peak_fraction: 0.0")
        print("RESULT: LOW")
        return
    frac = max(0.0, sps / peak_sps) if peak_sps > 0 else 0.0
    print(f"steps_per_second={sps:,.0f} peak_sps={peak_sps:,.0f}")
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
