"""SPS benchmark for grid + 3×MinGRU(h=256).

Times solution.run over the shape sweep. Score = geomean(achieved_sps / peak_sps).
Fusion is optional — multi-launch CUDA can still score; fused usually wins.
"""
import sys
import time
from math import exp, log
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    import reference
    import shapes
    import solution

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    peak_sps = float(meta.get("peak_sps", 5e7))
    num_perf_trials = int(meta.get("num_perf_trials", 3))

    if not hasattr(solution, "run"):
        print("FAIL: solution.py must define run(num_envs, horizon, seed)")
        sys.exit(1)

    ref_model = reference.Model().cuda()
    sol_model = solution.Model().cuda() if hasattr(solution, "Model") else None
    if sol_model is not None:
        sol_model.load_state_dict(ref_model.state_dict(), strict=True)

    fractions: list[float] = []

    for shape_idx, shape in enumerate(shapes.SHAPES):
        n = int(shape["num_envs"])
        h = int(shape["horizon"])
        seed = 2026 + shape_idx

        # Warmup
        try:
            solution.run(min(n, 1024), min(h, 4), seed, model=sol_model)
        except TypeError:
            solution.run(min(n, 1024), min(h, 4), seed)
        torch.cuda.synchronize()

        times = []
        for trial in range(num_perf_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                solution.run(n, h, seed + trial, model=sol_model)
            except TypeError:
                solution.run(n, h, seed + trial)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        # Median wall time
        times.sort()
        wall = times[len(times) // 2]
        steps = n * h
        sps = steps / max(wall, 1e-9)
        frac = sps / peak_sps
        fractions.append(frac)
        print(
            f"shape={shape_idx} variant=solution "
            f"num_envs={n} horizon={h} sps={sps:.3f} "
            f"wall_s={wall:.4f} peak_fraction={frac:.4f}",
            flush=True,
        )

        # Optional reference floor (once, smaller)
        if shape_idx == 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reference.run(min(n, 2048), min(h, 16), seed, model=ref_model)
            torch.cuda.synchronize()
            ref_wall = time.perf_counter() - t0
            ref_steps = min(n, 2048) * min(h, 16)
            ref_sps = ref_steps / max(ref_wall, 1e-9)
            print(
                f"shape={shape_idx} variant=eager_ref "
                f"sps={ref_sps:.3f} wall_s={ref_wall:.4f}",
                flush=True,
            )

    gmean = exp(sum(log(max(f, 1e-12)) for f in fractions) / len(fractions))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean >= 0.001 else 'LOW'}")


if __name__ == "__main__":
    main()
