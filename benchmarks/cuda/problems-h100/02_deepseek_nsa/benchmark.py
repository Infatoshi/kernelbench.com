"""Roofline benchmark for NSA-style sparse attention."""
import sys
from math import exp, log
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.roofline import compute_gbps, compute_tflops, peak_fraction  # noqa: E402
from src.eval.timing import benchmark_baselines_enabled, time_variant  # noqa: E402
from src.hardware import get as get_hw  # noqa: E402


def main():
    import reference
    import shapes
    import solution

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    hw = get_hw(meta["hardware"][0])
    peak_tflops = hw.peak_tflops_dense.get(meta["peak_tflops_key"], 0.0)
    num_perf_trials = int(meta.get("num_perf_trials", 15))
    device = torch.device("cuda:0")
    include_baselines = benchmark_baselines_enabled("02_DEEPSEEK_NSA")
    sol_fractions: list[float] = []

    for shape_idx, shape in enumerate(shapes.SHAPES):
        reference.B = shape["B"]
        reference.H = shape["H"]
        reference.S = shape["S"]
        reference.D = shape["D"]
        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        torch.manual_seed(2026)
        inputs = [t.to(device) for t in reference.get_inputs()]
        B, H, S, D = shape["B"], shape["H"], shape["S"], shape["D"]
        flops = 4.0 * B * H * S * S * D
        bytes_moved = B * H * S * D * 2 * 4
        ms_sol = time_variant(
            sol_model, inputs, shape_idx=shape_idx, variant="solution", iters=num_perf_trials
        )
        tflops = compute_tflops(flops, ms_sol)
        gbps = compute_gbps(bytes_moved, ms_sol)
        print(
            f"shape={shape_idx} variant=solution "
            f"tflops={tflops:.3f} gbps={gbps:.3f} ms={ms_sol:.3f}",
            flush=True,
        )
        if include_baselines and shape["S"] <= 1024:
            ms_e = time_variant(ref_model, inputs, shape_idx=shape_idx, variant="eager", iters=3)
            print(
                f"shape={shape_idx} variant=eager "
                f"tflops={compute_tflops(flops, ms_e):.3f} gbps={compute_gbps(bytes_moved, ms_e):.3f} "
                f"ms={ms_e:.3f}",
                flush=True,
            )
        frac = peak_fraction(tflops, peak_tflops)
        sol_fractions.append(frac)
        print(f"shape={shape_idx} solution_peak_fraction={frac:.4f}")

    gmean = exp(sum(log(max(f, 1e-9)) for f in sol_fractions) / len(sol_fractions))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean >= 0.001 else 'LOW'}")


if __name__ == "__main__":
    main()
