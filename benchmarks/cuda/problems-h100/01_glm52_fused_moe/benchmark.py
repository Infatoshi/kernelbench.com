"""Roofline benchmark for GLM-5.2 fused MoE."""
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
    peak_gbps = hw.peak_bandwidth_gb_s
    regime = meta.get("regime", "compute")
    num_perf_trials = int(meta.get("num_perf_trials", 15))
    device = torch.device("cuda:0")
    include_baselines = benchmark_baselines_enabled("01_GLM52_FUSED_MOE")
    sol_fractions: list[float] = []

    for shape_idx, shape in enumerate(shapes.SHAPES):
        reference.T = shape["T"]
        reference.E = shape["E"]
        reference.top_k = shape["top_k"]
        reference.n_shared = shape["n_shared"]
        reference.H = shape["H"]
        reference.I = shape["I"]

        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        try:
            sol_model.load_state_dict(ref_model.state_dict(), strict=True)
        except RuntimeError:
            pass

        torch.manual_seed(2026)
        inputs = [t.to(device) for t in reference.get_inputs()]
        T = shape["T"]
        top_k = shape["top_k"]
        n_shared = shape["n_shared"]
        H, I, E = shape["H"], shape["I"], shape["E"]
        flops = float(T * (n_shared + top_k) * (4 * H * I + 2 * I * H))
        bytes_moved = float(T * H * 2 * 2 + (E + n_shared) * (2 * I * H + H * I) * 2)

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
        if include_baselines and shape["T"] <= 256:
            ms_e = time_variant(ref_model, inputs, shape_idx=shape_idx, variant="eager", iters=3)
            print(
                f"shape={shape_idx} variant=eager "
                f"tflops={compute_tflops(flops, ms_e):.3f} "
                f"gbps={compute_gbps(bytes_moved, ms_e):.3f} ms={ms_e:.3f}",
                flush=True,
            )
        frac = (
            peak_fraction(tflops, peak_tflops)
            if regime == "compute"
            else peak_fraction(gbps, peak_gbps)
        )
        sol_fractions.append(frac)
        print(f"shape={shape_idx} solution_peak_fraction={frac:.4f}")

    gmean = exp(sum(log(max(f, 1e-9)) for f in sol_fractions) / len(sol_fractions))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean >= 0.01 else 'LOW'}")


if __name__ == "__main__":
    main()
