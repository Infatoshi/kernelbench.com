"""Roofline benchmark for Qwen3-0.6B decode-block megakernel."""
from __future__ import annotations

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


def _eval_formula(expr: str, vars_: dict) -> float:
    return float(eval(expr, {"__builtins__": {}}, vars_))


def main() -> None:
    import reference
    import shapes
    import solution

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    hw = get_hw(meta["hardware"][0])
    peak_tflops = hw.peak_tflops_dense.get(meta["peak_tflops_key"], 0.0)
    peak_gbps = hw.peak_bandwidth_gb_s
    regime = meta.get("regime", "memory")
    num_perf_trials = int(meta.get("num_perf_trials", 20))
    include_baselines = benchmark_baselines_enabled("01_QWEN3_DECODE_BLOCK")
    device = torch.device("cuda:0")
    sol_fractions: list[float] = []

    for shape_idx, shape in enumerate(shapes.SHAPES):
        reference.seq_len = int(shape["seq_len"])
        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        try:
            sol_model.load_state_dict(ref_model.state_dict(), strict=True)
        except RuntimeError:
            pass
        torch.manual_seed(2026)
        inputs = [t.to(device) for t in reference.get_inputs()]
        flops = _eval_formula(meta["flops_formula"], shape)
        bytes_moved = _eval_formula(meta["bytes_formula"], shape)
        # Solution first. Baselines are optional diagnostics and must not block scoring.
        ms_sol = time_variant(sol_model, inputs, shape_idx=shape_idx, variant="solution", iters=num_perf_trials)
        sol_tflops = compute_tflops(flops, ms_sol)
        sol_gbps = compute_gbps(bytes_moved, ms_sol)
        print(f"shape={shape_idx} variant=solution tflops={sol_tflops:.3f} gbps={sol_gbps:.3f} ms={ms_sol:.3f}", flush=True)
        # Optional torch.compile baseline marker for harness consistency.
        if include_baselines:
            ms_eager = time_variant(ref_model, inputs, shape_idx=shape_idx, variant="eager", iters=num_perf_trials)
            print(
                f"shape={shape_idx} variant=eager "
                f"tflops={compute_tflops(flops, ms_eager):.3f} gbps={compute_gbps(bytes_moved, ms_eager):.3f} ms={ms_eager:.3f}",
                flush=True,
            )
        frac = peak_fraction(sol_gbps if regime == "memory" else sol_tflops, peak_gbps if regime == "memory" else peak_tflops)
        sol_fractions.append(frac)
        print(f"shape={shape_idx} solution_peak_fraction={frac:.4f}")

    gmean = exp(sum(log(max(f, 1e-9)) for f in sol_fractions) / len(sol_fractions))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean >= 0.01 else 'LOW'}")


if __name__ == "__main__":
    main()
