"""Latency benchmark for the sort-free top-p mask.

This cell is ms-anchored (standing 2026-07-15 metric rule): the roofline
ceiling is launch-overhead-bound and unreadable, so the headline is per-shape
milliseconds and geomean speedup vs the eager sort-based reference. The eager
variant is timed on every run for the agent flywheel; the PUBLISHED anchor is
frozen at deck publication on the canonical H100.

Output lines the harness picks up:
  shape=<idx> variant=<name> tflops=<N> gbps=<N> ms=<N>
  shape=<idx> speedup_vs_eager=<N>
  geomean_speedup_vs_eager: <N>
  peak_fraction: <N>   (memory-regime gbps fraction; context only, never headline)
"""
import sys
from math import exp, log
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.roofline import compute_gbps, compute_tflops, peak_fraction  # noqa: E402
from src.eval.timing import time_variant  # noqa: E402
from src.hardware import get as get_hw  # noqa: E402


def _eval_formula(expr: str, vars: dict) -> float:
    return float(eval(expr, {"__builtins__": {}}, vars))


def main():
    import reference
    import shapes
    import solution

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    hw = get_hw(meta["hardware"][0])
    peak_gbps = hw.peak_bandwidth_gb_s
    num_perf_trials = int(meta.get("num_perf_trials", 50))

    device = torch.device("cuda:0")
    sol_fractions: list[float] = []
    speedups: list[float] = []

    for shape_idx, shape in enumerate(shapes.SHAPES):
        reference.B = shape["B"]
        reference.V = shape["V"]
        reference.P = shape["P"]

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

        ms_sol = time_variant(
            sol_model, inputs, shape_idx=shape_idx, variant="solution", iters=num_perf_trials
        )
        gbps = compute_gbps(bytes_moved, ms_sol)
        print(
            f"shape={shape_idx} variant=solution "
            f"tflops={compute_tflops(flops, ms_sol):.3f} gbps={gbps:.3f} ms={ms_sol:.4f}",
            flush=True,
        )

        # Eager anchor, timed every run (cheap: one sort + cumsum per call).
        ms_eager = time_variant(
            ref_model, inputs, shape_idx=shape_idx, variant="eager", iters=num_perf_trials
        )
        print(
            f"shape={shape_idx} variant=eager "
            f"tflops={compute_tflops(flops, ms_eager):.3f} "
            f"gbps={compute_gbps(bytes_moved, ms_eager):.3f} ms={ms_eager:.4f}",
            flush=True,
        )
        speedup = ms_eager / ms_sol if ms_sol > 0 else 0.0
        speedups.append(max(speedup, 1e-9))
        print(f"shape={shape_idx} speedup_vs_eager={speedup:.4f}")

        frac = peak_fraction(gbps, peak_gbps)
        sol_fractions.append(frac)
        print(f"shape={shape_idx} solution_peak_fraction={frac:.4f}")

    gmean_speedup = exp(sum(log(s) for s in speedups) / len(speedups))
    print(f"geomean_speedup_vs_eager: {gmean_speedup:.4f}")
    gmean = exp(sum(log(max(f, 1e-9)) for f in sol_fractions) / len(sol_fractions))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean_speedup >= 1.0 else 'LOW'}")


if __name__ == "__main__":
    main()
