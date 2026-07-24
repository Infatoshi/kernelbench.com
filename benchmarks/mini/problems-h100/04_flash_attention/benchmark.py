"""Roofline benchmark for causal flash attention.

Times the agent's solution on the full deck. Eager reference is opt-in
(KBH_BENCHMARK_BASELINES=1) and only runs at S<=4096 — it materializes O(S^2)
scores. The SDPA sota variant runs at every shape when baselines are enabled.

Output lines the harness picks up:
  shape=<idx> variant=<name> tflops=<N> gbps=<N> ms=<N>
  peak_fraction: <N>  (geomean over shapes; compute regime -> tflops / peak_bf16)
"""
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


def _eval_formula(expr: str, vars: dict) -> float:
    return float(eval(expr, {"__builtins__": {}}, vars))


def main():
    import reference
    import shapes
    import solution

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    hw = get_hw(meta["hardware"][0])
    peak_tflops = hw.peak_tflops_dense.get(meta["peak_tflops_key"], 0.0)
    num_perf_trials = int(meta.get("num_perf_trials", 15))

    device = torch.device("cuda:0")
    include_baselines = benchmark_baselines_enabled("04_FLASH_ATTENTION")

    has_sota = False
    if include_baselines:
        try:
            import sota as sota_mod

            has_sota = sota_mod.is_available()
        except Exception:
            has_sota = False

    sol_fractions: list[float] = []

    for shape_idx, shape in enumerate(shapes.SHAPES):
        reference.B = shape["B"]
        reference.H = shape["H"]
        reference.S = shape["S"]
        reference.D = shape["D"]

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
        tflops = compute_tflops(flops, ms_sol)
        gbps = compute_gbps(bytes_moved, ms_sol)
        print(
            f"shape={shape_idx} variant=solution "
            f"tflops={tflops:.3f} gbps={gbps:.3f} ms={ms_sol:.3f}",
            flush=True,
        )

        if include_baselines:
            if shape["S"] <= 4096:
                ms_e = time_variant(
                    ref_model, inputs, shape_idx=shape_idx, variant="eager", iters=3
                )
                print(
                    f"shape={shape_idx} variant=eager "
                    f"tflops={compute_tflops(flops, ms_e):.3f} "
                    f"gbps={compute_gbps(bytes_moved, ms_e):.3f} ms={ms_e:.3f}",
                    flush=True,
                )
            if has_sota:
                try:
                    def sota_fn(q, k, v):
                        return sota_mod.sota_forward(q, k, v)

                    ms_sota = time_variant(
                        sota_fn, inputs, shape_idx=shape_idx, variant="sota",
                        iters=num_perf_trials,
                    )
                    print(
                        f"shape={shape_idx} variant=sota "
                        f"tflops={compute_tflops(flops, ms_sota):.3f} "
                        f"gbps={compute_gbps(bytes_moved, ms_sota):.3f} ms={ms_sota:.3f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [sota unavailable] {type(e).__name__}: {e}")

        frac = peak_fraction(tflops, peak_tflops)
        sol_fractions.append(frac)
        print(f"shape={shape_idx} solution_peak_fraction={frac:.4f}")

    gmean = exp(sum(log(max(f, 1e-9)) for f in sol_fractions) / len(sol_fractions))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean >= 0.01 else 'LOW'}")


if __name__ == "__main__":
    main()
