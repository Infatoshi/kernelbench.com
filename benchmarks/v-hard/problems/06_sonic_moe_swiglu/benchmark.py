"""Roofline benchmark for Sonic-MoE up-projection (grouped GEMM + fused SwiGLU).

For each shape: times eager reference, compiled reference, SOTA (if available),
and the agent's solution. Reports achieved TFLOPS, GB/s, and peak_fraction.

Output lines the harness picks up:
  shape=<idx> variant=<name> tflops=<N> gbps=<N> ms=<N>
  peak_fraction: <N>  (geomean over shapes of solution's peak_fraction)
"""
import sys
from math import exp, log
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.roofline import compute_gbps, compute_tflops, peak_fraction  # noqa: E402
from src.eval.timing import time_fn  # noqa: E402
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
    peak_gbps = hw.peak_bandwidth_gb_s
    regime = meta.get("regime", "compute")
    flops_formula = meta["flops_formula"]
    bytes_formula = meta["bytes_formula"]
    num_perf_trials = int(meta.get("num_perf_trials", 20))

    device = torch.device("cuda:0")

    # Optional SOTA
    try:
        import sota as sota_mod
        has_sota = sota_mod.is_available()
    except Exception:
        has_sota = False

    sol_fractions: list[float] = []

    for shape_idx, shape in enumerate(shapes.SHAPES):
        reference.T_total = shape["T_total"]
        reference.H = shape["H"]
        reference.I = shape["I"]
        reference.E = shape["E"]
        reference.K = shape["K"]

        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        sd = ref_model.state_dict()
        try:
            sol_model.load_state_dict(sd, strict=True)
        except RuntimeError:
            pass

        torch.manual_seed(2026)
        inputs = [t.to(device) for t in reference.get_inputs()]

        flops = _eval_formula(flops_formula, shape)
        bytes_moved = _eval_formula(bytes_formula, shape)

        # Eager (slow Python loop in reference)
        ms_eager = time_fn(ref_model, inputs, iters=max(3, num_perf_trials // 4))

        # Compiled (best-effort)
        try:
            comp = torch.compile(ref_model, mode="reduce-overhead")
            ms_comp = time_fn(comp, inputs, iters=max(3, num_perf_trials // 4))
        except Exception as e:
            print(f"  [compile fallback] {type(e).__name__}: {e}")
            ms_comp = None

        # SOTA (sonic-moe). Wrap in try/except: SM120 path may be unavailable.
        ms_sota = None
        if has_sota:
            try:
                hidden_states, expert_offsets = inputs
                W_gate, W_up = ref_model.W_gate, ref_model.W_up

                def sota_fn(_x=hidden_states, _o=expert_offsets, _g=W_gate, _u=W_up):
                    return sota_mod.sota_forward(_x, _g, _u, _o)

                ms_sota = time_fn(sota_fn, [], iters=num_perf_trials)
            except Exception as e:
                print(f"  [sota unavailable] {type(e).__name__}: {e}")

        # Solution
        ms_sol = time_fn(sol_model, inputs, iters=num_perf_trials)

        for variant, ms in [
            ("eager", ms_eager),
            ("compiled", ms_comp),
            ("sota", ms_sota),
            ("solution", ms_sol),
        ]:
            if ms is None:
                continue
            tflops = compute_tflops(flops, ms)
            gbps = compute_gbps(bytes_moved, ms)
            print(f"shape={shape_idx} variant={variant} tflops={tflops:.3f} gbps={gbps:.3f} ms={ms:.3f}")

        sol_tflops = compute_tflops(flops, ms_sol)
        sol_gbps = compute_gbps(bytes_moved, ms_sol)
        if regime == "compute":
            frac = peak_fraction(sol_tflops, peak_tflops)
        else:
            frac = peak_fraction(sol_gbps, peak_gbps)
        sol_fractions.append(frac)
        print(f"shape={shape_idx} solution_peak_fraction={frac:.4f}")

    gmean = exp(sum(log(max(f, 1e-9)) for f in sol_fractions) / len(sol_fractions))
    print(f"peak_fraction: {gmean:.4f}")
    print(f"RESULT: {'OK' if gmean >= 0.1 else 'LOW'}")


if __name__ == "__main__":
    main()
