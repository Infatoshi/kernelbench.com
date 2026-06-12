"""Roofline benchmark for multi-axis RoPE pre-attention prep.

For each shape: times the agent's solution first. Optional eager, compiled,
and SOTA diagnostics are enabled with KBH_BENCHMARK_BASELINES=1. Reports achieved TFLOPS, GB/s, and peak_fraction.

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
from src.eval.timing import benchmark_baselines_enabled, time_variant  # noqa: E402
from src.hardware import get as get_hw  # noqa: E402


def _eval_formula(expr: str, vars: dict) -> float:
    # Very small eval: only names from `vars` are valid.
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
    num_perf_trials = int(meta.get("num_perf_trials", 30))

    device = torch.device("cuda:0")
    include_baselines = benchmark_baselines_enabled("09_FMHA_PREATTN_MROPE")

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
        reference.T = shape["T"]
        reference.Hq = shape["Hq"]
        reference.Hkv = shape["Hkv"]
        reference.D = shape["D"]
        reference.MROPE_SECTION = tuple(shape["mrope_section"])
        reference.MAX_POS = shape["max_pos"]

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

        # Theoretical work per call
        flops = _eval_formula(flops_formula, shape)
        bytes_moved = _eval_formula(bytes_formula, shape)

        # Solution first. Baselines are diagnostics and stay opt-in so they
        # cannot block scoring.
        ms_sol = time_variant(
            sol_model,
            inputs,
            shape_idx=shape_idx,
            variant="solution",
            iters=num_perf_trials,
        )
        tflops = compute_tflops(flops, ms_sol)
        gbps = compute_gbps(bytes_moved, ms_sol)
        print(
            f"shape={shape_idx} variant=solution "
            f"tflops={tflops:.3f} gbps={gbps:.3f} ms={ms_sol:.3f}",
            flush=True,
        )

        if include_baselines:
            ms_eager = time_variant(ref_model, inputs, shape_idx=shape_idx, variant="eager", iters=num_perf_trials)

            # Compiled (best-effort)
            try:
                comp = torch.compile(ref_model, mode="reduce-overhead")
                ms_comp = time_variant(comp, inputs, shape_idx=shape_idx, variant="compiled", iters=num_perf_trials)
            except Exception as e:
                print(f"  [compile fallback] {type(e).__name__}: {e}")
                ms_comp = None

            # SOTA
            ms_sota = None
            if has_sota:
                try:
                    q_in, k_in, pt, ph, pw = inputs
                    cc = ref_model.cos_cache
                    sc = ref_model.sin_cache
                    msec = ref_model.mrope_section

                    def sota_fn(_q=q_in, _k=k_in, _pt=pt, _ph=ph, _pw=pw,
                                _cc=cc, _sc=sc, _ms=msec):
                        return sota_mod.sota_forward(_q, _k, _pt, _ph, _pw, _cc, _sc, _ms)

                    ms_sota = time_variant(sota_fn, [], shape_idx=shape_idx, variant="sota", iters=num_perf_trials)
                except Exception as e:
                    print(f"  [sota unavailable] {type(e).__name__}: {e}")

            for variant, ms in [
                ("eager", ms_eager),
                ("compiled", ms_comp),
                ("sota", ms_sota),
            ]:
                if ms is None:
                    continue
                tflops = compute_tflops(flops, ms)
                gbps = compute_gbps(bytes_moved, ms)
                print(
                    f"shape={shape_idx} variant={variant} "
                    f"tflops={tflops:.3f} gbps={gbps:.3f} ms={ms:.3f}",
                    flush=True,
                )

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
