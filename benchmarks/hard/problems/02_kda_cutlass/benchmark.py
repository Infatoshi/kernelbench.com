"""Roofline benchmark for KDA forward (chunk form).

For each shape: times the agent's solution first and reports achieved TFLOPS,
GB/s, and peak_fraction. Optional diagnostics for eager reference, compiled
reference, and SOTA can be enabled with KBH_KDA_BENCHMARK_BASELINES=1.

Output lines the harness picks up:
  shape=<idx> variant=<name> tflops=<N> gbps=<N> ms=<N>
  peak_fraction: <N>  (geomean over shapes of solution's peak_fraction)
"""
import os
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


def _apply_shape(reference, shape):
    for k, v in shape.items():
        setattr(reference, k, v)


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

    include_baselines = os.environ.get("KBH_KDA_BENCHMARK_BASELINES") == "1"

    # Optional SOTA diagnostics.
    has_sota = False
    if include_baselines:
        try:
            import sota as sota_mod

            has_sota = sota_mod.is_available()
        except Exception:
            has_sota = False

    sol_fractions: list[float] = []

    for shape_idx, shape in enumerate(shapes.SHAPES):
        _apply_shape(reference, shape)

        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        sd = ref_model.state_dict()
        try:
            sol_model.load_state_dict(sd, strict=True)
        except RuntimeError:
            pass

        torch.manual_seed(2026)
        inputs = [t.to(device) if hasattr(t, "to") else t for t in reference.get_inputs()]

        # Theoretical work per call
        flops = _eval_formula(flops_formula, shape)
        bytes_moved = _eval_formula(bytes_formula, shape)

        # Solution first. The reference compile path is diagnostic and can be
        # much slower than the submitted kernel, so it must not block scoring.
        ms_sol = time_fn(sol_model, inputs, iters=num_perf_trials)
        sol_tflops = compute_tflops(flops, ms_sol)
        sol_gbps = compute_gbps(bytes_moved, ms_sol)
        print(
            f"shape={shape_idx} variant=solution "
            f"tflops={sol_tflops:.3f} gbps={sol_gbps:.3f} ms={ms_sol:.3f}",
            flush=True,
        )
        if regime == "compute":
            frac = peak_fraction(sol_tflops, peak_tflops)
        else:
            frac = peak_fraction(sol_gbps, peak_gbps)
        sol_fractions.append(frac)
        print(f"shape={shape_idx} solution_peak_fraction={frac:.4f}", flush=True)

        if not include_baselines:
            continue

        # Eager diagnostic.
        ms_eager = time_fn(ref_model, inputs, iters=num_perf_trials)

        # Compiled diagnostic (best-effort -- the chunk-form recurrence often defeats inductor)
        try:
            comp = torch.compile(ref_model, mode="reduce-overhead")
            ms_comp = time_fn(comp, inputs, iters=num_perf_trials)
        except Exception as e:
            print(f"  [compile fallback] {type(e).__name__}: {e}", flush=True)
            ms_comp = None

        # SOTA diagnostic.
        ms_sota = None
        if has_sota:
            try:
                scale = float(shape["K"]) ** -0.5

                def sota_fn(q, k, v, g, beta, _scale=scale):
                    return sota_mod.sota_forward(q, k, v, g, beta, scale=_scale)

                ms_sota = time_fn(sota_fn, inputs, iters=num_perf_trials)
            except Exception as e:
                print(f"  [sota unavailable] {type(e).__name__}: {e}", flush=True)

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

    gmean = exp(sum(log(max(f, 1e-9)) for f in sol_fractions) / len(sol_fractions))
    print(f"peak_fraction: {gmean:.4f}", flush=True)
    print(f"RESULT: {'OK' if gmean >= 0.1 else 'LOW'}", flush=True)


if __name__ == "__main__":
    main()
