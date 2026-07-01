"""Correctness runner for the grid-foraging PPO training megakernel.

Correctness here is the *learned return level*, not per-element allclose: a
different kernel implementation will never reproduce the reference trajectory
bit-for-bit (RNG stream and float reduction order differ), but a correct one
must implement the same MDP + PPO and learn the task to the same return level.

For each grading seed we train both the reference and the solution from
scratch and check that the solution's final-window mean episodic return lands
in a band around the reference's, and that it actually improved from its own
early baseline (so a pretrained or constant-return cheese fails).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


def _mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    try:
        import reference
        import shapes
        import solution
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    # Scan solution.py AND every local module it imports (recursively), so
    # forbidden ops / the megakernel escape hatches (CUDA graphs, torch.compile)
    # can't hide in a sidecar kernels.py. Harness files are excluded.
    import ast as _ast
    _HARNESS = {"reference", "baseline", "shapes", "check", "benchmark", "problem"}

    def _local_imports(_src):
        _mods = set()
        try:
            _tree = _ast.parse(_src)
        except Exception:
            return _mods
        for _node in _ast.walk(_tree):
            if isinstance(_node, _ast.Import):
                for _n in _node.names:
                    _mods.add(_n.name.split(".")[0])
            elif isinstance(_node, _ast.ImportFrom):
                if _node.module and _node.level == 0:
                    _mods.add(_node.module.split(".")[0])
        return _mods

    sol_src = Path("solution.py").read_text() if Path("solution.py").exists() else ""
    _seen = {"solution"}
    _queue = [
        _m for _m in _local_imports(sol_src)
        if _m not in _HARNESS and Path(_m + ".py").exists()
    ]
    while _queue:
        _m = _queue.pop()
        if _m in _seen:
            continue
        _seen.add(_m)
        try:
            _s = Path(_m + ".py").read_text()
        except Exception:
            continue
        sol_src += "\n" + _s
        for _mm in _local_imports(_s):
            if _mm not in _seen and _mm not in _HARNESS and Path(_mm + ".py").exists():
                _queue.append(_mm)
    for forbidden in meta.get("forbidden", []):
        if re.search(re.escape(forbidden), sol_src):
            print(f"FAIL: forbidden op used: {forbidden}")
            sys.exit(1)

    if not hasattr(solution, "train"):
        print("FAIL: solution.py must expose train(total_env_steps, seed) -> list[float]")
        sys.exit(1)

    band = meta["return_band"]
    fw = int(band["final_window"])
    ew = int(band["early_window"])
    steps = int(shapes.TOTAL_ENV_STEPS)

    for seed in shapes.GRADING_SEEDS:
        try:
            ref_curve = reference.train(steps, seed)
            sol_curve = solution.train(steps, seed)
        except Exception as e:
            print(f"FAIL: seed {seed}: training raised {type(e).__name__}: {e}")
            sys.exit(1)

        if not isinstance(sol_curve, (list, tuple)) or len(sol_curve) < len(ref_curve):
            print(
                f"FAIL: seed {seed}: solution returned {len(sol_curve) if hasattr(sol_curve, '__len__') else '?'} "
                f"iterations, reference ran {len(ref_curve)} (did it run the full step budget?)"
            )
            sys.exit(1)

        ref_final = _mean(ref_curve[-fw:])
        sol_final = _mean(sol_curve[-fw:])
        sol_early = _mean(sol_curve[:ew])

        lo = ref_final * (1.0 - float(band["band_low"]))
        hi = ref_final * (1.0 + float(band["band_high"]))
        need_improve = float(band["min_improvement"]) * ref_final

        if sol_final < lo:
            print(
                f"FAIL: seed {seed}: did not learn -- final return {sol_final:.3f} "
                f"below band floor {lo:.3f} (reference {ref_final:.3f})"
            )
            sys.exit(1)
        if sol_final > hi:
            print(
                f"FAIL: seed {seed}: final return {sol_final:.3f} above band ceiling {hi:.3f} "
                f"(reference {ref_final:.3f}) -- the MDP looks easier than the spec"
            )
            sys.exit(1)
        if sol_final - sol_early < need_improve:
            print(
                f"FAIL: seed {seed}: no learning curve -- final {sol_final:.3f} vs early "
                f"{sol_early:.3f}, need +{need_improve:.3f}"
            )
            sys.exit(1)

        print(
            f"seed {seed}: sol_final={sol_final:.3f} ref_final={ref_final:.3f} "
            f"band=[{lo:.3f},{hi:.3f}] early={sol_early:.3f} OK",
            flush=True,
        )

    _emit_framework_label(sol_src)
    print("PASS")


def _emit_framework_label(code: str) -> None:
    patterns = [
        ("ptx", r"asm\s+volatile|asm\s*\(|mma\.sync|tcgen05\."),
        ("cuda_raw", r"torch\.utils\.cpp_extension\.load_inline|__global__\s+void"),
        ("triton", r"import\s+triton\b|@triton\.jit|\btl\.dot\b"),
        ("cudagraph", r"CUDAGraph|cuda\.graphs|make_graphed_callables|capture_begin"),
        ("compile", r"torch\.compile"),
    ]
    label = "eager"
    for name, pat in patterns:
        if re.search(pat, code):
            label = name
            break
    Path("framework.txt").write_text(label + "\n")


if __name__ == "__main__":
    main()
