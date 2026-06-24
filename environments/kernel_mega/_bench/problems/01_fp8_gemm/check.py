"""Correctness runner for FP8 GEMM.

Runs solution.Model vs reference.Model across all shapes in shapes.py, 3 seeds
each, with per-dtype atol/rtol. Also rejects forbidden ops by grep.
"""
import re
import sys
from pathlib import Path

import torch
import yaml

# Make the repo's src/ importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.correctness import check_correctness  # noqa: E402
from src.eval.numeric_stress import (  # noqa: E402
    numeric_stress_cases,
    numeric_stress_context,
    tolerance_for_case,
)


def main():
    try:
        import reference
        import shapes
        import solution
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    problem_yaml = Path("problem.yaml")
    meta = yaml.safe_load(problem_yaml.read_text()) if problem_yaml.exists() else {}

    # --- Forbidden-op check ------------------------------------------------
    sol_src = Path("solution.py").read_text() if Path("solution.py").exists() else ""
    for forbidden in meta.get("forbidden", []):
        pat = re.escape(forbidden)
        if re.search(pat, sol_src):
            print(f"FAIL: forbidden op used: {forbidden}")
            sys.exit(1)

    ok, msg = _check_fp8_kernel_constraint(sol_src)
    if not ok:
        print(f"FAIL: fp8 kernel constraint: {msg}")
        sys.exit(1)

    device = torch.device("cuda:0")
    tol_override = _fp8_output_tolerance_override(meta)

    # --- Per-shape correctness --------------------------------------------
    all_shapes = shapes.SHAPES
    for shape_idx, shape in enumerate(all_shapes):
        # Rebuild reference module's module-level M/N/K shims so get_inputs /
        # get_init_inputs match this shape.
        reference.M = shape["M"]
        reference.N = shape["N"]
        reference.K = shape["K"]

        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()

        # Share weights. strict=True — if sol_model doesn't declare the same
        # parameters, correctness fails (this closes the "identity kernel"
        # cheat class).
        sd = ref_model.state_dict()
        try:
            sol_model.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"FAIL: state_dict mismatch at shape {shape_idx} ({shape}): {e}")
            sys.exit(1)

        for seed in (42, 123, 456):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            base_inputs = [t.to(device) for t in reference.get_inputs()]

            for case in numeric_stress_cases(meta.get("name", "")):
                with numeric_stress_context(ref_model, sol_model, base_inputs, case) as inputs:
                    with torch.no_grad():
                        ref_out = ref_model(*inputs)
                        sol_out = sol_model(*inputs)

                ok, msg = check_correctness(
                    ref_out, sol_out,
                    dtype=ref_out.dtype,
                    override=tolerance_for_case(tol_override, case),
                )
                if not ok:
                    print(f"FAIL: shape {shape_idx} {shape} seed {seed} case {case.name}: {msg}")
                    sys.exit(1)

    # --- Framework label (for stats) --------------------------------------
    _emit_framework_label()
    print("PASS")


def _fp8_output_tolerance_override(meta: dict) -> dict | None:
    """Apply declared fp8 tolerance to this problem's bf16 output.

    `problem.yaml` describes operand precision as fp8, but `check_correctness`
    looks up tolerance by the reference output dtype. This GEMM returns bf16, so
    the fp8_e4m3fn tolerance must also be available under bfloat16. Numeric
    stress cases can still override bfloat16 for their scale regime.
    """
    tolerance = meta.get("tolerance")
    if not isinstance(tolerance, dict):
        return None
    tolerance = dict(tolerance)
    tolerance.setdefault("bfloat16", tolerance.get("fp8_e4m3fn"))
    return tolerance


_FP8_KERNEL_PATTERNS = [
    r"__nv_fp8_e4m3",
    r"\bCUDA_R_8F_E4M3\b",
    r"\bfloat_e4m3(?:_t)?\b",
    r"\be4m3\b.*\bmma\b|\bmma\b.*\be4m3\b",
    r"\be4m3\b.*\btcgen05\b|\btcgen05\b.*\be4m3\b",
    r"@triton\.jit\b(?s:.)*\btl\.dot\b",
]

_BF16_DRESS_UP_PATTERNS = [
    ("fp8 operand cast to torch.bfloat16", r"\.to\s*\(\s*(?:dtype\s*=\s*)?torch\.bfloat16\s*\)"),
    ("fp8 operand cast with .bfloat16()", r"\.bfloat16\s*\("),
    ("BF16 CUDA storage or math type", r"\b__nv_bfloat16\b|\bnv_bfloat16\b|\bCUDA_R_16BF\b"),
    ("BF16 CUTLASS GEMM type", r"\bcutlass::bfloat16_t\b"),
    ("BF16 WMMA fragment", r"\bwmma::precision::bfloat16\b"),
]


def _check_fp8_kernel_constraint(solution_source: str) -> tuple[bool, str]:
    """Reject bf16-dressed GEMM solutions for the fp8 problem.

    The reference uses a bf16 matmul only as a correctness oracle. A submitted
    kernel must expose an FP8 execution path rather than upcasting the fp8
    operands and running a BF16 GEMM.
    """
    for label, pattern in _BF16_DRESS_UP_PATTERNS:
        if re.search(pattern, solution_source, flags=re.IGNORECASE | re.MULTILINE):
            return False, f"{label}; use fp8 tensor-core operands instead of bf16 GEMM"

    if not any(
        re.search(pattern, solution_source, flags=re.IGNORECASE | re.MULTILINE)
        for pattern in _FP8_KERNEL_PATTERNS
    ):
        return (
            False,
            "no FP8 kernel indicator found; expected an e4m3/tcgen05/CUTLASS FP8 "
            "path or a Triton tl.dot path over fp8 inputs",
        )

    return True, "ok"


def _emit_framework_label():
    """Write framework.txt with the detected kernel framework."""
    patterns = [
        ("ptx",       r"asm\s+volatile|asm\s*\(|mma\.sync|tcgen05\."),
        ("cutlass3",  r"\bcute::|cutlass/gemm/collective|cutlass::arch::Sm(9|10|12)"),
        ("cutlass2",  r"cutlass/gemm/device/gemm|cutlass::gemm::device"),
        ("cuda_wmma", r"\bnvcuda::wmma\b|wmma::fragment"),
        ("triton",    r"import\s+triton\b|@triton\.jit|\btl\.dot\b"),
        ("cuda_raw",  r"torch\.utils\.cpp_extension\.load_inline|__global__\s+void"),
    ]
    sol = Path("solution.py")
    if not sol.exists():
        return
    code = sol.read_text()
    label = "unknown"
    for name, pat in patterns:
        if re.search(pat, code):
            label = name
            break
    Path("framework.txt").write_text(label + "\n")


if __name__ == "__main__":
    main()
