"""Correctness runner for Qwen3-0.6B decode-block megakernel."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.correctness import check_correctness  # noqa: E402
from src.eval.numeric_stress import (  # noqa: E402
    numeric_stress_cases,
    numeric_stress_context,
    tolerance_for_case,
)


def main() -> None:
    try:
        import reference
        import shapes
        import solution
    except Exception as e:
        print(f"FAIL: import error: {e}")
        sys.exit(1)

    meta = yaml.safe_load(Path("problem.yaml").read_text())
    sol_src = Path("solution.py").read_text() if Path("solution.py").exists() else ""
    for forbidden in meta.get("forbidden", []):
        if re.search(re.escape(forbidden), sol_src):
            print(f"FAIL: forbidden op used: {forbidden}")
            sys.exit(1)

    device = torch.device("cuda:0")
    tol_override = meta.get("tolerance") or None

    for shape_idx, shape in enumerate(shapes.SHAPES):
        reference.seq_len = int(shape["seq_len"])
        init_args = reference.get_init_inputs()
        ref_model = reference.Model(*init_args).to(device).eval()
        sol_model = solution.Model(*init_args).to(device).eval()
        try:
            sol_model.load_state_dict(ref_model.state_dict(), strict=True)
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
                    ref_out,
                    sol_out,
                    dtype=ref_out.dtype,
                    override=tolerance_for_case(tol_override, case),
                )
                if not ok:
                    print(f"FAIL: shape {shape_idx} {shape} seed {seed} case {case.name}: {msg}")
                    sys.exit(1)

    _emit_framework_label()
    print("PASS")


def _emit_framework_label() -> None:
    patterns = [
        ("ptx", r"asm\s+volatile|asm\s*\(|mma\.sync|tcgen05\."),
        ("cutlass3", r"\bcute::|cutlass/gemm/collective|cutlass::arch::Sm(9|10|12)"),
        ("cutlass2", r"cutlass/gemm/device/gemm|cutlass::gemm::device"),
        ("cuda_wmma", r"\bnvcuda::wmma\b|wmma::fragment"),
        ("triton", r"import\s+triton\b|@triton\.jit|\btl\.dot\b"),
        ("cuda_raw", r"torch\.utils\.cpp_extension\.load_inline|__global__\s+void"),
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
