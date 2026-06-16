import importlib.util
from pathlib import Path


def _load_fp8_check_module():
    path = Path("problems/01_fp8_gemm/check.py")
    spec = importlib.util.spec_from_file_location("fp8_check", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fp8_constraint_accepts_cutlass_fp8_indicator() -> None:
    check = _load_fp8_check_module()
    source = """
from torch.utils.cpp_extension import load_inline
cuda_src = '''
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
'''
"""
    assert check._check_fp8_kernel_constraint(source) == (True, "ok")


def test_fp8_constraint_accepts_triton_dot_path() -> None:
    check = _load_fp8_check_module()
    source = """
import triton
import triton.language as tl

@triton.jit
def kernel(a, b, c):
    acc = tl.dot(a, b)
"""
    assert check._check_fp8_kernel_constraint(source) == (True, "ok")


def test_fp8_constraint_rejects_torch_bf16_dress_up() -> None:
    check = _load_fp8_check_module()
    source = "return x.to(torch.bfloat16) @ self.weight.to(torch.bfloat16).T"
    ok, msg = check._check_fp8_kernel_constraint(source)
    assert not ok
    assert "bf16 GEMM" in msg


def test_fp8_constraint_rejects_bf16_cutlass_gemm() -> None:
    check = _load_fp8_check_module()
    source = """
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
"""
    ok, msg = check._check_fp8_kernel_constraint(source)
    assert not ok
    assert "BF16 CUTLASS" in msg


def test_fp8_tolerance_applies_to_bfloat16_output() -> None:
    check = _load_fp8_check_module()
    meta = {"tolerance": {"fp8_e4m3fn": 0.15}}

    tolerance = check._fp8_output_tolerance_override(meta)

    assert tolerance["fp8_e4m3fn"] == 0.15
    assert tolerance["bfloat16"] == 0.15
