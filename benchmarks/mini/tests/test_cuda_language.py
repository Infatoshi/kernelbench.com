"""Unit tests for the CUDA-only language gate."""
from src.eval.cuda_language import check_cuda_language, detect_framework


def test_triton_fails():
    src = "import triton\nimport triton.language as tl\n@triton.jit\ndef k():\n    pass\n"
    ok, msgs, rep = check_cuda_language(src)
    assert not ok
    assert rep["triton_cheat"]
    assert any("Triton" in m for m in msgs)


def test_pytorch_only_fails():
    src = "import torch\nimport torch.nn.functional as F\ndef f(x):\n    return F.softmax(x, dim=-1)\n"
    ok, msgs, rep = check_cuda_language(src)
    assert not ok
    assert not rep["has_cuda_evidence"]


def test_load_inline_passes():
    src = '''
import torch
from torch.utils.cpp_extension import load_inline
src = r"""
#include <cuda_runtime.h>
__global__ void k(float* x) { x[threadIdx.x] = 1.f; }
"""
mod = load_inline(name="k", cpp_sources="", cuda_sources=src, functions=[])
'''
    ok, msgs, rep = check_cuda_language(src)
    assert ok, msgs
    assert rep["has_cuda_evidence"]
    assert detect_framework(src) == "cuda_raw"


def test_dsl_fails():
    src = "from cutlass.cute import something\n__global__ void k() {}\n"
    # has cuda evidence but DSL still fails
    ok, msgs, rep = check_cuda_language(src)
    assert not ok
    assert rep["dsl_cheat"]


def test_dead_cuda_next_to_triton_is_labelled_compound():
    """A solution carrying an unused load_inline/WMMA extension beside the
    Triton kernel it actually runs must not be reported as pure CUDA.
    Caught in the wild on the codex gpt-5.6-sol 01_dequant_gemv calibration
    cell (2026-07-24), where framework.txt read `cuda_wmma` for a kernel
    whose executed path was `tl.dot`."""
    src = (
        "import triton\nimport triton.language as tl\n"
        "_ext = load_inline(cuda_sources='using nvcuda::wmma; wmma::fragment<> f;')\n"
        "@triton.jit\ndef k(): tl.dot(a, b)\n"
    )
    label = detect_framework(src)
    assert "triton" in label
    assert "cuda_wmma" in label
    assert label != "cuda_wmma"


def test_single_framework_label_stays_scalar():
    assert detect_framework("@triton.jit\ndef k(): tl.dot(a,b)\n") == "triton"
    assert detect_framework("out = torch.nn.functional.silu(x)\n") == "pytorch_only"
