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
