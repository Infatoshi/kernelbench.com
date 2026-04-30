"""
Mixed-precision GEMM: FP8-like activations with FP16 weights/accumulation.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a_q: torch.Tensor, b_fp16: torch.Tensor, scale_a: torch.Tensor) -> torch.Tensor:
        a_fp16 = (a_q.float() * scale_a.float()).to(torch.float16)
        return a_fp16 @ b_fp16.to(torch.float16)


OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ["fp8", "fp16"]
HARDWARE_REQUIRED = ["H100", "B200"]
SPECIALIZED_LEVEL = 1


def get_inputs():
    m = 2048
    n = 2048
    k = 2048
    a_q = torch.randint(-127, 127, (m, k), dtype=torch.int8)
    b_fp16 = torch.randn(k, n, dtype=torch.float16)
    scale_a = torch.tensor(0.01, dtype=torch.float32)
    return [a_q, b_fp16, scale_a]


def get_init_inputs():
    return []
