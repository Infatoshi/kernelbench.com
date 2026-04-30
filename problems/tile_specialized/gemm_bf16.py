"""
BF16 GEMM baseline for tensor-core-optimized matmul paths.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.to(torch.bfloat16) @ b.to(torch.bfloat16)


OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX3090", "A100", "H100", "B200"]
SPECIALIZED_LEVEL = 1


def get_inputs():
    m = 2048
    n = 2048
    k = 2048
    a = torch.randn(m, k, dtype=torch.bfloat16)
    b = torch.randn(k, n, dtype=torch.bfloat16)
    return [a, b]


def get_init_inputs():
    return []
