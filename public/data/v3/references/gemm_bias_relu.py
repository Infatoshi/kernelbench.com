"""
GEMM + Bias + ReLU fusion target (epilogue fusion workload).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n: int = 4096):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(n, dtype=torch.float16) * 0.02)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        c = a.to(torch.float16) @ b.to(torch.float16)
        return F.relu(c + self.bias)


OP_TYPE = "gemm_epilogue"
SUPPORTED_PRECISIONS = ["fp16", "bf16"]
HARDWARE_REQUIRED = ["H100", "B200"]
SPECIALIZED_LEVEL = 2


def get_inputs():
    m = 2048
    n = 4096
    k = 2048
    return [torch.randn(m, k, dtype=torch.float16), torch.randn(k, n, dtype=torch.float16)]


def get_init_inputs():
    return []
