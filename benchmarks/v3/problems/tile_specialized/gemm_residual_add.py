"""
GEMM + residual add fusion target.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return (a.to(torch.float16) @ b.to(torch.float16)) + residual.to(torch.float16)


OP_TYPE = "gemm_epilogue"
SUPPORTED_PRECISIONS = ["fp16", "bf16"]
HARDWARE_REQUIRED = ["H100", "B200"]
SPECIALIZED_LEVEL = 2


def get_inputs():
    m = 2048
    n = 4096
    k = 2048
    a = torch.randn(m, k, dtype=torch.float16)
    b = torch.randn(k, n, dtype=torch.float16)
    residual = torch.randn(m, n, dtype=torch.float16)
    return [a, b, residual]


def get_init_inputs():
    return []
