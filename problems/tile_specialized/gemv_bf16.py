"""
BF16 GEMV for inference decoding workloads.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features: int = 4096, out_features: int = 14336):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16)
        return x_bf16 @ self.weight.t()


OP_TYPE = "gemv"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX3090", "A100", "H100", "B200"]
SPECIALIZED_LEVEL = 1


def get_inputs():
    return [torch.randn(32, 4096, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
