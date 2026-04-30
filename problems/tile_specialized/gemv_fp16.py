"""
FP16 GEMV for autoregressive decoding.

Each decoding step is effectively GEMV (single token projections), which is
memory bound and requires different tiling than large GEMMs.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features: int = 4096, out_features: int = 14336):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(dtype=torch.float16)
        return x_fp16 @ self.weight.t()


OP_TYPE = "gemv"
SUPPORTED_PRECISIONS = ["fp16"]
HARDWARE_REQUIRED = ["RTX3090", "A100", "H100", "B200"]
SPECIALIZED_LEVEL = 1


def get_inputs():
    return [torch.randn(32, 4096, dtype=torch.float16)]


def get_init_inputs():
    return []
