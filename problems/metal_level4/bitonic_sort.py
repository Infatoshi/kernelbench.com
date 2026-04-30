import torch
import torch.nn as nn

OP_TYPE = "sort"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 4


class Model(nn.Module):
    """Sort 1D arrays using PyTorch sort (baseline for bitonic sort kernel)."""

    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data.sort(dim=-1).values


def get_inputs():
    n = 2**18  # 262144 - power of 2 for bitonic sort
    return [torch.randn(16, n)]


def get_init_inputs():
    return []
