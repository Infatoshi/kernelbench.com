import torch
import torch.nn as nn

OP_TYPE = "sort"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 4


class Model(nn.Module):
    """Sort a batch of 1D arrays (ascending)."""

    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data.sort(dim=-1).values


def get_inputs():
    return [torch.randn(32, 262144)]


def get_init_inputs():
    return []
