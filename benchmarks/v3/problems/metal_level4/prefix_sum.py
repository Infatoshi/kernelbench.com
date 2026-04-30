import torch
import torch.nn as nn

OP_TYPE = "scan"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 4


class Model(nn.Module):
    """Inclusive prefix sum (cumulative sum) on a batch of 1D arrays."""

    def __init__(self):
        super().__init__()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data.cumsum(dim=-1)


def get_inputs():
    return [torch.randn(64, 1048576)]


def get_init_inputs():
    return []
