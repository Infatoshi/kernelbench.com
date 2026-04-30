import torch
import torch.nn as nn

OP_TYPE = "fft"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 1


class Model(nn.Module):
    """2D FFT magnitude spectrum."""

    def __init__(self):
        super().__init__()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.fft2(signal)
        return torch.abs(freq)


def get_inputs():
    return [torch.randn(8, 1, 512, 512)]


def get_init_inputs():
    return []
