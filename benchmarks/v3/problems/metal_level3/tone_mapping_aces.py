import torch
import torch.nn as nn

OP_TYPE = "elementwise"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 3


class Model(nn.Module):
    """ACES filmic tone mapping: HDR to LDR with S-curve response."""

    def __init__(self, exposure: float = 1.0):
        super().__init__()
        self.exposure = exposure

    def forward(self, hdr_image: torch.Tensor) -> torch.Tensor:
        x = hdr_image * self.exposure
        # ACES approximation by Krzysztof Narkowicz
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
        return mapped.clamp(0.0, 1.0)


def get_inputs():
    return [torch.rand(4, 3, 1024, 1024) * 5.0]


def get_init_inputs():
    return [1.0]
