import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "conv"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 1


class Model(nn.Module):
    """Bicubic image resize using PyTorch interpolation."""

    def __init__(self, output_h: int = 512, output_w: int = 512):
        super().__init__()
        self.output_h = output_h
        self.output_w = output_w

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            image,
            size=(self.output_h, self.output_w),
            mode="bicubic",
            align_corners=False,
        )


def get_inputs():
    return [torch.randn(4, 3, 1024, 1024)]


def get_init_inputs():
    return [512, 512]
