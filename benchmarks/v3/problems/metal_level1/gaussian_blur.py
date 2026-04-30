import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "conv"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 1


class Model(nn.Module):
    """Separable 2D Gaussian blur with configurable sigma."""

    def __init__(self, sigma: float = 2.0, kernel_size: int = 11):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        self.register_buffer("kernel_h", kernel_1d.view(1, 1, 1, -1))
        self.register_buffer("kernel_v", kernel_1d.view(1, 1, -1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        x = image.view(B * C, 1, H, W)
        x = F.conv2d(x, self.kernel_h, padding=(0, self.padding))
        x = F.conv2d(x, self.kernel_v, padding=(self.padding, 0))
        return x.view(B, C, H, W)


def get_inputs():
    return [torch.randn(4, 3, 1024, 1024)]


def get_init_inputs():
    return [2.0, 11]
