import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "conv"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 1


class Model(nn.Module):
    """Sobel edge detection: gradient magnitude from 3x3 Sobel operators."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        x = image.view(B * C, 1, H, W)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return magnitude.view(B, C, H, W)


def get_inputs():
    return [torch.rand(4, 1, 1024, 1024)]


def get_init_inputs():
    return []
