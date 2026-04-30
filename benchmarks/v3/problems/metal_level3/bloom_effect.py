import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "fused"
SUPPORTED_PRECISIONS = ["fp16", "fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 3


class Model(nn.Module):
    """HDR bloom: threshold extraction, multi-pass Gaussian blur, additive blend."""

    def __init__(self, threshold: float = 1.0, intensity: float = 0.6, blur_passes: int = 3, blur_kernel: int = 9):
        super().__init__()
        self.threshold = threshold
        self.intensity = intensity
        self.blur_passes = blur_passes
        self.blur_kernel = blur_kernel
        self.padding = blur_kernel // 2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        bright = torch.relu(image - self.threshold)

        blurred = bright
        for _ in range(self.blur_passes):
            blurred = F.avg_pool2d(
                blurred,
                kernel_size=self.blur_kernel,
                stride=1,
                padding=self.padding,
            )

        return torch.clamp(image + self.intensity * blurred, min=0.0, max=10.0)


def get_inputs():
    return [torch.randn(2, 3, 1024, 1024).abs() * 2.0]


def get_init_inputs():
    return [1.0, 0.6, 3, 9]
