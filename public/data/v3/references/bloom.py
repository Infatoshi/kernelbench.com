import torch
import torch.nn as nn
import torch.nn.functional as F



OP_TYPE = "fused"
SUPPORTED_PRECISIONS = ['fp16', 'bf16', 'fp32']
HARDWARE_REQUIRED = ['RTX3090']

GRAPHICS_LEVEL = 1


class Model(nn.Module):
    """Bloom effect approximation: threshold -> blur -> additive blend."""

    def __init__(self, threshold: float = 1.0, intensity: float = 0.6, blur_kernel: int = 5):
        super().__init__()
        self.threshold = threshold
        self.intensity = intensity
        self.blur_kernel = blur_kernel

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Extract bright regions and blur them.
        bright = torch.relu(image - self.threshold)
        blurred = F.avg_pool2d(bright, kernel_size=self.blur_kernel, stride=1, padding=self.blur_kernel // 2)
        return torch.clamp(image + self.intensity * blurred, min=0.0, max=10.0)


def get_inputs():
    # HDR-like image input.
    return [torch.randn(2, 3, 1024, 1024) * 1.5 + 0.2]


def get_init_inputs():
    return [1.0, 0.6, 5]
