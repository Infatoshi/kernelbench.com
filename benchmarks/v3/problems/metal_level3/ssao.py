import torch
import torch.nn as nn

OP_TYPE = "fused"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 3


class Model(nn.Module):
    """Screen-space ambient occlusion: hemisphere sampling around depth buffer."""

    def __init__(self, num_samples: int = 16, radius: float = 0.5, bias: float = 0.025):
        super().__init__()
        self.num_samples = num_samples
        self.radius = radius
        self.bias = bias
        torch.manual_seed(42)
        kernel = torch.randn(num_samples, 3)
        kernel = torch.nn.functional.normalize(kernel, dim=-1)
        scale = torch.linspace(0.1, 1.0, num_samples)
        kernel = kernel * scale.unsqueeze(-1) * radius
        self.register_buffer("kernel", kernel)

    def forward(self, depth: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = depth.shape
        occlusion = torch.zeros(B, 1, H, W, device=depth.device, dtype=depth.dtype)

        for s in range(self.num_samples):
            offset = self.kernel[s]
            offset_depth = offset[2].abs()
            sample_depth = depth + offset_depth * self.radius
            diff = sample_depth - depth
            occluded = (diff > self.bias).float()
            dot = (normals[:, 2:3] * offset[2]).clamp(min=0.0)
            occlusion += occluded * dot

        occlusion = 1.0 - (occlusion / self.num_samples)
        return occlusion.clamp(0.0, 1.0)


def get_inputs():
    depth = torch.rand(2, 1, 512, 512) * 10.0
    normals = torch.nn.functional.normalize(torch.randn(2, 3, 512, 512), dim=1)
    return [depth, normals]


def get_init_inputs():
    return [16, 0.5, 0.025]
