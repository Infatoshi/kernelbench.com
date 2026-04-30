import torch
import torch.nn as nn

OP_TYPE = "conv"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 1


class Model(nn.Module):
    """Edge-preserving bilateral filter with spatial and range weighting."""

    def __init__(self, kernel_size: int = 5, sigma_spatial: float = 2.0, sigma_range: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range
        self.pad = kernel_size // 2

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        ks = self.kernel_size
        padded = torch.nn.functional.pad(image, [self.pad] * 4, mode="reflect")

        output = torch.zeros_like(image)
        for dy in range(ks):
            for dx in range(ks):
                neighbor = padded[:, :, dy : dy + H, dx : dx + W]
                spatial_dist = ((dy - self.pad) ** 2 + (dx - self.pad) ** 2)
                spatial_w = torch.tensor(
                    (-spatial_dist / (2.0 * self.sigma_spatial**2)),
                    dtype=image.dtype,
                    device=image.device,
                ).exp()
                range_diff = (image - neighbor) ** 2
                range_w = (-range_diff / (2.0 * self.sigma_range**2)).exp()
                w = spatial_w * range_w
                output += w * neighbor

        norm = torch.zeros_like(image)
        for dy in range(ks):
            for dx in range(ks):
                neighbor = padded[:, :, dy : dy + H, dx : dx + W]
                spatial_dist = ((dy - self.pad) ** 2 + (dx - self.pad) ** 2)
                spatial_w = torch.tensor(
                    (-spatial_dist / (2.0 * self.sigma_spatial**2)),
                    dtype=image.dtype,
                    device=image.device,
                ).exp()
                range_diff = (image - neighbor) ** 2
                range_w = (-range_diff / (2.0 * self.sigma_range**2)).exp()
                norm += spatial_w * range_w

        return output / (norm + 1e-8)


def get_inputs():
    return [torch.rand(4, 3, 512, 512)]


def get_init_inputs():
    return [5, 2.0, 0.1]
