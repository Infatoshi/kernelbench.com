import torch
import torch.nn as nn

OP_TYPE = "reduction"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 1


class Model(nn.Module):
    """Histogram equalization on single-channel images."""

    def __init__(self, num_bins: int = 256):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        result = torch.empty_like(image)
        n_pixels = H * W
        for b in range(B):
            for c in range(C):
                channel = image[b, c]
                quantized = (channel.clamp(0.0, 1.0) * (self.num_bins - 1)).long()
                hist = torch.bincount(quantized.flatten(), minlength=self.num_bins).float()
                cdf = hist.cumsum(0)
                cdf_min = cdf[cdf > 0].min()
                cdf_norm = (cdf - cdf_min) / (n_pixels - cdf_min + 1e-8)
                result[b, c] = cdf_norm[quantized]
        return result


def get_inputs():
    return [torch.rand(4, 1, 512, 512)]


def get_init_inputs():
    return [256]
