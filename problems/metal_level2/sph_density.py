import torch
import torch.nn as nn

OP_TYPE = "simulation"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 2


class Model(nn.Module):
    """SPH density estimation with cubic spline kernel (brute-force neighbor search)."""

    def __init__(self, smoothing_length: float = 0.1, particle_mass: float = 1.0):
        super().__init__()
        self.h = smoothing_length
        self.mass = particle_mass

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        h = self.h
        norm_const = 8.0 / (3.14159265 * h**3)

        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        r = torch.sqrt((diff**2).sum(dim=-1) + 1e-10)  # (N, N)
        q = r / h

        w = torch.zeros_like(q)
        mask1 = q <= 0.5
        mask2 = (q > 0.5) & (q <= 1.0)
        w[mask1] = 1.0 - 6.0 * q[mask1] ** 2 + 6.0 * q[mask1] ** 3
        w[mask2] = 2.0 * (1.0 - q[mask2]) ** 3

        w = w * norm_const
        density = (self.mass * w).sum(dim=1)  # (N,)
        return density


def get_inputs():
    return [torch.randn(2048, 3)]


def get_init_inputs():
    return [0.1, 1.0]
