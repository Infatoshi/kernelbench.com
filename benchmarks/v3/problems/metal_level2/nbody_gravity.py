import torch
import torch.nn as nn

OP_TYPE = "simulation"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 2


class Model(nn.Module):
    """All-pairs gravitational N-body force computation."""

    def __init__(self, softening: float = 0.01, G: float = 1.0):
        super().__init__()
        self.softening = softening
        self.G = G

    def forward(self, positions: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        dist_sq = (diff**2).sum(dim=-1) + self.softening**2  # (N, N)
        inv_dist3 = dist_sq**(-1.5)
        force_magnitudes = self.G * masses.unsqueeze(0) * inv_dist3  # (N, N)
        forces = (force_magnitudes.unsqueeze(-1) * diff).sum(dim=1)  # (N, 3)
        return forces


def get_inputs():
    positions = torch.randn(4096, 3)
    masses = torch.rand(4096).abs() + 0.1
    return [positions, masses]


def get_init_inputs():
    return [0.01, 1.0]
