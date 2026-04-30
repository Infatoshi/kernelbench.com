import torch
import torch.nn as nn

OP_TYPE = "simulation"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 2


class Model(nn.Module):
    """Brute-force broad-phase collision detection: pairwise distance check."""

    def __init__(self, collision_radius: float = 0.5):
        super().__init__()
        self.collision_radius = collision_radius

    def forward(self, positions: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        dist = torch.sqrt((diff**2).sum(dim=-1) + 1e-10)  # (N, N)
        combined_radii = radii.unsqueeze(0) + radii.unsqueeze(1)  # (N, N)
        colliding = (dist < combined_radii).float()
        colliding.fill_diagonal_(0.0)
        return colliding


def get_inputs():
    positions = torch.randn(4096, 3)
    radii = torch.rand(4096).abs() * 0.3 + 0.1
    return [positions, radii]


def get_init_inputs():
    return [0.5]
