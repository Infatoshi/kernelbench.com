import torch
import torch.nn as nn

OP_TYPE = "reduction"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 4


class Model(nn.Module):
    """Monte Carlo pi estimation: count random points inside unit circle."""

    def __init__(self):
        super().__init__()

    def forward(self, random_points: torch.Tensor) -> torch.Tensor:
        x = random_points[:, 0]
        y = random_points[:, 1]
        inside = (x**2 + y**2 <= 1.0).float()
        pi_estimate = 4.0 * inside.mean()
        return pi_estimate.unsqueeze(0)


def get_inputs():
    return [torch.rand(10000000, 2)]


def get_init_inputs():
    return []
