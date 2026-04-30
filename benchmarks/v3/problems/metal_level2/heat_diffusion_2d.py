import torch
import torch.nn as nn

OP_TYPE = "stencil"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 2


class Model(nn.Module):
    """2D heat diffusion: one Jacobi iteration step with 5-point stencil."""

    def __init__(self, alpha: float = 0.25, num_steps: int = 10):
        super().__init__()
        self.alpha = alpha
        self.num_steps = num_steps

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        B, C, H, W = grid.shape
        u = grid.clone()
        for _ in range(self.num_steps):
            padded = torch.nn.functional.pad(u, [1, 1, 1, 1], mode="replicate")
            left = padded[:, :, 1:-1, :-2]
            right = padded[:, :, 1:-1, 2:]
            up = padded[:, :, :-2, 1:-1]
            down = padded[:, :, 2:, 1:-1]
            u = u + self.alpha * (left + right + up + down - 4.0 * u)
        return u


def get_inputs():
    grid = torch.zeros(4, 1, 512, 512)
    grid[:, :, 248:264, 248:264] = 1.0
    return [grid]


def get_init_inputs():
    return [0.25, 10]
