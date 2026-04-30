import torch
import torch.nn as nn

OP_TYPE = "simulation"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 2


class Model(nn.Module):
    """Spring-mass cloth simulation: one Verlet integration step on a grid."""

    def __init__(self, grid_size: int = 512, rest_length: float = 1.0, stiffness: float = 500.0, dt: float = 0.001):
        super().__init__()
        self.grid_size = grid_size
        self.rest_length = rest_length
        self.stiffness = stiffness
        self.dt = dt

    def forward(self, positions: torch.Tensor, prev_positions: torch.Tensor) -> torch.Tensor:
        G = self.grid_size
        pos = positions.view(G, G, 3)
        prev = prev_positions.view(G, G, 3)

        gravity = torch.tensor([0.0, -9.81, 0.0], device=positions.device, dtype=positions.dtype)
        forces = gravity.unsqueeze(0).unsqueeze(0).expand(G, G, 3).clone()

        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni = torch.arange(G, device=positions.device) + di
            nj = torch.arange(G, device=positions.device) + dj
            valid_i = (ni >= 0) & (ni < G)
            valid_j = (nj >= 0) & (nj < G)
            mask = valid_i.unsqueeze(1) & valid_j.unsqueeze(0)

            ni_c = ni.clamp(0, G - 1)
            nj_c = nj.clamp(0, G - 1)
            neighbor = pos[ni_c][:, nj_c]
            diff = neighbor - pos
            dist = torch.sqrt((diff**2).sum(dim=-1, keepdim=True) + 1e-8)
            spring = self.stiffness * (dist - self.rest_length) * diff / dist
            forces += spring * mask.unsqueeze(-1).float()

        new_pos = 2.0 * pos - prev + forces * self.dt**2
        return new_pos.view(-1, 3)


def get_inputs():
    positions = torch.randn(512 * 512, 3)
    prev_positions = positions + torch.randn_like(positions) * 0.001
    return [positions, prev_positions]


def get_init_inputs():
    return [512, 1.0, 500.0, 0.001]
