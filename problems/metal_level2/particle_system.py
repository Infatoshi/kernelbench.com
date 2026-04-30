import torch
import torch.nn as nn

OP_TYPE = "simulation"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 2


class Model(nn.Module):
    """Particle system: velocity/position update with boundary collision and damping."""

    def __init__(self, dt: float = 0.016, damping: float = 0.8, bounds: float = 10.0):
        super().__init__()
        self.dt = dt
        self.damping = damping
        self.bounds = bounds

    def forward(self, positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        gravity = torch.tensor([0.0, -9.81, 0.0], device=positions.device, dtype=positions.dtype)
        new_vel = velocities + gravity.unsqueeze(0) * self.dt
        new_pos = positions + new_vel * self.dt

        over_max = new_pos > self.bounds
        under_min = new_pos < -self.bounds
        new_vel = torch.where(over_max | under_min, -new_vel * self.damping, new_vel)
        new_pos = new_pos.clamp(-self.bounds, self.bounds)

        return torch.cat([new_pos, new_vel], dim=-1)


def get_inputs():
    positions = torch.randn(1000000, 3)
    velocities = torch.randn(1000000, 3)
    return [positions, velocities]


def get_init_inputs():
    return [0.016, 0.8, 10.0]
