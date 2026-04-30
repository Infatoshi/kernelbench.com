import torch
import torch.nn as nn



OP_TYPE = "fused"
SUPPORTED_PRECISIONS = ['fp16', 'bf16', 'fp32']
HARDWARE_REQUIRED = ['RTX3090']

GRAPHICS_LEVEL = 1


class Model(nn.Module):
    """Simple GPU particle integration step with boundary collisions."""

    def __init__(self, damping: float = 0.9):
        super().__init__()
        self.damping = damping

    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        dt: torch.Tensor,
        gravity: torch.Tensor,
    ) -> torch.Tensor:
        dt_value = dt.item() if isinstance(dt, torch.Tensor) else float(dt)
        new_vel = velocities + gravity * dt_value
        new_pos = positions + new_vel * dt_value

        # Bounce from normalized viewport bounds [-1, 1].
        over = new_pos.abs() > 1.0
        new_vel = torch.where(over, -new_vel * self.damping, new_vel)
        new_pos = torch.clamp(new_pos, -1.0, 1.0)

        # Return combined state as a single tensor for evaluator compatibility.
        return torch.cat([new_pos, new_vel], dim=-1)


def get_inputs():
    n = 1_048_576
    positions = torch.rand(n, 2) * 2.0 - 1.0
    velocities = torch.randn(n, 2) * 0.05
    dt = torch.tensor(1.0 / 60.0)
    gravity = torch.tensor([0.0, -9.81])
    return [positions, velocities, dt, gravity]


def get_init_inputs():
    return [0.9]
