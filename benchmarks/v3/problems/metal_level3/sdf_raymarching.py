import torch
import torch.nn as nn

OP_TYPE = "geometry"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 3


class Model(nn.Module):
    """SDF sphere tracing: march rays through a signed distance field of spheres."""

    def __init__(self, max_steps: int = 64, max_dist: float = 50.0, epsilon: float = 0.001):
        super().__init__()
        self.max_steps = max_steps
        self.max_dist = max_dist
        self.epsilon = epsilon

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
        sphere_centers: torch.Tensor,
        sphere_radii: torch.Tensor,
    ) -> torch.Tensor:
        R = ray_origins.shape[0]
        t = torch.zeros(R, device=ray_origins.device, dtype=ray_origins.dtype)

        for _ in range(self.max_steps):
            pos = ray_origins + t.unsqueeze(-1) * ray_dirs  # (R, 3)
            diffs = pos.unsqueeze(1) - sphere_centers.unsqueeze(0)  # (R, S, 3)
            dists = torch.sqrt((diffs**2).sum(dim=-1)) - sphere_radii.unsqueeze(0)  # (R, S)
            sdf = dists.min(dim=1).values  # (R,)

            still_marching = (sdf > self.epsilon) & (t < self.max_dist)
            t = t + sdf * still_marching.float()

        return t


def get_inputs():
    R = 100000
    S = 32
    ray_origins = torch.zeros(R, 3)
    ray_origins[:, 2] = -5.0
    ray_dirs = torch.nn.functional.normalize(torch.randn(R, 3) * 0.1 + torch.tensor([0.0, 0.0, 1.0]), dim=-1)
    sphere_centers = torch.randn(S, 3) * 3.0
    sphere_radii = torch.rand(S) * 1.0 + 0.3
    return [ray_origins, ray_dirs, sphere_centers, sphere_radii]


def get_init_inputs():
    return [64, 50.0, 0.001]
