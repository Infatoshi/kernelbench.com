import torch
import torch.nn as nn

OP_TYPE = "geometry"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 3


class Model(nn.Module):
    """Moller-Trumbore ray-triangle intersection for a batch of rays and triangles."""

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        v2: torch.Tensor,
    ) -> torch.Tensor:
        R = ray_origins.shape[0]
        T = v0.shape[0]
        eps = self.epsilon

        edge1 = (v1 - v0).unsqueeze(0).expand(R, T, 3)
        edge2 = (v2 - v0).unsqueeze(0).expand(R, T, 3)
        dirs = ray_dirs.unsqueeze(1).expand(R, T, 3)

        h = torch.cross(dirs, edge2, dim=-1)
        a = (edge1 * h).sum(dim=-1)

        parallel = a.abs() < eps

        f = 1.0 / (a + eps)
        s = ray_origins.unsqueeze(1).expand(R, T, 3) - v0.unsqueeze(0).expand(R, T, 3)
        u = f * (s * h).sum(dim=-1)
        q = torch.cross(s, edge1, dim=-1)
        v = f * (dirs * q).sum(dim=-1)
        t = f * (edge2 * q).sum(dim=-1)

        valid = (~parallel) & (u >= 0) & (v >= 0) & ((u + v) <= 1.0) & (t > eps)
        t_hit = torch.where(valid, t, torch.tensor(float("inf"), device=t.device))
        closest_t = t_hit.min(dim=1).values  # (R,)
        return closest_t


def get_inputs():
    R = 50000
    T = 128
    ray_origins = torch.randn(R, 3)
    ray_dirs = torch.nn.functional.normalize(torch.randn(R, 3), dim=-1)
    v0 = torch.randn(T, 3) * 5.0
    v1 = v0 + torch.randn(T, 3) * 0.5
    v2 = v0 + torch.randn(T, 3) * 0.5
    return [ray_origins, ray_dirs, v0, v1, v2]


def get_init_inputs():
    return [1e-7]
