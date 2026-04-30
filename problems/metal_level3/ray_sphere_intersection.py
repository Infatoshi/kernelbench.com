import torch
import torch.nn as nn

OP_TYPE = "geometry"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 3


class Model(nn.Module):
    """Batched ray-sphere intersection: returns closest hit distance per ray."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
        sphere_centers: torch.Tensor,
        sphere_radii: torch.Tensor,
    ) -> torch.Tensor:
        oc = ray_origins.unsqueeze(1) - sphere_centers.unsqueeze(0)  # (R, S, 3)
        a = (ray_dirs.unsqueeze(1) * ray_dirs.unsqueeze(1)).sum(dim=-1)  # (R, S)
        b = 2.0 * (oc * ray_dirs.unsqueeze(1)).sum(dim=-1)  # (R, S)
        c = (oc * oc).sum(dim=-1) - sphere_radii.unsqueeze(0) ** 2  # (R, S)

        discriminant = b**2 - 4.0 * a * c
        hit = discriminant >= 0
        sqrt_disc = torch.sqrt(discriminant.clamp(min=0.0))
        t = (-b - sqrt_disc) / (2.0 * a + 1e-8)

        t_valid = torch.where(hit & (t > 0), t, torch.tensor(float("inf"), device=t.device))
        closest_t = t_valid.min(dim=1).values  # (R,)
        return closest_t


def get_inputs():
    ray_origins = torch.randn(100000, 3)
    ray_dirs = torch.nn.functional.normalize(torch.randn(100000, 3), dim=-1)
    sphere_centers = torch.randn(64, 3) * 5.0
    sphere_radii = torch.rand(64) * 2.0 + 0.5
    return [ray_origins, ray_dirs, sphere_centers, sphere_radii]


def get_init_inputs():
    return []
