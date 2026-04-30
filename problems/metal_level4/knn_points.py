import torch
import torch.nn as nn

OP_TYPE = "geometry"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["M4MAX"]
METAL_LEVEL = 4


class Model(nn.Module):
    """Brute-force k-nearest neighbors on 3D point clouds."""

    def __init__(self, k: int = 8):
        super().__init__()
        self.k = k

    def forward(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        diff = query.unsqueeze(1) - reference.unsqueeze(0)  # (Q, R, 3)
        dist_sq = (diff**2).sum(dim=-1)  # (Q, R)
        _, indices = dist_sq.topk(self.k, dim=-1, largest=False)
        return indices


def get_inputs():
    query = torch.randn(4096, 3)
    reference = torch.randn(16384, 3)
    return [query, reference]


def get_init_inputs():
    return [8]
