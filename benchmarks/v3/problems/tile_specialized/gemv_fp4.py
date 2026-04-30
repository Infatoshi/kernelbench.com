"""
FP4-like GEMV reference using packed int4 semantics in int8 + scale.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features: int = 4096, out_features: int = 14336):
        super().__init__()
        # Simulate fp4 range in int8 storage.
        self.register_buffer("weight_q", torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.tensor(0.08, dtype=torch.float32))

    def forward(self, x_q: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        x_fp = x_q.float() * x_scale.float()
        w_fp = self.weight_q.float() * self.weight_scale
        return (x_fp @ w_fp.t()).to(torch.float16)


OP_TYPE = "gemv"
SUPPORTED_PRECISIONS = ["fp4"]
HARDWARE_REQUIRED = ["B200"]
SPECIALIZED_LEVEL = 1


def get_inputs():
    x_q = torch.randint(-8, 8, (32, 4096), dtype=torch.int8)
    x_scale = torch.tensor(0.08, dtype=torch.float32)
    return [x_q, x_scale]


def get_init_inputs():
    return []
