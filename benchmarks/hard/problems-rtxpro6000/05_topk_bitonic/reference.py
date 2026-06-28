"""Naive top-k reference: torch.topk over the last dim.

This is the correctness oracle. The agent's solution must produce the same
top-k values (and equivalent indices modulo ties) within the tolerance
declared in problem.yaml. Note that solution.py is FORBIDDEN from calling
torch.topk / torch.sort / torch.kthvalue (see problem.yaml).
"""
import torch
import torch.nn as nn

OP_TYPE = "topk"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


class Model(nn.Module):
    """Top-k over the last dim of a 2D tensor.

    Input:
        x: (batch, n) fp32
    Output:
        values:  (batch, k) fp32, sorted descending
        indices: (batch, k) int64, into the last dim of x
    """

    def __init__(self, batch: int, n: int, k: int):
        super().__init__()
        self.batch, self.n, self.k = batch, n, k
        # No learned parameters, but declare a dummy buffer so state_dict
        # is non-empty and load_state_dict(strict=True) is meaningful.
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, x: torch.Tensor):
        values, indices = torch.topk(x, k=self.k, dim=-1, largest=True, sorted=True)
        return values, indices


# Module-level shims rebuilt by check.py / benchmark.py per shape.
batch = 64
n = 8192
k = 8


def get_inputs():
    # fp32 input drawn from a roughly Gaussian distribution; ties unlikely
    # but possible. Seed is set by the caller.
    x = torch.randn(batch, n, dtype=torch.float32)
    return [x]


def get_init_inputs():
    return [batch, n, k]
