"""Sort-based top-p (nucleus) mask reference (correctness oracle).

Semantics: given logits (B, V) fp32 and a nucleus mass p, compute
probs = softmax(logits) per row, order tokens by probability descending
(stable: ties broken by lower index first), and keep the minimal prefix whose
cumulative mass reaches p (the token that crosses p is kept; top-1 is always
kept). Output is a bool mask (B, V), True = token is in the nucleus.

The reference uses torch.sort — solutions may NOT (see problem.yaml forbidden
list): the point of the problem is a sort-free device-side algorithm.

Grading note: check.py does not require bit-identical masks. It builds a
float64 oracle and only *forces* tokens whose exclusive cumulative mass is
clearly below or above p (outside a ±1e-3 band); tokens inside the band are
free, absorbing legitimate fp32 summation-order differences. Everything
outside the band is exact — there is no tolerance to game.
"""
from __future__ import annotations

import torch
import torch.nn as nn

OP_TYPE = "topp_mask"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["H100_SXM"]


def topp_mask(logits: torch.Tensor, p: float) -> torch.Tensor:
    """logits (B, V) fp32 -> bool mask (B, V)."""
    probs = torch.softmax(logits.float(), dim=-1)
    sp, idx = torch.sort(probs, dim=-1, descending=True, stable=True)
    cum = sp.cumsum(dim=-1)
    # Keep while the mass BEFORE this token is < p (minimal prefix reaching p).
    keep_sorted = (cum - sp) < p
    mask = torch.zeros_like(keep_sorted)
    mask.scatter_(-1, idx, keep_sorted)
    return mask


class Model(nn.Module):
    def __init__(self, B: int, V: int, P: float):
        super().__init__()
        self.B, self.V, self.P = B, V, P
        self.register_buffer("_dummy", torch.zeros(1, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return topp_mask(logits, self.P)


B = 1
V = 151936
P = 0.9


def get_inputs():
    logits = torch.randn(B, V, dtype=torch.float32) * 3.0
    return [logits]


def get_init_inputs():
    return [B, V, P]
