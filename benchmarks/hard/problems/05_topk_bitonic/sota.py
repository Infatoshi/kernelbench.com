"""SOTA reference for TopK: torch.topk itself.

torch.topk dispatches to a CUB-backed kernel that uses radix-select for
moderate k and a tuned bitonic sort for small n. It is the bar the agent's
hand-rolled bitonic kernel must beat. There is no obvious vendor library that
does better on the (batch, n, k) shape mix we evaluate — Faiss BlockSelect is
specialized for k>=32 with much larger n, and CUB's DeviceSegmentedRadixSort
sorts the full row (overkill for top-k).

This file is INTENTIONALLY allowed to call torch.topk because it is the SOTA
oracle, not the agent's submission. The agent's solution.py is forbidden from
using torch.topk (see problem.yaml.forbidden).
"""
from __future__ import annotations

import torch


def sota_forward(x: torch.Tensor, k: int):
    """Best-available top-k reference. x: (batch, n) fp32."""
    return torch.topk(x, k=k, dim=-1, largest=True, sorted=True)


def is_available() -> bool:
    return True
