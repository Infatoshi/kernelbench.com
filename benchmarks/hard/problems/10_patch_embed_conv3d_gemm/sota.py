"""SOTA reference for Conv3d-as-GEMM patch embedding.

Dispatches torch.nn.functional.conv3d (cuDNN). The agent's solution is
forbidden from calling conv3d directly; this file is only for the benchmark's
reference baseline.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sota_forward(x: torch.Tensor, weight: torch.Tensor,
                 stride: tuple[int, int, int]) -> torch.Tensor:
    return F.conv3d(x, weight, bias=None, stride=stride)


def is_available() -> bool:
    return True
