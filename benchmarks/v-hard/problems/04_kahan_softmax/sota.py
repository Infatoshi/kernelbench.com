"""SOTA reference for last-dim softmax.

Preference order:
  1. liger-kernel's Triton softmax (LigerSoftmaxFunction) — fused, fp32-acc
  2. torch.compile(torch.softmax) — Inductor generates a Triton kernel that
     is competitive on bandwidth-bound reductions

Agents are FORBIDDEN from importing either of these in solution.py (see
problem.yaml.forbidden). This file is only the benchmark's reference line.
"""
from __future__ import annotations

import torch

_compiled_softmax = None


def _liger_softmax(x: torch.Tensor) -> torch.Tensor | None:
    try:
        from liger_kernel.ops.softmax import LigerSoftmaxFunction
        return LigerSoftmaxFunction.apply(x)
    except Exception:
        return None


def _compiled(x: torch.Tensor) -> torch.Tensor:
    global _compiled_softmax
    if _compiled_softmax is None:
        _compiled_softmax = torch.compile(
            lambda t: torch.softmax(t, dim=-1),
            mode="reduce-overhead",
        )
    return _compiled_softmax(x)


def sota_forward(x: torch.Tensor) -> torch.Tensor:
    """Best-available softmax reference. x: (batch, vocab) fp32."""
    out = _liger_softmax(x)
    if out is not None:
        return out
    return _compiled(x)


def is_available() -> bool:
    return True  # torch.compile fallback is always available
