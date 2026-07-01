"""Numeric stress for multi-GPU correctness.

Rescales floating inputs small/large and reruns the same shapes/seeds. Defeats
zero-output and cached-nominal cheats: a kernel that returns a constant or a
stale buffer fails once the input magnitude changes. Does NOT add hidden shapes.

Disable for local debugging only with KBM_NUMERIC_STRESS=0.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StressCase:
    name: str
    scale: float


def stress_cases() -> list[StressCase]:
    if os.environ.get("KBM_NUMERIC_STRESS", "1") == "0":
        return [StressCase("nominal", 1.0)]
    return [
        StressCase("nominal", 1.0),
        StressCase("small", 1e-3),
        StressCase("large", 1e3),
    ]


def apply_scale(inputs: list[torch.Tensor], case: StressCase) -> list[torch.Tensor]:
    """Return scaled clones of floating inputs; integer/bool inputs pass through."""
    out = []
    for t in inputs:
        if t.is_floating_point() and case.scale != 1.0:
            out.append((t.float() * case.scale).to(t.dtype))
        else:
            out.append(t.clone())
    return out
