"""SOTA reference for FP8 GEMM: flashinfer.gemm.fp8_gemm.

If flashinfer is not installed or the SM120 path isn't supported, this falls
back to torch._scaled_mm which is the cuBLAS FP8 path. The benchmark treats
whichever succeeds as the SOTA reference line.

Agents are FORBIDDEN from using torch._scaled_mm in their solution (see
problem.yaml.forbidden). This file is only for the benchmark's reference line.
"""
from __future__ import annotations

import torch


def _try_flashinfer(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    try:
        import flashinfer  # noqa: F401
        # Note: flashinfer's FP8 GEMM API surface may differ; adapt if needed.
        # Placeholder call — replace with the actual flashinfer entry point
        # once validated on SM120.
        return None
    except ImportError:
        return None


def _scaled_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # torch._scaled_mm wants per-tensor scales. Use unit scales for the reference.
    scale_a = torch.tensor(1.0, device=x.device)
    scale_b = torch.tensor(1.0, device=x.device)
    out = torch._scaled_mm(
        x,
        w.T,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
    )
    return out if not isinstance(out, tuple) else out[0]


def sota_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Best-available FP8 GEMM reference. x: (M, K) fp8, w: (N, K) fp8."""
    out = _try_flashinfer(x, w)
    if out is not None:
        return out
    return _scaled_mm(x, w)


def is_available() -> bool:
    try:
        # Verify torch._scaled_mm is callable (smoke)
        return hasattr(torch, "_scaled_mm")
    except Exception:
        return False
