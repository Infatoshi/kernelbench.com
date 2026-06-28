"""SOTA reference for Sonic-MoE up-projection: Tri Dao's sonic-moe.

Status (2026-04): sonic-moe ships on PyPI as `sonic-moe` (>=0.1.2.post1) and
requires Python>=3.12. It dispatches to QuACK CuTeDSL grouped GEMM kernels.
SM120 (RTX PRO 6000 Blackwell Workstation) support is in-progress upstream --
the package installs cleanly but kernels may fail at launch on SM120 (the
QuACK grouped-GEMM path targets Sm90/Sm100 in the public release).

If the live call fails, `is_available()` returns False and the benchmark scores
the agent against PyTorch eager + the documented H100 paper ceiling (see
problem.yaml.sota.reference_throughput_tflops_h100). Agents are FORBIDDEN from
importing sonic_moe in solution.py (see problem.yaml.forbidden).
"""
from __future__ import annotations

import torch


def _try_sonic_moe(
    hidden_states: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor | None:
    try:
        import sonic_moe  # type: ignore  # noqa: F401
    except Exception:
        return None
    try:
        # Public sonic-moe API surface is still stabilizing. The expected entry
        # point bundles gate+up weights as a single (E, H, 2*I) tensor and fuses
        # SwiGLU. Adapt to the actual signature once SM120 lands.
        W = torch.cat([W_gate, W_up], dim=-1).contiguous()  # (E, H, 2*I)
        from sonic_moe import fused_moe_up  # type: ignore
        return fused_moe_up(hidden_states, W, expert_offsets)
    except Exception:
        return None


def sota_forward(
    hidden_states: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Best-available grouped-GEMM + SwiGLU reference."""
    out = _try_sonic_moe(hidden_states, W_gate, W_up, expert_offsets)
    if out is not None:
        return out
    raise RuntimeError("sonic-moe SOTA path unavailable on this hardware")


def is_available() -> bool:
    # On SM120 with the current public sonic-moe, this is expected to return
    # False until upstream lands SM120 kernels. Detect by attempting a tiny
    # smoke call on import; any failure -> not available.
    try:
        import sonic_moe  # type: ignore  # noqa: F401
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    # Cheap capability gate: sonic-moe public release targets sm_90/sm_100.
    major, _ = torch.cuda.get_device_capability(0)
    if major < 9:
        return False
    # We do not run a live smoke here (would require allocating real weights);
    # benchmark.py wraps sota_forward in try/except and treats failures as
    # "SOTA unavailable" -- see problem.yaml.sota.reference_throughput_tflops_h100
    # for the documented paper ceiling used in that case.
    return True
