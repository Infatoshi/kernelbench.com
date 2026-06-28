"""SOTA reference for KDA forward: fla.ops.kda.chunk_kda (Triton).

The agent's solution is forbidden from importing this module path (see
problem.yaml.forbidden). This file is only used by benchmark.py to draw
the SOTA reference line.

If FLA's Triton kernel does not run on SM120 (Blackwell consumer-lineage --
some Triton kernels in FLA target Hopper TMA), is_available() returns False
and benchmark.py omits the SOTA variant. The H100 reference is documented
in problem.yaml for context.
"""
from __future__ import annotations

import torch


def _import_fla():
    try:
        from fla.ops.kda import chunk_kda  # noqa: F401
        return chunk_kda
    except Exception:
        return None


def sota_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Run FLA's Triton chunk_kda. Returns o (B, T, H, V) in v's dtype."""
    chunk_kda = _import_fla()
    if chunk_kda is None:
        raise RuntimeError("fla.ops.kda.chunk_kda unavailable")
    # FLA's chunk_kda has a richer signature (A_log, dt_bias, l2norm, gates, ...).
    # We need the bare forward: pass A_log/dt_bias as None, gates off, no l2norm.
    # The wrapper expects fp32 g; q/k/v/beta in bf16/fp16.
    out = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
    )
    # chunk_kda returns (o, final_state) or just o depending on flags.
    return out[0] if isinstance(out, tuple) else out


def is_available() -> bool:
    if _import_fla() is None:
        return False
    # Probe a tiny call to confirm the kernel compiles on the current SM.
    try:
        device = torch.device("cuda:0")
        B, T, H, K, V = 1, 64, 1, 64, 64
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device)
        g = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 0.01
        beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.bfloat16, device=device))
        sota_forward(q, k, v, g, beta, scale=K ** -0.5)
        return True
    except Exception:
        return False
