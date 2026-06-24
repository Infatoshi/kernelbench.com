"""SOTA reference for multi-axis RoPE pre-attention.

Wraps transformers' apply_multimodal_rotary_pos_emb. The full pipeline (gather
per-axis cos/sin from cache, then call the vendor rotation) is timed, matching
what the reference and the agent solution do.

If transformers isn't installed or the impl signature has drifted, returns
is_available()=False and the benchmark omits the SOTA line.
"""
from __future__ import annotations

import torch


def _try_transformers():
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            apply_multimodal_rotary_pos_emb as _impl,
        )
        return _impl
    except Exception:
        return None


def sota_forward(
    q: torch.Tensor, k: torch.Tensor,
    pos_t: torch.Tensor, pos_h: torch.Tensor, pos_w: torch.Tensor,
    cos_cache: torch.Tensor, sin_cache: torch.Tensor,
    mrope_section: tuple[int, int, int],
):
    """q,k: (B, T, H, D); pos_*: (B, T); cos_cache,sin_cache: (max_pos, D)."""
    impl = _try_transformers()
    if impl is None:
        raise RuntimeError("transformers not available for SOTA")

    # Gather per-axis cos/sin and stack into (3, B, T, D) as transformers expects.
    cos = torch.stack([cos_cache[pos_t], cos_cache[pos_h], cos_cache[pos_w]], dim=0)
    sin = torch.stack([sin_cache[pos_t], sin_cache[pos_h], sin_cache[pos_w]], dim=0)

    # transformers wants (B, H, T, D) layout for q/k.
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    q_rot, k_rot = impl(q_t, k_t, cos, sin, list(mrope_section))
    return q_rot, k_rot


def is_available() -> bool:
    return _try_transformers() is not None
