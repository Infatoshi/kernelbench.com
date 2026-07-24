"""SOTA ceiling: torch SDPA (FlashAttention-2 / cuDNN fused path on SM90)."""
import torch
import torch.nn.functional as F


def is_available() -> bool:
    return torch.cuda.is_available()


def sota_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
