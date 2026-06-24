"""Naive PyTorch reference for Kimi Delta Attention (KDA) forward, chunk form.

This is the correctness oracle, NOT the SOTA baseline. It mirrors the
chunk-parallel formulation in fla/ops/kda/naive.py (Songlin Yang et al.)
without any Triton or CUDA optimization.

Inputs (per the FLA convention):
  q, k : (B, T, H, K)   bf16   -- queries / keys
  v    : (B, T, H, V)   bf16   -- values
  g    : (B, T, H, K)   fp32   -- per-channel log-decay (in-chunk cumsum applied)
  beta : (B, T, H)      bf16   -- write strength

Output:
  o    : (B, T, H, V)   bf16

The agent must reproduce this output (within bf16 tolerance) using a CUTLASS
CuTe kernel on SM120 -- NOT by calling fla.ops.chunk_kda directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

OP_TYPE = "linear_attention"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


def _naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
) -> torch.Tensor:
    """KDA forward, no initial state, no final state. Returns o with v's dtype."""
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    assert T % BT == 0, f"T={T} must be a multiple of chunk_size={BT}"
    NT = T // BT

    q, k, v, g, beta = (x.to(torch.float32) for x in (q, k, v, g, beta))
    q = q * scale

    q = rearrange(q, "b (n c) h d -> b h n c d", c=BT)
    k = rearrange(k, "b (n c) h d -> b h n c d", c=BT)
    v = rearrange(v, "b (n c) h d -> b h n c d", c=BT)
    g = rearrange(g, "b (n c) h d -> b h n c d", c=BT)
    beta = rearrange(beta, "b (n c) h -> b h n c", c=BT)

    g = g.cumsum(-2)

    # ---- Build A_kk (intra-chunk K-K interaction, lower-triangular w/ diag masked) ----
    mask_diag_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = torch.zeros(*q.shape[:-1], BT, dtype=torch.float32, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i + 1, :]
        A[..., i] = torch.einsum("... c d, ... d -> ... c", k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]
    A = -A.masked_fill(mask_diag_upper, 0)

    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float32, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    # ---- Recurrent inter-chunk pass ----
    S = q.new_zeros(B, H, K, V)
    o = torch.zeros_like(v)
    mask_strict_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        Aqk = torch.zeros(B, H, BT, BT, dtype=torch.float32, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j + 1, :]
            Aqk[..., j] = torch.einsum("... c d, ... d -> ... c", q_i * (g_i - g_j).exp(), k_j)
        Aqk = Aqk.masked_fill(mask_strict_upper, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), "b h k -> b h k 1")
        S = S + rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, "b h c k -> b h k c") @ v_i

    o = rearrange(o, "b h n c d -> b (n c) h d")
    return o.to(dtype)


class Model(nn.Module):
    """KDA forward (chunk form). No learned parameters; all inputs are activations."""

    def __init__(self, B: int, T: int, H: int, K: int, V: int, chunk_size: int = 64):
        super().__init__()
        self.B, self.T, self.H, self.K, self.V = B, T, H, K, V
        self.chunk_size = chunk_size
        self.scale = float(K) ** -0.5
        # No learned params; declare a dummy buffer so state_dict is well-defined.
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return _naive_chunk_kda(q, k, v, g, beta, scale=self.scale, chunk_size=self.chunk_size)


# Module-level shape shims (overridden by check.py / benchmark.py per shape).
B = 2
T = 1024
H = 8
K = 128
V = 128
CHUNK_SIZE = 64


def get_inputs():
    """Return a list of activations for one forward call.

    bf16 for q/k/v/beta; fp32 for the log-decay g (per FLA convention).
    """
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16) * 0.1
    # log-decay: small negative numbers so exp(g) is in (0, 1).
    g = (torch.randn(B, T, H, K, dtype=torch.float32) * 0.1 - 0.05)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.bfloat16))
    return [q, k, v, g, beta]


def get_init_inputs():
    return [B, T, H, K, V, CHUNK_SIZE]
