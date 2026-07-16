"""GLM-5.2-class fused MoE layer reference (correctness oracle).

Layout (vLLM-style fused weights, GLM-5.x routing structure):
  w1_routed:  (E, 2*I, H)  — gate|up packed per routed expert
  w2_routed:  (E, H, I)
  w1_shared:  (n_shared, 2*I, H)
  w2_shared:  (n_shared, H, I)

For each token:
  1. Shared experts always fire (sum over n_shared of silu_and_mul → down).
  2. top_k routed experts from expert_ids / expert_weights fire the same way.
  3. out = shared_sum + routed_weighted_sum

Routing inputs are given (not part of the kernel). n_shared=1, E=256, top_k=8
matches GLM-5 / 5.1 / 5.2 MoE layers (NVIDIA Megatron-Bridge notes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "glm52_fused_moe"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000"]

T, E, top_k, n_shared, H, I = 4096, 256, 8, 1, 4096, 2048


def _silu_mul_down(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """x (n,H), w1 (2I,H), w2 (H,I) → (n,H) float."""
    I_half = w1.shape[0] // 2
    gate = x @ w1[:I_half].T
    up = x @ w1[I_half:].T
    h = F.silu(gate) * up
    return h @ w2.T


class Model(nn.Module):
    def __init__(
        self,
        T: int,
        E: int,
        top_k: int,
        n_shared: int,
        H: int,
        I: int,
    ):
        super().__init__()
        self.T, self.E, self.top_k = T, E, top_k
        self.n_shared, self.H, self.I = n_shared, H, I
        self.w1_routed = nn.Parameter(torch.empty(E, 2 * I, H, dtype=torch.bfloat16))
        self.w2_routed = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16))
        self.w1_shared = nn.Parameter(torch.empty(n_shared, 2 * I, H, dtype=torch.bfloat16))
        self.w2_shared = nn.Parameter(torch.empty(n_shared, H, I, dtype=torch.bfloat16))
        for p in self.parameters():
            nn.init.normal_(p, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (T, H) bf16
        expert_ids: (T, top_k) int64 in [0, E)
        expert_weights: (T, top_k) bf16 (routed only; shared is unweighted sum)
        returns: (T, H) bf16
        """
        T, H = x.shape
        out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
        xf = x.float()

        # --- shared experts (always on) ---
        for s in range(self.n_shared):
            out = out + _silu_mul_down(
                xf, self.w1_shared[s].float(), self.w2_shared[s].float()
            )

        # --- routed experts ---
        wts = expert_weights.float()
        for e in range(self.E):
            mask = expert_ids == e
            if not mask.any():
                continue
            token_idx, k_idx = mask.nonzero(as_tuple=True)
            x_e = xf[token_idx]
            w_e = wts[token_idx, k_idx]
            y = _silu_mul_down(
                x_e, self.w1_routed[e].float(), self.w2_routed[e].float()
            )
            out.index_add_(0, token_idx, y * w_e.unsqueeze(1))

        return out.to(torch.bfloat16)


def get_init_inputs():
    return [T, E, top_k, n_shared, H, I]


def get_inputs():
    x = torch.randn(T, H, dtype=torch.bfloat16)
    # Mildly skewed logits → unbalanced expert loads (realistic)
    logits = torch.randn(T, E) + torch.linspace(0.3, 0.0, E).unsqueeze(0)
    vals, ids = torch.topk(logits, k=top_k, dim=-1)
    weights = torch.softmax(vals, dim=-1).to(torch.bfloat16)
    return [x, ids.to(torch.int64), weights]
