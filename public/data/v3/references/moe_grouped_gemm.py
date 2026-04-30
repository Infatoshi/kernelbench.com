"""
Grouped GEMM for Mixture of Experts.

MoE forward pass: route tokens to experts, compute expert GEMMs, combine outputs.
This problem targets CUTLASS/CuTe grouped-GEMM style optimization.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_experts: int = 8, hidden_dim: int = 1024, expert_dim: int = 3072):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.expert_up = nn.Parameter(torch.randn(num_experts, hidden_dim, expert_dim) * 0.02)
        self.expert_down = nn.Parameter(torch.randn(num_experts, expert_dim, hidden_dim) * 0.02)

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [batch, seq, hidden]
        expert_indices: [batch, seq, top_k]
        expert_weights: [batch, seq, top_k]
        """
        batch, seq, hidden = x.shape
        top_k = expert_indices.shape[-1]

        x_flat = x.reshape(batch * seq, hidden)
        idx_flat = expert_indices.reshape(batch * seq, top_k)
        w_flat = expert_weights.reshape(batch * seq, top_k)
        out_flat = torch.zeros_like(x_flat)

        # Baseline loops by expert; grouped GEMM kernels can fuse this routing pattern.
        for expert_id in range(self.num_experts):
            mask = idx_flat == expert_id
            if not torch.any(mask):
                continue

            token_ids, route_ids = torch.nonzero(mask, as_tuple=True)
            token_x = x_flat[token_ids]
            route_w = w_flat[token_ids, route_ids].unsqueeze(-1)

            up_out = token_x @ self.expert_up[expert_id]
            down_out = up_out @ self.expert_down[expert_id]
            out_flat.index_add_(0, token_ids, down_out * route_w)

        return out_flat.reshape(batch, seq, hidden)


OP_TYPE = "moe_grouped_gemm"
SUPPORTED_PRECISIONS = ["fp8", "bf16", "fp16"]
HARDWARE_REQUIRED = ["H100", "B200"]
SPECIALIZED_LEVEL = 1


def get_inputs():
    batch, seq, hidden = 4, 256, 1024
    top_k = 2
    num_experts = 8

    x = torch.randn(batch, seq, hidden)
    expert_indices = torch.randint(0, num_experts, (batch, seq, top_k))
    expert_weights = torch.softmax(torch.randn(batch, seq, top_k), dim=-1)
    return [x, expert_indices, expert_weights]


def get_init_inputs():
    return []
