"""Naive grouped GEMM + fused SwiGLU reference (correctness only, NOT the SOTA).

This is the up-projection of an MoE FFN. Each token i is assigned to K experts;
expert_indices[i*K + j] tells you which expert. Tokens are dispatched to experts
according to routing metadata; we compute, per expert e:

    h_e = silu(x_e @ W_gate[e])  *  (x_e @ W_up[e])

where x_e is the slice of permuted hidden states routed to expert e, with
expert_offsets[e]:expert_offsets[e+1] giving its row range in the permuted layout.

The reference loops over experts in Python. Slow, but pedagogically clear and
correct. Forbidden ops (torch.matmul, torch.bmm, F.linear, sonic_moe imports)
are NOT used here, but the reference is exempt — only solution.py is checked.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "grouped_gemm_swiglu"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


class Model(nn.Module):
    """Up-projection of a top-K MoE FFN with fused SwiGLU.

    Inputs at call time:
      hidden_states:    (T_perm, H)  bf16, already permuted to expert order
      expert_offsets:   (E+1,)       int32, prefix sums of token counts per expert
                                     so expert e owns rows [offsets[e]:offsets[e+1]]
                                     T_perm = T_total * K (each token visits K experts)

    Output:
      gated_up:         (T_perm, I)  bf16
    """

    def __init__(self, T_total: int, H: int, I: int, E: int, K: int):  # noqa: E741
        super().__init__()
        self.T_total = T_total
        self.H = H
        self.I = I
        self.E = E
        self.K = K
        # Two weight tensors per expert: gate (E, H, I) and up (E, H, I).
        self.W_gate = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16))
        self.W_up = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16))
        nn.init.normal_(self.W_gate, std=0.02)
        nn.init.normal_(self.W_up, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,   # (T_perm, H) bf16
        expert_offsets: torch.Tensor,  # (E+1,) int32
    ) -> torch.Tensor:
        T_perm, H = hidden_states.shape
        out = torch.empty(T_perm, self.I, dtype=torch.bfloat16, device=hidden_states.device)
        # Loop over experts. Each expert is a small dense GEMM on its slice.
        for e in range(self.E):
            start = int(expert_offsets[e].item())
            end = int(expert_offsets[e + 1].item())
            if end == start:
                continue
            x_e = hidden_states[start:end]                 # (n_e, H)
            gate = x_e @ self.W_gate[e]                    # (n_e, I)
            up = x_e @ self.W_up[e]                        # (n_e, I)
            out[start:end] = F.silu(gate) * up
        return out


# Module-level shape shims rewritten by check.py / benchmark.py per shape.
T_total = 32768
H = 4096
I = 1536  # noqa: E741
E = 128
K = 8


def _build_routing(T_total: int, E: int, K: int, device: str = "cpu") -> torch.Tensor:
    """Round-robin-ish routing metadata: balanced offsets summing to T_total*K."""
    T_perm = T_total * K
    # Even split with remainder distributed to first experts.
    base = T_perm // E
    rem = T_perm - base * E
    counts = torch.full((E,), base, dtype=torch.int32, device=device)
    counts[:rem] += 1
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def get_inputs():
    T_perm = T_total * K
    hidden_states = torch.randn(T_perm, H, dtype=torch.bfloat16) * 0.1
    expert_offsets = _build_routing(T_total, E, K)
    return [hidden_states, expert_offsets]


def get_init_inputs():
    return [T_total, H, I, E, K]
