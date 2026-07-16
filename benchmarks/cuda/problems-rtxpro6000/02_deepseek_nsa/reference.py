"""DeepSeek NSA-inspired sparse attention reference (simplified, correctness oracle).

For each query position t:
  1. Partition keys into blocks of size `block_size`.
  2. Block importance = mean over the block of (q_t · k_j / sqrt(D))  (no softmax).
  3. Keep the top_n_blocks by importance, always unioned with a local sliding
     window of the last `sliding_window` tokens (and causal: j <= t).
  4. Softmax attention over the selected key indices only; write o_t.

This is a *bench-faithful simplification* of NSA (compress/select/sliding), not
a bit-exact DeepSeek production kernel. Agents implement the same semantics.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

OP_TYPE = "deepseek_nsa"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000"]

B, H, S, D = 1, 16, 1024, 64
BLOCK_SIZE = 64
TOP_N_BLOCKS = 8
SLIDING_WINDOW = 64


def nsa_attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    top_n_blocks: int = TOP_N_BLOCKS,
    sliding_window: int = SLIDING_WINDOW,
) -> torch.Tensor:
    """q,k,v: (B,H,S,D) float or bf16 → o (B,H,S,D) float."""
    qf, kf, vf = q.float(), k.float(), v.float()
    B, H, S, D = qf.shape
    scale = 1.0 / math.sqrt(D)
    n_blocks = (S + block_size - 1) // block_size
    out = torch.zeros_like(qf)

    for b in range(B):
        for h in range(H):
            for t in range(S):
                # --- block importance (only causal keys) ---
                scores_full = (qf[b, h, t] @ kf[b, h, : t + 1].T) * scale  # (t+1,)
                block_imp = []
                for bi in range(n_blocks):
                    s0 = bi * block_size
                    if s0 > t:
                        block_imp.append((-1e9, bi))
                        continue
                    s1 = min((bi + 1) * block_size, t + 1)
                    block_imp.append((float(scores_full[s0:s1].mean()), bi))
                block_imp.sort(reverse=True)
                selected = set()
                for _, bi in block_imp[:top_n_blocks]:
                    s0 = bi * block_size
                    s1 = min((bi + 1) * block_size, t + 1)
                    if s0 <= t:
                        selected.update(range(s0, s1))
                # sliding window
                w0 = max(0, t + 1 - sliding_window)
                selected.update(range(w0, t + 1))
                if not selected:
                    selected.add(t)
                idx = torch.tensor(sorted(selected), device=q.device, dtype=torch.long)
                ks = kf[b, h].index_select(0, idx)
                vs = vf[b, h].index_select(0, idx)
                sc = (qf[b, h, t] @ ks.T) * scale
                att = torch.softmax(sc, dim=-1)
                out[b, h, t] = att @ vs
    return out


class Model(nn.Module):
    def __init__(self, B: int, H: int, S: int, D: int):
        super().__init__()
        self.B, self.H, self.S, self.D = B, H, S, D
        self.register_buffer("_dummy", torch.zeros(1, dtype=torch.bfloat16))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        o = nsa_attend(q, k, v, BLOCK_SIZE, TOP_N_BLOCKS, SLIDING_WINDOW)
        return o.to(torch.bfloat16)


def get_init_inputs():
    return [B, H, S, D]


def get_inputs():
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    return [q, k, v]
