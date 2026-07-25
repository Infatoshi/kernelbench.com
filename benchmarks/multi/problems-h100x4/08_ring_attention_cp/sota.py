"""FROZEN ANCHOR for 08: the production baseline a good engineer writes today.

NCCL all-gather of the K/V chunks, then one bf16 SDPA over the visible context —
the straightforward context-parallel implementation, with no ring, no overlap and
no load balancing. `anchor_ms` in problem.yaml is this module timed once on the
canonical node via `--mode anchor`, then FROZEN.

The causal mask is expressed as BOTTOM-RIGHT causal over the sliced K/V, not as a
dense bool `attn_mask`. Rank r owns queries [r*sl, (r+1)*sl) and may see keys
[0, (r+1)*sl), so with kv_len - q_len = r*sl the bottom-right convention gives
exactly `q_i attends k_j for j <= i + r*sl` — the same mask, without materializing
it. This matters: a dense attn_mask forces SDPA off the fused kernel onto the
score-materializing path, which measured 1.3-2.3x slower on the canonical node for
identical numerics (probe, 2026-07-25). Anchoring against that would have handed
every model a free speedup for doing nothing but avoiding a mistake no one ships.

Interface matches reference.Model (the anchor is timed, never correctness-checked).
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right


class Model(nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w = dist.get_world_size()
        r = dist.get_rank()
        sl = q.shape[0]

        gk = [torch.empty_like(k) for _ in range(w)]
        gv = [torch.empty_like(v) for _ in range(w)]
        dist.all_gather(gk, k.contiguous())
        dist.all_gather(gv, v.contiguous())
        kv_len = (r + 1) * sl                        # rank r sees only this prefix
        full_k = torch.cat(gk, dim=0)[:kv_len]
        full_v = torch.cat(gv, dim=0)[:kv_len]

        qh = q.transpose(0, 1).unsqueeze(0)
        kh = full_k.transpose(0, 1).unsqueeze(0)
        vh = full_v.transpose(0, 1).unsqueeze(0)

        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=causal_lower_right(sl, kv_len))
        return out.squeeze(0).transpose(0, 1).contiguous()


def is_available() -> bool:
    return True
