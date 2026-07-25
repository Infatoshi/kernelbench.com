"""Context-parallel (ring) causal attention — correctness oracle.

The sequence is split CONTIGUOUSLY across ranks: rank r owns positions
[r*seq_local, (r+1)*seq_local). Each rank holds q/k/v for its own chunk only, and
must produce the causal attention output for its own queries — which attend over
the entire prefix, most of which lives on other ranks.

The oracle gathers the full K/V and runs one masked fp32 SDPA. A real solution
must not do that: it passes K/V chunks around the fabric and accumulates with an
online-softmax (flash-style) rescale, overlapping each hop's transfer with the
attention math for the chunk already in hand.

The interesting part is the load imbalance. Under causal masking with contiguous
sharding, rank 0's queries attend to one chunk and rank 3's to four, so the
per-rank work is 1:2:3:4 while the collective runs in lockstep — a naive ring
leaves rank 0 idle 75% of the time and the SLOWEST rank gates the measured time
(the harness max-reduces it). Rebalancing means moving work or queries across
ranks, and how you do that is the open part of the problem.

Local attention math may use any library (SDPA / flash-attn / cuDNN / a custom
kernel). The chunk exchange may not use a c10d collective.
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

seq_local = 2048       # per-rank contiguous sequence chunk
heads = 32
head_dim = 128


class Model(nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: (seq_local, heads, head_dim) for THIS rank's chunk.
        w = dist.get_world_size()
        r = dist.get_rank()
        sl = q.shape[0]

        gk = [torch.empty_like(k) for _ in range(w)]
        gv = [torch.empty_like(v) for _ in range(w)]
        dist.all_gather(gk, k.contiguous())
        dist.all_gather(gv, v.contiguous())
        full_k = torch.cat(gk, dim=0)                     # (w*sl, heads, head_dim)
        full_v = torch.cat(gv, dim=0)

        # (1, heads, seq, head_dim) for SDPA; fp32 oracle, single downcast at the end.
        qh = q.float().transpose(0, 1).unsqueeze(0)
        kh = full_k.float().transpose(0, 1).unsqueeze(0)
        vh = full_v.float().transpose(0, 1).unsqueeze(0)

        # Causal mask against GLOBAL positions: my query i is at r*sl + i.
        qpos = torch.arange(sl, device=q.device) + r * sl
        kpos = torch.arange(w * sl, device=q.device)
        mask = kpos.unsqueeze(0) <= qpos.unsqueeze(1)     # (sl, w*sl) True = attend

        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        return out.squeeze(0).transpose(0, 1).contiguous().to(torch.bfloat16)


def get_inputs():
    q = (torch.randn(seq_local, heads, head_dim) * 0.1).to(torch.bfloat16)
    k = (torch.randn(seq_local, heads, head_dim) * 0.1).to(torch.bfloat16)
    v = (torch.randn(seq_local, heads, head_dim) * 0.1).to(torch.bfloat16)
    return [q, k, v]


def get_init_inputs():
    return [heads, head_dim]
