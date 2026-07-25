"""FROZEN ANCHOR for 08: the production baseline a good engineer writes today.

NCCL all-gather of the K/V chunks, then one masked bf16 SDPA over the full
context — the straightforward context-parallel implementation, with no ring, no
overlap, and no load balancing. `anchor_ms` in problem.yaml is this module timed
once on the canonical node via `--mode anchor`, then FROZEN.

Interface matches reference.Model (the anchor is timed, never correctness-checked).
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


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
        full_k = torch.cat(gk, dim=0)
        full_v = torch.cat(gv, dim=0)

        qh = q.transpose(0, 1).unsqueeze(0)
        kh = full_k.transpose(0, 1).unsqueeze(0)
        vh = full_v.transpose(0, 1).unsqueeze(0)

        qpos = torch.arange(sl, device=q.device) + r * sl
        kpos = torch.arange(w * sl, device=q.device)
        mask = kpos.unsqueeze(0) <= qpos.unsqueeze(1)

        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        return out.squeeze(0).transpose(0, 1).contiguous()


def is_available() -> bool:
    return True
