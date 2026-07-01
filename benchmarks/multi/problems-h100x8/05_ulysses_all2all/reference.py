"""Ulysses seq<->head all-to-all repartition primitive (correctness oracle).

Sequence-parallel attention (DeepSpeed-Ulysses) needs to switch the activation
from sequence-sharded to head-sharded before attention and back after. This is
the all-to-all that does it — NO attention math here, just the repartition, so
the problem stays comms-dominated (single-GPU compute is not the subject).

Input per rank: (seq_local, heads, head_dim), sequence-sharded (each rank owns
seq_local of the full sequence, all heads). Output per rank:
(seq_local*world, heads/world, head_dim), head-sharded (full sequence, this
rank's slice of heads). Expressed via all_gather + slice (gloo-supported). The
SOLUTION must do the all-to-all over NVLink without any c10d collective.

Inputs are RANK-DISTINCT.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

seq = 2048        # per-rank sequence shard
heads = 32
head_dim = 128


class Model(nn.Module):
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_local, heads, head_dim) — seq-sharded.
        w = dist.get_world_size()
        r = dist.get_rank()
        hl = x.shape[1] // w                          # heads per rank after repartition
        allx = [torch.empty_like(x) for _ in range(w)]
        dist.all_gather(allx, x.contiguous())
        # gather full sequence, keep only this rank's head slice
        out = torch.cat([allx[s][:, r * hl:(r + 1) * hl, :] for s in range(w)], dim=0)
        return out.to(torch.bfloat16)                 # (seq_local*world, heads/world, head_dim)


def get_inputs():
    x = (torch.randn(seq, heads, head_dim) * 0.1).to(torch.bfloat16)
    return [x]


def get_init_inputs():
    return [heads, head_dim]
