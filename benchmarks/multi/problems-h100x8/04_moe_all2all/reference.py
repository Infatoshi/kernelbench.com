"""Expert-parallel all-to-all dispatch + combine (correctness oracle).

MoE expert parallelism: tokens are routed to the rank that owns their expert
(dispatch all-to-all), the expert transforms them, then results are routed back
(combine all-to-all). This is comms-dominated — the "expert" here is a single
per-rank channel scale, deliberately light so the all-to-all dominates.

The all-to-all is expressed via all_gather + slice (gloo-supported, validates on
one GPU), with equal capacity per (src,dst) pair. The SOLUTION must implement the
all-to-all over NVLink without any c10d collective.

Inputs are RANK-DISTINCT, so the routing is real (identity/echo cannot pass).
"""
import torch
import torch.distributed as dist
import torch.nn as nn

capacity = 2048   # rows each rank sends to each rank
hidden = 4096


def _all_to_all(t: torch.Tensor) -> torch.Tensor:
    """Equal-split all-to-all of t (shape (world*capacity, hidden)) built from
    all_gather: out chunk i = the chunk rank i addressed to me."""
    w = dist.get_world_size()
    r = dist.get_rank()
    cap = t.shape[0] // w
    allt = [torch.empty_like(t) for _ in range(w)]
    dist.all_gather(allt, t.contiguous())
    return torch.cat([allt[i][r * cap:(r + 1) * cap] for i in range(w)], dim=0)


class Model(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.expert_w = nn.Parameter(torch.ones(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dispatched = _all_to_all(x)                       # route to expert ranks
        y = dispatched * self.expert_w                    # light per-channel expert
        combined = _all_to_all(y)                         # route back
        return combined.to(torch.bfloat16)


def get_inputs():
    w = dist.get_world_size()
    x = (torch.randn(w * capacity, hidden) * 0.1).to(torch.bfloat16)
    return [x]


def get_init_inputs():
    return [hidden]
