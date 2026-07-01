"""SP reduce-scatter + RMSNorm reference (correctness oracle, NOT a fast kernel).

Sequence-parallel building block: each rank holds a full (tokens, hidden) partial
sum; a reduce-scatter sums across ranks and scatters the token rows so each rank
owns tokens/world rows; then RMSNorm over hidden. The semantics are expressed
here with all_reduce + slice (which gloo supports, so this validates on a
single-GPU box) — that is identical to reduce_scatter. The SOLUTION must move the
bytes with fine-grained NVLink, not any c10d collective.

Inputs are RANK-DISTINCT, so the reduce-scatter cannot be faked.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

tokens = 4096
hidden = 8192


class Model(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full = x.float().clone()
        dist.all_reduce(full, op=dist.ReduceOp.SUM)  # == reduce_scatter (sum) here
        w = dist.get_world_size()
        r = dist.get_rank()
        local = full.shape[0] // w
        chunk = full[r * local:(r + 1) * local]      # this rank's scattered shard
        rms = chunk * torch.rsqrt(chunk.pow(2).mean(-1, keepdim=True) + self.eps)
        return (rms * self.weight.float()).to(torch.bfloat16)


def get_inputs():
    x = (torch.randn(tokens, hidden) * 0.1).to(torch.bfloat16)
    return [x]


def get_init_inputs():
    return [hidden]
