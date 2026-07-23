"""TP all-reduce + residual reference (correctness oracle, NOT a fast kernel).

Tensor-parallel building block: after a row-parallel matmul each rank holds a
partial sum of the output; an all-reduce sums them across the TP group, then a
residual is added. The reference uses NCCL (`dist.all_reduce`) — that is exactly
what the solution is FORBIDDEN from calling. A real solution implements the
cross-GPU sum with fine-grained NVLink (symmetric-memory one-shot/two-shot,
NVSHMEM, put/get) and beats NCCL's busbw.

Inputs are RANK-DISTINCT (the harness seeds each rank differently), so a kernel
cannot fake the all-reduce by assuming symmetric inputs.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

# module-level shape vars (set by the harness from shapes.py)
tokens = 4096
hidden = 8192


class Model(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # x: (tokens, hidden) bf16 — this rank's partial output.
        # fp32 accumulation: the oracle is the mathematically exact sum with a
        # single downcast at the end, NOT NCCL's order-dependent in-type bf16
        # reduction (measured on 4xH100: in-type bf16 orderings disagree with
        # each other by up to ~2e-2 rel, so an order-dependent oracle would
        # fail honest kernels — see scripts/numerics_probe.py).
        y = x.float()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)   # FORBIDDEN in solution.py
        return (y + residual.float()).to(torch.bfloat16)


def get_inputs():
    x = (torch.randn(tokens, hidden) * 0.1).to(torch.bfloat16)
    residual = (torch.randn(tokens, hidden) * 0.1).to(torch.bfloat16)
    return [x, residual]


def get_init_inputs():
    return [hidden]
