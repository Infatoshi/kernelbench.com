"""FROZEN ANCHOR for 07: the production baseline a good engineer writes today.

bf16 tensor-core GEMM through cuBLAS, then a NCCL all-reduce, then the residual
add — sequential, no overlap. This is what `metric: speedup` in problem.yaml is
measured against; `anchor_ms` is this module timed once on the canonical node via
`--mode anchor` and pinned into problem.yaml, so historical cells never re-grade.

Interface matches reference.Model (the anchor is timed by the same benchmark
path, never correctness-checked).
"""
import math

import torch
import torch.distributed as dist
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, k_local: int, hidden: int):
        super().__init__()
        rank = dist.get_rank() if dist.is_initialized() else 0
        gen = torch.Generator().manual_seed(20260725 + rank * 7919)
        w = torch.randn(k_local, hidden, generator=gen) / math.sqrt(k_local)
        self.weight = nn.Parameter(w.to(torch.bfloat16))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        partial = x @ self.weight                       # cuBLAS bf16, fp32 accum
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)  # NCCL
        return partial + residual


def is_available() -> bool:
    return True
