"""Row-parallel TP GEMM + all-reduce + residual (correctness oracle).

Tensor-parallel MLP down-projection: the full (K, hidden) weight is sharded
across ranks along K, so every rank computes a PARTIAL (tokens, hidden) product
over its own K shard. The partials are summed across ranks, then the residual is
added. This is the shape of every TP layer in every large model.

Why this problem exists: the partial product is produced tile by tile by the MMA
pipeline, and the sum is a collective over the same tiles. A sequential
implementation (finish the whole GEMM, then all-reduce) pays gemm_ms + comm_ms.
A fused one starts pushing tiles to peers while the pipeline is still running and
pays roughly max(gemm_ms, comm_ms). The shapes are chosen so those two terms are
the same order of magnitude, which is where the overlap is worth real time.

The weight is RANK-DISTINCT by construction (generator seeded from the rank), so
sum_r (x_r @ W_r) != (sum_r x_r) @ W. You cannot all-reduce the smaller
activation first and do a single GEMM — the reduction has to happen on the
(tokens, hidden) product.

Oracle numerics: accumulate the product in fp32 and sum in fp32, downcasting once
at the end (see DEVLOG 2026-07-22 — an in-type bf16 reduction ORDER is not a
valid oracle; it fails honest fp32-accumulate kernels).
"""
import math

import torch
import torch.distributed as dist
import torch.nn as nn

tokens = 4096
k_local = 2048
hidden = 8192


class Model(nn.Module):
    def __init__(self, k_local: int, hidden: int):
        super().__init__()
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Rank-distinct shard of the full (world*k_local, hidden) weight. Seeded
        # from the rank so it is deterministic and reproducible, but NOT equal
        # across ranks — that is what forbids the all-reduce-the-input shortcut.
        gen = torch.Generator().manual_seed(20260725 + rank * 7919)
        w = torch.randn(k_local, hidden, generator=gen) / math.sqrt(k_local)
        self.weight = nn.Parameter(w.to(torch.bfloat16))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # x: (tokens, k_local) — this rank's shard of the activation.
        partial = x.float() @ self.weight.float()       # this rank's partial sum
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)  # sum partials across ranks
        return (partial + residual.float()).to(torch.bfloat16)


def get_inputs():
    x = (torch.randn(tokens, k_local) * 0.1).to(torch.bfloat16)
    residual = (torch.randn(tokens, hidden) * 0.1).to(torch.bfloat16)
    return [x, residual]


def get_init_inputs():
    return [k_local, hidden]
