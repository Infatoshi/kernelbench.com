"""
FP4 GEMM reference using int4-like values packed in int8 + scale.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, m: int = 2048, n: int = 2048, k: int = 2048):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k

    def forward(
        self,
        a_q: torch.Tensor,
        b_q: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        a_fp = a_q.float() * scale_a.float()
        b_fp = b_q.float() * scale_b.float()
        return (a_fp @ b_fp).to(torch.float16)


OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ["fp4"]
HARDWARE_REQUIRED = ["B200"]
SPECIALIZED_LEVEL = 1


def get_inputs():
    m = 2048
    n = 2048
    k = 2048
    a_q = torch.randint(-8, 8, (m, k), dtype=torch.int8)
    b_q = torch.randint(-8, 8, (k, n), dtype=torch.int8)
    scale_a = torch.tensor(0.08, dtype=torch.float32)
    scale_b = torch.tensor(0.08, dtype=torch.float32)
    return [a_q, b_q, scale_a, scale_b]


def get_init_inputs():
    return []
