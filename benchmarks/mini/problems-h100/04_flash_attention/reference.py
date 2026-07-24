"""Naive causal attention reference (correctness only).

Standard multi-head causal attention forward:

  q, k, v: (B, H, S, D)  bf16
  out:     (B, H, S, D)  bf16

  scores = (q @ k^T) / sqrt(D), causal mask, softmax in fp32, @ v.

The reference materializes the full (S, S) score matrix in fp32 — O(S^2)
memory. A real solution is a flash-style streaming kernel: tiled, online
softmax, never materializing scores. check.py uses small S; benchmark.py runs
the full deck against the solution only (the eager reference at S=16384 would
allocate ~8.6 GB of scores).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

OP_TYPE = "flash_attention_causal"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["H100_SXM"]


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q,k,v (B,H,S,D) -> o (B,H,S,D), fp32 softmax, bf16 out."""
    qf, kf, vf = q.float(), k.float(), v.float()
    S = qf.shape[2]
    scale = 1.0 / math.sqrt(qf.shape[3])
    scores = (qf @ kf.transpose(-1, -2)) * scale
    causal = torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    att = torch.softmax(scores, dim=-1)
    return (att @ vf).to(torch.bfloat16)


class Model(nn.Module):
    def __init__(self, B: int, H: int, S: int, D: int):
        super().__init__()
        self.B, self.H, self.S, self.D = B, H, S, D
        self.register_buffer("_dummy", torch.zeros(1, dtype=torch.bfloat16))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return causal_attention(q, k, v)


B = 1
H = 32
S = 4096
D = 128


def get_inputs():
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    return [q, k, v]


def get_init_inputs():
    return [B, H, S, D]
