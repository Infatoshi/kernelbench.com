"""Naive W4A16 weight-only quantized GEMM reference (correctness only).

AWQ/GPTQ-style scheme:
  x:      (M, K)               bf16
  w_q:    (K // 2, N)          uint8   -- two int4 weights packed per byte (low nibble = even-K, high = odd-K)
  scales: (K // group, N)      bf16
  zeros:  (K // group, N)      bf16    -- asymmetric (stored already as float zero-point)
  out:    (M, N)                bf16

Dequant (per group along K):
  w_bf[k, n] = (w_q[k, n] - zeros[k // group, n]) * scales[k // group, n]
where w_q[k, n] is the unpacked 4-bit value (0..15).

This reference unpacks to a full bf16 matrix and then runs torch.matmul. Slow and
memory-heavy on the dequant; the agent's solution must fuse unpack+GEMM.
"""
from __future__ import annotations

import torch
import torch.nn as nn

OP_TYPE = "gemm_w4a16"
SUPPORTED_PRECISIONS = ["int4_bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]

GROUP_SIZE = 128


def _pack_int4(w_q: torch.Tensor) -> torch.Tensor:
    """Pack (K, N) uint8 in [0,15] into (K//2, N) uint8.

    Even rows go in the low nibble, odd rows in the high nibble.
    """
    K, N = w_q.shape
    assert K % 2 == 0
    lo = w_q[0::2].to(torch.uint8) & 0xF
    hi = w_q[1::2].to(torch.uint8) & 0xF
    return (lo | (hi << 4)).contiguous()


def _unpack_int4(w_packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack (K//2, N) uint8 -> (K, N) uint8 in [0,15]."""
    Kh, N = w_packed.shape
    assert Kh * 2 == K
    out = torch.empty((K, N), dtype=torch.uint8, device=w_packed.device)
    out[0::2] = w_packed & 0xF
    out[1::2] = (w_packed >> 4) & 0xF
    return out


class Model(nn.Module):
    """W4A16 GEMM: y = x @ dequant(w_q, scales, zeros).

    Buffers are registered (not Parameters) so state_dict carries them across to
    the agent's solution. Initialization picks scales/zeros from a normal weight,
    then quantizes deterministically.
    """

    def __init__(self, M: int, N: int, K: int, group_size: int = GROUP_SIZE):
        super().__init__()
        assert K % group_size == 0, "K must be divisible by group_size"
        assert K % 2 == 0, "K must be even (int4 packing)"
        self.M, self.N, self.K = M, N, K
        self.group_size = group_size
        n_groups = K // group_size

        # Synthetic quant: take a random bf16 weight, compute per-group asym
        # quant params, then pack. This produces a *correct* set of (w_q, s, z)
        # triples that round-trip cleanly under the dequant formula.
        torch.manual_seed(0xC0DE ^ (M * 1315423911 + N * 2654435761 + K))
        w_full = torch.randn(K, N, dtype=torch.float32) * 0.02

        w_g = w_full.view(n_groups, group_size, N)
        w_min = w_g.min(dim=1, keepdim=True).values  # (n_groups, 1, N)
        w_max = w_g.max(dim=1, keepdim=True).values
        scales = (w_max - w_min).clamp_min(1e-8) / 15.0  # (n_groups, 1, N)
        zeros = (-w_min / scales).round().clamp(0, 15)   # (n_groups, 1, N)
        # Quantize
        w_q = ((w_g / scales) + zeros).round().clamp(0, 15).to(torch.uint8)
        w_q = w_q.view(K, N)

        scales_2d = scales.squeeze(1).to(torch.bfloat16)         # (n_groups, N)
        zeros_2d = zeros.squeeze(1).to(torch.bfloat16)            # (n_groups, N)
        w_packed = _pack_int4(w_q)                                # (K//2, N)

        self.register_buffer("w_q", w_packed)        # uint8
        self.register_buffer("scales", scales_2d)    # bf16
        self.register_buffer("zeros", zeros_2d)      # bf16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Naive: unpack -> dequant -> matmul.
        K = self.K
        w_unpacked = _unpack_int4(self.w_q, K).to(torch.bfloat16)  # (K, N) in [0,15]
        # Broadcast scales/zeros along the group axis.
        scales = self.scales.repeat_interleave(self.group_size, dim=0)  # (K, N) bf16
        zeros = self.zeros.repeat_interleave(self.group_size, dim=0)    # (K, N) bf16
        w_bf = (w_unpacked - zeros) * scales  # (K, N) bf16
        return x.to(torch.bfloat16) @ w_bf  # (M, N) bf16


M = 1
N = 12288
K = 4096


def get_inputs():
    x = torch.randn(M, K, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [M, N, K]
