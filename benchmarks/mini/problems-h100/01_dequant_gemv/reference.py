"""Naive W4A16 gated dequant-GEMV reference (correctness only).

AWQ-style asymmetric int4, but with GROUP SIZE 96 along K — not 128. Most K
values are not multiples of 96, so the LAST GROUP IS RAGGED (shorter than 96).
Copy-paste group-128 kernels are wrong here by construction.

Op (a gated-MLP up projection at decode time):
  x:        (M, K)                  bf16
  w_q_*:    (K // 2, N)             uint8  -- two int4 per byte (low nibble = even-K row)
  scales_*: (ceil(K / 96), N)       bf16
  zeros_*:  (ceil(K / 96), N)       bf16   (float zero-point)
  out:      (M, N)                  bf16

  gate = x @ dequant(w_q_gate)      up = x @ dequant(w_q_up)
  out  = silu(gate) * up

Dequant: w_bf[k, n] = (unpack(w_q)[k, n] - zeros[k // 96, n]) * scales[k // 96, n].

This reference unpacks to full bf16 matrices then matmuls. Slow on purpose; a
real solution fuses unpack + both GEMVs + the silu-mul epilogue.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

OP_TYPE = "dequant_gemv_gated"
SUPPORTED_PRECISIONS = ["int4_bf16"]
HARDWARE_REQUIRED = ["H100_SXM"]

GROUP_SIZE = 96


def _pack_int4(w_q: torch.Tensor) -> torch.Tensor:
    """Pack (K, N) uint8 in [0,15] into (K//2, N) uint8 (low nibble = even row)."""
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


def _quantize(w_full: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group asymmetric int4 quant with a ragged final group."""
    K, N = w_full.shape
    n_groups = (K + group_size - 1) // group_size
    w_q = torch.empty(K, N, dtype=torch.uint8)
    scales = torch.empty(n_groups, N, dtype=torch.float32)
    zeros = torch.empty(n_groups, N, dtype=torch.float32)
    for g in range(n_groups):
        s0, s1 = g * group_size, min((g + 1) * group_size, K)
        blk = w_full[s0:s1]
        mn = blk.min(dim=0).values
        mx = blk.max(dim=0).values
        sc = (mx - mn).clamp_min(1e-8) / 15.0
        zp = (-mn / sc).round().clamp(0, 15)
        w_q[s0:s1] = ((blk / sc) + zp).round().clamp(0, 15).to(torch.uint8)
        scales[g] = sc
        zeros[g] = zp
    return w_q, scales.to(torch.bfloat16), zeros.to(torch.bfloat16)


class Model(nn.Module):
    """Gated W4A16 GEMV: out = silu(x @ Wg) * (x @ Wu), group size 96."""

    def __init__(self, M: int, N: int, K: int, group_size: int = GROUP_SIZE):
        super().__init__()
        assert K % 2 == 0, "K must be even (int4 packing)"
        self.M, self.N, self.K = M, N, K
        self.group_size = group_size

        torch.manual_seed(0x9C ^ (M * 1315423911 + N * 2654435761 + K))
        for name in ("gate", "up"):
            w_full = torch.randn(K, N, dtype=torch.float32) * 0.02
            w_q, scales, zeros = _quantize(w_full, group_size)
            self.register_buffer(f"w_q_{name}", _pack_int4(w_q))  # uint8 (K//2, N)
            self.register_buffer(f"scales_{name}", scales)         # bf16 (G, N)
            self.register_buffer(f"zeros_{name}", zeros)           # bf16 (G, N)

    def _dequant(self, name: str) -> torch.Tensor:
        w = _unpack_int4(getattr(self, f"w_q_{name}"), self.K).to(torch.bfloat16)
        g_idx = torch.arange(self.K, device=w.device) // self.group_size
        scales = getattr(self, f"scales_{name}").index_select(0, g_idx)  # (K, N)
        zeros = getattr(self, f"zeros_{name}").index_select(0, g_idx)    # (K, N)
        return (w - zeros) * scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x.to(torch.bfloat16)
        gate = xb @ self._dequant("gate")  # (M, N) bf16
        up = xb @ self._dequant("up")      # (M, N) bf16
        return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


M = 1
N = 11008
K = 4096


def get_inputs():
    x = torch.randn(M, K, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [M, N, K]
