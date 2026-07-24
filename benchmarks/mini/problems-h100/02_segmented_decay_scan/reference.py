"""Segmented exponential-decay scan reference (correctness only).

A gated linear recurrence with per-token episode resets:

  h_t = x_t                      if resets[t]
  h_t = decay_t * h_{t-1} + x_t  otherwise        (h_{-1} = 0)

  x:      (B, T, D)  bf16
  decay:  (B, T, D)  fp32 in (0, 1)   -- per-token per-channel decay
  resets: (B, T)     bool             -- episode boundaries
  out:    (B, T, D)  bf16             -- the full state trajectory h

The recurrence is associative once the reset is folded into the decay (a reset
is decay=0 for that step), so it admits a parallel scan — but the standard
tl.associative_scan / cumprod recipes do not handle the reset mask as-is.
Reference is a sequential fp32 loop over T; the whole trajectory is written out,
so the op is memory-bound at roughly 8 bytes per element of traffic.
"""
from __future__ import annotations

import torch
import torch.nn as nn

OP_TYPE = "segmented_decay_scan"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["H100_SXM"]


class Model(nn.Module):
    def __init__(self, B: int, T: int, D: int):
        super().__init__()
        self.B, self.T, self.D = B, T, D
        self.register_buffer("_dummy", torch.zeros(1, dtype=torch.bfloat16))

    def forward(
        self, x: torch.Tensor, decay: torch.Tensor, resets: torch.Tensor
    ) -> torch.Tensor:
        xf = x.float()
        out = torch.empty_like(xf)
        h = torch.zeros(xf.shape[0], xf.shape[2], dtype=torch.float32, device=x.device)
        for t in range(xf.shape[1]):
            keep = (~resets[:, t]).float().unsqueeze(-1)  # (B, 1)
            h = keep * decay[:, t] * h + xf[:, t]
            out[:, t] = h
        return out.to(torch.bfloat16)


B = 8
T = 4096
D = 2048
RESET_PROB = 1.0 / 64.0


def get_inputs():
    x = torch.randn(B, T, D, dtype=torch.bfloat16)
    decay = torch.rand(B, T, D, dtype=torch.float32) * 0.945 + 0.05  # (0.05, 0.995)
    resets = torch.rand(B, T) < RESET_PROB
    return [x, decay, resets]


def get_init_inputs():
    return [B, T, D]
