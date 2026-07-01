"""fp8 gradient reduce-scatter (correctness oracle).

FSDP/ZeRO gradient reduction with fp8 compression: each rank produces a full
(tokens, hidden) bf16 gradient, quantizes it to fp8_e4m3, and the comm sums the
fp8-compressed gradients across ranks and scatters the shards (reduce-scatter) so
each rank owns tokens/world rows of the summed gradient. fp8 on the wire halves
the traffic.

The reference models the fp8 quantization (so the tolerance target includes the
compression error), then expresses the reduce-scatter as all_reduce + slice
(gloo-supported, validates on one GPU). The SOLUTION must move fp8 bytes over
NVLink and reduce them without any c10d collective.

Inputs are RANK-DISTINCT.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

tokens = 4096
hidden = 8192
E4M3_MAX = 448.0


def _fp8_roundtrip(g: torch.Tensor) -> torch.Tensor:
    """Per-rank fp8 fake-quant: models the compression the comm applies."""
    amax = g.abs().amax().clamp(min=1e-12)
    scale = amax / E4M3_MAX
    q = (g / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.float() * scale


class Model(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, grad: torch.Tensor) -> torch.Tensor:
        q = _fp8_roundtrip(grad.float())              # each rank's fp8-compressed grad
        full = q.clone()
        dist.all_reduce(full, op=dist.ReduceOp.SUM)   # == fp8 reduce-scatter (sum)
        w = dist.get_world_size()
        r = dist.get_rank()
        local = full.shape[0] // w
        return full[r * local:(r + 1) * local].to(torch.bfloat16)


def get_inputs():
    grad = (torch.randn(tokens, hidden) * 0.1).to(torch.bfloat16)
    return [grad]


def get_init_inputs():
    return [hidden]
