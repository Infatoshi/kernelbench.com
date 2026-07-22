"""All-gather of fp8 shards + on-the-fly dequant (correctness oracle).

FSDP/TP weight-gather pattern: each rank holds a bf16 shard; it fp8-quantizes the
shard (amax-scaled e4m3), the comm assembles the full tensor by gathering the fp8
BYTES across ranks (1 byte/elem — half a bf16 gather), and each shard is
dequantized back to bf16 with its source scale. Moving fp8 on the wire is the
whole point.

Expressed with all_gather over a uint8 byte-view (gloo-supported, so this
validates on a single GPU). The SOLUTION must move the fp8 bytes over NVLink
without any c10d collective. The model quantizes internally so the scale adapts
to input magnitude (robust to numeric stress).

Inputs are RANK-DISTINCT.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

tokens = 2048   # per-rank shard rows
hidden = 8192
E4M3_MAX = 448.0


class Model(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dist.get_world_size()
        g = x.float()
        amax = g.abs().amax().clamp(min=1e-12)
        scale = (amax / E4M3_MAX).reshape(1)
        x_fp8 = (g / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        u8 = x_fp8.view(torch.uint8).contiguous()
        gathered = [torch.empty_like(u8) for _ in range(w)]
        dist.all_gather(gathered, u8)                 # fp8 bytes over the wire
        scales = [torch.empty_like(scale) for _ in range(w)]
        dist.all_gather(scales, scale)
        outs = [g_.view(torch.float8_e4m3fn).float() * s_.float()
                for g_, s_ in zip(gathered, scales)]
        return torch.cat(outs, dim=0).to(torch.bfloat16)   # (w*tokens, hidden)


def get_inputs():
    return [(torch.randn(tokens, hidden) * 0.1).to(torch.bfloat16)]


def get_init_inputs():
    return [hidden]
