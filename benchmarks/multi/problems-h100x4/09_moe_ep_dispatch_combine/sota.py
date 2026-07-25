"""FROZEN ANCHOR for 09: the production baseline a good engineer writes today.

The honest NCCL path for irregular EP routing: sort tokens by destination,
exchange the per-destination counts, all_to_all_single the fp8 payload with
variable splits, run the expert on the received batch, all_to_all_single the bf16
results back, then unpermute. Same quantization as the reference, so the agent's
win has to come from the kernel and not from the wire format.

`anchor_ms` in problem.yaml is this module timed once on the canonical node via
`--mode anchor`, then FROZEN.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

E4M3_MAX = 448.0


class Model(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        rank = dist.get_rank() if dist.is_initialized() else 0
        gen = torch.Generator().manual_seed(20260725 + rank * 6151)
        w = 0.5 + torch.rand(hidden, generator=gen)
        self.expert_w = nn.Parameter(w.to(torch.bfloat16))

    def forward(self, x: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
        w = dist.get_world_size()
        n, hidden = x.shape
        dev = x.device

        # Quantize (per-token e4m3) and permute into destination-contiguous order.
        amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = amax / E4M3_MAX
        q = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

        order = torch.argsort(dest, stable=True)
        send_counts = torch.bincount(dest, minlength=w)
        q_send = q[order].view(torch.uint8).contiguous()
        s_send = scale[order].contiguous()

        # Exchange counts so every rank can size its receive buffers.
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        sc = send_counts.tolist()
        rc = recv_counts.tolist()
        total_recv = int(sum(rc))

        q_recv = torch.empty(total_recv, hidden, dtype=torch.uint8, device=dev)
        s_recv = torch.empty(total_recv, 1, dtype=torch.float32, device=dev)
        dist.all_to_all_single(q_recv, q_send, rc, sc)
        dist.all_to_all_single(s_recv, s_send, rc, sc)

        # Expert on the received batch, in canonical (src_rank, src_index) order.
        xr = q_recv.view(torch.float8_e4m3fn).float() * s_recv
        if total_recv > 0:
            prev = (torch.arange(total_recv, device=dev) - 1) % total_recv
            yr = 0.5 * (xr + xr[prev]) * self.expert_w.float()
        else:
            yr = xr
        yr = yr.to(torch.bfloat16)

        # Combine back to origin, then unpermute into the origin's token order.
        y_back = torch.empty(n, hidden, dtype=torch.bfloat16, device=dev)
        dist.all_to_all_single(y_back, yr.contiguous(), sc, rc)
        out = torch.empty_like(y_back)
        out[order] = y_back
        return out


def is_available() -> bool:
    return True
