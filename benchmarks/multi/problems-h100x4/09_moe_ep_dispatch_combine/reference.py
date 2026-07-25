"""Expert-parallel MoE dispatch + combine with IRREGULAR routing (oracle).

Each rank owns one expert and holds `tokens` tokens. Every token carries a
destination expert produced by the gate (an input, not a parameter), and the
per-destination counts are DATA-DEPENDENT and unequal — which is the whole
difficulty. The kernel must:

  1. dispatch  — send each token to the rank owning its expert, fp8 on the wire,
  2. run the expert on the received batch,
  3. combine   — return each result to its origin rank, in the origin's token order.

Two design decisions here are anti-cheat measures, both learned from the deck's
earlier MoE problem (`04_moe_all2all`, retired 2026-07-25), where an equal-split
all-to-all with a rank-identical per-channel expert was algebraically a scaled
identity on local data — a zero-communication solution would have passed.

  * The expert weight is RANK-DISTINCT (generator seeded from the rank).
  * The expert is NOT a per-token map. Its output for a token depends on the
    NEIGHBOURING token in the expert's received batch, ordered canonically by
    (src_rank, src_index). Because that neighbour is some other rank's token, in
    full width, the dependency cannot be collapsed into a scalar or a gathered
    weight — no local shortcut reproduces it. The bytes have to move, both ways.

fp8 on the dispatch leg is likewise not optional: the reference quantizes with a
PER-TOKEN e4m3 scale before routing, so a solution that ships bf16 differs from
the oracle by the full quantization error and fails the 2.5e-2 gate. Quantization
is deterministic given the input, so an honest fp8 solution reproduces it exactly.
The combine leg is bf16.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

tokens = 2048          # tokens held per rank
hidden = 4096
E4M3_MAX = 448.0


def _fp8_per_token(x: torch.Tensor) -> torch.Tensor:
    """Per-token e4m3 fake-quant: models the compression the dispatch applies."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / E4M3_MAX
    q = (x / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q.float() * scale


class Model(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        rank = dist.get_rank() if dist.is_initialized() else 0
        gen = torch.Generator().manual_seed(20260725 + rank * 6151)
        w = 0.5 + torch.rand(hidden, generator=gen)      # rank-distinct expert
        self.expert_w = nn.Parameter(w.to(torch.bfloat16))

    def forward(self, x: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
        # x: (tokens, hidden) bf16 — this rank's tokens.
        # dest: (tokens,) int64 — destination expert (== owning rank) per token.
        w = dist.get_world_size()
        r = dist.get_rank()
        n = x.shape[0]

        xq = _fp8_per_token(x.float())                   # fp8 dispatch leg

        gx = [torch.empty_like(xq) for _ in range(w)]
        dist.all_gather(gx, xq.contiguous())
        big_x = torch.cat(gx, dim=0)                     # (w*n, hidden), src-major
        gd = [torch.empty_like(dest) for _ in range(w)]
        dist.all_gather(gd, dest.contiguous())
        big_d = torch.cat(gd, dim=0)                     # (w*n,)
        gw = [torch.empty_like(self.expert_w) for _ in range(w)]
        dist.all_gather(gw, self.expert_w.contiguous())
        big_w = torch.stack(gw, dim=0).float()           # (w, hidden)

        # Canonical received order per expert: stable sort by destination keeps
        # (src_rank, src_index) order within each expert's batch, because the
        # gathered buffer is src-major.
        order = torch.argsort(big_d, stable=True)
        d_sorted = big_d[order]
        counts = torch.bincount(d_sorted, minlength=w)
        starts = torch.cumsum(counts, 0) - counts
        pos_in_batch = torch.arange(big_d.numel(), device=big_d.device) - starts[d_sorted]
        prev = starts[d_sorted] + (pos_in_batch - 1) % counts[d_sorted].clamp(min=1)

        x_sorted = big_x[order]
        y_sorted = 0.5 * (x_sorted + x_sorted[prev]) * big_w[d_sorted]

        y = torch.empty_like(y_sorted)
        y[order] = y_sorted                              # combine: back to origin order
        return y[r * n:(r + 1) * n].to(torch.bfloat16)


def get_inputs():
    w = dist.get_world_size() if dist.is_initialized() else 4
    x = (torch.randn(tokens, hidden) * 0.1).to(torch.bfloat16)
    # Gate logits with a per-call, per-rank bias: destination counts come out
    # genuinely unbalanced (real MoE routing is never uniform) and change from
    # trial to trial, so a solution cannot hard-code a split.
    logits = torch.randn(tokens, w) * 1.5 + torch.randn(w) * 1.2
    dest = logits.argmax(dim=-1).to(torch.int64)
    return [x, dest]


def get_init_inputs():
    return [hidden]
