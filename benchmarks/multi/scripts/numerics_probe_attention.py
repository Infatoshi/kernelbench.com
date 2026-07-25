"""Calibrate the correctness tolerance for 08_ring_attention_cp, empirically.

Same methodology as `numerics_probe.py` did for the reduction decks (DEVLOG
2026-07-22): measure what HONEST implementations disagree with the fp32 oracle by,
measure what the cheapest CHEATS disagree by, and set the tolerance in the gap.

The gate is scale-aware: tol = atol * rms(ref) + rtol * |ref|. For each variant we
report `atol_min` — the smallest atol that would let it pass with rtol fixed —
so honest variants and cheats can be compared on one axis.

    torchrun --nproc_per_node=4 scripts/numerics_probe_attention.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

BENCH_ROOT = Path(__file__).resolve().parents[1]
PROBLEM = BENCH_ROOT / "problems-h100x4" / "08_ring_attention_cp"
RTOL = 0.025


def _gather(t: torch.Tensor, world: int) -> torch.Tensor:
    buf = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(buf, t.contiguous())
    return torch.cat(buf, dim=0)


def _mask(sl: int, total: int, rank: int, device) -> torch.Tensor:
    qpos = torch.arange(sl, device=device) + rank * sl
    kpos = torch.arange(total, device=device)
    return kpos.unsqueeze(0) <= qpos.unsqueeze(1)


def oracle(q, k, v, rank, world):
    """fp32 one-shot masked SDPA over the full context — the reference."""
    sl = q.shape[0]
    fk, fv = _gather(k, world), _gather(v, world)
    qh = q.float().transpose(0, 1).unsqueeze(0)
    kh = fk.float().transpose(0, 1).unsqueeze(0)
    vh = fv.float().transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=_mask(sl, world * sl, rank, q.device))
    return out.squeeze(0).transpose(0, 1).contiguous()


def honest_sdpa_bf16(q, k, v, rank, world):
    """What the anchor does: one-shot SDPA in bf16 (fp32 softmax accumulation
    internally, bf16 PV matmul). The noisiest honest implementation."""
    sl = q.shape[0]
    fk, fv = _gather(k, world), _gather(v, world)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), fk.transpose(0, 1).unsqueeze(0),
        fv.transpose(0, 1).unsqueeze(0), attn_mask=_mask(sl, world * sl, rank, q.device),
    )
    return out.squeeze(0).transpose(0, 1).contiguous().float()


def _ring(q, k, v, rank, world, acc_dtype, drop_last=False, no_rescale=False):
    """Chunk-by-chunk accumulation with an online softmax — the numerics of a real
    ring, without the comm (the chunks are gathered, the MERGE ORDER is what we
    are measuring). `drop_last` / `no_rescale` are the cheats."""
    sl, h, d = q.shape
    dev = q.device
    fk, fv = _gather(k, world), _gather(v, world)
    qh = q.float().transpose(0, 1)                       # (h, sl, d)
    scale = 1.0 / (d ** 0.5)
    acc = torch.zeros(h, sl, d, dtype=acc_dtype, device=dev)
    m = torch.full((h, sl, 1), -float("inf"), dtype=torch.float32, device=dev)
    lse = torch.zeros(h, sl, 1, dtype=torch.float32, device=dev)
    qpos = torch.arange(sl, device=dev) + rank * sl

    steps = world - 1 if drop_last else world
    for c in range(steps):
        kc = fk[c * sl:(c + 1) * sl].float().transpose(0, 1)
        vc = fv[c * sl:(c + 1) * sl].float().transpose(0, 1)
        kpos = torch.arange(sl, device=dev) + c * sl
        cm = (kpos.unsqueeze(0) <= qpos.unsqueeze(1))
        if not cm.any():
            continue
        s = (qh @ kc.transpose(-1, -2)) * scale
        s = s.masked_fill(~cm.unsqueeze(0), -float("inf"))
        m_c = s.amax(dim=-1, keepdim=True)
        m_c = torch.where(torch.isinf(m_c), torch.zeros_like(m_c), m_c)
        p = torch.exp(s - m_c)
        l_c = p.sum(dim=-1, keepdim=True)
        o_c = (p.to(acc_dtype) @ vc.to(acc_dtype))
        if no_rescale:                                   # CHEAT: ignore running max
            acc = acc + o_c
            lse = lse + l_c
            continue
        m_new = torch.maximum(m, m_c)
        a = torch.exp(m - m_new).nan_to_num(0.0)
        b = torch.exp(m_c - m_new)
        acc = acc * a.to(acc_dtype) + o_c * b.to(acc_dtype)
        lse = lse * a + l_c * b
        m = m_new
    return (acc.float() / lse.clamp(min=1e-30)).transpose(0, 1).contiguous()


def atol_min(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Smallest atol that would let `got` pass, with rtol fixed."""
    err = (ref - got).abs()
    rms = ref.pow(2).mean().sqrt()
    need = (err - RTOL * ref.abs()).clamp(min=0.0) / rms
    return float(need.max())


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world)

    sys.path.insert(0, str(PROBLEM))
    import reference

    rows = []
    for shape in [{"seq_local": 2048, "heads": 32, "head_dim": 128},
                  {"seq_local": 512, "heads": 32, "head_dim": 128}]:
        for key, val in shape.items():
            setattr(reference, key, val)
        for scale_name, scale in [("nominal", 1.0), ("small", 1e-3), ("large", 1e3)]:
            torch.manual_seed(1000 + rank * 1_000_003)
            q, k, v = [(t * scale).to(torch.bfloat16).to(dev) for t in reference.get_inputs()]
            with torch.no_grad():
                ref = oracle(q, k, v, rank, world)
                variants = {
                    "honest sdpa bf16 (anchor)": honest_sdpa_bf16(q, k, v, rank, world),
                    "honest ring fp32 merge": _ring(q, k, v, rank, world, torch.float32),
                    "honest ring bf16 acc": _ring(q, k, v, rank, world, torch.bfloat16),
                    "CHEAT drop last chunk": _ring(q, k, v, rank, world, torch.float32, drop_last=True),
                    "CHEAT no online rescale": _ring(q, k, v, rank, world, torch.float32, no_rescale=True),
                }
            for name, got in variants.items():
                a = atol_min(ref, got)
                t = torch.tensor([a], device=dev, dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                if rank == 0:
                    rows.append((shape["seq_local"], scale_name, name, float(t.item())))

    if rank == 0:
        print(f"{'seq_local':>10} {'scale':>8} {'variant':<28} {'atol_min':>10}")
        for sl, sc, name, a in rows:
            print(f"{sl:>10} {sc:>8} {name:<28} {a:>10.4f}")
        honest = max(a for _, _, n, a in rows if not n.startswith("CHEAT"))
        cheat = min(a for _, _, n, a in rows if n.startswith("CHEAT"))
        print(f"\nworst honest atol_min = {honest:.4f}")
        print(f"cheapest cheat atol_min = {cheat:.4f}")
        print(f"separation = {cheat / max(honest, 1e-9):.1f}x")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
