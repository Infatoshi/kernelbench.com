"""NCCL-best busbw ceiling sweep for KernelBench-Multi.

Measures the genuine NCCL collective (the thing the solution must BEAT, with
NCCL itself forbidden in solutions) across a message-size sweep, for each
collective family in the deck:

    all_reduce        -> 01_allreduce_residual
    reduce_scatter    -> 02_reducescatter_rmsnorm, 06_fp8_reducescatter_grad
    all_gather (u8)   -> 03_allgather_fp8
    all_to_all        -> 04_moe_all2all, 05_ulysses_all2all

For each size it reports achieved busbw and the fraction of the NVLink4 peak
(900 GB/s). The point is the SHAPE of the curve: where NCCL saturates (large,
bandwidth-bound -> little headroom for a custom kernel) vs where it is
latency-bound (small -> big headroom for one-shot symmetric-memory).

Run on the 4xH100 node:
    torchrun --nproc_per_node=4 scripts/nccl_ceiling.py
Env: KBM_WARMUP (default 200), KBM_ITERS (default 50).
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

PEAK = 450.0  # NVLink4 per-GPU UNIDIRECTIONAL (NCCL busbw convention), GB/s
HIDDEN = 8192
# per-rank "tokens" sweep at fixed hidden=8192 bf16 => message 0.25MB .. 512MB
TOKENS = [16, 64, 256, 1024, 4096, 16384, 32768]


def _time(fn, warmup, iters, device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms_local = start.elapsed_time(end) / iters
    t = torch.tensor([ms_local], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # slowest rank gates
    return float(t.item())


def _row(rank, fam, tokens, msg_mb, busbw_bytes, ms):
    if rank != 0:
        return
    gbps = busbw_bytes / (ms / 1e3) / 1e9
    print(f"{fam:14s} tokens={tokens:6d} msg={msg_mb:8.2f}MB  ms={ms:8.4f}  "
          f"busbw={gbps:7.1f}GB/s  frac={gbps/PEAK:.3f}", flush=True)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    warmup = int(os.environ.get("KBM_WARMUP", 200))
    iters = int(os.environ.get("KBM_ITERS", 50))

    if rank == 0:
        print(f"# NCCL-best ceiling sweep  world={world}  peak={PEAK}GB/s  "
              f"warmup={warmup} iters={iters}", flush=True)

    for tok in TOKENS:
        n = tok * HIDDEN
        msg_mb = n * 2 / 1e6

        # all_reduce (bf16)
        x = torch.randn(tok, HIDDEN, device=device, dtype=torch.bfloat16)
        ms = _time(lambda: dist.all_reduce(x, op=dist.ReduceOp.SUM), warmup, iters, device)
        _row(rank, "all_reduce", tok, msg_mb, 2 * (world - 1) / world * n * 2, ms)

        # reduce_scatter (bf16) — tokens divisible by world
        if tok % world == 0:
            inp = torch.randn(tok, HIDDEN, device=device, dtype=torch.bfloat16)
            out = torch.empty(tok // world, HIDDEN, device=device, dtype=torch.bfloat16)
            ms = _time(lambda: dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM),
                       warmup, iters, device)
            _row(rank, "reduce_scatter", tok, msg_mb, (world - 1) / world * n * 2, ms)

        # all_gather of fp8 bytes (uint8): per-rank shard = tok rows
        u8 = torch.randint(0, 255, (tok, HIDDEN), device=device, dtype=torch.uint8)
        outg = torch.empty(world * tok, HIDDEN, device=device, dtype=torch.uint8)
        ms = _time(lambda: dist.all_gather_into_tensor(outg, u8), warmup, iters, device)
        _row(rank, "all_gather_u8", tok, n * 1 / 1e6, (world - 1) / world * world * n * 1, ms)

        # all_to_all_single (bf16) — first dim divisible by world
        if tok % world == 0:
            a = torch.randn(tok, HIDDEN, device=device, dtype=torch.bfloat16)
            b = torch.empty_like(a)
            ms = _time(lambda: dist.all_to_all_single(b, a), warmup, iters, device)
            _row(rank, "all_to_all", tok, msg_mb, (world - 1) / world * n * 2, ms)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
