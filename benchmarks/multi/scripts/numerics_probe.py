"""Empirical probe: can two HONEST all-reduce implementations disagree beyond
the bench tolerance purely from summation order / accumulation dtype?

Context: the first 4xH100 smoke run monkey-patched the reference oracle to
fp32-accumulate, arguing NCCL's in-type bf16 reduction is ~1 ulp lossy and
order-dependent, which the 5e-3 gate cannot absorb. The tampering was illegal;
this script measures whether the underlying numerics claim is true.

Variants (all mathematically correct all-reduce implementations):
    nccl_bf16   NCCL all_reduce on bf16 in-type       (the reference path)
    nccl_fp32   upcast fp32 -> NCCL all_reduce -> bf16 (honest fp32 kernel)
    exact       all_gather -> fp64 ordered sum -> bf16 (ground truth)
    tree_bf16   (x0+x1)+(x2+x3) in bf16               (order variant 1)
    ring_bf16   ((x0+x1)+x2)+x3 in bf16               (order variant 2)

For each pair we apply the EXACT bench gate (compare.py: atol=rtol=5e-3 on
bf16, one bad element fails) and report violations. Residual add mimics
problem 01's epilogue. Run: torchrun --nproc_per_node=4 scripts/numerics_probe.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.compare import compare  # noqa: E402

_PRIME = 1_000_003
SHAPES = [(4096, 8192), (64, 8192)]
SCALES = [("nominal", 1.0), ("large", 1e3), ("small", 1e-3)]
TRIALS = 5


def variants(x: torch.Tensor, world: int) -> dict[str, torch.Tensor]:
    out = {}
    a = x.clone()
    dist.all_reduce(a)
    out["nccl_bf16"] = a

    b = x.float()
    dist.all_reduce(b)
    out["nccl_fp32"] = b.to(torch.bfloat16)

    gathered = [torch.empty_like(x.float()) for _ in range(world)]
    dist.all_gather(gathered, x.float())
    g64 = [t.double() for t in gathered]

    e = torch.zeros_like(g64[0])
    for t in g64:
        e = e + t
    out["exact"] = e.to(torch.bfloat16)

    gb = [t.to(torch.bfloat16) for t in gathered]
    out["tree_bf16"] = (gb[0] + gb[1]) + (gb[2] + gb[3]) if world == 4 else out["exact"]
    r = gb[0]
    for t in gb[1:]:
        r = r + t
    out["ring_bf16"] = r

    return out


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    backend = os.environ.get("KBM_BACKEND", "nccl")
    device = torch.device("cpu") if backend == "gloo" else torch.device(f"cuda:{local}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend, rank=rank, world_size=world)

    pairs = [
        ("nccl_bf16", "nccl_fp32"),   # reference vs honest fp32 kernel
        ("nccl_bf16", "exact"),       # reference vs ground truth
        ("nccl_fp32", "exact"),       # fp32 kernel vs ground truth
        ("nccl_bf16", "ring_bf16"),   # reference vs same-dtype other order
        ("tree_bf16", "ring_bf16"),   # pure order sensitivity, same dtype
    ]
    # aggregate: pair -> [n_fail_trials, worst_n_bad, worst_max_abs, worst_max_rel]
    agg: dict[tuple[str, str, str, str, bool], list] = {}

    for tokens, hidden in SHAPES:
        for scale_name, scale in SCALES:
            for trial in range(TRIALS):
                torch.manual_seed(1000 + trial + rank * _PRIME)
                x = (torch.randn(tokens, hidden, device=device) * scale).to(torch.bfloat16)
                torch.manual_seed(1000 + trial + rank * _PRIME + 7)
                resid = (torch.randn(tokens, hidden, device=device) * scale).to(torch.bfloat16)
                v = variants(x, world)
                for use_resid in (False, True):
                    vv = {k: (t + resid if use_resid else t) for k, t in v.items()}
                    for a, b in pairs:
                        ok, msg = compare(vv[a], vv[b])
                        key = (f"{tokens}x{hidden}", scale_name, a, b, use_resid)
                        rec = agg.setdefault(key, [0, 0, 0.0, 0.0])
                        if not ok:
                            rec[0] += 1
                            # parse n_bad / max_abs / max_rel out of the gate msg
                            parts = msg.split()
                            n_bad = int(parts[0])
                            max_abs = float(msg.split("max_abs=")[1].split()[0])
                            max_rel = float(msg.split("max_rel=")[1].split()[0])
                            rec[1] = max(rec[1], n_bad)
                            rec[2] = max(rec[2], max_abs)
                            rec[3] = max(rec[3], max_rel)

    # slowest-rank style: a violation on ANY rank counts; reduce max over ranks
    for key in sorted(agg):
        rec = agg[key]
        t = torch.tensor(rec, dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if rank == 0:
            shape, scale_name, a, b, use_resid = key
            n_fail, n_bad, max_abs, max_rel = t.tolist()
            tag = "resid" if use_resid else "plain"
            verdict = "FAIL" if n_fail else "pass"
            print(
                f"{shape:>10s} {scale_name:7s} {tag:5s} {a:>9s} vs {b:9s} "
                f"{verdict}  fail_trials={int(n_fail)}/{TRIALS} "
                f"worst_bad_elems={int(n_bad)} max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
