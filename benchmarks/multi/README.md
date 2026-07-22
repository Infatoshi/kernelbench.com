# KernelBench-Multi

**Can LLMs write fast multi-GPU (NVLink) collective kernels?**

A small, hand-curated, reward-hack-audited benchmark where frontier coding agents
turn a PyTorch + NCCL reference for a distributed op into a fast, fine-grained
NVLink implementation (CUDA / Triton / NVSHMEM / CUDA symmetric memory /
ParallelKittens) on an **4×H100 SXM** node.

Sibling to KernelBench-Hard. Hard tests single-GPU kernels; Multi tests the
**inter-GPU fabric**. The graded number is **busbw** (NVLink bandwidth
efficiency), never TFLOPS — single-GPU compute is deliberately not the subject.
See `SPEC.md` for methodology.

> Design inspired by and partially adapting **ParallelKernelBench** (Together AI
> / Hazy Research), Apache-2.0. See `NOTICE`.

## The deck (`problems-h100x4/`)

| NN | problem | collective |
| -- | ------- | ---------- |
| 01 | `allreduce_residual` | all-reduce + residual |
| 02 | `reducescatter_rmsnorm` | reduce-scatter + RMSNorm |
| 03 | `allgather_fp8` | all-gather + fp8 dequant |
| 04 | `moe_all2all` | expert-parallel all-to-all |
| 05 | `ulysses_all2all` | seq↔head repartition |
| 06 | `fp8_reducescatter_grad` | fp8 reduce-scatter |

Each problem dir holds: `reference.py` (PyTorch+NCCL oracle), `shapes.py`,
`problem.yaml` (busbw formula, world_size, forbidden collective, tolerances),
`sota.py` (NCCL ceiling), `check.py`, `benchmark.py`, `PROMPT.txt`. The agent's
`solution.py` is gitignored output.

## Local validation (single GPU, free)

No NVLink needed to validate correctness plumbing — run `torchrun` with the gloo
backend on CPU:

```bash
# from a problem dir, e.g. problems-h100x4/01_allreduce_residual/
KBM_BACKEND=gloo KBM_DEVICE=cpu KBM_WORLD_SIZE=4 python check.py
```

This exercises the launcher, per-rank compare, rank-asymmetry seeding, and the
anti-hack gates. Real busbw numbers require the 4×H100 node.

## On the 4×H100 node (real eval)

```bash
# correctness (real NCCL, world_size from problem.yaml)
python check.py

# busbw benchmark (TK2 rigor: 500 warmup / 100 timed)
python benchmark.py
```

## Status

Bootstrapping. Decks and harness are being validated on a single-GPU box via
gloo before any node is rented. FSDP2 / pipeline-parallel "full training step"
problems are intentionally **out of scope** here — they are compute/orchestration
heavy and belong in a separate future training benchmark.
