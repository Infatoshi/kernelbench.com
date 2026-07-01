# KernelBench-Multi: Design Specification

Last updated: 2026-06-27.

## Purpose

A small, hand-curated **multi-GPU** kernel benchmark. Frontier coding agents take
a PyTorch + NCCL reference for a distributed op and rewrite it as a fast,
fine-grained NVLink implementation (CUDA / Triton / NVSHMEM / CUDA symmetric
memory / ParallelKittens) that beats the NCCL reference on an 8×H100 NVLink node.

Sibling to KernelBench-Hard. Same philosophy: a tiny deck where every cell is
audited, the opposite of breadth. The difference is the axis under test.

## What this bench measures — and what it deliberately does not

**It measures inter-GPU (NVLink) efficiency.** The graded quantity is **busbw**
(achieved NVLink bus bandwidth ÷ NVLink peak), never TFLOPS. A model that writes
a blazing tensor-core kernel but moves data inefficiently across the wire scores
poorly; a model that saturates NVLink scores well.

**It deliberately does not measure single-GPU compute.** Every problem is chosen
to be **communication-dominated**: the inter-GPU bytes dominate the local FLOPs,
so end-to-end time is gated by the wire, not by the local tensor core / attention
kernel. This is why one SKU suffices (see below) and why the deck excludes
compute-dominated problems like full ring/Ulysses attention — there the local
flash-attention kernel would dominate, which is not the point.

## One SKU: 8×H100 SXM (NVLink)

We grade only on **8×H100 SXM** (NVLink4, 900 GB/s per-GPU bidirectional).
Rationale: the benchmark targets the NVLink fabric and the collective algorithms
that ride it, which are architecturally the same across Hopper and Blackwell.
The single-GPU compute engine (where H100 and B200 differ) is explicitly *not*
the subject. Adding 8×B200 would mostly re-measure single-GPU compute at far
higher hourly cost, so it is out of scope.

Deck dir: `problems-h100x8/`. (Hardware-scoped, matching the hard bench's
`problems-h100` / `problems-rtxpro6000` convention.)

## Metric: busbw roofline

```
busbw_achieved = busbw_bytes(shape, world_size, dtype) / time_seconds
score          = busbw_achieved / nvlink_peak_busbw      # peak from src/hardware/h100x8.py
```

`busbw_bytes` already folds in the collective's bandwidth factor (NCCL
convention), declared per problem in `problem.yaml.busbw_bytes_formula`:

| collective       | factor on message size |
| ---------------- | ---------------------- |
| all-reduce       | `2*(n-1)/n`            |
| all-gather       | `(n-1)/n`              |
| reduce-scatter   | `(n-1)/n`              |
| all-to-all       | `(n-1)/n`              |
| broadcast/reduce | `1`                    |

- **Timing:** ThunderKittens-2 rigor — 500 warmup, 100 timed iterations.
- **Slowest rank gates:** the collective finishes when the last rank finishes, so
  per-shape time is the **max** over ranks. busbw uses that.
- **Shape sweep:** 3–5 canonical shapes in `shapes.py`; score is the **geometric
  mean** of per-shape `score`, penalizing single-shape hyperspecialization.

## Correctness: per-rank, rank-asymmetric

`check.py` launches the problem under `torchrun` at the problem's `world_size`.
Each rank builds **rank-distinct** inputs (the worker seeds `base_seed +
rank*PRIME`), runs the reference and the solution on the same local inputs, and
compares per-rank outputs within per-dtype tolerance. A single mismatch on any
rank, seed, or shape fails. NaN/inf is an automatic fail.

- Tolerances (tighter than hard, because loose tol enables rank hacks): fp32
  `1e-4`; bf16/fp16 `5e-3`; fp8 `5e-2`.
- 5 RNG trials per shape (rank-distinct each trial).
- `strict=True` state-dict load — missing params fail (kills identity kernels).
- Numeric stress: rescale inputs small/large and rerun (defeats zero-output /
  cached-nominal cheats), same idea as hard's `numeric_stress.py`.

## Reward-hack resistance (new classes vs single-GPU)

Multi-GPU adds hacks that only exist with >1 rank. Each has an automatic gate:

- **Rank symmetry.** If inputs were identical across ranks, all-reduce collapses
  to `x*n`, all-to-all to identity, reduce-scatter to a slice. → inputs are
  **rank-distinct every trial**, enforced in the worker. Mandatory.
- **Skip-a-rank / local-only.** Return the local shard, pass on loose tol. →
  tight tol + asymmetry.
- **Bare collective call** (the `torch._scaled_mm` analog of this bench). Just
  calling `torch.distributed.all_reduce` defeats the purpose. → per-problem
  `forbidden` list grepped in `check.py`; the movement must be implemented with
  fine-grained NVLink primitives.

Plus everything inherited from hard: contamination guard, `kb lint` tripwire, and
the **mandatory manual + subagent audit** of every published cell. For any
caching / symmetric-memory / graph-replay pattern, the empirical buffer-overwrite
recompute test is required.

## Algorithmic-bytes rule

Communication volume is counted as the **dense-equivalent** the reference moves.
An agent cannot win by dropping ranks, sparsifying the message, or lossily
compressing beyond what the reference/tolerance allows. (fp8/quantized-comms
problems declare their compressed volume explicitly in `problem.yaml`.)

## Prompt design

Same voice as hard (`problems-*/PROMPT.txt` is the human-voice query sent to the
agent under test). Hardware parenthetical is now `8×H100 SXM, NVLink4
(~900 GB/s/GPU)`. The forbidden bare-collective is named explicitly. Suggested
paths named by reputation only: NVSHMEM, CUDA symmetric memory / one-shot &
two-shot all-reduce, ParallelKittens, distributed CUTLASS. No busbw numbers, no
recipes, no "you are being evaluated" framing.

## The deck (6, comms-dominated)

| NN | name | collective | what's fused |
| -- | ---- | ---------- | ------------ |
| 01 | `allreduce_residual` | all-reduce | + residual add (TP output) |
| 02 | `reducescatter_rmsnorm` | reduce-scatter | + RMSNorm epilogue (SP) |
| 03 | `allgather_fp8` | all-gather | + on-the-fly fp8 dequant |
| 04 | `moe_all2all` | all-to-all | expert-parallel dispatch+combine (permutation) |
| 05 | `ulysses_all2all` | all-to-all | seq↔head repartition primitive (no attention math) |
| 06 | `fp8_reducescatter_grad` | reduce-scatter | fp8 gradient compression |

All busbw-graded, all forbid their bare collective.

## Harness / execution model

On-node agent flywheel on the rented 8×H100 (KBH-faithful: implement → profile
with `nsys`/`ncu` → benchmark → iterate). Cost is bounded by: prebaked image,
a node-wide 8-GPU lock so concurrent agents stay busy on API latency, a hard
per-cell wall-clock budget (the metered analog of hard's unlimited time), a
<2-min fail-fast preflight (deps + 8-GPU all-reduce + NVLink topo/busbw sanity),
continuous artifact streaming, and a watchdog auto-teardown. See README for the
Brev run flow.

## Local validation (free, single-GPU)

The whole harness and all correctness logic validate on a single-GPU box via
`torchrun` with the **gloo** backend on CPU tensors
(`KBM_BACKEND=gloo KBM_DEVICE=cpu`), at a small `KBM_WORLD_SIZE`. This exercises
the launcher, per-rank compare, rank-asymmetry seeding, and every anti-hack gate
functionally. Real NVLink perf numbers require the 8×H100 node. The node must
never see a correctness bug for the first time.
