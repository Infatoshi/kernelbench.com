# KernelBench v3

Hardware-centric GPU kernel optimization benchmark. LLM agents write CUDA/Triton/MLX kernels in sandboxed environments, competing against PyTorch baselines.

Inspired by the original [KernelBench](https://github.com/ScalingIntelligence/KernelBench) from Scaling Intelligence and the benchmarking methodology in ["This Kernel Was Faster Yesterday" — In Pursuit of High-Fidelity GPU Kernel Benchmarking](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/) from Standard Kernel.

## Hardware Benchmarks

| Benchmark        | GPU                    | Problems | Approach                   | Status             |
| ---------------- | ---------------------- | -------- | -------------------------- | ------------------ |
| **RTX3090Bench** | NVIDIA RTX 3090 (24GB) | 43       | CUDA C++ or Triton         | Tested             |
| **M4MaxBench**   | Apple M4 Max (128GB)   | 63       | MLX                        | Tested             |
| **H100Bench**    | NVIDIA H100 (80GB)     | 54       | CUDA/Triton/CUTLASS        | Coming soon        |
| **B200Bench**    | NVIDIA B200 (192GB)    | 57       | CUDA/Triton/CUTLASS/CuTile | Coming soon        |

Only RTX 3090 and Apple M4 Max have been evaluated end-to-end so far. H100 and B200 scaffolding exists but has not been run at scale — the author does not currently have sustained access to those GPUs.

**Looking for sponsors**: open to compute credits (Modal, Lambda, CoreWeave, Together, Fireworks, etc.) and API credits (OpenAI, Anthropic, Google, xAI, Moonshot, Z.AI, Minimax, DeepSeek, Alibaba) to extend H100/B200 coverage and keep the leaderboard current as new models ship. Reach out if you'd like your lab's model or hardware represented.


## Quick Start

```bash
# List available hardware targets
uv run python bench.py list-hardware

# List models
uv run python bench.py list-models

# Dry-run
uv run python bench.py run rtx3090 --models google/gemini-3-flash-preview --levels 1 --problems-per-level 1 --dry-run

# Full run
uv run python bench.py run rtx3090 --models google/gemini-3-flash-preview --levels 1,2,3,4 --workers 4

# View results
uv run python bench.py summary outputs/batch_eval/run_XXXXXXXX
```

## Problem Levels

- **L1** (15): Simple ops — matmul, softmax, conv, norms
- **L2** (15): Fused ops — matmul+activation chains
- **L3** (3): Architecture blocks — attention, transformer
- **L4** (8): Novel layers — MLA, MoE, GQA, FP8, INT4

M4MaxBench adds 26 Metal-specific problems (image processing, physics, rendering, scientific compute).

## How It Works

1. Agent reads `reference.py` (PyTorch baseline)
2. Writes `solution.py` using GPU kernels (CUDA, Triton, or MLX)
3. Verifies correctness against reference (`torch.allclose`)
4. Solution is benchmarked for speedup vs PyTorch

Every result includes hardware fingerprinting (GPU model, driver version, VRAM, etc.) for reproducibility.

## For LLM Agents

See [docs/LLMs.md](docs/LLMs.md) for comprehensive agent instructions.