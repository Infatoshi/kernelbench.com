# KernelBench-CUDA: Design Specification

Last updated: 2026-07-16.

## Purpose

Four hard **CUDA-only** problems. Hard/Mega stay frozen. Language gate fails
Triton/DSL and pure PyTorch without a kernel.

## Perfect stack (v1)

| NN | problem | skill |
| -- | --- | --- |
| 01 | `glm52_fused_moe` | GLM-5.2 MoE (256+shared, top-8) fused; ban Triton/vllm |
| 02 | `deepseek_nsa` | Chinese weird arch: block top-n + sparse attn |
| 03 | `megaqwen_decode` | Improve known MegaQwen CUDA megakernel geometry |
| 04 | `grid_mingru_sps` | Non-LLM RL sim SPS; fusion optional |

## Language gate

`src/eval/cuda_language.py` — Triton/DSL fail; need `load_inline` / `__global__` / `.cu` / PTX / CUTLASS C++.

## Metrics and shapes

Score is always a **geomean over a shape sweep** (Hard FP8-style), including
off-alignment / serving tails (e.g. T=4127, S not multiple of NSA block_size).

- 01, 02: roofline peak_fraction (dense-eq FLOPs where relevant)
- 03: **decode-only** tok/s at ctx ∈ {2k,8k,32k,128k}; prefill untimed; pure
  numeric last_hidden (no tokenizer)
- 04: SPS vs `peak_sps` (150M)


## Harness

```bash
cd benchmarks/cuda
uv run kbh run grok grok-4.5 problems-rtxpro6000/01_vllm_fused_moe
```

## Non-goals

- Softmax/RMSNorm-only tutorial cells
- Paged-attn rematch of Hard
- MLA isolate (use mega decode instead)
- Lightning / Mamba in v1
