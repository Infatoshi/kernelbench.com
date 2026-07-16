# KernelBench-CUDA

CUDA-only writing bench (4 problems). Hard/Mega frozen.

| NN | problem | note |
| -- | ------- | ---- |
| 01 | `glm52_fused_moe` | GLM-5.2 MoE (E=256, top_k=8, 1 shared); fused; ban Triton |
| 02 | `deepseek_nsa` | NSA-inspired block-select sparse attention |
| 03 | `megaqwen_decode` | Qwen3-0.6B geometry; improve [MegaQwen](https://github.com/Infatoshi/MegaQwen) |
| 04 | `grid_mingru_sps` | grid + 3×MinGRU(h=256) SPS |

```bash
cd benchmarks/cuda && uv sync && ./scripts/patch_torch.sh
uv run kbh run grok grok-4.5 problems-rtxpro6000/01_glm52_fused_moe
```

See `SPEC.md`, `DEVLOG.md`.
