# KernelBench-Hard

Per-op GPU kernel bench: frontier coding agents write competitive CUDA or
Triton kernels, unlimited wall-clock, roofline-graded, reward-hack audited.
Default deck is `problems-rtxpro6000/` (RTX PRO 6000); variants exist as
`problems-h100/` and `problems-b200/`.

| NN | problem | note |
| -- | ------- | ---- |
| 01 | `01_fp8_gemm` | FP8 e4m3 GEMM, off-alignment shapes |
| 02 | `02_kda_cutlass` | Kimi Delta Attention via CUTLASS CuTe |
| 03 | `03_paged_attention` | paged attention decode |
| 05 | `05_topk_bitonic` | top-k via bitonic sort (ms-anchored) |
| 06 | `06_sonic_moe_swiglu` | Sonic-MoE grouped GEMM + SwiGLU |
| 07 | `07_w4a16_gemm` | W4A16 weight-only quantized GEMM |

```bash
uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/01_fp8_gemm
```

See `SPEC.md`, `DEVLOG.md`, and repo-root `AGENTS.md`.
