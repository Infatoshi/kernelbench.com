# v4 RTX 3090 Benchmark Spec

Locked contract for the v4 RTX 3090 (Ampere SM86) problem slate.

## Baseline policy

Every `benchmark.py` grades against:
- **Primary (graded)**: `torch.compile(model, mode="reduce-overhead", fullgraph=False)` — represents production. Fallback to eager if compile raises.
- **Supplementary (reported, ungraded)**: plain torch eager.

Speedup printed for both; leaderboard uses primary.

Warmups (3 for compile trigger + 5 timed-warmup), then median of 30 timed iterations. No AMP autocast — dtype comes from `get_inputs()`. No `torch.jit.script`. No `max-autotune`.

## Guardrails

Existing FORBIDDEN list only (`torch.matmul/mm/bmm/conv2d`, `F.linear/conv2d`, `torch.compile`, `@torch.jit.script`, `flash_attn`, `xformers`, `fla.ops`, dtype-cast-wrappers-only). **No additional guardrails.** Reward hacking is detected post-hoc via inspection and optional LLM judge.

## Framework gating

Each problem specifies `FRAMEWORK_GATE` in its `reference.py`. Enforcement:
1. `setup_workspace.py` injects the gate rule into the generated `CLAUDE.md`.
2. After run, `src/eval/framework_detect.py` identifies the framework used in `solution.py`.
3. Solutions not matching the gate are marked FAIL even if `check.py` passes.

Gate labels: `triton` | `cutlass2` | `cutlass3` | `cuda_wmma` | `ptx` | `cutile` | `cuda_raw` | `no_triton` | `None` (open).

## Dropped from 3090 slate

These problems depend on FLA kernels that crash on Ampere SM86 (Triton autotune `IndexError`). Deferred to H100/B200 track:

- Gated DeltaNet (fla.ops.fused_recurrent_gated_delta_rule)
- Kimi Delta Attention (fla.ops.chunk_kda)
- FLA chunked linear attention (fla.ops.linear_attn)

Final 3090 slate: **37 problems**.

## Problem slate (37)

### GEMM compute-bound (5)
| # | Name | Gate | Baseline source |
|---|---|---|---|
| 1 | GEMM FFN up-proj (Llama-70B) | cuda_wmma | torch.matmul |
| 2 | GEMM square attn proj | ptx | torch.matmul |
| 3 | GEMM + bias + SiLU + store (epilogue) | cuda_wmma | unfused torch |
| 4 | W4A16 dequant-fused GEMM | cutlass2 | naive dequant + torch.matmul |
| 5 | Grouped GEMM (MoE variable M) | cutlass2 | for-loop over experts |

### GEMV / decode (3)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 6 | GEMV decode projection | triton | torch.matmul |
| 7 | Fused QKV GEMV + RoPE | no_triton | unfused torch |
| 8 | W4A16 GEMV | cuda_wmma | naive dequant + matmul |

### Attention classic (4)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 9 | FA2 fwd causal | triton | naive torch softmax(QK)V |
| 10 | FA2 bwd causal | cutlass2 | autograd of (9) |
| 11 | GQA fwd (Llama-3) | triton | repeat_interleave + SDPA-like torch |
| 12 | Paged attention decode | triton | gather + torch SDPA |

### Attention novel (4)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 13 | MLA fwd (DeepSeek-V3) | no_triton | naive torch unroll |
| 14 | DSA fwd (DeepSeek-V3.2) | triton | top-k + dense mask |
| 15 | Lightning Attention (MiniMax) | triton | quadratic φ(Q)φ(K)ᵀV torch |
| 18 | POD-Attention (mixed batch) | no_triton | for-loop per-req SDPA |

### Non-attention sequence (2)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 19 | Mamba-2 selective scan | triton | sequential python loop |
| 20 | Triangle multiplication | triton | einsum |

### Norms + activations fused (4)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 21 | Liger RMSNorm + RoPE + QKV | no_triton | unfused torch |
| 22 | RMSNorm + residual add fwd | open | torch rmsnorm + add |
| 23 | RMSNorm bwd (dW + dX) | open | autograd |
| 24 | SwiGLU (gate·up·down fused) | open | unfused torch |

### MoE (3)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 25 | Top-K routing + softmax + permute | open | torch gather+softmax |
| 26 | Fused expert FFN per-expert | cutlass2 | for-loop expert FFN |
| 27 | ScatterMoE permutation-free | triton | naive gather-compute-scatter |

### Serving (3)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 28 | Punica SGMV (LoRA) | cutlass2 | row-by-row LoRA loop |
| 29 | Sort-free top-p sampling | open | torch.sort + cumsum |
| 30 | KV cache append (paged) | open | scatter via indexing |

### Position / embed (2)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 31 | RoPE fused with QK | open | torch elementwise RoPE |
| 32 | Embedding gather + scale | open | torch.embedding + mul |

### Quant utility (2)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 33 | GPTQ-style groupwise dequant | open | torch unpack loop |
| 34 | Blockwise INT8 quant + scale | open | torch clamp+round |

### Reduction / scan (2)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 35 | Segmented reduction (varlen) | open | scatter_reduce loop |
| 36 | Inclusive scan (top-k path) | ptx | torch.cumsum |

### Backward (2)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 37 | GEMM bwd (dW + dX) | cuda_wmma | torch.matmul.backward |
| 39 | MLA bwd | no_triton | autograd of MLA fwd |

### Graphics holdout (1)
| # | Name | Gate | Baseline |
|---|---|---|---|
| 40 | Bloom or particles (legacy) | open | naive torch |

## Gate distribution

| Gate | Count | Intent |
|---|---|---|
| triton | 10 | force Triton fluency for attention + scan |
| cuda_wmma | 4 | force raw-CUDA tensor-core work |
| cutlass2 | 5 | force CUTLASS 2.x / CuTe on 3090 |
| ptx | 2 | force inline `mma.sync` PTX |
| no_triton | 5 | force any CUDA dialect except Triton |
| open | 11 | let model pick |
| **total** | **37** | |

## Implementation status

- [x] infrastructure (`src/eval/framework_detect.py`, `FRAMEWORK_GATE` in setup_workspace)
- [x] 3 templates: 18 POD-Attention, 19 Mamba, 28 Punica
- [ ] 34 remaining reference.py files
- [ ] `scripts/calibrate_tolerance.py` (FP32 oracle procedure)
- [ ] benchmark.py template: pin to `torch.compile(reduce-overhead)` as graded, eager supplementary
- [ ] Gate-violation rejection wired into `run_harness.sh` result JSON
