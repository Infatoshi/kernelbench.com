# V4 Novel Ops Research Briefs

Compiled 2026-04-14 from 10 parallel subagent research passes. Each brief
identifies the canonical upstream, the best tuned baseline available on
Ampere SM86, default shapes, and numerical tolerance hints.

This file is the working reference for building `problems/v4_rtx3090/`
kernels. Each op has a companion `<n>_<name>.py` in that directory that
exercises the math with a naive PyTorch implementation; the tuned baseline
listed here is what we grade solutions against.

---

## 1. DeepSeek Multi-head Latent Attention (MLA)

- **Upstream**: github.com/deepseek-ai/DeepSeek-V3, github.com/deepseek-ai/FlashMLA, arxiv 2412.19437
- **Math**: low-rank KV compression into `c_kv` (kv_lora_rank=512), decompress to per-head K,V at compute time; RoPE applied only to `qk_rope_head_dim=64` subset of Q/K, `qk_nope_head_dim=128` bypasses; weight absorption trick folds `W_up` into query projection offline
- **Best Ampere baseline**: SGLang hybrid_attn_backend Triton MLA (FlashMLA is Hopper-only)
- **Shapes (DeepSeek-V3)**: heads=128, qk_nope=128, qk_rope=64, v_head=128, q_lora=1536, kv_lora=512
- **Ampere notes**: BF16 sufficient, no FP8 TC needed for forward; softmax non-determinism under reduction order
- **Tolerance**: atol=1e-2, rtol=1e-2 for BF16 vs FP32 reference

## 2. DeepSeek Sparse Attention (DSA)

- **Upstream**: huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp (Sept 2025), arxiv 2512.02556
- **Math**: two-stage indexer+selector. FP8 "lightning indexer" scores Q vs prior tokens (L×L); top-k=2048 keys selected per query; full attention runs only on selected subset. Dynamic per-query sparsity
- **Best Ampere baseline**: no native DSA kernel for Ampere yet. Fallback: SGLang NSA backend (dense + mask) or FlashInfer block-sparse
- **Shapes**: heads=128, head_dim=192 (128 nope + 64 rope), top-k=2048, seqlen typical 163k
- **Ampere notes**: FP8 indexer needs manual blockwise rescaling; BF16 safe for main attn; top-k ties break deterministically with stable sort
- **Tolerance**: atol=1e-2, rtol=1e-3 for single-kernel validation

## 3. MiniMax Lightning Attention

- **Upstream**: github.com/MiniMax-AI/MiniMax-01, arxiv 2501.08313 (MiniMax-01), arxiv 2401.04658 (Lightning-2)
- **Math**: linear attention recurrence `o = φ(q)·(S + φ(k)ᵀv)` with SiLU feature map; intra-block dense + inter-block recurrent state; chunked O(Nd²) training
- **Best Ampere baseline**: github.com/OpenNLPLab/lightning-attention Triton kernel; fla.ops.lightning_attn as alternative
- **Shapes (MiniMax-01)**: heads=48, head_dim=128, chunk=512
- **Ampere notes**: FP32 accumulator for `k^T v` state required at seqlen > 100k; BF16 fine at shorter seqs
- **Tolerance**: BF16→FP32 rtol=1e-3 atol=1e-5 (upstream test); inter-block state loosens to ~1-2% at 1M tokens

## 4. Gated DeltaNet (Kimi/NVLabs/FLA)

- **Upstream**: github.com/fla-org/flash-linear-attention (gated_delta_rule), arxiv 2406.06484 (Yang et al), NVLabs ICLR 2025, arxiv 2510.26692 (Kimi Linear)
- **Math**: fixed-size state S∈R^(d_k×d_v). Per-step forget S←S⊙exp(-β_t), delta update S+=qᵀkv, readout o=softmax(q)·S·1. Chunked parallel form splits dense intra + low-rank inter
- **Best Ampere baseline**: `fla.fused_recurrent_gated_delta_rule` Triton; vLLM has variant. SM86 support not formally verified — test empirically
- **Shapes (fla tests)**: heads=32, head_dim=128, seqlen=2048-4096, batch=8
- **Ampere notes**: BF16 state accumulation error compounds linearly with T; FLA uses L2Warp. Chunked vs recurrent diverge slightly
- **Tolerance**: FLA suite: atol=1e-2, rtol=1e-2 fwd; 1e-3 bwd

## 5. Flash Linear Attention (chunked)

- **Upstream**: github.com/fla-org/flash-linear-attention, arxiv 2312.06635 (Gated Linear Attention), arxiv 2503.14376 (Tiled FLA)
- **Math**: linear attn with chunkwise parallel: within-chunk dense φ(Q)·(φ(K)ᵀV), across-chunk recurrent state transfer. Feature map typically `elu+1` or exp-variant
- **Best Ampere baseline**: `fla.ops.linear_attn` Triton kernel (parallel_attn_fwd_kernel). SM86 supported
- **Shapes**: heads=8-16, head_dim=64-128, seqlen=512-4096, chunk=64-128
- **Ampere notes**: Chunk=64 or 128 fits SM86 SRAM (~96KB/SM). BF16 has ~10x higher drift than FP32 — keep FP32 accumulators even with BF16 io
- **Tolerance**: FP32 atol=1e-5 rtol=1e-3; BF16 relaxed to atol=1e-2 rtol=5e-2

## 6. POD-Attention (FlashInfer)

- **Upstream**: github.com/flashinfer-ai/flashinfer, arxiv 2410.18038 (ASPLOS'25)
- **Math**: standard causal SDPA with paged KV, but one kernel schedules mixed batch (prefill qL=512-8192 and decode qL=1) on same SMs via runtime CTA assignment. Virtual decode CTAs partition shared memory
- **Best Ampere baseline**: `flashinfer.pod_attention()` — supports SM86. Compare against naive `batch_prefill_with_paged_kv_cache` + separate decode kernel launches
- **Shapes**: B=8-32, mix of 4 prefill + 12 decode, H=8-32, H_kv=1-8 (GQA), D=128, page=16
- **Ampere notes**: BF16 preferred over FP16 on SM86 TC; softmax in FP32 accumulator
- **Tolerance**: rtol=1e-2 atol=1e-3 for BF16 vs FP32 reference

## 7. Mamba-2 Selective Scan

- **Upstream**: github.com/state-spaces/mamba, arxiv 2405.21060 (Mamba-2), tridao.me SSD blog
- **Math**: h_t = A_t·h_{t-1} + B_t·x_t; y_t = C_t·h_t. Mamba-2 SSD decomposes into 4 steps (3 matmul-heavy + 1 short scan over N=64-256)
- **Best Ampere baseline**: `selective_scan_cuda` (SM80+ binary compat should cover SM86), or `mamba_chunk_scan_combined` Triton (5 fused kernels)
- **Shapes (Mamba-2.7B)**: d_model=4096, d_state=128, chunk=256, seqlen=2048-4096
- **Ampere notes**: store h in FP32 even with BF16 compute — recurrence is cancellation-sensitive. Official code uses independent cumsum batches to avoid subtraction-based cumsum pitfalls
- **Tolerance**: initial atol=1e-5 rtol=1e-4 FP32; BF16 relax per empirical

## 8. ScatterMoE

- **Upstream**: github.com/shawntan/scattermoe, arxiv 2403.08245
- **Math**: top-k routing; sort tokens by expert; single fused `scatter2scatter` loops experts + gather/matmul/accumulate + weighted scatter with gate coefficients. No explicit permute tensors
- **Best Ampere baseline**: `scattermoe.kernels.ops.scatter2scatter` Triton (BLOCK_N=128, BLOCK_K=32, num_stages=4)
- **Shapes (upstream tests)**: E=8, top_k=2-4, D_in=128-1000, hidden=4·D_in/k, tokens=1-512
- **Ampere notes**: FP32 stable at 1e-4; FP16 has Triton SM86 `.dot()` regression (issue #9830); BF16 avoids native dot accel on some Triton versions; ALLOW_TF32=True default
- **Tolerance**: FP32 atol=1e-4, FP16/BF16 atol=1e-2

## 9. Punica SGMV

- **Upstream**: github.com/punica-ai/punica, arxiv 2310.18547 (MLSys 2024)
- **Math**: per-row adapter-indexed low-rank update. `y[i] = base[i] + scale·x[i]@A[adapter[i]]@B[adapter[i]]`. Distinct from BGMV (single adapter batch) and grouped GEMM (variable M). Split as sgmv_shrink (dense→rank) + sgmv_expand (rank→dense)
- **Best Ampere baseline**: `punica.ops.sgmv` CUTLASS kernel; vLLM punica_wrapper wraps it. SM80+ supported; 3090 validated in paper
- **Shapes**: N=8-64 requests, K=4-64 adapters, D=4096 (Llama), R=8-64
- **Ampere notes**: BF16 preferred; FP16 accumulator should upcast to FP32 at D>1024. TF32 inherited from SM80
- **Tolerance**: atol=1e-4 rtol=1e-3 BF16 (industry default; Punica tests not publicly documented)

## 10. Triangle Multiplication (AlphaFold TriMul)

- **Upstream**: github.com/aqlaboratory/openfold (TriangleMultiplicativeUpdate), deepmind/alphafold, arxiv 2510.18870, NVIDIA cuEquivariance
- **Math**: pair tensor z∈R^(N×N×C). LayerNorm→project a,b→gate via sigmoid→outgoing contraction h_ij = Σ_k a_ik⊙b_jk→LayerNorm→project→gate→final project. 3D outer-product reduction
- **Best Ampere baseline**: NVIDIA cuEquivariance `triangle_multiplicative_update` (hidden_dim must be multiple of 32; Ampere supported). No mature OSS Triton port
- **Shapes (AlphaFold MSA Evoformer)**: N=128-512 for 3090 VRAM, c_z=128, c_hidden=32
- **Ampere notes**: memory-bound, O(N²·C) intermediates. N=512 C=128 ≈ 256 GB/s footprint, bandwidth-bound on 3090. BF16 safe for reduction; sigmoid gating numerically stable
- **Tolerance**: FP32 atol=1e-5 rtol=1e-4; BF16 atol=1e-4 rtol=1e-3

---

## Cross-op Ampere kernel availability

| Op | Ampere tuned kernel? | Baseline |
|---|---|---|
| MLA | ⚠️ no dedicated kernel | SGLang Triton fallback |
| DSA | ❌ too new | naive dense+mask |
| Lightning | ✅ | OpenNLPLab Triton |
| Gated DeltaNet | ⚠️ probable | FLA (verify SM86) |
| FLA linear | ✅ | fla.ops.linear_attn |
| POD-Attention | ✅ | flashinfer.pod_attention |
| Mamba scan | ✅ | selective_scan_cuda |
| ScatterMoE | ✅ | scatter2scatter Triton |
| Punica SGMV | ✅ | punica.ops.sgmv |
| TriMul | ⚠️ proprietary only | cuEquivariance |

## Common tolerance pattern

For all ops: `atol=1e-2 rtol=1e-2` is a reasonable default for BF16 solutions vs FP32 reference, with per-op calibration via the FP32 oracle procedure.
