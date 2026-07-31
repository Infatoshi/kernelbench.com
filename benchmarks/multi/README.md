# KernelBench-Multi

4×H100 NVLink multi-GPU bench (WIP). Agents turn a PyTorch + NCCL reference
into a fine-grained NVLink kernel; graded on busbw / speedup, not single-GPU
TFLOPS. **Frontier-only** roster gate at launch (`roster.yaml`); off-roster
needs `KBM_ALLOW_OFF_ROSTER=1` and is not publishable. Deck: `problems-h100x4/`.

| NN | problem | note |
| -- | ------- | ---- |
| 01 | `01_allreduce_residual` | pure all-reduce + residual (busbw) |
| 07 | `07_gemm_allreduce_overlap` | TP GEMM overlapped with all-reduce |
| 08 | `08_ring_attention_cp` | context-parallel ring attention |
| 09 | `09_moe_ep_dispatch_combine` | MoE EP dispatch/combine (fp8) |

```bash
cd benchmarks/multi && ./scripts/run_agent.sh grok grok-4.5 01_allreduce_residual high
# full sequential wave: ./scripts/sweep_wave.sh grok grok-4.5 high
```

See `SPEC.md`, `DEVLOG.md`, and repo-root `AGENTS.md`.
