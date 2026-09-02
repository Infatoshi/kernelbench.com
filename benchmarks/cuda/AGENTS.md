# KernelBench-CUDA

CUDA-only writing bench (4 problems). Hard and Mega are frozen live boards, so this is a separate deck with the same harness DNA: `src/eval/cuda_language.py` is the hard fail, and a numeric PASS with Triton, a kernel DSL, or pure torch still fails the problem (`cuda_language.json` records `framework` / `triton_cheat`). Live board: `/cuda`. Default deck `problems-rtxpro6000/`; `problems-h100/` and `problems-h100sxm/` exist.

| NN | problem | note |
| -- | ------- | ---- |
| 01 | `01_glm52_fused_moe` | GLM-5.2 MoE (E=256, top_k=8, 1 shared); fused; ban Triton |
| 02 | `02_deepseek_nsa` | NSA-inspired block-select sparse attention; ms-anchored (dense-eq roofline is unreadable) |
| 03 | `03_megaqwen_decode` | Qwen3-0.6B geometry; improve [MegaQwen](https://github.com/Infatoshi/MegaQwen); decode-only timed at ctx 2k to 128k; `BASELINE.md` in the problem dir |
| 04 | `04_grid_mingru_sps` | grid + 3xMinGRU(h=256) steps/sec; `peak_sps` 150M is an aspirational ceiling, do not fit it to the best kernel |

```bash
kb -b cuda run|sweep|audit|lint|traces-to-hf ...        # from the Mac, remote worker
cd benchmarks/cuda && uv sync && ./scripts/patch_torch.sh   # on the GPU box
uv run kbh run grok grok-4.5 problems-rtxpro6000/01_glm52_fused_moe
```

Deck is frozen at four; spec-decode tree attention was pitched as a fifth and rejected (2026-07-15), do not re-pitch it. The retired v0 problems (`01_rmsnorm_residual`, `02_online_softmax`) stay retired.

Everything else is shared with Hard and lives in `benchmarks/hard/AGENTS.md`: layout, adding a problem, correctness and numeric stress, results and thin archives, tests, torch policy, the audit YAML schema (`results/annotations/`), and the `KBH_` variables. Harness routes, the shared runner, rented workers: `kbtool/AGENTS.md`. Methodology, including the latency-anchored scoring for `02_deepseek_nsa`: `SPEC.md`. History: `DEVLOG.md`. Publish gates: root `AGENTS.md`.
