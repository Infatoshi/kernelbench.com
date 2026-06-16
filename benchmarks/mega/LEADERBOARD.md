# KernelBench-Hard Leaderboard

Hardware: **RTX PRO 6000 Blackwell Workstation** (sm_120, 96 GB GDDR7, 1.8 TB/s peak DRAM bandwidth).

**23 evaluated rows × 8 problems.** Cells show `peak_fraction` of the published throughput peak (1.0 = saturating the relevant tensor-core or memory bandwidth limit) only when correctness and benchmarking both completed. `BENCH` means correctness passed but benchmark timing did not finish, so the cell is unscored. `NO PERF` means correctness passed but no `peak_fraction` was recorded. `FAIL` means a solution was written but failed correctness. `ERR` means no solution was produced.

Annotations (`★`) attached to specific cells live in `results/annotations/<run_id>.yaml`. Some annotations are clean implementation notes; others flag cells where the number does not measure what the problem name implies.

## Cross-model grid

| model | 01 | 02 | 03 | 05 | 06 | 07 | 09 | 10 | scored |
|---|---|---|---|---|---|---|---|---|---|
| gpt-5.5 [xhigh] | 0.423 ★ | 0.032 | 0.497 ★ | 0.042 | 0.251 ★ | 0.159 ★ | 0.666 | 0.248 | 8/8 |
| claude-opus-4-7 [max] | 0.534 ★ | 0.033 | 0.602 ★ | 0.020 | FAIL | 0.184 ★ | 0.722 | 0.212 | 7/8 |
| deepseek/deepseek-v4-flash | FAIL | 0.009 | 0.167 ★ | FAIL | 0.083 | 0.134 ★ | 0.577 | 0.196 | 6/8 |
| deepseek/deepseek-v4-pro | FAIL | FAIL | 0.027 | 0.011 | 0.108 ★ | 0.125 ★ | 0.667 | 0.232 | 6/8 |
| or/xiaomi/mimo-v2.5-pro | 0.434 ★ | FAIL | ERR | 0.017 | 0.211 ★ | 0.137 ★ | 0.516 | 0.190 | 6/8 |
| claude-opus-4-8 [2026-05-28 opus48-grok max] | 0.533 | 0.117 | 0.652 | 0.046 | 0.251 | 0.113 | - | - | 6/6 |
| kimi-k2.6 | FAIL | 0.022 | 0.432 ★ | 0.014 | 0.161 ★ | 0.220 ★ | ERR | ERR | 5/8 |
| MiniMax M3 [2026-06-01] | 0.533 ★ | 0.111 | 0.029 | 0.043 ★ | 0.254 ★ | 0.108 | - | - | 6/6 |
| or/qwen/qwen3.6-max-preview | 0.429 ★ | 0.011 | ERR | FAIL | 0.004 | 0.110 ★ | 0.395 | ERR | 5/8 |
| zai/glm-5.1 | FAIL | 0.005 | ERR | ERR | 0.238 ★ | 0.180 ★ | 0.589 | 0.182 | 5/8 |
| claude-opus-4-7 [2026-05-28 finish max] | 0.524 | 0.117 | 0.026 | 0.045 | 0.247 | 0.100 | - | - | 6/6 |
| gpt-5.5 [2026-05-28 finish xhigh] | 0.537 | 0.009 | 0.664 | FAIL | 0.254 | 0.095 | - | - | 5/6 |
| droid/zai/glm-5.1 [2026-05-08] | 0.414 | ERR | 0.252 | ERR | 0.149 | 0.086 | ERR | ERR | 4/8 |
| gemini/gemini-3.5-flash [2026-05-28 finish] | FAIL | FAIL | 0.225 | 0.010 | 0.183 | 0.077 | - | - | 4/6 |
| or/qwen/qwen3.6-plus | 0.431 ★ | ERR | 0.022 | FAIL | 0.040 | 0.125 ★ | 0.569 | FAIL | 5/8 |
| cursor/composer-2.5-fast [2026-05-28 finish] | FAIL | 0.069 | 0.625 | 0.032 | FAIL | 0.119 | - | - | 4/6 |
| or/minimax/minimax-m2.7 | ERR | ERR | FAIL | FAIL | 0.076 | 0.030 | 0.113 | FAIL | 3/8 |
| or/qwen/qwen3.6-27b | ERR | FAIL | FAIL | FAIL | 0.082 | ERR | 0.436 | 0.105 | 3/8 |
| zai/glm-5.1 [2026-05-08] | FAIL | 0.003 | ERR | ERR | 0.215 | ERR | ERR | 0.174 | 3/8 |
| grok/grok-build [2026-05-28 opus48-grok max] | FAIL | 0.118 | FAIL | FAIL | FAIL | FAIL | - | - | 1/6 |
| zai/glm-5.1 [2026-05-28 finish] | ERR | ERR | ERR | ERR | ERR | ERR | - | - | 0/6 |
| zai-claude/glm-5.1 [2026-05-13] | FAIL | FAIL | 0.222 | 0.003 | 0.111 | ERR | FAIL | 0.147 | 4/8 |
| zai-claude/glm-5.1 [2026-05-28 finish] | ERR | ERR | ERR | ERR | ERR | ERR | - | - | 0/6 |

## Per-problem ceilings

| problem | best peak | best model | n scored |
|---|---|---|---|
| 01_fp8_gemm | 0.537 | gpt-5.5 [2026-05-28 finish xhigh] | 10/23 |
| 02_kda_cutlass | 0.118 | grok/grok-build [2026-05-28 opus48-grok max] | 13/23 |
| 03_paged_attention | 0.664 | gpt-5.5 [2026-05-28 finish xhigh] | 14/23 |
| 05_topk_bitonic | 0.046 | claude-opus-4-8 [2026-05-28 opus48-grok max] | 11/23 |
| 06_sonic_moe_swiglu | 0.254 | MiniMax M3 [2026-06-01] | 18/23 |
| 07_w4a16_gemm | 0.220 | kimi-k2.6 | 17/23 |
| 09_fmha_preattn_mrope | 0.722 | claude-opus-4-7 [max] | 10/14 |
| 10_patch_embed_conv3d_gemm | 0.248 | gpt-5.5 [xhigh] | 9/14 |

## Benchmark design flaws — read these before citing numbers

One published problem still has insufficient guardrails to enforce the algorithmic skill its name implies. The cells below are technically valid scored cells (correctness succeeded and a peak_fraction was recorded), but the peak fractions don't measure what the problem promises.

### 01 fp8_gemm — bf16 dressup

Every passing solution at peak fraction >= 0.4 (claude-opus-4-7 0.534, MiniMax-M3 0.533, mimo-v2.5-pro 0.434, qwen3.6-plus 0.431, qwen3.6-max-preview 0.429, gpt-5.5 0.423) casts the fp8 inputs to bf16 inside the kernel and runs a bf16 GEMM. Opus, MiniMax-M3, and gpt-5.5 explicitly pin to `cutlass::arch::Sm80` (Ampere), so they don't touch SM120 FP8 tensor cores at all.

This is technically valid because the reference computes `x.to(bf16) @ w.to(bf16)`, so the model's bf16 GEMM matches reference arithmetic. But the problem name is FP8 GEMM and the prompt suggests using SM120 FP8 tensor cores. The peak fractions on this row reflect bf16 kernel optimization quality on fp8-typed inputs, not FP8 tensor core skill.

**To fix**: tighten tolerance to a value where bf16-via-cast and real fp8-tensor-core math diverge visibly, or explicitly forbid the cast in the prompt with a static-analysis check. Until fixed, this row is `★`-flagged.

### Until fixed: how to read this row

- **01 fp8_gemm peaks** — read as bf16-on-fp8-inputs kernel quality, not FP8 tensor core kernel quality.

Future problem-set revisions should close this leak before treating the row as uncaveated.

## Source data

- `results/leaderboard.json` — machine-readable cross-model grid with full peak fractions, run IDs, and per-problem ceilings.
- `results/annotations/<run_id>.yaml` — per-run human commentary (verdicts: `clean`, `rubric_leak`, `reward_hack`, `interesting`, `bug`). 27 annotations as of this revision.
- `results/annotations/SCHEMA.md` — annotation file format spec.
- `outputs/runs/<run_id>/` — per-run artifacts (transcripts, solution.py, check.log, result.json). Local only; gitignored.
- `DEVLOG.md` — running record of design decisions, dead ends, and lessons.
