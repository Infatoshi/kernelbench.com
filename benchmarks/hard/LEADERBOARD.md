# KernelBench-Hard Leaderboard

Hardware: **RTX PRO 6000 Blackwell Workstation** (sm_120, 96 GB GDDR7, 1.8 TB/s peak DRAM bandwidth).

**23 evaluated rows × 9 problems.** Cells show `peak_fraction` of the published throughput peak (1.0 = saturating the relevant tensor-core or memory bandwidth limit) only when correctness and benchmarking both completed. `BENCH` means correctness passed but benchmark timing did not finish, so the cell is unscored. `NO PERF` means correctness passed but no `peak_fraction` was recorded. `FAIL` means a solution was written but failed correctness. `ERR` means no solution was produced.

Annotations (`★`) attached to specific cells live in `results/annotations/<run_id>.yaml`. Two cell verdicts mean the cell number does not measure what the problem name implies — see the **Benchmark design flaws** section below.

## Cross-model grid

| model | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 09 | 10 | scored |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-5.5 [xhigh] | 0.423 ★ | 0.032 | 0.497 ★ | 0.363 ★ | 0.042 | 0.251 ★ | 0.159 ★ | 0.666 | 0.248 ★ | 9/9 |
| claude-opus-4-7 [max] | 0.534 ★ | NO PERF | 0.602 ★ | 0.317 ★ | 0.020 | FAIL | 0.184 ★ | 0.722 ★ | 0.212 | 7/9 |
| deepseek/deepseek-v4-flash | FAIL | 0.009 | 0.167 ★ | 0.138 ★ | FAIL | 0.083 | 0.134 ★ | 0.577 | 0.196 | 7/9 |
| deepseek/deepseek-v4-pro | FAIL | FAIL | 0.027 | 0.101 ★ | 0.011 | 0.108 ★ | 0.125 ★ | 0.667 | 0.232 | 7/9 |
| or/xiaomi/mimo-v2.5-pro | 0.434 ★ | FAIL | ERR | 0.121 ★ | 0.017 | 0.211 ★ | 0.137 ★ | 0.516 | 0.190 | 7/9 |
| claude-opus-4-8 [2026-05-28 opus48-grok max] | 0.533 | BENCH | 0.652 | 0.352 | 0.046 | 0.251 | 0.113 | - | - | 6/7 |
| kimi-k2.6 | FAIL | 0.022 | 0.432 ★ | 0.118 ★ | 0.014 | 0.161 ★ | 0.220 ★ | ERR | ERR | 6/9 |
| MiniMax M3 [2026-06-01] | 0.533 ★ | BENCH | 0.029 | 0.236 ★ | 0.043 ★ | 0.254 ★ | 0.108 | - | - | 6/7 |
| or/qwen/qwen3.6-max-preview | 0.429 ★ | 0.011 | ERR | 0.077 | FAIL | 0.004 | 0.110 ★ | 0.395 | ERR | 6/9 |
| zai/glm-5.1 | FAIL | 0.005 | ERR | 0.125 ★ | ERR | 0.238 ★ | 0.180 ★ | 0.589 | 0.182 | 6/9 |
| claude-opus-4-7 [2026-05-28 finish max] | 0.524 | BENCH | 0.026 | FAIL | 0.045 | 0.247 | 0.100 | - | - | 5/7 |
| gpt-5.5 [2026-05-28 finish xhigh] | 0.537 | BENCH | 0.664 | 0.113 | FAIL | 0.254 | 0.095 | - | - | 5/7 |
| droid/zai/glm-5.1 [2026-05-08] | 0.414 | ERR | 0.252 | 0.234 | ERR | 0.149 | 0.086 | ERR | ERR | 5/9 |
| gemini/gemini-3.5-flash [2026-05-28 finish] | FAIL | FAIL | 0.225 | 0.081 | 0.010 | 0.183 | 0.077 | - | - | 5/7 |
| or/qwen/qwen3.6-plus | 0.431 ★ | ERR | 0.022 | ERR | FAIL | 0.040 | 0.125 ★ | 0.569 | FAIL ★ | 5/9 |
| zai-glm-5.1 [2026-05-13] | INVALID ★ | FAIL | 0.222 | 0.337 | 0.003 | 0.111 | ERR | FAIL | 0.147 | 5/9 |
| cursor/composer-2.5-fast [2026-05-28 finish] | FAIL | BENCH | 0.625 | 0.315 | 0.032 | FAIL | 0.119 | - | - | 4/7 |
| or/minimax/minimax-m2.7 | ERR | ERR | FAIL | 0.034 | FAIL | 0.076 | 0.030 | 0.113 | FAIL ★ | 4/9 |
| or/qwen/qwen3.6-27b | ERR | FAIL | FAIL | ERR | FAIL | 0.082 | ERR | 0.436 | 0.105 | 3/9 |
| zai/glm-5.1 [2026-05-08] | FAIL | NO PERF | ERR | 0.056 | ERR | 0.215 | ERR | ERR | 0.174 | 3/9 |
| grok/grok-build [2026-05-28 opus48-grok max] | FAIL | BENCH | FAIL | 0.037 | FAIL | FAIL | FAIL | - | - | 1/7 |
| zai/glm-5.1 [2026-05-28 finish] | ERR | ERR | ERR | 0.039 | ERR | ERR | ERR | - | - | 1/7 |
| zai-glm-5.1 [2026-05-28 finish] | ERR | ERR | ERR | ERR | ERR | ERR | ERR | - | - | 0/7 |

## Per-problem ceilings

| problem | best peak | best model | n scored |
|---|---|---|---|
| 01_fp8_gemm | 0.537 | gpt-5.5 [2026-05-28 finish xhigh] | 10/23 |
| 02_kda_cutlass | 0.032 | gpt-5.5 [xhigh] | 5/23 |
| 03_paged_attention | 0.664 | gpt-5.5 [2026-05-28 finish xhigh] | 14/23 |
| 04_kahan_softmax | 0.363 | gpt-5.5 [xhigh] | 19/23 |
| 05_topk_bitonic | 0.046 | claude-opus-4-8 [2026-05-28 opus48-grok max] | 11/23 |
| 06_sonic_moe_swiglu | 0.254 | MiniMax M3 [2026-06-01] | 18/23 |
| 07_w4a16_gemm | 0.220 | kimi-k2.6 | 17/23 |
| 09_fmha_preattn_mrope | 0.722 | claude-opus-4-7 [max] | 10/14 |
| 10_patch_embed_conv3d_gemm | 0.248 | gpt-5.5 [xhigh] | 9/14 |

## Benchmark design flaws — read these before citing numbers

Two of the published problems have insufficient guardrails to enforce the algorithmic skill their names imply. The cells below are technically valid scored cells (correctness succeeded and a peak_fraction was recorded), but the peak fractions don't measure what the problem promises.

### 01 fp8_gemm — bf16 dressup

Every passing solution at peak fraction ≥ 0.4 (claude-opus-4-7 0.534, MiniMax-M3 0.533, mimo-v2.5-pro 0.434, qwen3.6-plus 0.431, qwen3.6-max-preview 0.429, gpt-5.5 0.423) casts the fp8 inputs to bf16 inside the kernel and runs a bf16 GEMM. Opus, MiniMax-M3, and gpt-5.5 explicitly pin to `cutlass::arch::Sm80` (Ampere) — they don't touch SM120 FP8 tensor cores at all.

This is technically valid because the reference computes `x.to(bf16) @ w.to(bf16)`, so the model's bf16 GEMM matches reference arithmetic. But the problem name is FP8 GEMM and the prompt suggests using SM120 FP8 tensor cores. The peak fractions on this row reflect bf16 kernel optimization quality on fp8-typed inputs, not FP8 tensor core skill.

**To fix**: tighten tolerance to a value where bf16-via-cast and real fp8-tensor-core math diverge visibly, or explicitly forbid the cast in the prompt with a static-analysis check. Until fixed, this row is `★`-flagged.

### 04 kahan_softmax — Kahan compensation skipped

Six of seven passing solutions skipped the Kahan compensated summation entirely, including both top-tier scores (gpt-5.5 0.363, opus 4.7 max 0.317). Only deepseek-v4-pro — the **lowest** passing peak at 0.101 — actually implemented the algorithm the problem name describes.

Tolerance is loose enough that naive softmax fits within it. So the rubric leaks: models recognize the easy path and take it; the model that does the right algorithmic thing scores lowest because compensated summation has real overhead.

**To fix**: tighten tolerance to a value where naive vs Kahan produce visibly different results on the test inputs, or write a check that detects Kahan-pattern code in solution.py.

### Until fixed: how to read these two rows

- **01 fp8_gemm peaks** — read as bf16-on-fp8-inputs kernel quality, not FP8 tensor core kernel quality.
- **04 kahan_softmax peaks** — read as fast naive softmax, not numerically-stable Kahan softmax. The 0.101 deepseek-v4-pro cell is the only one that measures the targeted algorithm.

These flaws were discovered post-hoc by reading the high-peak solutions. We're publishing the leaderboard with the flaws documented rather than iterating on problem design until perfect — diminishing returns past a point. Future problem-set revisions will close these specific leaks.

## Source data

- `results/leaderboard.json` — machine-readable cross-model grid with full peak fractions, run IDs, and per-problem ceilings.
- `results/annotations/<run_id>.yaml` — per-run human commentary (verdicts: `clean`, `rubric_leak`, `reward_hack`, `interesting`, `bug`). 13 annotations as of this revision.
- `results/annotations/SCHEMA.md` — annotation file format spec.
- `outputs/runs/<run_id>/` — per-run artifacts (transcripts, solution.py, check.log, result.json). Local only; gitignored.
- `DEVLOG.md` — running record of design decisions, dead ends, and lessons.
