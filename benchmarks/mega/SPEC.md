# MK-Bench: Design Specification

Last updated: 2026-06-05.

## Purpose

MK-Bench is a small, hand-curated GPU megakernel benchmark where frontier coding agents attempt to build large fused kernels on specific hardware. It inherits the KernelBench-Hard harness style, archive format, and roofline reporting, but its deck starts with one full megakernel problem instead of many operation-level kernels.

## Active Problem

`problems/01_qwen3_decode_block` is the initial MK-Bench problem. It asks agents to implement one Qwen3-0.6B transformer block decode step for batch=1 and one new token, matching the dimensions and spirit of `Infatoshi/megaqwen`:

- hidden size: 1024
- intermediate size: 3072
- Q heads: 16
- KV heads: 8
- head dim: 128
- static BF16 projection weights
- sequence-length sweep: 32, 128, 512

The operation includes input RMSNorm, Q/K/V projection, Q/K RMSNorm, RoPE, cache append, GQA attention decode, O projection, residual, post-attention RMSNorm, SwiGLU MLP, down projection, and final residual.

## Metric

Primary score is fraction of the RTX PRO 6000 memory roofline for the active shape sweep. B=1 decode is dominated by streamed weights, KV-cache traffic, phase boundaries, and intermediate movement rather than dense tensor-core peak. `problem.yaml` still records dense-equivalent FLOPs for telemetry.

The score is the geometric mean across the declared decode sequence lengths. Each input cache has `seq_len - 1` prior tokens, so the current-token attention scan is exactly `seq_len`.

## Correctness

- BF16 tolerance for the first problem is `atol=rtol=0.08`.
- Three seeds per shape: 42, 123, 456.
- `check.py` loads the reference state dict with `strict=True`.
- NaN or inf output fails through the shared correctness helper.
- Forbidden framework shortcuts are declared in `problem.yaml` and rejected by `check.py`.

## Prompt Design

Each `problems/<X>/PROMPT.txt` is a single human-voice task prompt. It names the hardware, points at `reference.py` and `solution.py`, inlines the required semantics and shape sweep, bans obvious vendor/framework shortcuts, and tells the agent to implement, profile, run `check.py`, run `benchmark.py`, and iterate.

## Harness

Use `scripts/run_hard.sh` for all model smoke tests and sweeps. It stages a disposable archive-local workspace, preserves problem definitions, isolates CUDA/Triton/Torch caches, and serializes GPU-facing checks through `outputs/gpu.lock`.

## Adding a New Problem

1. Create `problems/<NN>_<name>/`.
2. Write `reference.py`, `shapes.py`, `problem.yaml`, `check.py`, `benchmark.py`, `sota.py`, and `PROMPT.txt`.
3. Keep prompts human-voice and benchmark definitions immutable after a published run.
4. Smoke-test with `./scripts/run_hard.sh codex gpt-5.5 problems/<NN>_<name> xhigh`.
5. Run `uv run ruff check . --fix && uv run pytest` before declaring repository-level changes complete.
