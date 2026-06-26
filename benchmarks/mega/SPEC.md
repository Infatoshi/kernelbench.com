# KernelBench-Mega: Design Specification

Last updated: 2026-06-05.

## Purpose

KernelBench-Mega is a small, hand-curated GPU megakernel benchmark where frontier coding agents attempt to build large fused kernels on specific hardware. It inherits the KernelBench-Hard harness style, archive format, and roofline reporting, but its deck is full fused megakernels instead of many operation-level kernels.

## Active Problems

The deck is two megakernel problems:

- `problems/01_rl_grid_ppo` — a PufferLib-style grid-foraging PPO training megakernel: fused environment step, rollout collection, and PPO update in one vectorized GPU kernel. Reference/check/prompt are in place; no published board yet.
- `problems/02_kimi_linear_decode` — a Kimi-Linear W4A16 decode megakernel: whole-block fused decode with W4A16 weight dequant and linear-attention state update across a sequence-length sweep. This is the published, GPU-scored board (3 GPUs x ~9 models).

See each problem's `problem.yaml` and `PROMPT.txt` for exact shapes, dtypes, tolerances, and forbidden shortcuts.

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
