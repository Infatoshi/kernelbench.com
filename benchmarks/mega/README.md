# KernelBench-Mega

Megakernel-focused GPU benchmark scaffold. KernelBench-Mega uses the same native-harness, transcript, and roofline machinery as KernelBench-Hard, but its deck is full fused megakernels instead of the old operation-level Hard deck.

The deck is two megakernel problems: a PufferLib-style grid-foraging PPO training megakernel and a Kimi-Linear W4A16 decode megakernel. The live, GPU-scored board (`/mega`) is the Kimi-Linear decode problem.

## PR policy

This repository is published for transparency: it documents the exact prompts, harnesses, traces, kernels, and scoring code used to evaluate models. It is not an open benchmark track, and PRs that change the canonical problems, hardware target, scoring, prompts, or results are not accepted.

## Problem deck

| # | Problem | Hardware | What it tests |
|---|---------|----------|---------------|
| 01 | Grid-foraging PPO training megakernel | RTX PRO 6000 (SM120) | PufferLib-style vectorized GPU RL: fused env step + rollout + PPO update in one kernel |
| 03 | Kimi-Linear W4A16 decode megakernel | RTX PRO 6000 (SM120) | Whole-block fused decode, W4A16 dequant, linear-attention state update, seq-len sweep |

The Kimi-Linear decode problem (`03`) is the published, GPU-scored board (3 GPUs x ~9 models, see `/mega`). Problem `01` (PPO) has reference/check/prompt in place but no published board yet.

## Hardware

- **RTX PRO 6000 Blackwell Workstation** (SM120, 96GB GDDR7, 1.8 TB/s, ~200 BF16 / ~400 FP8 / ~800 FP4 TFLOPS dense)

Required: CUDA 13.x (symlink `/usr/local/cuda-13`), torch 2.11+cu130, Python 3.11+.

## Active model matrix

One harness per model, each pinned to the highest-fidelity native endpoint. See `scripts/sweep.sh` for the current matrix.

## Quick start

```bash
uv sync
./scripts/patch_torch.sh
./scripts/run_hard.sh codex gpt-5.5 problems/02_kimi_linear_decode xhigh
./scripts/sweep.sh
uv run python scripts/roofline_plot.py outputs/runs/<run_dir>
```

## Viewing transcripts in a browser

Each run produces a `transcript.jsonl` or `codex_session.jsonl`. Generate a self-contained HTML viewer with:

```bash
uv run python -m src.viewer outputs/runs/<run_dir>
```

Output: `outputs/runs/<run_dir>/index.html`.
