# KernelBench-Mega

Megakernel-focused GPU benchmark scaffold. KernelBench-Mega uses the same native-harness, transcript, and roofline machinery as KernelBench-Hard, but starts with one full megakernel problem instead of the old operation-level Hard deck.

The first problem is a MegaQwen-style Qwen3-0.6B batch=1 decode-block megakernel on the RTX PRO 6000 Blackwell workstation.

## PR policy

This repository is published for transparency: it documents the exact prompts, harnesses, traces, kernels, and scoring code used to evaluate models. It is not an open benchmark track, and PRs that change the canonical problems, hardware target, scoring, prompts, or results are not accepted.

## Problem deck

| # | Problem | Hardware | What it tests |
|---|---------|----------|---------------|
| 01 | Qwen3-0.6B decode block megakernel | RTX PRO 6000 (SM120) | Whole-block fusion, B=1 decode, seq-len-scaled GQA attention, RoPE, RMSNorm, SwiGLU MLP, cooperative scheduling |

Historical KernelBench-Hard-derived problem directories may still exist in this repo while the scaffold is being converted. They are not part of the active KernelBench-Mega deck unless listed above or included by the sweep scripts.

## Hardware

- **RTX PRO 6000 Blackwell Workstation** (SM120, 96GB GDDR7, 1.8 TB/s, ~200 BF16 / ~400 FP8 / ~800 FP4 TFLOPS dense)

Required: CUDA 13.x (symlink `/usr/local/cuda-13`), torch 2.11+cu130, Python 3.11+.

## Active model matrix

One harness per model, each pinned to the highest-fidelity native endpoint. See `scripts/sweep.sh` for the current matrix.

## Quick start

```bash
uv sync
./scripts/patch_torch.sh
./scripts/run_hard.sh codex gpt-5.5 problems/01_qwen3_decode_block xhigh
./scripts/sweep.sh
uv run python scripts/roofline_plot.py outputs/runs/<run_dir>
```

## Viewing transcripts in a browser

Each run produces a `transcript.jsonl` or `codex_session.jsonl`. Generate a self-contained HTML viewer with:

```bash
uv run python -m src.viewer outputs/runs/<run_dir>
```

Output: `outputs/runs/<run_dir>/index.html`.
