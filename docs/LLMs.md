# KernelBench v3 — LLM Agent Guide

Read this file first. Everything needed to run evaluations autonomously.

## CLI

```bash
uv run python bench.py <command> [args]
```

Commands:
- `run <hardware>` — run benchmark (rtx3090, h100, b200, m4max)
- `list-models` — show registered models
- `list-hardware` — show hardware targets with problem counts
- `list-problems <hardware>` — show problems for a target
- `summary <run_dir>` — print results for a completed run

## Running Evaluations

```bash
# RTX 3090 (local GPU)
uv run python bench.py run rtx3090 --models google/gemini-3-flash-preview --levels 1,2,3,4 --workers 4

# H100 (Modal cloud)
uv run python bench.py run h100 --models openai/gpt-5.3-codex --levels 1,2,3,4 --workers 4

# M4 Max (run from macbook)
cd ~/MetalBench && uv run python bench.py run m4max --models minimax/minimax-m2.5 --levels 1,2,3,4 --workers 4

# Dry-run
uv run python bench.py run rtx3090 --models minimax/minimax-m2.5 --levels 1 --problems-per-level 1 --dry-run
```

## Hardware Targets

| Target | GPU | VRAM | Problems | Sandbox |
|--------|-----|------|----------|---------|
| rtx3090 | RTX 3090 | 24GB | 43 | Local |
| h100 | H100 | 80GB | 54 | Modal |
| b200 | B200 | 192GB | 57 | Modal |
| m4max | M4 Max | 128GB | 63 | Local (macOS) |

## Architecture

```
bench.py                    CLI entry point
src/
  hardware/                 Hardware target registry (rtx3090, h100, b200, m4max)
  eval/
    agent.py                Agent loop (Gemini/standard/reasoning)
    context.py              Workspace context + correctness self-check
    results.py              EvalResult dataclass
    guardrails.py           Unified solution validation
    benchmark.py            CUDA + Metal benchmark templates
    fingerprint.py          Hardware metadata capture
  batch.py                  Batch orchestration
  models.py                 Model registry
  api.py                    API communication, prompt caching, token tracking
  tools.py                  Tool schemas + dispatch (read_file, write_file, edit_file, bash, submit)
  prompts.py                System prompts (NVIDIA + Metal variants)
  parsing.py                XML/code parsing
  agent/                    Sandbox implementations (Local, Modal, Metal)
  config/                   Precision matrix, GPU validation
problems/                   Problem definitions (85 files)
outputs/                    Run artifacts (gitignored)
```

## Rules

- Use `uv run` for all Python. Never bare `python` or `pip`.
- Before completing: `uv run --with ruff ruff check . --fix && uv run pytest`
- Models choose their own approach (CUDA C++, Triton, or any compilable kernel)
- No PyTorch operator fallbacks allowed (guardrails enforce this)

## Future Levels (TODO)

- **PTXBench**: Hand-written PTX assembly via CUDA Driver API
- **Multi-GPU**: Tensor parallelism, pipeline parallelism
- **Distributed**: All-reduce, ring-allreduce, NCCL custom kernels
- **FlashInfer-Bench**: Port attention/decode kernels from flashinfer-bench
- **Blackwell-specific**: More FP4/FP6 block-scaled GEMMs, sparse narrow-precision, MoE with tcgen05.mma
