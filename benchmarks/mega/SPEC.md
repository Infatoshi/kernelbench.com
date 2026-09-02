# KernelBench-Mega: Design Specification

Last updated: 2026-09-02.

## Purpose

KernelBench-Mega is a small, hand-curated GPU megakernel benchmark where frontier coding agents attempt to build large fused kernels on specific hardware. It inherits the KernelBench-Hard harness style, archive format, and roofline reporting, but its deck is full fused megakernels instead of many operation-level kernels.

## Active Problems

The deck is one megakernel problem:

- `problems/02_kimi_linear_decode` — a Kimi-Linear W4A16 decode megakernel: whole-block fused decode with W4A16 weight dequant and linear-attention state update across a sequence-length sweep. This is the published, GPU-scored board (3 GPUs). Score is `baseline_latency / solution_latency`, speedup over `baseline.py` (optimized PyTorch); `reference.py` is the correctness oracle only.

`01_rl_grid_ppo` (grid-foraging PPO training megakernel) was removed 2026-07-21; the RL-sim skill moved to the CUDA bench.

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

## Megakernel authenticity: judge gate + advisory tripwires

v2.1 (2026-07-01). A scored path must be one genuinely fused kernel. The two ways a submission fakes this: hidden launches (a Python loop of small kernels replayed under `torch.cuda.CUDAGraph` or a `torch.compile` region, so the replay is one launch but nothing is fused on-chip) and no kernel at all (eager PyTorch with a docstring claiming fusion).

A substring ban on those tokens was tried (v2) and killed: the red-team battery in `tests/test_megakernel_evidence.py` (cases A1 to A7) showed it is leaky (`getattr(torch.cuda, "CUDAGra"+"ph")`, `importlib` runtime codegen) and brittle (an honest "no torch.compile, no CUDA graphs" comment false-fails). Authenticity is decided by an LLM judge reading the code, fed deterministic advisory evidence.

- Bright line, hard fail in `check.py`: importing a prebuilt library (`transformers`, `vllm`, `marlin`, `pufferlib`, ...), matched by AST import statements over `solution.py` plus every local module it imports, recursively. Naming a lib in a comment does not fail.
- Advisory tripwires, never auto-reject (`src/eval/megakernel.py`, CLI `scripts/megakernel_evidence.py`): `kernel_count` (`@triton.jit`, `load_inline`, `__global__ void`); `graph` and `compile` on comment-and-string-stripped code; `codegen` (`exec`/`eval`/`compile`, `importlib.import_module`, writing a `.py`/`.cu`/`.so`); `obfuscation` (`getattr(x, "a"+"b")`, string-concat folding into a banned token, AST-level). `check.py` writes `framework.txt` (`eager`/`triton`/`cuda_raw`/`cudagraph`/`compile`/`ptx`) as a coarse label.
- Judge gate: the mandatory pre-publish audit renders the prompt from `render_judge_prompt(...)` (`scripts/megakernel_evidence.py <run_dir> --prompt --problem 02_kimi_linear_decode`), reasons from the code treating tripwires as hints and docstrings as untrusted, and records the verdict in the audit YAML: `megakernel_authentic: true|false` plus `authenticity_reason:`. Omitted means not yet judged; judge before you publish. `scripts/build_mega_leaderboard.py` excludes `false` alongside the contamination exclusion and renders the `megakernel` column on `/mega`.
- Known gap: the tripwires and judge do not measure launch count. The airtight complement is a profiler launch-count gate (warm up, profile one step, assert one launch, handling graph replay and memcpy/memset); needs on-GPU validation before wiring in.

Code comments in `check.py`, `problem.yaml`, and `src/eval/megakernel.py` still cite `docs/megakernel_authenticity_judge.md`; that file was folded into this section on 2026-09-02 (the frozen problem files are not edited for a comment).

## Adding a New Problem

1. Create `problems/<NN>_<name>/`.
2. Write `reference.py`, `shapes.py`, `problem.yaml`, `check.py`, `benchmark.py`, `sota.py`, and `PROMPT.txt`.
3. Keep prompts human-voice and benchmark definitions immutable after a published run.
4. Smoke-test with `./scripts/run_hard.sh codex gpt-5.5 problems/<NN>_<name> xhigh`.
5. Run `uv run pytest` before declaring repository-level changes complete.
