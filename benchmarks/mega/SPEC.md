# KernelBench-Mega: Design Specification

Last updated: 2026-11-16 (grader v3).

## Purpose

KernelBench-Mega is a small, hand-curated GPU megakernel benchmark where frontier coding agents attempt to build large fused kernels on specific hardware. It inherits the KernelBench-Hard harness style, archive format, and roofline reporting, but its deck is full fused megakernels instead of many operation-level kernels.

## Active Problems

The deck is one megakernel problem:

- `problems/02_kimi_linear_decode` — a Kimi-Linear W4A16 decode megakernel: whole-block fused decode with W4A16 weight dequant and linear-attention state update across a sequence-length sweep. This is the published, GPU-scored board (3 GPUs). Score is `baseline_latency / solution_latency`, speedup over `baseline.py` (optimized PyTorch); `reference.py` is the correctness oracle only.

`01_rl_grid_ppo` (grid-foraging PPO training megakernel) was removed 2026-07-21; the RL-sim skill moved to the CUDA bench.

See each problem's `problem.yaml` and `PROMPT.txt` for exact shapes, dtypes, tolerances, and forbidden shortcuts.

## Metric

The score is the speedup over `baseline.py` (optimized eager PyTorch that materializes the dequantized weights): baseline latency divided by solution latency per decode step at batch 1, geometric mean over the context sweep in `shapes.py` (2048, 8192, 16384). It is stored in `peak_fraction` as a multiple, not a roofline fraction.

Grader v3 (2026-11-16): `benchmark.py` times the two sides in separate processes, solution first. The parent creates no CUDA context; the baseline process never imports the solution and starts only after the solution process and everything it spawned have exited, so device state a solution leaves behind (L2 persistence, stream attributes, stray processes) cannot slow the baseline. Before v3 the baseline ran in the solution's process and read up to 10% slow after a kernel that pinned L2. `baseline.py` is part of the graded-surface digest.

The baseline is CPU-bound eager PyTorch, so the same kernel scores differently on hosts with different CPUs (one kernel read 29.9x on a rented box and 26.9x on tetra under the same benchmark). Rows of one GPU board are comparable only when timed on one host; the RTX PRO 6000 board was re-timed on tetra under v3 for that reason. The H100 and B200 boards were graded on rented boxes under the pre-v3 grader and stay that way until replayed there.

## Correctness

- Per step, not a trajectory: reference and solution get an identical state and one new token and each runs one decode step. The next-token hidden, the KDA recurrent state and the MLA latent cache must each reach cosine 0.98 (`problem.yaml` `tolerance.cos_sim`). NaN or inf fails.
- Graded cache: seeds 0, 1, 2 at context 2048 and 8192 on the random cache from `reference.init_state`. On that cache the MLA softmax is near uniform and attention moves the output by about 0.1%, so it cannot tell a right attention path from a wrong one.
- Attention-sensitive cases (v3, 2026-11-16): seeds 27, 64, 76 (`shapes.ATTENTION_SEEDS`) at the same two contexts on `reference.init_state_attn`, where 64 cache rows have their latent and rope key scaled by 60 so attention carries about 90% of the output. In the 2026-11-16 replay of the RTX PRO 6000 board the 29 passing kernels read 0.9968 to 1.0000 on these cases and the 11 that fail read 0.21 to 0.96 (every one a confirmed kernel defect, `DEVLOG.md` 2026-11-16); `baseline.py` holds 0.996 or better from gain 1 to 120. Seeds were chosen so no MoE router has a near-tie at its top-8 cut (honest rounding can flip a tie, and these caches amplify the flip).
- The check proves each attention case before using it: the reference's output with the cache history zeroed must differ (cosine at most 0.7) and every router's top-8 margin must be at least 2.5%. A failure there prints `GRADER ERROR` and exits 2; it is the grader's fault, not the solution's.
- `benchmark.py` runs a one-step gate on both cache kinds at every shape before timing; a failure scores 0.0 and the baseline is never run.
- `check.py` loads the reference state dict with `strict=True`.
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
