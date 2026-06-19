"""kernel_hard: verifiers/Prime-RL environment wrapping the NATIVE KernelBench-Hard
agentic harness (`anvil:~/kernelbench.com/benchmarks/hard`).

~10 operation-level hard CUDA kernels (FP8 GEMM, KDA/CUTLASS, paged attention,
TopK bitonic, sonic-MoE SwiGLU, W4A16 GEMM, FMHA pre-attn mRoPE, conv3d patch
embed, ...). The model iterates in a persistent workspace with the native tools
(write_solution -> run_check -> run_benchmark), and reward = the native
benchmark's `peak_fraction` (% of roofline) gated on the native `check.py` PASS.

This is the SIBLING of kernel_mega: identical native-harness machinery, only the
problem deck differs. Both wrap `kernel_native_harness.build_environment`; this
module just binds `bench_root` to the hard checkout.

CPU-import-safe: torch/CUDA only run inside the native check.py/benchmark.py
SUBPROCESS (the benchmark's own uv env), never in this module's interpreter.

TODO(vendoring): for Hub publishing, vendor benchmarks/hard's `src/` and
`problems/` (+ pyproject/uv.lock) into this env dir and list them in
[tool.hatch.build].include; today we point at the anvil checkout via
KERNEL_HARD_ROOT.
"""

from __future__ import annotations

import os

import verifiers as vf

import kernel_native_harness as knh

HARD_ROOT = os.environ.get(
    "KERNEL_HARD_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "hard")),
)


def load_environment(
    *,
    bench_root: str | None = None,
    hardware: str = "RTX_PRO_6000",
    deck: list[str] | None = None,
    eval_frac: float = 0.2,
    max_turns: int = 12,
    check_timeout_s: int = 600,
    bench_timeout_s: int = 900,
    enable_judge: bool = False,
    judge_model: str = "z-ai/glm-5.2",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    max_concurrent: int = 1,
    **kwargs,
) -> vf.Environment:
    """Build the kernel_hard verifiers environment.

    kwargs:
        bench_root: KernelBench-Hard checkout (default $KERNEL_HARD_ROOT or repo-relative).
        hardware: gpu_sku for the roofline (default RTX_PRO_6000 / sm_120).
        deck: optional explicit list of problem dir names; default = all problems.
        eval_frac: deterministic hash-based held-out eval split (disjoint from train).
        max_turns: max agentic turns (write/check/benchmark iterations).
        check_timeout_s / bench_timeout_s: per-script subprocess timeouts.
        enable_judge: opt-in judge veto (off by default; only zeros a correct reward).
        judge_model / judge_base_url / judge_api_key: judge config (default GLM-5.2 / OpenRouter).
        max_concurrent: cap concurrent GPU scoring subprocesses (reward GPU, not a training GPU).
    """
    return knh.build_environment(
        bench_root=bench_root or HARD_ROOT,
        hardware=hardware,
        deck=deck,
        eval_frac=eval_frac,
        max_turns=max_turns,
        check_timeout_s=check_timeout_s,
        bench_timeout_s=bench_timeout_s,
        enable_judge=enable_judge,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        max_concurrent=max_concurrent,
        **kwargs,
    )
