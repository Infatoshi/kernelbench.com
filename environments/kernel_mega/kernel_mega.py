"""kernel_mega: verifiers/Prime-RL environment wrapping the NATIVE KernelBench-Mega
agentic harness (`anvil:~/kernelbench.com/benchmarks/mega`).

Two-problem MEGAKERNEL deck: `01_rl_grid_ppo` (PufferLib-style grid-foraging PPO
training megakernel) and `02_kimi_linear_decode` (Kimi-Linear W4A16 decode
megakernel). KernelBench-Mega is its own benchmark, separate from
KernelBench-Hard. It reuses this repo's native-harness, transcript, and roofline
machinery (the shared `kernel_native_harness.build_environment` core), binding
`bench_root` to the mega checkout.

The model iterates in a persistent workspace with the native tools
(write_solution -> run_check -> run_benchmark); reward = the native benchmark's
`peak_fraction` (% of roofline) gated on `check.py` PASS. Megakernel problems
are `regime: memory` (peak_fraction off the HBM-bandwidth roofline) and can
report raw peak_fraction > 1 (dense-FLOPS/bytes formula overcounts skipped
work); the reward CLAMPS to [0,1] (raw value kept as `raw_peak_fraction`).

CPU-import-safe: torch/CUDA only run inside the native check.py/benchmark.py
SUBPROCESS, never in this module's interpreter.

SELF-CONTAINED (option 1): benchmarks/mega's `src/` and `problems/` are VENDORED
into this env dir at `_bench/` (listed in [tool.hatch.build]). The native scoring
subprocess runs check.py/benchmark.py with the ENV's own interpreter (sys.executable
in kernel_native_harness.run_native; torch + benchmark deps declared in this env's
pyproject), NOT the benchmark's `uv run python`. No checkout dependency.
"""

from __future__ import annotations

import os

import verifiers as vf

import kernel_native_harness as knh

MEGA_ROOT = os.environ.get(
    "KERNEL_MEGA_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench"),
)


def load_environment(
    *,
    bench_root: str | None = None,
    hardware: str = "RTX_PRO_6000",
    deck: list[str] | None = None,
    eval_frac: float = 0.2,
    max_turns: int = 12,
    check_timeout_s: int = 900,
    bench_timeout_s: int = 1200,
    enable_judge: bool = False,
    judge_model: str = "z-ai/glm-5.2",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    max_concurrent: int = 1,
    **kwargs,
) -> vf.Environment:
    """Build the kernel_mega verifiers environment.

    kwargs:
        bench_root: KernelBench-Mega checkout (default $KERNEL_MEGA_ROOT or repo-relative).
        hardware: gpu_sku for the roofline (default RTX_PRO_6000 / sm_120).
        deck: optional explicit list of problem dir names; default = all problems.
            e.g. deck=["02_kimi_linear_decode"] for just the published megakernel.
        eval_frac: deterministic hash-based held-out eval split (disjoint from train).
        max_turns: max agentic turns (write/check/benchmark iterations).
        check_timeout_s / bench_timeout_s: per-script subprocess timeouts (megakernels are slow).
        enable_judge: opt-in judge veto (off by default; only zeros a correct reward).
        judge_model / judge_base_url / judge_api_key: judge config (default GLM-5.2 / OpenRouter).
        max_concurrent: cap concurrent GPU scoring subprocesses (reward GPU, not a training GPU).
    """
    return knh.build_environment(
        bench_root=bench_root or MEGA_ROOT,
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
