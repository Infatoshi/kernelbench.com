"""kernel_mega: verifiers/Prime-RL environment wrapping the NATIVE KernelBench-Mega
agentic harness (`anvil:~/kernelbench.com/benchmarks/mega`).

Full MEGAKERNEL deck (Qwen3-0.6B decode-block megakernel, Kimi-Linear decode,
RL-grid PPO, ... plus operation-level problems shared with Hard). Per the Mega
README it uses the SAME native-harness, transcript, and roofline machinery as
KernelBench-Hard; only the `problems/` deck differs. So this module is the
SIBLING of kernel_hard: both wrap `kernel_native_harness.build_environment`;
this one binds `bench_root` to the mega checkout.

The model iterates in a persistent workspace with the native tools
(write_solution -> run_check -> run_benchmark); reward = the native benchmark's
`peak_fraction` (% of roofline) gated on `check.py` PASS. Megakernel problems
are `regime: memory` (peak_fraction off the HBM-bandwidth roofline) and can
report raw peak_fraction > 1 (dense-FLOPS/bytes formula overcounts skipped
work); the reward CLAMPS to [0,1] (raw value kept as `raw_peak_fraction`).

CPU-import-safe: torch/CUDA only run inside the native check.py/benchmark.py
SUBPROCESS (the benchmark's own uv env).

TODO(vendoring): for Hub publishing, vendor benchmarks/mega's `src/` and
`problems/` (+ pyproject/uv.lock) into this env dir; today we point at the anvil
checkout via KERNEL_MEGA_ROOT.
"""

from __future__ import annotations

import os

import verifiers as vf

import kernel_native_harness as knh

MEGA_ROOT = os.environ.get(
    "KERNEL_MEGA_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "mega")),
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
            e.g. deck=["01_qwen3_decode_block"] for just the active megakernel.
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
