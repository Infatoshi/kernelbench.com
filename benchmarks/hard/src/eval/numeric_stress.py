"""Numeric distribution stress cases for correctness checks.

These cases are not hidden shapes. They rerun the same shape/seed validation
under a few scale regimes that catch kernels that only work for the nominal
N(0, 1)-ish inputs.
"""
from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

ToleranceOverride = Mapping[str, float | Mapping[str, float]]


@dataclass(frozen=True)
class NumericStressCase:
    name: str
    input_scales: Mapping[int, float] = field(default_factory=dict)
    state_scales: Mapping[str, float] = field(default_factory=dict)
    tolerance: ToleranceOverride | None = None


NOMINAL_CASE = NumericStressCase("nominal")

_SMALL_BF16 = {"bfloat16": {"atol": 1e-4, "rtol": 5e-2}}
_MED_BF16 = {"bfloat16": {"atol": 5e-4, "rtol": 5e-2}}
_LARGE_BF16 = {"bfloat16": {"atol": 2e-1, "rtol": 5e-2}}
_TINY_FP32 = {"float32": {"atol": 1e-7, "rtol": 1e-4}}
_LARGE_FP32 = {"float32": {"atol": 1e-1, "rtol": 1e-4}}

_CASES: dict[str, tuple[NumericStressCase, ...]] = {
    "01_fp8_gemm": (
        # fp8 x fp8 has accumulation-order noise vs the bf16-matmul reference that
        # scales with input magnitude; atols are calibrated to absorb it (measured
        # fp8-MMA residual x ~1.5 margin) while rtol=5e-2 still catches gross error.
        NumericStressCase("small_input", input_scales={0: 1e-3},
                          tolerance={"bfloat16": {"atol": 5e-4, "rtol": 5e-2}}),
        NumericStressCase("large_input", input_scales={0: 64.0},
                          tolerance={"bfloat16": {"atol": 12.0, "rtol": 5e-2}}),
        NumericStressCase(
            "small_weight",
            state_scales={"weight": 1e-2},
            tolerance={"bfloat16": {"atol": 3e-3, "rtol": 5e-2}},
        ),
    ),
    "02_kda_cutlass": (
        NumericStressCase(
            "small_qkv",
            input_scales={0: 1e-2, 1: 1e-2, 2: 1e-2},
            tolerance=_MED_BF16,
        ),
        NumericStressCase(
            "large_qkv",
            input_scales={0: 2.0, 1: 2.0, 2: 2.0},
            tolerance={"bfloat16": {"atol": 5e-2, "rtol": 5e-2}},
        ),
    ),
    "03_paged_attention": (
        NumericStressCase(
            "small_q_kv",
            input_scales={0: 1e-2, 1: 1e-2},
            tolerance=_MED_BF16,
        ),
        NumericStressCase(
            "large_q_kv",
            input_scales={0: 8.0, 1: 8.0},
            tolerance={"bfloat16": {"atol": 5e-2, "rtol": 5e-2}},
        ),
    ),
    "05_topk_bitonic": (
        NumericStressCase("tiny_values", input_scales={0: 1e-4}, tolerance=_TINY_FP32),
        NumericStressCase("large_values", input_scales={0: 1e3}, tolerance=_LARGE_FP32),
    ),
    "06_sonic_moe_swiglu": (
        NumericStressCase("small_hidden", input_scales={0: 1e-2}, tolerance=_MED_BF16),
        NumericStressCase(
            "large_hidden",
            input_scales={0: 8.0},
            tolerance={"bfloat16": {"atol": 1e-1, "rtol": 5e-2}},
        ),
    ),
    "07_w4a16_gemm": (
        NumericStressCase("small_activation", input_scales={0: 1e-3}, tolerance=_SMALL_BF16),
        NumericStressCase(
            "large_activation",
            input_scales={0: 64.0},
            tolerance={"bfloat16": {"atol": 1.0, "rtol": 5e-2}},
        ),
    ),
}


def numeric_stress_cases(problem_name: str) -> tuple[NumericStressCase, ...]:
    if os.environ.get("KBH_NUMERIC_STRESS", "1").lower() in {"0", "false", "no"}:
        return (NOMINAL_CASE,)
    return (NOMINAL_CASE, *_CASES.get(problem_name, ()))


def tolerance_for_case(
    base: dict | None,
    case: NumericStressCase,
) -> dict | None:
    if case.tolerance is None:
        return base
    merged = dict(base or {})
    merged.update(case.tolerance)
    return merged


@contextmanager
def numeric_stress_context(
    ref_model: torch.nn.Module,
    sol_model: torch.nn.Module,
    inputs: Sequence[object],
    case: NumericStressCase,
) -> Iterator[list[object]]:
    backups = _scale_states([ref_model, sol_model], case.state_scales)
    try:
        yield _scale_inputs(inputs, case.input_scales)
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            for tensor, original in backups:
                tensor.copy_(original)


def _scale_inputs(inputs: Sequence[object], scales: Mapping[int, float]) -> list[object]:
    out = list(inputs)
    for idx, scale in scales.items():
        if idx >= len(out):
            raise IndexError(f"input scale index {idx} out of range for {len(out)} inputs")
        value = out[idx]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"input {idx} is not a tensor and cannot be scaled")
        out[idx] = _scale_tensor(value, scale)
    return out


def _scale_states(
    models: Sequence[torch.nn.Module],
    scales: Mapping[str, float],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    backups: list[tuple[torch.Tensor, torch.Tensor]] = []
    if not scales:
        return backups
    for model in models:
        state = dict(model.named_parameters())
        state.update(dict(model.named_buffers()))
        for name, scale in scales.items():
            if name not in state:
                raise KeyError(f"state scale target {name!r} not found")
            tensor = state[name]
            if not torch.is_floating_point(tensor):
                raise TypeError(f"state scale target {name!r} is not floating point")
            backups.append((tensor, tensor.detach().clone()))
            with torch.no_grad():
                tensor.copy_(_scale_tensor(tensor, scale))
    return backups


def _scale_tensor(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"cannot scale non-floating tensor dtype={tensor.dtype}")
    return (tensor.float() * scale).to(dtype=tensor.dtype, device=tensor.device)
