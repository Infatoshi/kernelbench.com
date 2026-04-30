"""EvalResult dataclass and result processing helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvalResult:
    model: str = ""
    gpu: str = ""
    problem: str = ""
    level: int = 0

    compiled: bool = False
    correct: bool = False
    speedup: Optional[float] = None
    error: Optional[str] = None

    turns: int = 0
    submitted: bool = False

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    estimated_cost_usd: Optional[float] = None

    ref_ms: Optional[float] = None
    sol_ms: Optional[float] = None
    ref_kernels: Optional[int] = None
    sol_kernels: Optional[int] = None

    precision_used: Optional[str] = None
    tolerance_atol: Optional[float] = None
    tolerance_rtol: Optional[float] = None
    has_nan: bool = False
    has_inf: bool = False
    is_deterministic: bool = True
    baseline_type: Optional[str] = None
    op_type: Optional[str] = None

    achieved_tflops: Optional[float] = None
    ref_tflops: Optional[float] = None
    pct_of_peak: Optional[float] = None

    solution_path: Optional[str] = None
    solution_hash: Optional[str] = None
    reasoning_effort: Optional[str] = None
    elapsed_seconds: float = 0.0

    judge_legitimate: Optional[bool] = None
    judge_reason: Optional[str] = None
    judge_model: Optional[str] = None

    hardware_fingerprint: Dict[str, Any] = field(default_factory=dict)


def attach_solution_metadata(result: EvalResult, solution_path: str, sandbox) -> None:
    result.solution_path = solution_path
    code = sandbox.read_file(solution_path.replace("/workspace/", ""))
    if code:
        result.solution_hash = hashlib.sha256(code.encode()).hexdigest()[:16]


def apply_benchmark_metrics(result: EvalResult, benchmark_result: dict) -> None:
    result.compiled = benchmark_result.get("compiled", False)
    result.correct = benchmark_result.get("correct", False)
    result.speedup = benchmark_result.get("speedup")
    result.error = benchmark_result.get("error")
    result.ref_ms = benchmark_result.get("ref_ms")
    result.sol_ms = benchmark_result.get("sol_ms")
    result.ref_kernels = benchmark_result.get("ref_kernels")
    result.sol_kernels = benchmark_result.get("sol_kernels")
    result.precision_used = benchmark_result.get("precision_used")
    result.tolerance_atol = benchmark_result.get("tolerance_atol")
    result.tolerance_rtol = benchmark_result.get("tolerance_rtol")
    result.has_nan = benchmark_result.get("has_nan", False)
    result.has_inf = benchmark_result.get("has_inf", False)
    result.is_deterministic = benchmark_result.get("is_deterministic", True)
    result.baseline_type = benchmark_result.get("baseline_type")
    result.op_type = benchmark_result.get("op_type")
    result.achieved_tflops = benchmark_result.get("achieved_tflops")
    result.ref_tflops = benchmark_result.get("ref_tflops")
    result.pct_of_peak = benchmark_result.get("pct_of_peak")
