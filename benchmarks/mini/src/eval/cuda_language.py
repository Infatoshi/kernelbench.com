"""CUDA-only language gate for KernelBench-CUDA.

This bench grades *CUDA kernel writing*. Triton, ThunderKittens Python, CuteDSL,
and similar high-level kernel DSLs are forbidden. Pure PyTorch op chains are also
not enough — the solution must contain evidence of a real CUDA C++/PTX kernel.

`check_cuda_language(sol_src, meta)` returns (ok, messages, report_dict).
`report_dict` is written to `cuda_language.json` for leaderboard sidecars:
  - framework: detected backend label
  - triton_cheat: bool
  - dsl_cheat: bool
  - has_cuda_evidence: bool
  - forbidden_hits: list[str]
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Hard fails — models reach for these instead of writing CUDA.
TRITON_PATTERNS: list[tuple[str, str]] = [
    ("import_triton", r"(?m)^\s*import\s+triton\b|from\s+triton\b"),
    ("triton_jit", r"@triton\.jit\b"),
    ("triton_language", r"\btriton\.language\b|\btl\.(?:program_id|load|store|dot)\b"),
]

DSL_PATTERNS: list[tuple[str, str]] = [
    ("thunderkittens", r"\bimport\s+thunderkittens\b|\bfrom\s+thunderkittens\b|\bimport\s+tk\b.*thunder"),
    ("cute_dsl", r"\bimport\s+cutlass\.cute\b|\bfrom\s+cutlass\.cute\b|\bcute\.dsl\b"),
    ("tilelang", r"\bimport\s+tilelang\b|\bfrom\s+tilelang\b"),
    ("cuda_python_kernel", r"\bfrom\s+cuda\.cooperative\b|\bnumba\.cuda\.jit\b"),
]

# Positive evidence that a real CUDA/PTX path exists.
CUDA_EVIDENCE_PATTERNS: list[tuple[str, str]] = [
    ("load_inline", r"torch\.utils\.cpp_extension\.load_inline|cpp_extension\.load\b"),
    ("global_kernel", r"__global__\s+(?:void|__launch_bounds__)"),
    ("cuda_header", r"#include\s*<cuda(?:_runtime)?\.h>|#include\s*<cuda_fp16\.h>"),
    ("ptx_inline", r"asm\s+volatile|mma\.sync|tcgen05\."),
    ("wmma", r"\bnvcuda::wmma\b|wmma::fragment"),
    ("cutlass_cpp", r"cutlass::|cute::|#include\s*<cutlass/"),
    ("cu_file", r"""load(?:_inline)?\s*\([^)]*['"][^'"]+\.cu['"]"""),
]

FRAMEWORK_PRIORITY: list[tuple[str, str]] = [
    ("ptx", r"asm\s+volatile|mma\.sync|tcgen05\."),
    ("cutlass", r"cutlass::|cute::|#include\s*<cutlass/"),
    ("cuda_wmma", r"\bnvcuda::wmma\b|wmma::fragment"),
    ("cuda_raw", r"torch\.utils\.cpp_extension\.load_inline|__global__\s+void|cpp_extension\.load\b"),
    ("triton", r"import\s+triton\b|@triton\.jit|\btl\.dot\b"),
    ("pytorch_only", r"torch\.(?:nn\.functional|ops)"),
]


def _scan(code: str, patterns: list[tuple[str, str]]) -> list[str]:
    hits: list[str] = []
    for name, pat in patterns:
        if re.search(pat, code):
            hits.append(name)
    return hits


def detect_framework(code: str) -> str:
    for name, pat in FRAMEWORK_PRIORITY:
        if re.search(pat, code):
            return name
    return "unknown"


def check_cuda_language(
    sol_src: str,
    meta: dict[str, Any] | None = None,
    *,
    require_cuda_evidence: bool = True,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate that solution.py is CUDA-path code, not Triton/DSL/pytorch-only.

    `meta` may set:
      language: "cuda" (default for this bench)
      require_cuda_evidence: bool (default True)
      allow_triton: bool (default False) — escape hatch for experiments only
    """
    meta = meta or {}
    allow_triton = bool(meta.get("allow_triton", False))
    if "require_cuda_evidence" in meta:
        require_cuda_evidence = bool(meta["require_cuda_evidence"])

    triton_hits = _scan(sol_src, TRITON_PATTERNS)
    dsl_hits = _scan(sol_src, DSL_PATTERNS)
    cuda_hits = _scan(sol_src, CUDA_EVIDENCE_PATTERNS)
    framework = detect_framework(sol_src)

    messages: list[str] = []
    ok = True

    if triton_hits and not allow_triton:
        ok = False
        messages.append(
            f"FAIL: Triton is forbidden on KernelBench-CUDA (hits: {', '.join(triton_hits)}). "
            "Write a CUDA C++/PTX kernel via load_inline or a .cu extension."
        )
    if dsl_hits:
        ok = False
        messages.append(
            f"FAIL: kernel DSL forbidden on KernelBench-CUDA (hits: {', '.join(dsl_hits)}). "
            "Write CUDA C++/PTX."
        )
    if require_cuda_evidence and not cuda_hits:
        ok = False
        messages.append(
            "FAIL: no CUDA kernel evidence in solution.py. "
            "Expected load_inline / __global__ / .cu / inline PTX / CUTLASS C++. "
            "A pure torch.nn.functional chain is not enough on this bench."
        )

    report = {
        "framework": framework,
        "triton_cheat": bool(triton_hits) and not allow_triton,
        "dsl_cheat": bool(dsl_hits),
        "has_cuda_evidence": bool(cuda_hits),
        "cuda_evidence": cuda_hits,
        "forbidden_hits": triton_hits + dsl_hits,
        "ok": ok,
    }
    return ok, messages, report


def enforce_and_write(
    sol_path: Path = Path("solution.py"),
    meta: dict[str, Any] | None = None,
    report_path: Path = Path("cuda_language.json"),
    framework_path: Path = Path("framework.txt"),
) -> None:
    """Run the gate; write report files; sys.exit(1) on failure.

    Call from problem check.py after the forbidden-op grep and before/after
    numeric correctness (language fail is independent of numeric PASS).
    """
    import sys

    sol_src = sol_path.read_text() if sol_path.exists() else ""
    ok, messages, report = check_cuda_language(sol_src, meta)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    framework_path.write_text(report["framework"] + "\n")
    if not ok:
        for m in messages:
            print(m)
        sys.exit(1)
    print(
        f"cuda_language: ok framework={report['framework']} "
        f"evidence={','.join(report['cuda_evidence']) or 'none'}"
    )


def collect_solution_sources(root: Path = Path(".")) -> str:
    """Concatenate solution.py plus local .cu/.cuh/.cpp/.h it may load.

    Agents often put the kernel in a sidecar .cu; scan the problem dir so a
    pure-Python wrapper around a real .cu still counts as CUDA evidence.
    """
    chunks: list[str] = []
    sol = root / "solution.py"
    if sol.exists():
        chunks.append(sol.read_text())
    for pattern in ("*.cu", "*.cuh", "*.cpp", "*.cc", "*.h", "*.hpp"):
        for p in sorted(root.glob(pattern)):
            try:
                chunks.append(p.read_text(errors="ignore"))
            except OSError:
                pass
    return "\n".join(chunks)
