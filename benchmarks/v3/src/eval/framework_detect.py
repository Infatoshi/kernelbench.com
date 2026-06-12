"""Detect which kernel framework a solution.py uses.

Returns a single canonical label from:
  triton | cutlass3 | cutlass2 | cuda_wmma | ptx | cuda_raw | cutile | mlx | metal | unknown

Detection precedence: PTX > CuTe/CUTLASS3 > CUTLASS2 > WMMA > Triton > cuTile > raw CUDA.
Sources checked: string content (imports, headers, includes, kernel source strings).
"""

from __future__ import annotations

import re
from pathlib import Path


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Order matters: more specific first.
    ("ptx",       re.compile(r'asm\s+volatile|asm\s*\(|\.ptx\b|wgmma\.mma_async|mma\.sync|tcgen05\.', re.IGNORECASE)),
    ("cutlass3",  re.compile(r'\bcute::|cutlass/gemm/collective|cutlass/gemm/kernel/sm(9|10)|cutlass::arch::Sm(9|10)')),
    ("cutlass2",  re.compile(r'cutlass/gemm/device/gemm|cutlass::gemm::device|cutlass::epilogue::thread')),
    ("cuda_wmma", re.compile(r'\bnvcuda::wmma\b|#include\s*<mma\.h>|wmma::fragment|wmma::load_matrix|wmma::mma_sync')),
    ("cutile",    re.compile(r'\bcutile::|#include\s*<cutile|cutile/gemm')),
    ("triton",    re.compile(r'import\s+triton\b|@triton\.jit|triton\.language\b|\btl\.dot\b')),
    ("mlx",       re.compile(r'import\s+mlx\b|mlx\.core\b|mx\.fast\.|mx\.compile')),
    ("metal",     re.compile(r'#include\s*<metal_stdlib>|using\s+namespace\s+metal\b|simdgroup_')),
    ("cuda_raw",  re.compile(r'torch\.utils\.cpp_extension\.load_inline|__global__\s+void|<<<[^>]+>>>')),
]


def detect_framework(code: str) -> str:
    """Return the dominant framework label for a solution's source text."""
    for label, pat in _PATTERNS:
        if pat.search(code):
            return label
    return "unknown"


def detect_frameworks_all(code: str) -> list[str]:
    """Return all framework labels that match (useful for hybrid solutions)."""
    hits = []
    for label, pat in _PATTERNS:
        if pat.search(code):
            hits.append(label)
    return hits or ["unknown"]


def detect_from_path(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    return detect_framework(p.read_text())


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        print(f"{arg}: {detect_from_path(arg)} (all={detect_frameworks_all(Path(arg).read_text())})")
