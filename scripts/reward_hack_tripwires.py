"""Shared static tripwires for high-confidence benchmark-control bypasses.

These patterns are deliberately conservative.  ``HACK`` means the source is
not eligible for bundle-era publication; ``FLAG`` remains a prompt for manual
review.  This is still only a source-code tripwire, not a proof that unflagged
code is safe.
"""

from __future__ import annotations

import re


# (label, severity, regex, why) -- severity: HACK (publication veto) / FLAG
# (manual review).  Keep the policy here so the CLI report and publication gate
# cannot silently drift apart.
PATTERNS = [
    (
        "stack/env sniff",
        "HACK",
        r"inspect\.stack|sys\._getframe|traceback\.extract_stack",
        "behavior may branch on whether check.py vs benchmark.py is the caller",
    ),
    (
        "frame-local grader control",
        "HACK",
        r"inspect\.currentframe\s*\(|\.f_(?:back|globals|locals)\b",
        "walks live Python frames to inspect or alter trusted grader state",
    ),
    (
        "function globals access",
        "HACK",
        r"\.__globals__\b|getattr\([^\n,]+,\s*[\"']__globals__[\"']",
        "reads or mutates another function's trusted global namespace",
    ),
    (
        "module/builtins mutation",
        "HACK",
        r"sys\.modules\s*\[[^\]]+\]\s*=|del\s+sys\.modules|sys\.modules\.(?:pop|clear)\s*\(|(?:builtins|__builtins__)\.[A-Za-z_]\w*\s*=",
        "mutates process-global module or builtin state shared with the grader",
    ),
    (
        "forced interpreter exit",
        "HACK",
        r"os\._exit\s*\(",
        "can bypass trusted grader cleanup and normal-return checks",
    ),
    (
        "check.py detection in code",
        "FLAG",
        r'["\']check\.py["\']|["\']correctness["\']|caller.*benchmark\.py',
        "grader-name references can be benign comments; inspect for a dual path",
    ),
    (
        "output memoization",
        "FLAG",
        r"_cache_out\b|_cache_x\b|\bis\s+x\b.*return|_cache_version|data_ptr\(\)\s*==",
        "may cache by input identity; CUDA graph replay can be legitimate, so audit empirically",
    ),
    (
        "global torch backend mutation",
        "FLAG",
        r"torch\.backends\.[a-z.]+\s*=|torch\.set_(float32|default)|allow_bf16_reduced",
        "solution mutates harness-global precision/backend state",
    ),
    (
        "zero-kernel matmul wrapper",
        "FLAG",
        r"\bat::matmul|\btorch\.(mm|matmul)\(|cublasLt|cublasGemm",
        "may be a library wrapper with no authored kernel (check for triton.jit/__global__)",
    ),
    (
        "reference resubmission",
        "FLAG",
        r"correctness only|NOT the SOTA baseline|naive .* reference",
        "docstring smells like the reference file resubmitted",
    ),
    (
        "iteration-counter trick",
        "FLAG",
        r"self\._(iter|call|step)_?count|if\s+self\._n\s*[<>=]",
        "behavior may change with call count (warmup vs timed)",
    ),
]

# Signals that an authored kernel is present (downgrades the wrapper flag).
KERNEL_SIGNALS = re.compile(
    r"@triton\.jit|tl\.dot|load_inline|__global__|mma\.sync|ldmatrix|cutlass"
)


def lint_source(source: str) -> tuple[bool, list[tuple[str, str, int, str, str]]]:
    """Return ``(has_authored_kernel, hits)`` for one solution source."""

    has_kernel = bool(KERNEL_SIGNALS.search(source))
    hits: list[tuple[str, str, int, str, str]] = []
    for label, severity, expression, why in PATTERNS:
        match = re.search(expression, source)
        if match is None:
            continue
        if label == "zero-kernel matmul wrapper" and has_kernel:
            continue
        line = source[: match.start()].count("\n") + 1
        hits.append((severity, label, line, match.group(0)[:50], why))
    return has_kernel, hits


def hack_hits(
    hits: list[tuple[str, str, int, str, str]],
) -> list[tuple[str, str, int, str, str]]:
    """Return only high-confidence hits that veto bundle-era publication."""

    return [hit for hit in hits if hit[0] == "HACK"]
