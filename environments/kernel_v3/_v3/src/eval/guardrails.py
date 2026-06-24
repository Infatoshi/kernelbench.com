"""Unified solution validation guardrails."""

from __future__ import annotations

import re
from typing import Optional

NVIDIA_FORBIDDEN = [
    (re.compile(r"torch::\s*(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("), "Forbidden C++ PyTorch fallback"),
    (re.compile(r"(?:^|[^\w])torch\.(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("), "Forbidden Python PyTorch fallback"),
    (re.compile(r"(?:^|[^\w])F\.(?:linear|conv1d|conv2d|conv3d)\s*\("), "Forbidden torch.nn.functional fallback"),
    (re.compile(r"(?:^|[^\w])torch\.compile\s*\("), "Forbidden torch.compile"),
    (re.compile(r"@torch\.jit\.script"), "Forbidden torch.jit.script"),
]

# External libraries that do the heavy compute -- using these isn't "optimizing"
NVIDIA_LIBRARY_FORBIDDEN = [
    (re.compile(r"from\s+fla\.ops\b"), "Forbidden external library: fla.ops (must write custom kernel)"),
    (re.compile(r"from\s+flash_attn\b"), "Forbidden external library: flash_attn (must write custom kernel)"),
    (re.compile(r"from\s+xformers\b"), "Forbidden external library: xformers (must write custom kernel)"),
]

# Patterns that indicate custom kernel work exists in the solution
_CUSTOM_KERNEL_INDICATORS = [
    "triton.jit",
    "@triton.jit",
    "load_inline",
    "__global__",
    "tl.load(",
    "tl.store(",
    "tl.dot(",
]

METAL_FORBIDDEN = [
    (re.compile(r"(?:^|[^\w])import\s+torch\b"), "Forbidden: torch in Metal solution"),
    (re.compile(r"(?:^|[^\w])torch\."), "Forbidden: PyTorch usage in Metal solution"),
    (re.compile(r"(?:^|[^\w])import\s+triton\b"), "Forbidden: triton in Metal solution"),
    (re.compile(r"torch\.utils\.cpp_extension|(?:^|[^\w])load_inline\s*\("), "Forbidden: CUDA extension in Metal solution"),
]


def _has_custom_kernel(code: str) -> bool:
    """Check if solution contains at least one custom kernel."""
    return any(indicator in code for indicator in _CUSTOM_KERNEL_INDICATORS)


def validate_nvidia(code: str) -> Optional[str]:
    for pattern, message in NVIDIA_FORBIDDEN:
        match = pattern.search(code)
        if match:
            return f"{message}: `{match.group(0).strip()}`"

    for pattern, message in NVIDIA_LIBRARY_FORBIDDEN:
        match = pattern.search(code)
        if match:
            return f"{message}: `{match.group(0).strip()}`"

    if not _has_custom_kernel(code):
        return "No custom kernel found: solution must contain at least one Triton or CUDA kernel"

    return None


def validate_metal(code: str) -> Optional[str]:
    for pattern, message in METAL_FORBIDDEN:
        match = pattern.search(code)
        if match:
            return f"{message}: `{match.group(0).strip()}`"
    if "import mlx.core as mx" not in code:
        return "Missing MLX import: expected `import mlx.core as mx`"
    if "def solution(" not in code:
        return "Missing required interface: `def solution(*inputs)`"
    return None


def validate_solution(code: str, is_metal: bool = False) -> Optional[str]:
    return validate_metal(code) if is_metal else validate_nvidia(code)
