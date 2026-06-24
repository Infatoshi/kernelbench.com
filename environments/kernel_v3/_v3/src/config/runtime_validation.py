"""Shared runtime validation helpers for benchmark batch entrypoints."""

from __future__ import annotations

import platform
import sys
from typing import Iterable, Sequence


ERROR_MESSAGES = {
    "gpu_not_allowed": "GPU '{gpu}' is not supported for {benchmark}. Allowed: {allowed}",
    "platform_mismatch": "{benchmark} requires {required_platform}, but running on {current_platform}",
    "missing_dependency": "{benchmark} requires {dep} which is not installed",
    "hardware_unavailable": "GPU '{gpu}' is not available in the current environment",
}

NVIDIA_GPUS = {"RTX3090", "H100", "B200", "A100", "L40S", "LOCAL"}


def _default_gpus_for_platform(allowed_gpus: Sequence[str]) -> list[str]:
    """Pick a sensible default when --gpus is omitted."""
    system = platform.system()
    if system == "Darwin":
        darwin_gpus = [gpu for gpu in allowed_gpus if gpu == "M4MAX"]
        if darwin_gpus:
            return darwin_gpus
    non_metal = [gpu for gpu in allowed_gpus if gpu != "M4MAX"]
    if non_metal:
        return non_metal
    return list(allowed_gpus[:1])


def parse_requested_gpus(argv: Sequence[str], allowed_gpus: Sequence[str]) -> list[str]:
    """Parse --gpus argument from argv and normalize into a list."""
    for i, arg in enumerate(argv):
        if arg != "--gpus" or i + 1 >= len(argv):
            continue
        raw = argv[i + 1].strip()
        if not raw:
            return []
        if raw.lower() == "all":
            return list(allowed_gpus)
        return [gpu.strip() for gpu in raw.split(",") if gpu.strip()]
    return _default_gpus_for_platform(allowed_gpus)


def ensure_gpu_arg(argv: Sequence[str], requested: Sequence[str]) -> list[str]:
    """Return argv with --gpus set when it is missing."""
    args = list(argv)
    if "--gpus" in args:
        return args
    return ["--gpus", ",".join(requested), *args]


def validate_gpus(
    requested: Iterable[str],
    allowed: Sequence[str],
    benchmark: str,
    reason: str,
) -> None:
    """Fail fast on unsupported GPU values with a benchmark-specific message."""
    invalid = [gpu for gpu in requested if gpu not in allowed]
    if not invalid:
        return

    for gpu in invalid:
        print(
            ERROR_MESSAGES["gpu_not_allowed"].format(
                gpu=gpu,
                benchmark=benchmark,
                allowed=", ".join(allowed),
            )
        )
    print(f"Reason: {reason}")
    sys.exit(1)


def validate_platform(requested: Iterable[str], benchmark: str) -> None:
    """Fail on impossible GPU/platform combinations."""
    system = platform.system()
    requested_set = set(requested)

    if "M4MAX" in requested_set and system != "Darwin":
        print(
            ERROR_MESSAGES["platform_mismatch"].format(
                benchmark=benchmark,
                required_platform="macOS (Darwin) for Metal",
                current_platform=system,
            )
        )
        print("Reason: M4MAX targets Apple Metal and must run on Apple Silicon.")
        sys.exit(1)

    if any(gpu in NVIDIA_GPUS for gpu in requested_set) and system == "Darwin":
        print(
            ERROR_MESSAGES["platform_mismatch"].format(
                benchmark=benchmark,
                required_platform="Linux for NVIDIA CUDA/Modal execution",
                current_platform=system,
            )
        )
        print("Reason: NVIDIA benchmark paths are currently Linux-only in this harness.")
        sys.exit(1)
