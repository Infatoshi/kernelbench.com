"""Hardware peak lookup for KernelBench-Multi.

Only one SKU is graded: 4xH100 SXM (NVLink4). See SPEC.md for why a single SKU
suffices (the bench targets the NVLink fabric, not single-GPU compute).
"""
from __future__ import annotations

from . import h100x4

_TABLE = {
    "H100x4": h100x4,
    "h100x4": h100x4,
    "4xH100": h100x4,
}


def get(name: str):
    if name not in _TABLE:
        raise KeyError(f"unknown hardware '{name}'; known: {sorted(_TABLE)}")
    return _TABLE[name]
