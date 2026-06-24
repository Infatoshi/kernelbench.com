"""Hardware peak-throughput lookup tables."""
from src.hardware.m4_max import M4_MAX
from src.hardware.rtx_pro_6000 import RTX_PRO_6000

TARGETS = {
    "RTX_PRO_6000": RTX_PRO_6000,
    "M4_MAX": M4_MAX,
}


def get(name: str):
    if name not in TARGETS:
        raise ValueError(f"Unknown hardware {name!r}; available: {list(TARGETS)}")
    return TARGETS[name]
