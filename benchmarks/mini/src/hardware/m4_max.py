"""Apple M4 Max — unified memory, 16-core GPU, Metal 3."""
from src.hardware.rtx_pro_6000 import HardwareTarget

M4_MAX = HardwareTarget(
    name="Apple M4 Max",
    sm="metal3",
    vram_gb=36,  # unified
    peak_bandwidth_gb_s=546.0,
    peak_tflops_dense={
        # Apple doesn't publish formal TFLOPS numbers the way NVIDIA does.
        # These are community estimates from chip-level benchmarks.
        "fp32": 17.0,
        "fp16": 34.0,
        "bf16": 34.0,
        "int8": 68.0,
    },
)
