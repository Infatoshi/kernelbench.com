"""NVIDIA H100 PCIe (80GB HBM2e) — Hopper GH100, SM90.

Dense tensor-core peaks from the official NVIDIA H100 datasheet (PCIe column).
The datasheet quotes structured-sparsity figures; dense = half: fp8 3026->1513,
bf16/fp16 1513->756, tf32 756->378. Memory 2 TB/s, 80GB. These are the PCIe SKU
numbers (the SXM5 part is higher: ~989 bf16 / 1979 fp8 dense, 3.35 TB/s); do not
mix them up. Well-tuned kernels reach ~60-85% of these, so they serve as the
roofline ceiling: the 2026-06-18 8-model sweep produced no peak_fraction > 1.0.
"""
from src.hardware.rtx_pro_6000 import HardwareTarget

H100 = HardwareTarget(
    name="NVIDIA H100 PCIe",
    sm="sm_90a",
    vram_gb=80,
    peak_bandwidth_gb_s=2039.0,  # 80GB HBM2e, ~2.0 TB/s
    peak_tflops_dense={
        # H100 PCIe dense tensor-core peaks (sparse = 2x). No FP4 on Hopper.
        "fp8": 1513.0,
        "fp6": 1513.0,
        "bf16": 756.0,
        "fp16": 756.0,
        "tf32": 378.0,
        "fp32": 51.0,  # non-tensor-core SIMT fp32
        "int8": 1513.0,
    },
)
