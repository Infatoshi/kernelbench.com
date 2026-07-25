"""NVIDIA H100 SXM5 (80GB HBM3) — Hopper GH100, SM90.

Dense tensor-core peaks from the official NVIDIA H100 datasheet (SXM5 column).
The datasheet quotes structured-sparsity figures; dense = half: fp8 3958->1979,
bf16/fp16 1979->989.5, tf32 989->494.7. Memory 3.35 TB/s HBM3, 80GB. These are
the SXM numbers — Lambda gpu_1x_h100_sxm5 is this SKU. Do not mix them up with
the PCIe part in h100.py (756 bf16 dense, 2.0 TB/s).
"""
from src.hardware.rtx_pro_6000 import HardwareTarget

H100_SXM = HardwareTarget(
    name="NVIDIA H100 SXM5",
    sm="sm_90a",
    vram_gb=80,
    peak_bandwidth_gb_s=3350.0,  # 80GB HBM3
    peak_tflops_dense={
        # H100 SXM5 dense tensor-core peaks (sparse = 2x). No FP4 on Hopper.
        "fp8": 1979.0,
        "fp6": 1979.0,
        "bf16": 989.5,
        "fp16": 989.5,
        "tf32": 494.7,
        "fp32": 67.0,  # non-tensor-core SIMT fp32
        "int8": 1979.0,
    },
)
