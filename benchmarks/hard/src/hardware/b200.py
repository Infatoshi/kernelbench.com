"""NVIDIA B200 (Blackwell datacenter, GB100) — SM100, HBM3e.

Peak tensor-core throughputs are dense-matrix advertised peaks (NVIDIA B200
datasheet; the headline FP4/FP8 figures there include 2x sparsity, so dense =
half). Real well-tuned kernels see ~60-85% of peak. This box reports ~180GB
HBM3e (the 180GB B200 SKU) at ~8 TB/s.

Empirical verification pending: confirm cuBLAS achieved TFLOPS land at
~0.75-0.85 of these (fp8 ~3400-3800, bf16 ~1700-1900) when baselines run on the
box; bump the table if any real kernel exceeds peak_fraction 1.0.
"""
from src.hardware.rtx_pro_6000 import HardwareTarget

B200 = HardwareTarget(
    name="NVIDIA B200",
    sm="sm_100a",
    vram_gb=180,
    peak_bandwidth_gb_s=8000.0,  # HBM3e, ~8 TB/s
    peak_tflops_dense={
        # B200 dense tensor-core peaks (datasheet halves: sparse = 2x).
        "fp4": 9000.0,
        "nvfp4": 9000.0,
        "mxfp4": 9000.0,
        "fp6": 4500.0,
        "fp8": 4500.0,
        "bf16": 2250.0,
        "fp16": 2250.0,
        "tf32": 1100.0,
        "fp32": 80.0,  # non-tensor-core SIMT fp32 (estimate; memory-regime problems ignore this)
        "int8": 4500.0,
        "int4": 9000.0,
    },
)
