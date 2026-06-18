"""NVIDIA B200 (Blackwell datacenter, GB100) — SM100, HBM3e.

Dense tensor-core peaks from the official NVIDIA B200 datasheet. The headline
figures are with structured sparsity; dense = half: fp4 18->9000, fp8 9->4500,
bf16/fp16 4.5->2250 PFLOPS->TFLOPS, tf32 2.2->1100. Memory ~8 TB/s HBM3e (NVIDIA
spec is 192GB; this cloud box reports ~180GB, a binned SKU - bandwidth used for
the roofline is the 8 TB/s spec). Well-tuned kernels reach ~60-85% of these, so
they serve as the roofline ceiling: the 2026-06-18 8-model sweep produced no
peak_fraction > 1.0.
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
