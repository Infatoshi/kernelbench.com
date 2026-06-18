"""NVIDIA H100 PCIe (80GB HBM2e) — Hopper GH100, SM90.

Peak tensor-core throughputs are dense-matrix advertised peaks (NVIDIA H100
PCIe datasheet; the sparse figures there are 2x these). Real well-tuned kernels
see ~60-85% of peak, and cuBLAS lands ~75-82% — so dense peaks keep
peak_fraction < 1.0 for genuine kernels. These are the PCIe SKU numbers (the
SXM5 part is higher: ~989 bf16 / 1979 fp8 dense, 3.35 TB/s); do not mix them up.

Empirical verification pending: confirm cuBLAS achieved TFLOPS on 4096^3 land
at ~0.75-0.85 of these (fp8 ~1150-1300, bf16 ~570-640) when baselines run on
the box; bump the table if any real kernel exceeds peak_fraction 1.0.
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
