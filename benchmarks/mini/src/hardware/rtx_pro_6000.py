"""RTX PRO 6000 Blackwell Workstation — SM120, consumer-lineage Blackwell.

Peak tensor-core throughputs are dense-matrix advertised peaks. Actual kernels
will see 60-85% of peak on well-tuned code.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareTarget:
    name: str
    sm: str
    vram_gb: int
    peak_bandwidth_gb_s: float  # DRAM
    peak_tflops_dense: dict[str, float]  # dtype -> TFLOPS


RTX_PRO_6000 = HardwareTarget(
    name="RTX PRO 6000 Blackwell Workstation",
    sm="sm_120a",
    vram_gb=96,
    peak_bandwidth_gb_s=1800.0,
    peak_tflops_dense={
        # Blackwell GB202 dense tensor peaks. Derived from NVIDIA's headline
        # 4000 fp4-sparse AI TOPS -> fp4 dense 2000 -> fp8 1000 -> bf16 500
        # (each precision step = 2x; sparse = 2x dense). Verified empirically:
        # cuBLAS hits fp8 773 / bf16 412 TFLOPS on 4096^3 (~77-82% of these,
        # normal cuBLAS efficiency). The prior table was ~2.5x too low, which
        # produced peak_fraction > 1.0 for real fp8 kernels. fp32 is the 125
        # TFLOPS SIMT figure from the spec sheet.
        "fp4": 2000.0,
        "nvfp4": 2000.0,
        "mxfp4": 2000.0,
        "fp6": 1000.0,
        "fp8": 1000.0,
        "bf16": 500.0,
        "fp16": 500.0,
        "tf32": 250.0,
        "fp32": 125.0,  # non-tensor-core SIMT fp32 (spec sheet)
        "int8": 1000.0,
        "int4": 2000.0,
    },
)
