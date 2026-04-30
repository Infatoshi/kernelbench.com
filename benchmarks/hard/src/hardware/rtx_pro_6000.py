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
        "fp4": 800.0,
        "nvfp4": 800.0,
        "mxfp4": 800.0,
        "fp6": 800.0,
        "fp8": 400.0,
        "bf16": 200.0,
        "fp16": 200.0,
        "tf32": 100.0,
        "fp32": 12.0,  # non-tensor-core SIMT fp32
        "int8": 400.0,
        "int4": 800.0,
    },
)
