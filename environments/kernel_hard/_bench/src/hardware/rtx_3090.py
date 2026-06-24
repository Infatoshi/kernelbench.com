"""NVIDIA GeForce RTX 3090 — GA102, SM86 (consumer Ampere), GDDR6X.

Peak tensor-core throughputs are dense advertised peaks (NVIDIA Ampere GA102
whitepaper; sparse = 2x dense, not listed here). Real well-tuned kernels see
~60-85% of peak.

Notes for grading on this part:
  - No FP8 and no FP4 tensor cores on Ampere (those debut on Ada sm_89 /
    Hopper sm_90). Those keys are intentionally absent: a problem that looks
    them up will KeyError, which correctly flags it as un-gradable here
    (01_fp8_gemm). bf16/fp16/int8/int4 tensor cores DO exist.
  - bf16/fp16 are the FP32-accumulate rate (71 TFLOPS dense). Consumer Ampere
    halves the FP32-accumulate tensor path vs the FP16-accumulate path (142
    dense); kernels held to the bf16 atol=0.02 tolerance need FP32 accumulate,
    so 71 is the realistic peak to grade against. cuBLAS bf16 GEMM lands at
    ~60 TFLOPS (~85% of 71), so peak_fraction stays <= 1.0.
  - The four memory-bound problems (03/05/07/09) grade on bandwidth, not these
    TFLOPS, so 936 GB/s is the number that matters for them.
"""
from src.hardware.rtx_pro_6000 import HardwareTarget

RTX_3090 = HardwareTarget(
    name="NVIDIA GeForce RTX 3090",
    sm="sm_86",
    vram_gb=24,
    peak_bandwidth_gb_s=936.0,  # 24GB GDDR6X, 384-bit @ 19.5 Gbps
    peak_tflops_dense={
        # Ampere GA102 dense tensor-core peaks (FP32-accumulate path).
        "bf16": 71.0,
        "fp16": 71.0,
        "tf32": 71.0,
        "fp32": 35.6,  # non-tensor-core SIMT fp32 (spec sheet)
        "int8": 284.0,
        "int4": 568.0,
        # No "fp8"/"fp4"/"nvfp4"/"mxfp4"/"fp6": Ampere has no such hardware.
    },
)
