"""4xH100 SXM (NVLink4) peak figures.

The graded ceiling for this bench is NVLink **busbw**, not compute. NVLink4 on
H100 SXM is 900 GB/s per GPU bidirectional (450 GB/s each direction), fully
connected via NVSwitch. The NCCL busbw convention measures against the
UNIDIRECTIONAL rate — a ring all-reduce sends and receives concurrently, and
its busbw tops out at the 450 GB/s one-way rate, never 900 — so the roofline
ceiling is 450 GB/s. Standard ring/tree collectives realistically reach ~70-85%
of it (measured on poseidon 4xH100 NVSwitch, 2026-07-22: NCCL c10d all-reduce
348 GB/s = 0.77 at 512 MB), so a peak_fraction near ~0.8 is "saturating the
fabric."

Compute peaks are listed for reference only (e.g. to sanity-check that a problem
is genuinely comms-bound) and are NOT used for scoring.
"""
from __future__ import annotations

name = "4xH100 SXM (NVLink4)"
num_gpus = 4

# NVLink4: 18 links x 25 GB/s per direction = 450 GB/s per GPU unidirectional
# (900 GB/s bidir marketing figure). busbw grades against the one-way rate.
peak_nvlink_busbw_gb_s = 450.0

# Per-GPU HBM3 bandwidth (reference only).
peak_hbm_bandwidth_gb_s = 3350.0

# Per-GPU dense tensor-core peaks (reference only; NOT graded).
peak_tflops_dense = {
    "bf16": 989.0,
    "fp16": 989.0,
    "fp8": 1979.0,
}
