"""8xH100 SXM (NVLink4) peak figures.

The graded ceiling for this bench is NVLink **busbw**, not compute. NVLink4 on
H100 SXM is 900 GB/s per GPU, bidirectional, fully connected via NVSwitch. The
roofline uses that 900 GB/s as the bus-bandwidth ceiling; standard ring/tree
collectives realistically reach ~70-85% of it, so a peak_fraction near ~0.8 is
"saturating the fabric."

Compute peaks are listed for reference only (e.g. to sanity-check that a problem
is genuinely comms-bound) and are NOT used for scoring.
"""
from __future__ import annotations

name = "8xH100 SXM (NVLink4)"
num_gpus = 8

# NVLink4: 18 links x 25 GB/s x 2 (bidir) = 900 GB/s per GPU.
peak_nvlink_busbw_gb_s = 900.0

# Per-GPU HBM3 bandwidth (reference only).
peak_hbm_bandwidth_gb_s = 3350.0

# Per-GPU dense tensor-core peaks (reference only; NOT graded).
peak_tflops_dense = {
    "bf16": 989.0,
    "fp16": 989.0,
    "fp8": 1979.0,
}
