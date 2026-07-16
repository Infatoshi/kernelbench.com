"""GLM-5.2-class MoE layer shape sweep (fused).

Geometry from GLM-5 / GLM-5.1 / GLM-5.2 MoE layers (Z.ai):
  - 256 routed experts
  - top-8 routing
  - 1 shared expert (always on for every token)
  - moe intermediate I=2048 (DeepSeek-V3-class expert width used by GLM-5)
  - hidden H=4096 (bench-practical single-GPU slice of the production H)

No Mixtral (E=8, top_k=2). Score = geomean over shapes (Hard FP8 technique):
aligned prefill, T misaligned, decode T=1, long prefill, off-alignment combos.
"""

# Canonical GLM-5.2 MoE layer knobs (fixed across shapes unless noted).
E = 256
TOP_K = 8
N_SHARED = 1
H = 4096
I = 2048

SHAPES = [
    # Prefill-aligned token count
    {"T": 4096, "E": E, "top_k": TOP_K, "n_shared": N_SHARED, "H": H, "I": I},
    # T not multiple of 128 — pack / tile tails (Hard K=4127 analogue)
    {"T": 4127, "E": E, "top_k": TOP_K, "n_shared": N_SHARED, "H": H, "I": I},
    # Single-token decode microbatch (still 8 routed + 1 shared)
    {"T": 1, "E": E, "top_k": TOP_K, "n_shared": N_SHARED, "H": H, "I": I},
    # Longer prefill
    {"T": 8192, "E": E, "top_k": TOP_K, "n_shared": N_SHARED, "H": H, "I": I},
    # Short batch, still full expert table
    {"T": 512, "E": E, "top_k": TOP_K, "n_shared": N_SHARED, "H": H, "I": I},
    # Off-alignment T
    {"T": 1000, "E": E, "top_k": TOP_K, "n_shared": N_SHARED, "H": H, "I": I},
]
