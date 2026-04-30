"""Canonical shape sweep for KDA forward (chunk form).

Mix of:
  - short-context training-step scale (T=1024)
  - mid-context (T=2048) which is the headline benchmark
  - long-context that stresses the inter-chunk recurrence (T=4096)
  - thin-batch decode-style (B=1, T=2048, fewer heads)

Constraints:
  - T % chunk_size == 0 (chunk_size = 64)
  - K, V are the per-head channel dims; KDA in Kimi Linear uses K=V=128
"""

SHAPES = [
    {"B": 2, "T": 1024, "H": 8, "K": 128, "V": 128, "CHUNK_SIZE": 64},
    {"B": 2, "T": 2048, "H": 8, "K": 128, "V": 128, "CHUNK_SIZE": 64},
    {"B": 1, "T": 4096, "H": 8, "K": 128, "V": 128, "CHUNK_SIZE": 64},
    {"B": 1, "T": 2048, "H": 4, "K": 128, "V": 128, "CHUNK_SIZE": 64},
]
