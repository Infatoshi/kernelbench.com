"""Canonical shapes for context-parallel ring attention.

Total sequence is world_size * seq_local, so shape 1 is a 32k-token context. The
axes that matter: how long the ring is relative to the per-hop attention math
(seq_local), and how much of the per-hop cost is KV bytes vs flops (heads).

  shape 0: 8k context, balanced
  shape 1: 16k context, fewer heads — longer ring hops, less math to hide them
  shape 2: 2k context — latency-bound, ring startup dominates
  shape 3: off-alignment seq_local (not a multiple of any nice tile)
"""

SHAPES = [
    {"seq_local": 2048, "heads": 32, "head_dim": 128},
    {"seq_local": 4096, "heads": 16, "head_dim": 128},
    {"seq_local": 512, "heads": 32, "head_dim": 128},
    {"seq_local": 2000, "heads": 32, "head_dim": 128},
]
