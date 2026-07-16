"""MegaQwen decode-only context sweep.

Protocol per shape (see reference.run / benchmark.py):
  1. PREFILL to ctx_len (untimed) — fill KV caches with real block semantics.
  2. DECODE for decode_steps tokens starting at position=ctx_len (timed).

We do NOT grade tokens. If last_hidden (and intermediate numerics) match the
reference within tolerance, greedy tokens would match — that is exact for a
deterministic argmax head, and the right contract for a kernel bench.

ctx_len grid: 2k / 8k / 32k / 128k.
"""

SHAPES = [
    {"ctx_len": 2048, "decode_steps": 64},
    {"ctx_len": 8192, "decode_steps": 64},
    {"ctx_len": 32768, "decode_steps": 32},
    {"ctx_len": 131072, "decode_steps": 16},
]

# Short shapes for check.py only (full 128k prefill is too slow for a
# correctness gate). Same protocol, tiny lengths.
CHECK_SHAPES = [
    {"ctx_len": 128, "decode_steps": 8},
    {"ctx_len": 256, "decode_steps": 8},
    {"ctx_len": 512, "decode_steps": 4},
]
