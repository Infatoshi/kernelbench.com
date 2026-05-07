"""Canonical shape sweep for multi-axis RoPE pre-attention.

Sized for Qwen2-VL-style vision-language inference: long T, GQA, mrope_section
splitting D/2 across temporal/height/width axes. Mix of base, long-context,
small-head-dim, and very-long-context.
"""

SHAPES = [
    # Qwen2-VL base: T=4k, head_dim=128, GQA 32:8, mrope [16,24,24]
    {"B": 1, "T": 4096,  "Hq": 32, "Hkv": 8, "D": 128,
     "mrope_section": (16, 24, 24), "max_pos": 32768},

    # Qwen2-VL 7B-style long context, narrower kv heads
    {"B": 1, "T": 8192,  "Hq": 28, "Hkv": 4, "D": 128,
     "mrope_section": (16, 24, 24), "max_pos": 32768},

    # Smaller head_dim and B=2 (tests batch dim and reduced D)
    {"B": 2, "T": 2048,  "Hq": 16, "Hkv": 2, "D": 64,
     "mrope_section": (8, 12, 12), "max_pos": 16384},

    # Very long context (16k tokens)
    {"B": 1, "T": 16384, "Hq": 32, "Hkv": 8, "D": 128,
     "mrope_section": (16, 24, 24), "max_pos": 65536},
]
