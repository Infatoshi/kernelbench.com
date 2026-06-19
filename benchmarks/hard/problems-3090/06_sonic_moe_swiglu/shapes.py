"""Shape sweep for Sonic-MoE up-projection (grouped GEMM + fused SwiGLU).

Defaults match the sonic-moe paper's headline configuration. We add:
  - a smaller shape for fast iteration during agent development
  - a wider intermediate (different aspect ratio) to stress N-tile selection
"""

SHAPES = [
    # Headline sonic-moe shape: 32K tokens, 128 experts, top-8.
    {"T_total": 32768, "H": 4096, "I": 1536, "E": 128, "K": 8},

    # Fast-iteration shape (~16x cheaper). Same expert count to keep the
    # variable-length grouped layout meaningful, but smaller token / hidden dims.
    {"T_total": 4096, "H": 2048, "I": 1024, "E": 64, "K": 4},

    # Different aspect ratio: smaller H, wider I (intermediate-heavy FFN).
    # Forces tiles to handle larger N relative to K.
    {"T_total": 16384, "H": 2048, "I": 4096, "E": 64, "K": 8},
]
