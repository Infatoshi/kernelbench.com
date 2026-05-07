"""Canonical shape sweep for Conv3d-as-GEMM patch embedding.

Sized for Qwen2-VL / ViT-style patch embedding. embed_dim=1280 matches Qwen2-VL
ViT; embed_dim=768 is a ViT-B/16 reference. Patch sizes are the canonical
14x14 spatial / 2-frame temporal used in modern VL encoders.
"""

SHAPES = [
    # Qwen2-VL ViT base: 224x224x2-frame video, 14x14x2 patches, embed_dim=1280
    {"B": 1, "C": 3, "T": 2,  "H": 224, "W": 224, "kT": 2, "kH": 14, "kW": 14, "embed_dim": 1280},

    # Same patch config, batch 2, 4-frame video
    {"B": 2, "C": 3, "T": 4,  "H": 224, "W": 224, "kT": 2, "kH": 14, "kW": 14, "embed_dim": 1280},

    # Larger spatial (336x336), 8-frame — exercises larger num_patches
    {"B": 1, "C": 3, "T": 8,  "H": 336, "W": 336, "kT": 2, "kH": 14, "kW": 14, "embed_dim": 1280},

    # ViT-B/16 (image-mode): kT=1, kH=kW=16, embed_dim=768
    {"B": 4, "C": 3, "T": 1,  "H": 224, "W": 224, "kT": 1, "kH": 16, "kW": 16, "embed_dim": 768},
]
