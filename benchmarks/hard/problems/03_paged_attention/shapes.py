"""Shape sweep for paged attention decode.

Mix targets:
  - small batch / long context (Llama-3 8B-style decode)
  - large batch / medium context (server batched decode)
  - GQA wide ratio (Llama-3 70B: 64 heads / 8 kv-heads)
  - non-power-of-2 seq_len (forces predicated tail handling)
  - head_dim=64 small-head case
"""

SHAPES = [
    # (B, H, Hkv, D, L, P)
    {"batch": 8,  "num_heads": 32, "num_kv_heads": 8,  "head_dim": 128, "seq_len": 1024, "page_size": 16},
    {"batch": 32, "num_heads": 32, "num_kv_heads": 8,  "head_dim": 128, "seq_len": 2048, "page_size": 16},
    {"batch": 4,  "num_heads": 64, "num_kv_heads": 8,  "head_dim": 128, "seq_len": 4096, "page_size": 16},
    {"batch": 16, "num_heads": 32, "num_kv_heads": 8,  "head_dim": 128, "seq_len": 1535, "page_size": 16},  # non-pow2
    {"batch": 8,  "num_heads": 16, "num_kv_heads": 4,  "head_dim": 64,  "seq_len": 2000, "page_size": 16},  # small-D, non-pow2
]
