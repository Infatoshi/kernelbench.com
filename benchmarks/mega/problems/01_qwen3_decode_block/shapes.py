"""Shape sweep for Qwen3-0.6B batch=1 decode-block megakernel.

`seq_len` is the full decode context length seen by the new token. The input
KV cache contains the previous `seq_len - 1` tokens, and the freshly projected
current K/V makes the attention/GEMV decode scan length exactly `seq_len`.
"""

SHAPES = [
    {"seq_len": 32},
    {"seq_len": 128},
    {"seq_len": 512},
]
