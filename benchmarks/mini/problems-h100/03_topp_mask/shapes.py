"""Shape sweep for the top-p mask.

Vocab sizes from real tokenizers (Qwen 151936, Llama-3 128256, Llama-2 32000)
plus one odd V. Small B keeps the op in launch-overhead territory — the
headline for this problem is milliseconds, not peak fraction.
"""

SHAPES = [
    {"B": 1, "V": 151936, "P": 0.9},   # single-stream decode, Qwen vocab
    {"B": 8, "V": 128256, "P": 0.95},  # small serving batch, Llama-3 vocab
    {"B": 64, "V": 32000, "P": 0.9},   # large batch, Llama-2 vocab
    {"B": 16, "V": 100003, "P": 0.92}, # odd vocab: no power-of-two tricks
]
