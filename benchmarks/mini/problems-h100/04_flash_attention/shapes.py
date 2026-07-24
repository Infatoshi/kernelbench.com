"""Shape sweep for causal flash attention.

S=4097 forces a ragged tile tail; S=16384 with B=1, H=8 is the long-context
stress where an O(S^2)-memory strategy simply cannot run.
"""

SHAPES = [
    {"B": 1, "H": 32, "S": 4096, "D": 128},  # llama-7B-ish prefill
    {"B": 1, "H": 32, "S": 4097, "D": 128},  # odd S: ragged tile tail
    {"B": 4, "H": 8, "S": 8192, "D": 128},   # batched long prefill
    {"B": 8, "H": 16, "S": 2048, "D": 64},   # small heads, D=64 tile geometry
    {"B": 1, "H": 8, "S": 16384, "D": 128},  # long context: streaming or bust
]
