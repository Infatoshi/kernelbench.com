"""Shape sweep for the gated W4A16 GEMV.

Group size is 96, so most K values leave a ragged last group (4096 = 42*96 + 64,
5120 = 53*96 + 32, 2560 = 26*96 + 64); 3072 = 32*96 exactly, so the aligned case
is covered too. N=4607 is the off-alignment output width.
"""

SHAPES = [
    {"M": 1, "N": 11008, "K": 4096},  # decode, ragged last group (64 rows)
    {"M": 1, "N": 4607, "K": 3072},   # decode, aligned K, odd N
    {"M": 1, "N": 13824, "K": 5120},  # decode, ragged last group (32 rows)
    {"M": 4, "N": 8192, "K": 2560},   # tiny speculative batch, ragged (64 rows)
]
