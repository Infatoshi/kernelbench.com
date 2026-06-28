"""Shape sweep for W4A16 GEMM.

Llama-style up_proj / qkv_proj shapes. Decode (M=1) is the bandwidth-bound
case every inference engine optimizes -- it's the bar to beat.
"""

SHAPES = [
    {"M": 1,   "N": 12288, "K": 4096},   # decode: memory-bound on int4 weight read
    {"M": 32,  "N": 12288, "K": 4096},   # small prefill: mixed regime
    {"M": 256, "N": 12288, "K": 4096},   # larger prefill: approaching compute
    {"M": 1,   "N": 4096,  "K": 4096},   # decode: square shape
    {"M": 16,  "N": 14336, "K": 4096},   # speculative-decode-ish
]
