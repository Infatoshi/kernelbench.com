"""NSA shape sweep — long context + block-boundary stress.

block_size=64 in problem.yaml. Include S not multiple of 64 so the last
partial block and top-n selection tails are forced (same role as K=4127
on Hard FP8 GEMM). Score = geomean over shapes.
"""

SHAPES = [
    # Comfortable prefill, aligned to block_size
    {"B": 1, "H": 16, "S": 2048, "D": 64},
    # S not multiple of 64
    {"B": 1, "H": 16, "S": 4127, "D": 64},
    # Longer context, fewer heads
    {"B": 1, "H": 8, "S": 8192, "D": 64},
    # Long + partial last block
    {"B": 1, "H": 8, "S": 8191, "D": 128},
    # Batched short (server multi-seq microbatch)
    {"B": 4, "H": 8, "S": 1024, "D": 64},
    # Odd S mid-length
    {"B": 2, "H": 8, "S": 3000, "D": 64},
]
