"""Shape sweep for the segmented decay scan.

T=8191 forces a ragged chunk tail for chunked-scan strategies; T=16384 with
B=1 removes batch parallelism so speed must come from parallelizing over T.
"""

SHAPES = [
    {"B": 8, "T": 4096, "D": 2048},   # comfortable batched trainer shape
    {"B": 4, "T": 8191, "D": 1024},   # odd T: ragged chunk tail
    {"B": 16, "T": 2048, "D": 4096},  # wide channels, short sequences
    {"B": 1, "T": 16384, "D": 512},   # single long sequence: scan-parallel or bust
]
