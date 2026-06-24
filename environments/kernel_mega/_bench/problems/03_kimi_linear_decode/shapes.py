"""Decode shapes for the Kimi-Linear hybrid unit.

Score is latency speedup over baseline.py, so the sweep is over decode context
length. The long-context shape is where the architecture's design (fixed-size
KDA state + compressed MLA latent cache) pays off and where the absorb-vs-
materialize fork in MLA decides the whole thing -- that is the shippable "wow".

`n_experts` is 64 for v0 (real Kimi-Linear-48B-A3B is 256); bump it once the
problem is calibrated. `steps` is how many tokens to decode autoregressively
when timing (fewer at long context to bound wall time).
"""

SHAPES = [
    {"context_len": 2048, "steps": 32, "n_experts": 64},
    {"context_len": 8192, "steps": 16, "n_experts": 64},
    {"context_len": 16384, "steps": 16, "n_experts": 64},
]

# Correctness is checked on these seeds (one decode step + a few AR steps each).
GRADING_SEEDS = (0, 1, 2)
