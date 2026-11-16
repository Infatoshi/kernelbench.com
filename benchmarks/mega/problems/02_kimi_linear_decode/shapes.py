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

# Correctness is checked on these seeds (one decode step each).
GRADING_SEEDS = (0, 1, 2)

# Seeds for the attention-sensitive cases (reference.init_state_attn), where MLA
# attention carries the output and so amplifies any upstream difference. Chosen
# so that every MoE router's top-8 cut has a relative margin of >= 3% in the
# reference (no near-tie for honest rounding to flip); check.py re-verifies it.
# benchmark.py's gate uses the first one.
ATTENTION_SEEDS = (27, 64, 76)
