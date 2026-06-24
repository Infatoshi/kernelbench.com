"""Grading configuration for the grid-foraging PPO training megakernel.

Unlike the roofline problems, this one is graded on training throughput
(environment steps per second) at a single fixed task size, with correctness
gated on the learned return level across several seeds. There is no shape
sweep; these constants are the grading knobs check.py and benchmark.py read.
"""

# Total environment steps a single training run executes, from scratch.
# = ROLLOUT (32) * NUM_ENVS (4096) * 40 iterations.
TOTAL_ENV_STEPS = 32 * 4096 * 40

# Correctness is checked on these seeds (reference vs solution return level).
GRADING_SEEDS = (0, 1, 2)

# Throughput is measured on this seed.
BENCH_SEED = 7
