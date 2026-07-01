# Mega "megakernel" audit — 2026-07-01

## Question
The mega deck is branded a **megakernel** bench (both `display_name`s say
"Megakernel"). Do the submissions actually fuse the work into a single kernel
launch, or do they hit the leaderboard some other way?

## Method
Static scan of every archived mega run's `solution.py` **plus any local module it
imports** (e.g. a sidecar `kernels.py`): count custom kernels (`@triton.jit` +
`load_inline`/`__global__`), and flag CUDA-graph (`CUDAGraph`/`cuda.graph`/
`graph.replay`) and `torch.compile` usage. Spot-checked the headline cells by
reading the solutions (Opus 19.35x, Sonnet 4.03x, and a "1-kernel" Gemini cell)
to confirm the static signal.

## Headline finding
**Zero of the 36 decode submissions (and the 1 RL submission) are true
single-launch megakernels.** The bench's defining premise is unmet across the
board:

- **Every high-scoring decode cell wins via CUDA graphs, not fusion.** All 7 of
  the top cells (19.35x → 11.14x) are `MULTI+CUDAGRAPH`: 5–16 separate kernels
  stitched under a captured graph. A CUDA graph replaying a dozen kernels
  eliminates launch overhead without fusing anything — it is not a megakernel.
- **The genuine single-custom-kernel attempts are mostly-eager, not
  megakernels, and score low.** The runs with exactly 1 custom kernel (Gemini
  2.74x, DeepSeek 1.56x, GLM 0.86x) pair that one kernel with a pile of eager
  PyTorch ops (einsum/softmax/silu/cat/topk…) in the step path — verified on the
  Gemini 2.74x cell. They are "1 fused GEMV helper + eager everything else," not
  a fused decode step.
- **The Sonnet 5 vs Opus 4.03x → 19.35x gap is entirely this axis.** Same
  algebra (near-identical int4 dequant-GEMV, MLA absorption, delta-rule). Opus
  captured the whole step as a CUDA graph (≈1 replay/token); Sonnet drove ~9
  kernels/token from Python. Neither fused into a real megakernel.

### Decode verdict distribution (36 runs with source)
- `MULTI+CUDAGRAPH`: 9  ← all the top cells
- `MULTI_EAGER` (many kernels, no graph): 20
- single custom kernel + eager rest (scored ≤2.74x): 4
- `MULTI+COMPILE` (torch.compile): 1
- no custom kernel at all: 2

### RL (`01_rl_grid_ppo`)
Only **one** submission exists total (codex gpt-5.5), using `torch.compile`,
scoring 0.32x of the (provisional) ceiling. The RL problem is effectively
unbenched.

## Why this happened
The prompts and graders never required a megakernel. The decode `PROMPT.txt`
asked only for "fast batch-1 decode" + "write your own fused int4 GEMV"; the
grader times `step()` in a Python loop; the forbidden list was libraries only.
CUDA graphs / `torch.compile` were unmentioned and unforbidden, so models
rationally reached for the cheapest launch-overhead fix instead of fusion. The
mandatory reward-hack/contamination audit never checked "did the solution do the
kind of thing the bench claims to test."

## Actions taken (benchmark surface v2)
Both mega problems were versioned to enforce the premise:

1. **`PROMPT.txt` (both):** explicit hard mandate.
   - Decode: the timed `step()` must be a **single GPU kernel launch**; the whole
     per-token forward fused into one custom kernel. CUDA graphs / `torch.compile`
     / per-op kernel loops are banned as substitutes.
   - RL: launches must **not scale with env-steps/horizon/minibatches**; the whole
     rollout (all HORIZON steps × all envs + policy + sampling + advantage) runs
     in one persistent kernel launch, PPO update fused in or a small fixed number
     of extra launches per iteration. Coarse fusion (many rollouts per launch) is
     the goal — only per-step/per-minibatch loops are ruled out.
2. **`problem.yaml` (both):** added `torch.compile`, `CUDAGraph`, `cuda.graph`,
   `make_graphed_callables` to `forbidden`, plus a documented `constraints:` block.
3. **`check.py` (both):** the forbidden scan now covers `solution.py` **and every
   local module it imports** (recursively), so the escape hatches can't hide in a
   sidecar `kernels.py`.

These give immediate teeth against the two launch-overhead workarounds (graph +
compile), enforced by `check.py`.

## Remaining gap (follow-up)
String-forbidding graph/compile does **not** by itself guarantee "exactly one
launch" — a `MULTI_EAGER` many-small-kernels solution would still pass `check.py`
while not being a megakernel. The airtight fix is a **profiler-based launch-count
gate** in `check.py`: warm up a step, profile one step, assert exactly 1 kernel
launch (decode) / launches that don't scale with steps (RL). This needs on-GPU
validation of the counting method (CUDA-graph replay and memcpy/memset must be
handled) before wiring in.

## Implication for any prior "mega" numbers
The existing mega board reflects CUDA-graph launch-overhead elimination, not
megakernel fusion. Under v2 rules the high-scoring graph/compile cells would fail
`check.py`. A re-sweep under v2 is required before reporting a real megakernel
leaderboard.
