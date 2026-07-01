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

## Actions taken (benchmark surface v2 → v2.1)
Both mega problems were versioned to enforce the premise. The first pass (v2)
tried to string-forbid `torch.compile`/`CUDAGraph` in `check.py`; a red-team
battery (below) showed that gate is both leaky and brittle, so v2.1 moved
authenticity to a **judge gate fed by advisory tripwires**. Final state:

1. **`PROMPT.txt` (both):** explicit hard mandate.
   - Decode: the timed `step()` must be a **single GPU kernel launch**; the whole
     per-token forward fused into one custom kernel. CUDA graphs / `torch.compile`
     / per-op kernel loops do not count — a **post-run authenticity judge** reads
     the final source (recursively) and rejects them; obfuscation is itself a flag.
   - RL: launches must **not scale with env-steps/horizon/minibatches**; the whole
     rollout in one persistent kernel launch, PPO update fused in or a small fixed
     number of extra launches per iteration. Coarse fusion is the goal — only
     per-step/per-minibatch loops are ruled out. Same judge gate.
2. **`problem.yaml` (both):** graph/compile are **not** in `forbidden` (they
   false-positive on honest disclaimers and miss obfuscation). `forbidden` is now
   libraries only; a `constraints:` block documents `authenticity_gate: judge` and
   the advisory tripwires.
3. **`check.py` (both):** the one bright-line hard fail is **importing a banned
   library**, matched by **AST import statements** (not substring) over
   `solution.py` + every local module it imports (recursively, incl. `scratch/`).
4. **`src/eval/megakernel.py` + `scripts/megakernel_evidence.py`:** deterministic
   advisory evidence — recursive source, kernel count, and graph/compile/codegen/
   obfuscation tripwires (graph/compile checked on comment+string-stripped code so
   a disclaimer doesn't trip; obfuscation caught at the AST level).
5. **The judge gate:** the mandatory pre-publish audit renders the judge prompt
   from the evidence and records `megakernel_authentic: true|false` in
   `results/annotations/<run_id>.yaml`. `build_mega_leaderboard.py` excludes any
   run marked `false` (like the contamination exclusion). See
   `docs/megakernel_authenticity_judge.md`.

## Red-team result (why v2.1, not v2)
`tests/test_megakernel_evidence.py` runs 7 adversarial cases. A raw substring gate
on graph/compile **false-fails** an honest solution that merely disclaims the
techniques in a comment (A7), and **misses** `getattr(torch.cuda,"CUDAGra"+"ph")`
(A5) and `importlib`-based runtime codegen (A6). The deterministic tripwires now
catch A5 (obfuscation) and A6 (codegen) and stay quiet on A7; the LLM judge
resolved every case correctly (PASS A1, FAIL A2–A6, FAIL A7-as-eager). Verified
on real archives: the Opus 19.35x cell trips the graph tripwire (9 kernels +
CUDA graph), Sonnet's cell is graph-clean (9 Triton kernels, no graph) — both
correctly flagged as *not a single fused megakernel* for the judge to rule on.

## Remaining gap (follow-up)
The tripwires + judge do not *measure* launch count. The airtight complement is a
**profiler-based launch-count gate**: warm up a step, profile one step, assert 1
launch (decode) / launches that don't scale with steps (RL), handling CUDA-graph
replay and memcpy/memset. Needs on-GPU validation before wiring in; the judge gate
is the enforcement mechanism until then.

## Implication for any prior "mega" numbers
The existing mega board reflects CUDA-graph launch-overhead elimination, not
megakernel fusion. Under v2.1 the high-scoring graph/compile cells (incl. the Opus
19.35x decode cell) must be annotated `megakernel_authentic: false` and are
excluded by the builder, or re-done as a true single fused kernel. A re-sweep is
required before reporting a real megakernel leaderboard. Mega is not on the public
site yet, so nothing user-facing is currently wrong.
