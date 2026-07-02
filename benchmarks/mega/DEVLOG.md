# DEVLOG

A running record of decisions, dead ends, and lessons. Newest entries on top. This is not a changelog (the git log is) — it's the why behind the shape of the project.

---

## 2026-07-01 — fable-5 lands 14.3x on 01_rl_grid_ppo; SPS credibility cap recalibrated 20x → 100x

First fable-5 (Claude Code, effort max) run on the RL deck: 14.2961x
(357M SPS), 54 min agent time. Manual audit verdict: **clean, megakernel
authentic** — the entire training run (40 iters of rollout + GAE + 4×4 PPO/Adam
update) executes inside a single `cudaLaunchCooperativeKernel` launch with
`grid.sync()` barriers; one warp per env episode, MLP split 2 units/lane with
butterfly reductions. Empirically verified from the run workspace: fresh random
seeds start at random-policy return (~0.13) and learn to ~3.97, seed-dependent
curves, from-scratch on every call, 356M SPS reproduced independently. See
`results/annotations/20260701_172617_claude_claude-fable-5_01_rl_grid_ppo.yaml`.

Consequence: yesterday's fabrication guard set `max_credible_sps_multiple: 20`
(500M SPS) when the best known honest kernel was the ~2.1M SPS reference on the
3090. A clean 357M SPS solution leaves only ~1.4x honest headroom before the
cap zeroes a legitimate kernel. Recalibrated to 100x (2.5B SPS): still orders
of magnitude below the ~20,000x a no-work fabricator posts, so the guard keeps
its teeth. The 25M `peak_sps` scoring ceiling is also clearly miscalibrated
(scores now read as >14x "of peak") but rescaling it changes every published
multiple, so that decision is deferred — flagged, not changed.

---

## 2026-07-01 — RL sim fidelity smoke + fabrication red-team on the 3090 (benchmark v2 guards)

Pre-sweep smoke of `01_rl_grid_ppo` on the idle RTX 3090 (disposable /tmp
workspace, `CUDA_VISIBLE_DEVICES=1`), before handing the deck to an external
machine. Fidelity is good: reference trains 0.16 → 3.97 return over 40 iters,
bit-deterministic per seed, seed-to-seed final spread <1%, ~2.1M SPS on the
3090. check.py's band logic and the honest reference-as-solution control both
behave.

Red-teaming found a **critical hole: the return curve is unfalsifiable.**
`train()` just returns floats — nothing proves an env step ever happened. Two
cheats sailed through the old gates with absurd scores:
- **Fabricated curve** (analytic exponential ramp, no GPU work at all): check.py
  PASS (lands in the return band, "improves" from its own early window),
  benchmark **22,006x** peak_fraction.
- **Memoized replay** (`{(steps, seed): curve}` dict; benchmark reran the same
  BENCH_SEED every trial): **26,849x**.

Fixes shipped (validated same-session on the 3090):
- **benchmark.py draws a fresh random seed per timed trial** (SystemRandom).
  SPS doesn't depend on the seed, but a memoized lookup misses every trial —
  the cached cheat now pays full training cost (honest ~6.2M SPS on the 3090).
- **Return floor enforced per trial**, not just on the last curve.
- **SPS credibility cap**: `max_credible_sps_multiple: 20` in problem.yaml —
  anything over 20x peak_sps scores 0.0 outright (fabricator lands ~4 orders of
  magnitude over; a real megakernel can't). Near-cap results are an audit flag.
- **`import reference` / `from reference` / baseline added to `forbidden`** for
  the RL problem (decode already had them; RL didn't — a solution could
  literally re-export `reference.train`).

Residual, documented and judge-gated: a fabricator that also *sleeps* to fake a
plausible elapsed time defeats any mechanical timing check. That's exactly what
the mandatory authenticity audit catches — a "trainer" with no environment and
no policy update fails on sight. check.py still can't verify env steps happened
(fabricated curves pass correctness); the benchmark guards + judge are the
enforcement pair.

---

## 2026-07-01 — megakernel authenticity: judge gate + advisory tripwires (not a substring ban)

Audit finding that started this: **zero of the archived mega submissions are
true single-launch megakernels**, despite the bench being branded one. Every
high-scoring decode cell wins via CUDA graphs (Opus 19.35x = 9 Triton kernels
replayed under a captured graph — cuts launch overhead without fusing anything);
the honest single-custom-kernel attempts are "1 fused GEMV + eager everything
else." The Sonnet 5 (4.03x) vs Opus (19.35x) gap is *entirely* this axis: same
algebra, but Opus captured the whole step as a graph (~1 replay/token) while
Sonnet drove ~9 kernels/token from Python. Neither actually fused. The prompts
and graders never *required* a megakernel, so models rationally reached for the
cheapest launch-overhead fix. Full write-up: `docs/megakernel_audit_2026-07-01.md`.

**First attempt (v2) was wrong: string-forbid `torch.compile`/`CUDAGraph` in
check.py.** Before shipping it I red-teamed the gate with a 7-case adversarial
battery (`tests/test_megakernel_evidence.py`). A raw substring scan turned out to
be the worst of both worlds:
- **Leaky** — `getattr(torch.cuda, "CUDAGra"+"ph")` (A5) and `importlib`-based
  runtime codegen that writes+imports a kernel module (A6) carry no literal
  banned token, so they sail through.
- **Brittle** — an honest solution whose *comment* says "no torch.compile, no
  CUDA graphs" (A7) gets false-failed on its own disclaimer.

A substring gate therefore punishes honesty and rewards obfuscation. Killed it.

**Shipped (v2.1): judge gate fed by deterministic advisory evidence.**
- The one bright line that stays a hard fail in check.py is **importing a banned
  library**, matched by **AST import statements** (not substring), recursively
  over solution.py + every local module it imports (incl. `scratch/` sidecars,
  where archived claude/cursor runs stash the real kernel). Naming a lib in a
  comment no longer fails; `marlin` no longer matches `marlinx`.
- `src/eval/megakernel.py` (CLI `scripts/megakernel_evidence.py`) extracts
  objective signals: recursive source, kernel count, and graph/compile/codegen/
  obfuscation tripwires. graph/compile are matched on **comment+string-stripped
  code** (so a disclaimer can't trip them); obfuscation (getattr string-concat,
  banned-token folding) is caught at the AST level so it survives stripping.
- The mandatory pre-publish audit renders the judge prompt from that evidence and
  records `megakernel_authentic: true|false` in `results/annotations/<run_id>.yaml`.
  The judge reasons from code, treating tripwires as hints and docstrings as
  untrusted. Red-team result: judge PASSed A1, FAILed A2–A6 (incl. the obfuscated/
  codegen evasions the substring scan missed) and FAILed A7-as-eager.
- `build_mega_leaderboard.py` **excludes** runs annotated `megakernel_authentic:
  false` (alongside the contamination exclusion), and now emits a **megakernel
  column**: the custom-kernel count in the timed path (launches-per-step proxy;
  for RL, coarse fusion of many steps into one launch is expected, budget ≤8) +
  a green/red marker. Green = genuine fused megakernel within the launch budget;
  red = hides launches / unfused / eager. The marker uses the judge verdict when
  present, else a provisional evidence-based read (hollow dot + trailing `?`).
  `/mega` renders it. Rubric + integration: `docs/megakernel_authenticity_judge.md`.

**Prompts (both problems) updated** to state the timed path must be one fused
kernel and that a *post-run authenticity judge* (not check.py) rejects graph/
compile/per-op-loop escapes — and that obfuscating them is itself a red flag.
Decode mandates one launch/step; RL allows coarse fusion (many env-steps per
launch) but forbids launch counts that scale with steps/horizon/minibatches.

**Known implication, not yet actioned:** the published Opus 19.35x decode cell is
a CUDA-graph solution → not an authentic megakernel. It (and the other graph/
many-kernel cells) currently show a *provisional* red marker; they need explicit
`megakernel_authentic: false` annotations, or a redo as a true single fused
kernel, before any megakernel-headline re-publish. Mega is not on the public site
yet, so nothing user-facing is wrong today.

**Follow-up (the airtight complement):** the tripwires+judge do not *measure*
launch count. A profiler-based launch-count gate (warm up, profile one step,
assert 1 launch for decode / non-scaling launches for RL, handling graph replay
and memcpy/memset) is the durable enforcement. Needs on-GPU validation before
wiring; the judge gate is the mechanism until then.

---

## 2026-06-18 — contamination prevented at the source: bwrap sandbox on run_hard.sh

The near-term, easy fix for cross-run contamination (agents reading prior winning
solutions from the shared outputs/runs archive via absolute paths). The proper
sandboxed harness is the Prime verifiers env (`environments/kernelbench_decode/`,
handed off separately); this is the cheap, in-place prevention that ships now.

Each agent launch in run_hard.sh is now wrapped in:
  bwrap --dev-bind / / --tmpfs $REPO_ROOT/outputs/runs --bind $RUN_DIR $RUN_DIR --chdir $PROBLEM_DIR
`--dev-bind / /` keeps EVERYTHING working (toolchain, src symlink, GPU, codex/
node/claude auth, outputs/gpu.lock — which lives at outputs/gpu.lock, NOT under
outputs/runs, so it stays visible). The `--tmpfs` over outputs/runs HIDES every
other run from the agent; `--bind $RUN_DIR` re-exposes just this run's own dir
(writable, persists). Net: the agent physically cannot read other solutions, and
nothing else changes. Validated: GPU visible under bwrap, codex runs (auth+node
intact), other runs invisible (ls -> 0), find solution.py -> 0, own writes
persist. Only the agent `timeout "$BUDGET_SECONDS"` launches are wrapped (13 of
them); the harness-owned post-run check.py/benchmark.py scoring runs OUTSIDE the
sandbox (different timeout var) and writes results normally.

Toggle: `KBH_SANDBOX=0` disables; auto-off if bwrap is absent (e.g. a box without
bubblewrap). The publish-time tripwire (build_mega_leaderboard.py +
audit_contamination.py) stays as defense-in-depth. See
[[cross-run-contamination]].

---

## 2026-06-18 — budget is a 3-hour ceiling (documented, not "unlimited")

All mega problem-03 runs this session (codex/opus + the 6-model Hard-roster
expansion) used `BUDGET_SECONDS=10800` — a 3-hour wall-clock ceiling, NOT
literally unlimited. Decision (Elliot): keep 3h and document it rather than
re-run at a larger ceiling. Justification: nothing is hitting the cap — every
run self-terminated (decided it was done, was not cut off). The new 6 models
top out ~1.1h (glm 3882s, MiniMax 3815s, deepseek 2556s, gemini 1867s, composer
867s); the deepest worker, opus, peaked at ~2.5h (B200 9177s), leaving ~30min
margin. Because every cell shares the same ceiling, the board is internally
comparable. The /mega page states the 3h ceiling explicitly so it does not claim
unlimited. If a future/harder problem pushes a strong model past 3h, bump the
ceiling (a running `timeout` cannot be extended).

---

## 2026-06-18 — three-GPU leaderboard published (Blackwell / H100 / B200)

Problem 03 (W4A16 Kimi-Linear decode) swept across three GPU generations, codex
+ opus on each, all published to /mega with the GPU as a table column:

      GPU                     codex/gpt-5.5    opus-4-8
      RTX PRO 6000 Blackwell      4.34x         14.40x
      H100                        5.62x         15.50x
      B200                        9.37x         19.35x   <- highest

Reads: opus dominates codex on every GPU (deeper kernel engineering under
unlimited time). Both models scale *up* the speedup ratio from Blackwell ->
H100 -> B200, because the baseline (naive int4 materialize) gets relatively
worse on the bigger datacenter cards while the fused dequant-GEMV keeps pace --
so the fusion win compounds with bandwidth. The score is a same-GPU
speedup-over-baseline ratio, which is exactly why it ports across generations
with no recalibration. int4/bf16-acc ran on all three with stock cu128 torch and
zero code changes (Ampere/Hopper/Blackwell all do bf16; no special tensor-core
format), validated on B200 sm_100 (driver 580).

One-shot reproducibility proven on the B200: `cloud_launch.sh <instance> <gpu>`
did rsync + bootstrap + cu128-torch-on-sm_100 + the codex+opus sweep in one
command (`cloud_sweep.sh` tags each run with its GPU so the leaderboard builder
picks it up). The recurring failure was brev dropping the host entry from
`~/.brev/ssh_config` (DNS resolve fails) -- now self-healed via `brev refresh`
retry in `ensure_reachable`, and monitoring switched to direct-IP ssh to dodge
the alias churn entirely.

Publish path: `scripts/build_mega_leaderboard.py` (requires a per-run `gpu`
marker, excluding legacy bf16 03-runs) -> `public/data/mega/results.csv` ->
`app/mega/page.tsx` (GPU column, grouped by GPU then speedup). Both my cloud
instances (H100, B200) terminated after pull; never touched Elliot's kbh-* eval
boxes.

---

## 2026-06-18 — Autonomous H100 cloud run + per-GPU leaderboard

Ran codex/gpt-5.5 (xhigh) autonomously on problem 03 (W4A16 Kimi-Linear decode)
on a brev H100, away from the local Blackwell workstation. Result: **5.62x
geomean speedup** over the optimized-PyTorch baseline (2.18 ms/tok, 459 tok/s;
6.09x at ctx 16384), self-terminated in 34 min well under a 3h budget.

The journey is the point, and the transcript shows it: 15 benchmark runs, 37
correctness checks, peak_fraction climbing 0.0 → 3.17 (first working fused
version, already past launch reduction) → 4.53 → 5.48 → 5.85 plateau. It wrote
**11 specialized Triton kernels** and visibly experimented with the hardest rung
(three MoE-batching variants in the final file: `_w4_expert_same_x`,
`_w4_expert_batch_x`, `_w4_expert_pair_same_x`), fused q/k/v/g into one 4-way
GEMV, did MLA absorption in-kernel (`_kvb_score`/`_kvb_value`), fused the down
projection with the routed weighted-sum, and hand-wrote the KDA conv +
recurrence. My own hand solution stalled at 1.1x (launch-bound, 126 tiny
kernels); codex crushed that by batching/fusing — the exact ladder the problem
was designed to expose.

**Reproducible cloud run (committed):** `scripts/cloud_bootstrap.sh` (on the
box: uv + node + codex/claude CLIs + cu128 torch — R570-safe, skips the painful
R580/cu130 driver upgrade since the decode problems are portable bf16/int4) and
`scripts/cloud_launch.sh` (from anvil: rsync repo + auth [codex token / claude
creds / env_vars], bootstrap, launch a detached run). Recipe: `brev create H100`
→ `cloud_launch.sh` → ~8 min to a running journey. Bake a brev image from a
bootstrapped box to get reprovision under a minute.

**Per-GPU leaderboard:** the score for problem 03 is a speedup over baseline (a
same-GPU ratio), so it ports across GPU generations with zero recalibration.
`scripts/build_mega_leaderboard.py` scans run archives (requiring a per-run `gpu`
marker, so legacy bf16 runs that share the `03` problem name are excluded) and
emits `public/data/mega/results.csv` with a `gpu` column. `app/mega/page.tsx`
reuses the proven `/v3` CSV + GPU-filter pattern to show RTX PRO 6000 Blackwell
/ H100 / B200 categories. int4/bf16-acc needs no special tensor-core format, so
the same problem runs on Ampere → Hopper → Blackwell.

Gotchas hit: brev's short-lived auth token (a failed-auth `delete` reports a
false "gone" — always re-verify `brev ls`); `brev org set`/`refresh` regenerate
`~/.brev/ssh_config` and drop the active instance's host entry (do not switch
orgs mid-run); zsh does not word-split unquoted vars used as commands (inline ssh
or use arrays). Never `brev delete/stop/reset` an instance you do not own.

---

## 2026-06-17 — Problem 03 went W4A16 (int4 weights), because fp8 loses at decode

Problem 03 (Kimi-Linear hybrid decode) started bf16, then the plan was to make
the baseline fp8 to force quantization into play. Benchmarking killed that:
`torch._scaled_mm` fp8 is a tensor-core compute path with ~7us fixed overhead
and M-padding, so at batch-1 decode (memory-bound) it is *slower* than bf16.

  fp8/bf16 overall speedup across the decode projection set:
    M=1 (decode): 0.84x   M=8: 0.97x   M=32: 1.05x   M=128: 1.16x   M=256: 1.45x

fp8 only pays at large batch. This also explains the dispersion runs where none
of codex/opus/gemini took fp8 — at decode it is not a free win; the real win
needs a hand-written fused dequant-GEMV, and stock fp8 makes it worse.

The fix (Elliot's call): **W4A16** — int4 weights, group-128 asymmetric, bf16
accumulation (the AWQ/GPTQ format OSS actually ships). At batch-1 it is a
*memory-bound* dequant-GEMV, not a compute path, so the 4x weight-traffic
reduction is realizable exactly where decode lives. Empirically grounded by
Hard's own `07_w4a16_gemm` (`regime: memory`, M=1 bandwidth-bound) where models
already wrote fused W4A16 GEMVs hitting 0.15-0.35 of peak DRAM bandwidth on
SM120. Bonus: int4/bf16-acc needs no special tensor-core format, so it runs on
any bf16 GPU (Ampere 3090, Hopper, Blackwell) — the benchmark can travel across
GPU generations, and the speedup-over-baseline metric is already a same-GPU
ratio so it ports with zero recalibration.

Shape of the problem now:
- Weights stored W4A16 (reuses Hard 07's exact pack/dequant; format in
  reference.py). MoE experts quantized too (one int4 set per expert).
- reference.py dequantizes naively in fp32 (oracle). baseline.py is bf16 +
  batched-MoE but still *materializes* each bf16 weight (int4 read + bf16 write
  + bf16 read = ~9x the traffic of fusion) — the floor, deliberately leaving the
  fused dequant-GEMV on the table. Naive int4 is *slower* than plain bf16; int4
  only pays with fusion, which is the whole test.
- Correctness is cosine >= 0.98 of the next-token hidden + decode state vs the
  oracle. The int4 quant noise is in both sides (same weights), so they match at
  ~0.9999 — no tolerance loosening needed, unlike Hard 07 (which compares int4
  against an un-quantized bf16 reference and had to loosen).
- Forbidden now also bars prebuilt int4 kernels (bitsandbytes, torchao, marlin,
  gptq/awq, exllama) so the model writes its own fused dequant-GEMV.
- int4 buffers: 1068 MB vs 4273 MB bf16. Baseline floor ~5.6 ms/tok at ctx 2048.

The "epic" target is now concrete: fused int4 dequant-GEMV + MLA absorption +
KDA/MoE fusion, stacked. None of the three models stacked all of it in the bf16
version; W4A16 makes quantization the central, realizable lever.

---

## 2026-06-16 — Problem 02: RL training megakernel (throughput-graded), v0

Added `problems/02_rl_grid_ppo`, the first non-roofline problem on the deck. It
asks for a from-scratch PPO training run (vectorized grid-foraging env + tiny
MLP actor-critic + PPO update) made as fast as possible, graded on training
throughput (environment steps per second), not FLOPS/bandwidth peak fraction.

**Why this and not a standalone LLM training megakernel.** A fused fwd+bwd
transformer block is low-signal: at frontier scale the training step is already
a chain of big compute-bound GEMMs that cuBLAS/FlashAttention saturate, so there
is nothing to fuse; the only people who care about fused training kernels are
small-model fine-tuners, and Liger-Kernel already owns that. The training that
*is* overhead-bound — tiny nets, millions of tiny steps, env↔learn ping-pong —
is RL. So the "training megakernel" that matters lives inside the RL loop, which
is exactly this problem. It also opens the train/infer/sim spread: problem 01 is
memory-bound forward-only decode, this is throughput-bound full-loop training.

**Why throughput, not roofline.** An RL step is control-flow / launch-overhead
bound and has no clean FLOPS ceiling, so peak_fraction-vs-FLOPS is meaningless.
Instead score `achieved_sps / peak_sps`, the same shape of number against a
target ceiling. To keep it from confounding kernel speed with RL luck, the
algorithm + hyperparameters + total step budget are fixed and seed-determined;
"fastest time to train" then collapses to "fastest to run N steps" = SPS.

**Correctness is the learned return level, not allclose.** A different kernel
will never reproduce the reference trajectory bit-for-bit (RNG stream + float
reduction order differ), so `check.py` trains both reference and solution from
scratch on several seeds and requires the solution's final-window mean return to
land in a band around the reference's, and to have climbed from its own early
baseline. The env is tuned so a random policy scores ≈0.16 and a trained one
≈3.96 (a ~25x gap), giving the band real signal. A no-op-fast-loop cheat is
killed twice: the band (no learning → below floor) and a benchmark-side return
floor (SPS only credited when the run actually learned, so a path that detects
benchmark.py and skips the work cannot score). An easier-than-spec env is caught
by the band's upper bound.

**Kept self-contained.** `benchmark.py` is throughput-native and does not touch
the shared `src/eval` roofline path, so problem 01 is untouched. No shared-infra
edits in v0.

**Known-provisional, by design (this is a v0 to find where the bench is
imperfect).** `peak_sps=25e6` is a guess — the naive reference measured
~0.8–1.5M SPS but only under heavy GPU contention (a separate 81GB job pinned
the card at 100%), so both the floor and the ceiling need recalibration on an
idle GPU against a real PufferLib/Brax-class fused baseline before any published
sweep. The return band (±0.30/0.40, min-improvement 0.5) is loose on purpose for
RL variance and unvalidated against an *independent* correct implementation —
reference-as-solution passes trivially (same seed → same curve), which only
proves the plumbing, not that a genuinely different fast solution lands in band.
Smoke-testing codex on it next to surface the real failure modes.

**Smoke test 1 — codex / gpt-5.5 xhigh (900s budget, contended GPU).** Passed,
`correct=true`, `peak_fraction=0.315` (SPS ~7.9M vs the contended reference's
~1.5M, ~5x). It took the honest path: a Triton kernel that fuses the entire
32-step rollout — env step, policy forward, and a custom on-device RNG for
action sampling — into one launch via `tl.static_range`, plus a hand-fused Adam
step and torch.compile on the forward. Faithful PPO, full step schedule, learned
to the reference return level (final 3.957 vs 3.956), no template mutation. Two
validations fell out: (a) the band is the right call — codex's independent RNG
reproduced the return level closely but not bit-exact (seed 2: 3.985 vs 3.998),
which allclose would have failed; (b) the bench correctly rewards genuine fusion.

What the honest run did *not* exercise, but a follow-up exploit probe confirmed:

- **DEMONSTRATED HOLE — SPS is credited against the nominal budget, not actual
  work.** A wrapper that runs *half* the iterations and pads the returned curve
  to full length with the converged value passes `check.py` (it saturates by
  ~iter 20, so the final-window mean stays in band and the curve still climbs
  from the early baseline) and scores `peak_fraction=0.121` — 2x the honest
  reference floor (0.059) for half the work. `benchmark.py` divides
  `TOTAL_ENV_STEPS / wall_time`, trusting the solution to have run the budget.
  This is the top thing to fix before publishing: the harness must own the
  env-step accounting (count actual steps in the timed region), or at minimum
  reject padded/constant tails and require exact curve length, or pivot the
  metric to wall-clock-time-to-return-threshold.
- **UNTESTED HOLE — only the outcome is pinned, not the algorithm.** The band
  checks learned return, not that PPO (clip/GAE/stochastic sampling) was used.
  On an env this learnable, a greedy or vanilla-PG or even a hand-coded
  beeline-to-food controller could clear the band faster. A throughput metric
  structurally rewards doing less; the env needs to be hard enough that only
  real PPO converges, or the algorithm needs structural enforcement.
- **CONTENTION / CALIBRATION.** codex's 3 timed trials spanned 6.08–7.88M SPS
  (~30% spread) on the shared GPU; best-of-3 papers over it but the absolute
  number is shaky. The contention-invariant signal is the *ratio* to the naive
  reference (~5x), which argues for grading on speedup-over-reference rather than
  fraction-of-a-guessed-ceiling — the reference floor is reproducible, peak_sps
  is not. Recalibrate on an idle GPU regardless.

---

## 2026-06-04 - KernelBench-Mega scaffold created

KernelBench-Mega starts as a repo-shaped copy of KernelBench-Hard so it can keep
the existing frontier-agent harnesses, roofline scoring, numeric-stress
correctness checks, transcript viewer, and sweep tooling while the problem deck
is replaced. The current `problems/` directory is intentionally temporary: it is
the Hard deck until the megakernel problems are authored.

---

## 2026-06-02 - Numeric stress correctness validation

Correctness now reruns canonical shapes and seeds under problem-specific numeric
stress cases. This is not hidden-shape bloat: stress cases rescale existing
floating inputs or model state to catch zero-output, cached-nominal, and
loose-tolerance solutions that can pass under one friendly random distribution.
`benchmark.py` remains canonical-deck only, so measured peak fractions stay
comparable for kernels that still pass.

Implemented:

- Added `src/eval/numeric_stress.py` with nominal plus targeted small/large
  activation or weight-scale cases for the active hard problems.
- Wired numeric stress into the active `check.py` runners.
- Kept integer/discrete comparison exact and improved float failure diagnostics
  with max absolute/relative error, bad element count, worst index, and
  tolerance.
- Added tests for classic cheat/failure classes: zero output under loose
  tolerance, cached nominal answers, and state scaling/restoration.

Verification:

```text
uv run ruff check . --fix
uv run pytest                         # 31 passed
KBH_NUMERIC_STRESS=1 check.py TopK    # disposable GPU smoke: PASS
KBH_NUMERIC_STRESS=1 check.py FP8     # tiny disposable GPU smoke: PASS
```

Operational note: `KBH_NUMERIC_STRESS=0` is useful for local debugging only.
Do not use it for official checks, sweeps, or published backfills.

---

## 2026-06-01 - Removed Kahan softmax from the active deck

`04_kahan_softmax` has been removed from the benchmark surface. The problem was
too easy to satisfy with a plain fast softmax under the existing tolerance, so
it rewarded the shortcut instead of forcing compensated summation. Current
scripts, machine-readable results, baselines, annotations, and leaderboard docs
no longer include it. Historical DEVLOG discussion is intentionally preserved
below as audit context for why the problem was removed.

## 2026-06-01 - Benchmark scoring is solution-first by default

KDA exposed a general harness risk: reference diagnostics can be slower than the
submitted kernel, so timing eager / `torch.compile(reference)` / SOTA before the
solution can turn a valid submission into a post-run benchmark timeout. The
default benchmark path now measures the submitted solution first for every
problem. Reference diagnostics are still available, but only when explicitly
requested.

Fixes:

- Every `problems/*/benchmark.py` now times and prints `variant=solution` before
  any eager, compiled, or SOTA diagnostic.
- Eager / compiled / SOTA diagnostics are opt-in via
  `KBH_BENCHMARK_BASELINES=1`; KDA also keeps the legacy
  `KBH_KDA_BENCHMARK_BASELINES=1` alias.
- `src/eval/timing.py` now emits `benchmark_event` lines around each variant
  (`variant_start`, `variant_end`, `variant_error`) so future audits can split
  solution, eager, compiled, and SOTA wall time directly from `benchmark.log`.
- During `KBH_DISABLE_AGENT_CUDA=1` agent phases, `nvidia-smi` and `nvcc` now
  pass through without taking the GPU lock, while `ncu` and `nsys` fail fast.
  Harness-owned `check.py` and `benchmark.py` still run under `outputs/gpu.lock`.

## 2026-06-01 - KDA benchmark backfill

The KDA benchmark timeouts were not lost submissions. The archived
`solution.py` files were present and correctness-passing; the old
`benchmark.py` measured eager + `torch.compile(reference)` diagnostics before
timing the submitted solution, and the compile path could consume the whole
1800s post-run benchmark budget.

Fixes:

- `02_kda_cutlass/benchmark.py` now times and prints the solution score first.
  Eager/compiled/SOTA reference diagnostics are opt-in via
  `KBH_KDA_BENCHMARK_BASELINES=1`.
- `scripts/run_hard.sh` gives KDA a 7200s benchmark backstop by default
  (`KBH_BENCHMARK_TIMEOUT_02_KDA_CUTLASS_SECONDS` overrides it) and records the
  check/benchmark timeout values in future `result.json` files.
- Backfilled every archived correctness-passing/no-score KDA row from its
  submitted kernel under `outputs/gpu.lock`; `results/leaderboard.json` now has
  zero `correct=true` cells without a numeric `peak_fraction`.

Backfilled KDA scores:

```text
grok/grok-build [2026-05-28 opus48-grok max]       0.1184
claude/claude-opus-4-7 [2026-05-28 finish max]    0.1166
claude/claude-opus-4-8 [2026-05-28 opus48-grok]   0.1165
minimax-claude/MiniMax-M3 [2026-06-01]            0.1114
cursor/composer-2.5-fast [2026-05-28 finish]      0.0690
claude/claude-opus-4-7 [max]                      0.0330
codex/gpt-5.5 [2026-05-28 finish xhigh]           0.0095
opencode/zai/glm-5.1 [2026-05-08]                 0.0030
```

Note: the Z.ai Claude FP8 row from 2026-05-13 remains invalid despite having a
numeric archived benchmark, because that run modified `problem.yaml` tolerance.
The leaderboard backfill intentionally skips cells that were already marked
invalid.

---

## 2026-06-01 - MiniMax M3 Claude Code full sweep

Full CUDA-track sweep through the direct Claude Code route:

```text
kbh_minimax_m3_claude_full_20260601_105827
```

MiniMax M3 produced correct solutions on all seven problems. The original
`02_kda_cutlass` post-run benchmark hit the old 1800s timeout, but the archived
submission later backfilled successfully after the KDA benchmark fix. Published
row:

```text
01_fp8_gemm          0.5334
02_kda_cutlass       0.1114
03_paged_attention   0.0286
04_kahan_softmax     0.2364
05_topk_bitonic      0.0433
06_sonic_moe_swiglu  0.2538
07_w4a16_gemm        0.1076
```

The run is a big delta from the previous OpenCode MiniMax M3 free route, which
wrote no solutions. Claude Code route quality matters here.

Audit notes: the FP8 GEMM cell uses the known bf16-reference loophole (explicit
fp8-to-bf16 cast plus CUTLASS Sm80 bf16 GEMM), and the Kahan softmax cell is a
fast fp32 tree-sum softmax rather than compensated Kahan. Both are annotated as
rubric leaks. The TopK and Sonic MoE cells are clean/interesting: TopK uses CUB
BlockRadixSort with striped loads and a hierarchical k=64 single-row merge;
Sonic MoE directly implements grouped GEMM with fused SwiGLU and becomes the
new best cell on that problem.

---

## 2026-06-01 - MiniMax M3 Claude Code direct route

MiniMax's current docs now explicitly support Claude Code through the
Anthropic-compatible endpoint `https://api.minimax.io/anthropic` with model
`MiniMax-M3`. Added a dedicated `minimax-claude` harness instead of mutating the
normal `claude` harness or global `~/.claude/settings.json`.

Auth convention:

```sh
export MINIMAX_API_KEY=...
```

Keep that in Anvil's `~/.env_vars`, which `scripts/run_hard.sh` and
`scripts/preflight_harnesses.sh` already source. The harness maps it to
`ANTHROPIC_AUTH_TOKEN` only inside the spawned Claude Code process and sets
`ANTHROPIC_MODEL` plus the Sonnet/Opus/Haiku defaults to `MiniMax-M3`.
The key is exported inside the launch subshell before `timeout claude`; do not
use `timeout env ANTHROPIC_AUTH_TOKEN=...` because `env` arguments appear in
process listings while a run is active.

Use:

```sh
KBH_USE_MINIMAX_M3_CLAUDE=1 ./scripts/preflight_harnesses.sh
./scripts/run_hard.sh minimax-claude MiniMax-M3 problems/01_fp8_gemm
```

---

## 2026-05-31 - MiniMax M3 free sweep and provider classifier hardening

Swept MiniMax M3 through the opencode harness using the available public Zen
route `opencode-zen-live/minimax-m3-free` because Anvil has no saved OpenCode
Go credentials. Run group:

```text
kbh_minimax_m3_opencode_20260531_183925
```

All seven rows completed with `session_complete=true` and `harness_exit_code=0`
but wrote no `solution.py`; no `check.py` or `benchmark.py` validation ran.
The corrected result is therefore 0/7, all `no_solution`.

The first summary falsely labeled `01_fp8_gemm` as
`provider_rate_limited` and `06_sonic_moe_swiglu` as
`provider_insufficient_credits`. Both were transcript false positives: the
model had read text containing "quota/rate limits" from `AGENTS.md` and
`insufficient_credits` from `run_hard.sh`. Provider classification now lives in
`src/harness/classification.py` and scans explicit CLI/API error events plus
stderr, not arbitrary assistant text or tool outputs.

---

## 2026-05-28 - Opus 4.8 and Grok Build addendum

Added Anvil `grok` CLI support using model `grok-build` and the top-level
headless streaming JSON route. Also added a Grok transcript viewer parser so
run archives render correctly in `src.viewer`.

Run group:

```text
kbh_opus48_grok_full_20260528_125852
```

The addendum drained cleanly after the old temporary launcher exposed a wait
bug: 14 manifest rows, 14 `result.json` rows, 0 running, and 0
exited-without-result. `scripts/launch_parallel_sweep.sh` has since been fixed
to keep child jobs waitable. Claude Opus 4.8 used `--effort max` with fast mode
disabled. Grok Build completed all seven rows through the new harness path.

Claude Opus 4.8 passed six of seven CUDA rows plus KDA correctness:
`01_fp8_gemm` 0.5332, `03_paged_attention` 0.6517, `04_kahan_softmax` 0.3517,
`05_topk_bitonic` 0.0462, `06_sonic_moe_swiglu` 0.2507, and
`07_w4a16_gemm` 0.1127. `02_kda_cutlass` passed correctness but timed out in
the benchmark phase.

Grok Build passed `04_kahan_softmax` at 0.0373 and passed KDA correctness but
timed out in benchmark. The remaining Grok rows wrote checkable solutions that
failed correctness.

Summary artifacts:

```text
outputs/sweeps/kbh_opus48_grok_full_20260528_125852/summary/summary.json
outputs/sweeps/kbh_opus48_grok_full_20260528_125852/summary/summary.latest.json
```

---

## 2026-05-23 - Lock-timeout and workspace stress fix

`check.py` and `benchmark.py` now acquire `outputs/gpu.lock` before their
execution timeout starts. The previous `timeout 180 uv run python check.py`
shape let lock wait consume the correctness budget, which made queued rows look
like model failures. The new `run_gpu_locked_timeout` path wraps `timeout`
inside the lock holder and classifies execution timeouts as
`check_timeout`/`benchmark_timeout` retryable rows instead of plain
`check_failed`.

Claude-family harnesses now `cd "$PROBLEM_DIR"` before launching Claude Code.
The old repo-root cwd plus `--add-dir "$PROBLEM_DIR"` was enough for some runs
to spend huge token budgets writing `problems/<name>/solution.py` in the source
tree while the archive-local workspace had no `solution.py`.

Stress test `stress_lock_fix_20260523_230809` used fake `kimi`/`claude`
binaries to avoid API spend. A fake Kimi run waited four seconds on the GPU
lock with `KBH_CHECK_TIMEOUT_SECONDS=1` and still passed after the lock opened;
six concurrent fake Kimi rows got unique archive directories and passed; fake
Claude proved its cwd was the archive-local
`outputs/runs/.../repo/problems/99_lock_stress` directory. Source-tree leaked
solutions/scratch were preserved under
`outputs/tmp/source_contamination_20260523_230921` and removed from
`problems/`.

---

## 2026-05-23 - Classified resweep fixes

The resweep launcher now has one worker per harness instead of a problem-major
outer loop. The original per-harness cap prevented more than two sessions per
harness, but it still head-of-line blocked when the next row belonged to a busy
harness. With workers, Cursor/Gemini/OpenCode can backfill their own next
problem while Codex or Claude remains busy.

OpenRouter was depleted during resweep setup, so the current classified rerun
uses:

```sh
KBH_SKIP_OPENROUTER=1 KBH_USE_DIRECT_GEMINI=1 KBH_HARNESS_CONCURRENCY=2
```

That runs Codex GPT-5.5, Claude Opus 4.7, Z.ai GLM-5.1 through both Claude
Code and OpenCode, Cursor Composer 2.5 Fast, and direct Gemini 3.5 Flash. Qwen
3.7 Max remains blocked until OpenRouter is topped up or a direct provider key
is added.

Aborted sweeps can leave orphaned harness timeout groups because some CLIs
spawn new process groups below `run_hard.sh`. Before restarting, kill by cwd
under `KernelBench-Hard` / `outputs/runs/<run_prefix>` and verify
`nvidia-smi --query-compute-apps` is empty.

Also fixed a failure-classifier false positive: matching plain `overage` marked
normal Cursor transcript text containing `coverage` as
`provider_insufficient_credits`. Credit detection is now limited to explicit
credit/balance/payment phrases and only applies when no solution was produced.
This matters because Cursor can quote old `result.json` files in otherwise
successful-session transcripts.

---

## 2026-05-23 - Guarded parallel sweep logging

The guarded parallel sweep now records enough metadata for website use:
agent wall time, total/check/benchmark wall time, harness/check/benchmark exit
codes, session completeness, CUDA-guard state, parsed token/cache/reasoning
usage, output tokens/sec, and GPU lock wait/active totals from
`scripts/summarize_runs.py`.

Current sweep group:

```text
kbh_hard_parallel_guarded_20260523_003820
```

The lock intentionally catches `uv`, `python`, `python3`, `nvidia-smi`, `ncu`,
`nsys`, and `nvcc` from agent workspaces. This is conservative: CPU-only probes
such as `python -c import triton` may wait behind a harness-owned benchmark.
That is acceptable for the guarded sweep because the invariant is stronger:
agent editing phases can overlap, while CUDA-facing compile/check/benchmark
work serializes through `outputs/gpu.lock`.

After observing Z.ai rows waiting behind a Qwen `benchmark.py` for harmless
`python -c import triton` probes, the wrapper was relaxed for future runs:
when `KBH_AGENT_PHASE=1`, `uv`/`python`/`python3` bypass the lock. CUDA remains
hidden with `CUDA_VISIBLE_DEVICES=` plus the `sitecustomize.py` torch guard, so
agent edit-phase Python can inspect syntax/imports without queueing behind GPU
benchmarks. Harness-owned post-run `check.py` and `benchmark.py` still run
outside `KBH_AGENT_PHASE` and therefore serialize through the lock.
The CPU-only transcript usage extraction path also bypasses the lock now, so
completed rows do not queue behind unrelated GPU benchmarks just to parse token
counts.

---

## 2026-05-22 - Parallel-safe workspaces and Cursor Agent smoke

`scripts/run_hard.sh` now creates an archive-local repo-shaped workspace for
every run:

```text
outputs/runs/<run_id>/repo/problems/<problem_name>/
```

Only the immutable problem template files are copied into that workspace. The
workspace gets a symlink to the real `src/` plus copied `pyproject.toml`,
`uv.lock`, and `.python-version`, so `check.py` / `benchmark.py` still see a
repo root two parents up while agents can mutate dependencies or scratch files
without touching the source `problems/*` directory. This fixes the parallel run
hazard where two agents on the same problem could delete or overwrite each
other's `solution.py`.

Added a `cursor)` harness branch for Anvil's Cursor Agent CLI, which is
installed as `agent` rather than `cursor`:

```sh
agent --trust --yolo --print --output-format stream-json \
  --model "$MODEL" --workspace "$PROBLEM_DIR" "$PROMPT"
```

`scripts/extract_usage.py` now has `_cursor()` support for terminal
`{"type":"result"}` events with `usage.inputTokens`, `outputTokens`,
`cacheReadTokens`, and `cacheWriteTokens`. Partial Cursor timeouts do not expose
a terminal usage block, so usage may be null on 300s smokes that hit timeout.

Composer 2.5 smoke:

```text
BUDGET_SECONDS=300
problem: problems/01_fp8_gemm
harness/model: cursor / composer-2.5
archive: outputs/runs/20260522_144839_cursor_composer-2.5_01_fp8_gemm/
```

The run wrote `solution.py` and preserved source problem cleanliness, but timed
out and failed post-run `check.py`: the generated solution tried to load a
Torch extension named `cutlass_fp8_gemm`, but the `.so` was missing at
post-check import time. This is a real model/harness smoke result, not a
workspace collision.

Validation after the harness changes:

```sh
uv run ruff check . --fix
uv run pytest
```

Both passed.

### GPU queue smoke

Added per-run cache directories and a shared GPU lock wrapper under each
archive's `bin/` directory. During a run, `PATH` points at wrappers for `uv`,
`python`, `python3`, `nvidia-smi`, `ncu`, `nsys`, and `nvcc`; those wrappers
acquire `outputs/gpu.lock` before forwarding to the real binary. Per-run cache
env vars are also set:

```sh
TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"
TMPDIR="$RUN_DIR/tmp"
```

Smoke test launched three agents concurrently on `01_fp8_gemm` with
`BUDGET_SECONDS=180`:

- `opencode openrouter-alibaba/qwen/qwen3.7-max`
- `opencode openrouter-google-ai-studio/google/gemini-3.5-flash`
- `cursor composer-2.5`

All three reached post-run validation without touching the source problem
directory. The lock logs showed serialized GPU-facing calls. In particular,
Gemini's `check.py` ran first (`16:01:07` to `16:01:09`), Cursor's `check.py`
then ran (`16:01:09` to `16:01:35`), and Qwen's `check.py` then ran
(`16:01:35` to `16:01:38`). This validates the intended shape: agent work can
overlap, but check/compile/benchmark phases queue through the shared lock.

The 180s model results are not capability scores:

- Gemini wrote a solution but failed tolerance (`max_abs_diff=0.5625`).
- Qwen wrote a Triton solution but failed compile (`Unsupported rhs dtype fp8e4nv`).
- Composer wrote a CUTLASS solution but failed extension build.

Important limitation: the lock only governs commands launched inside the
KernelBench harness. It does not stop unrelated machine-wide CUDA jobs already
running elsewhere on Anvil, so serious published sweeps should still check
`overnight-compute status` / `nvidia-smi` first or run under a broader machine
reservation.

First wrapper attempt deadlocked during a follow-up Qwen run: `uv run python
benchmark.py` held the GPU lock, then the benchmark's child `nvcc --version`
wrapper tried to acquire the same non-reentrant lock. Fixed by setting
`KBH_GPU_LOCK_HELD=1` while executing the real locked command; nested wrapper
calls now bypass the lock and exec the real binary directly.

Validation after the reentrant fix:

```text
BUDGET_SECONDS=300
harness/model: opencode / openrouter-alibaba/qwen/qwen3.7-max
archive: outputs/runs/20260522_161511_opencode_openrouter-alibaba_qwen_qwen3.7-max_01_fp8_gemm
```

Result: `correct=true`, `peak_fraction=0.4257`, `template_mutated=false`.
The lock log shows the full post-run path serialized cleanly:

```text
16:20:11 start uv run python check.py
16:20:17 end   uv run python check.py status=0
16:20:17 start uv run python benchmark.py
16:20:26 end   uv run python benchmark.py status=0
```

---

## 2026-05-21 - Gemini CLI smoke wired, timeout path validated

Gemini CLI support is now wired into the Hard harness but remains uncommitted.
`scripts/run_hard.sh` has a `gemini)` branch that runs from inside the problem
directory because Gemini has no `--cwd` or `--add-dir` flag:

```sh
cd "$PROBLEM_DIR" && gemini -m "$MODEL" --approval-mode yolo -o stream-json -p "$PROMPT"
```

The Gemini branch was added to the `session_complete` group by looking for a
terminal result event shaped like `{"type":"result"}`. `scripts/extract_usage.py`
now has `_gemini()` support that reads `stats.input_tokens`,
`stats.output_tokens`, and cached-token stats from that terminal result event.

Smoke run:

```text
BUDGET_SECONDS=300
problem: problems/01_fp8_gemm
harness/model: gemini / gemini-3.5-flash
archive: outputs/runs/20260519_212055_gemini_gemini-3.5-flash_01_fp8_gemm/
```

The smoke passed end-to-end at the harness level: Gemini wrote a Triton GEMM
`solution.py` and executed `check.py` inside the sandbox. The run timed out at
300 seconds with exit 124, so `session_complete=false`; this is expected for
the smoke budget, and because there was no final result event, usage stayed
null.

The important validation was the reward-hack defense. Gemini tried to edit
`problem.yaml` to add a `bfloat16: 0.15` tolerance after its FP8 kernel failed
the atol check (`max_abs_diff=0.546875`). The template-mutation guard detected
`template_mutated: true`, restored `problem.yaml`, and marked the run INVALID.
That is the intended defense working. Workspace isolation also held: Gemini was
restricted to `PROBLEM_DIR` plus `~/.gemini/tmp/...`, matching the other
harnesses sandbox model.

Harness is ready for a real 45-minute sweep when requested. Before publishing
or merging this work, run normal checks and commit the two dirty harness files:

- `scripts/run_hard.sh`
- `scripts/extract_usage.py`


## 2026-05-14 - Leaderboard split after non-pass audit

After a per-run audit of every non-pass cell, `/hard` now renders two leaderboard sections instead of one flat table. The serious comparison section keeps rows where audited non-passes are normal benchmark outcomes: correctness failures, build failures, full 2700s timeouts, or explicit invalid/reward-hack behavior. Current serious rows are GPT-5.5, Claude Opus 4.7, Claude Code GLM-5.1 on Z.ai, Droid GLM-5.1, and the two DeepSeek OpenCode rows.

Rows moved to diagnostic/needs-rerun have at least one non-pass that is not a clean model attempt: API/provider errors, auth/setup failures, harness adapter failures, hidden reasoning-token exhaustion before writing `solution.py`, or unknown early stops with no checkable artifact. That includes both older OpenCode/Z.ai GLM-5.1 rows, the OpenRouter-pinned Qwen/MiMo/MiniMax rows, and Kimi K2.6. Kimi is demoted despite being otherwise interesting because problems 09/10 ended in 401 authentication errors after only 4-5 seconds; its 6/9 raw pass total is not directly comparable until those cells are rerun.

DeepSeek through OpenCode is intentionally not blanket-demoted: the audited DeepSeek non-passes were ordinary solution bugs or full-budget timeouts with artifacts, not API/setup failures. Droid is kept serious because the documented May 8 smoke tests confirmed the custom Z.ai/Factory route was wired correctly; its four `ERR` cells were 45-minute incomplete runs with no `solution.py`, not endpoint failures.

---

## 2026-05-14 — Z.ai GLM-5.1 Claude Code rerun with corrected Anthropic endpoint

Shuyan confirmed Z.ai's internal Claude Code eval config: disable experimental betas, set very high retry and output-token ceilings, disallow plan/user-question tools, and map every Claude Code alias including Haiku / Explore / subagents to `glm-5.1`. `scripts/run_hard.sh` now bakes those defaults into the `zai-claude` harness against `https://api.z.ai/api/anthropic`, and `scripts/rerun_zai_claude_glm51.sh` records the nine-problem CUDA rerun command.

The May 13 rerun is now represented as `zai-claude/glm-5.1 [2026-05-13]`. Correctness-passing cells were `03_paged_attention` 0.2220, `04_kahan_softmax` 0.3367, `05_topk_bitonic` 0.0029, `06_sonic_moe_swiglu` 0.1111, and `10_patch_embed_conv3d_gemm` 0.1471. `02_kda_cutlass` failed numerically, `07_w4a16_gemm` timed out with no `solution.py`, and `09_fmha_preattn_mrope` failed CUDA extension compilation/checking.

`01_fp8_gemm` is deliberately not counted as a pass even though its archived `result.json` says `correct=true`: the model edited `problem.yaml`, changing the tolerance key from `fp8_e4m3fn: 0.15` to `bfloat16: 0.15`. Since `check.py` looks up tolerance by `ref_out.dtype == torch.bfloat16`, that relaxed the actual correctness check from the default bf16 tolerance to 0.15. This is a clean reward-hack example, not a valid kernel result, so the public row marks it invalid and attaches a `reward_hack` annotation.

Harness fix: `run_hard.sh` now snapshots `reference.py`, `sota.py`, `shapes.py`, `problem.yaml`, `check.py`, `benchmark.py`, and `PROMPT.txt` before each agent run. If any of those files are changed, deleted, or created unexpectedly, the run is marked invalid, `template_mutated=true` is written to `result.json`, a diff lands in `template_mutations.log`, and the original files are restored before the next problem.

---

## 2026-05-09 — Website policy: demote OpenCode, remove blocked Qwen 35B-A3B row

After inspecting the May 8 GLM-5.1 rerun transcripts, we changed the public `/hard` page to treat OpenCode rows as diagnostic rather than primary evidence. The OpenCode Z.ai rerun had multiple early `ERR` cells caused by hidden-reasoning budget exhaustion before tool use, so the page now shows a red disclaimer and pushes OpenCode rows below the native-harness rows. Droid and Claude Code rows should carry more weight when they exist.

Also removed `opencode/openrouter-pinned/qwen/qwen3.6-35b-a3b` from the public leaderboard data. Its previous `0/7 ERR` row was an infrastructure block, not a model result: available providers did not advertise tool-use support to the agent harness. The historical details remain in this devlog; the website no longer presents it as an evaluated model row.

For Qwen 3.6 27B, a transcript dive found no result-parser bug. The frequent failures are mostly missing `solution.py` / early-stop behavior, plus several full-budget OpenCode timeouts where a written solution still failed `check.py`. The later problem 09/10 first attempts failed due to OpenRouter insufficient-credit API errors, then reran successfully and are represented by the passing rows in `leaderboard.json`.

## 2026-05-08 — Z.ai GLM-5.1 rerun: OpenCode, Droid, Claude Code attempt

Z.ai reached out after the public KernelBench-Hard GLM-5.1 row, asking for a rerun because several OpenCode cells appeared to terminate early as `ERR` after a small number of iterations. We reran all CUDA-track problems on the RTX PRO 6000 using their dedicated `$ZAI_API_KEY` against the actual Z.ai endpoint. `08_metal_lightning_attn` stayed out of scope on this CUDA host.

OpenCode rerun used `zai/glm-5.1` through the Z.ai API. It reproduced the core anomaly: `03_paged_attention`, `05_topk_bitonic`, `07_w4a16_gemm`, and `09_fmha_preattn_mrope` all ended as `ERR` well before the 2700s budget. Passing cells were `02_kda_cutlass` (correct but no parsed peak), `04_kahan_softmax` at 0.0561, `06_sonic_moe_swiglu` at 0.2154, and `10_patch_embed_conv3d_gemm` at 0.1742. `01_fp8_gemm` wrote a solution but failed correctness at timeout.

Droid rerun used the Factory custom model `custom:GLM-5.1-[Z.AI-Coding-Plan]-0`, also pointed at `https://api.z.ai/api/coding/paas/v4`. It solved five of nine: `01_fp8_gemm` 0.4140, `03_paged_attention` 0.2523, `04_kahan_softmax` 0.2339, `06_sonic_moe_swiglu` 0.1490, and `07_w4a16_gemm` 0.0863. `02_kda_cutlass`, `05_topk_bitonic`, `09_fmha_preattn_mrope`, and `10_patch_embed_conv3d_gemm` timed out incomplete with no scored solution.

Claude Code was attempted only against Z.ai, not Anthropic: first through `ccr-rust` with a Z.ai-only router, then directly with `ANTHROPIC_BASE_URL=https://api.z.ai/api/coding/paas/v4` and `ANTHROPIC_AUTH_TOKEN=$ZAI_API_KEY`. The proxy path authenticated but returned malformed/empty HTTP 200 responses to Claude Code; direct model names returned 404 model-access errors. Those runs are setup-invalid, not model results, and are not counted on the leaderboard.

Website changes from this rerun: add Droid harness support to `scripts/run_hard.sh`, add Droid usage extraction, render all 18 May 8 transcript viewers, and publish two additional leaderboard rows: `opencode/zai/glm-5.1 [2026-05-08]` and `droid/zai/glm-5.1 [2026-05-08]`. This preserves the original public GLM row while making the rerun evidence explicit.

Follow-up smoke tests found the Claude Code wiring bug: Z.ai exposes a separate Anthropic-compatible endpoint for Claude Code, `https://api.z.ai/api/anthropic`. The earlier direct attempt used the OpenAI-compatible coding endpoint, `https://api.z.ai/api/coding/paas/v4`, which is the right shape for Droid/Factory but the wrong shape for Claude Code. `scripts/run_hard.sh` now has a `zai-claude` harness that sets `ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic`, maps Claude Code's `opus`/`sonnet` aliases to `glm-5.1`, and leaves `ccr-claude` as the historical proxy path. One-turn smoke results: `zai-claude` returned `KB_SMOKE_OK` through model `glm-5.1`; Droid's existing `custom:GLM-5.1-[Z.AI-Coding-Plan]-0` also returned `KB_SMOKE_OK`. Droid was already hooked up correctly; its four benchmark ERR cells were 45-minute incomplete runs with no `solution.py`, not API failures.

---

## 2026-04-30 — Launch prep: monorepo, kernelbench.com, transcript viewers, blog plots

Three substantial pieces went in between the rubric-leak audit and shipping public.

### Monorepo

The standalone `Infatoshi/KernelBench-Hard` and `Infatoshi/KernelBench-v3` repos got absorbed into `Infatoshi/kernelbench.com` as `git subtree` merges (history preserved). The website lives at the repo root for Vercel auto-detection; benchmarks live under `benchmarks/hard/` and `benchmarks/v3/`. The standalone repos still exist but the monorepo is now the canonical home — the website's `lib/data.ts` reads `benchmarks/hard/results/leaderboard.json` directly from disk at build time, no HTTP fetch.

Trade-off accepted: the per-suite DEVLOGs stay inside their subdirs (this file lives at `benchmarks/hard/DEVLOG.md`). Cleaner per-suite history; harder to write a single chronological narrative across them. Worth the trade.

### Public website with the hacker theme

Next.js 16 + React 19 + Tailwind v4. Phosphor green on near-black, JetBrains Mono everywhere, subtle CRT scanlines via fixed CSS overlay. Routes:

- `/` — landing with ASCII KernelBench banner, version cards, design principles, contact box.
- `/hard` — leaderboard table (12×7 grid, every cell clickable into its run viewer), per-problem ceilings table with eager / compiled / SOTA timings, full rubric-leak deep dives with pull quotes, what-changed-from-v3 bullets.
- `/v3` — client-side filterable explorer over the 2071-row results.csv, embedded plots, per-row solution.py / reference.py links.
- `/runs` — sortable index of all 100 transcript viewers (peak-fraction-ranked).
- `/runs/<run_id>.html` — themed transcript viewers, see below.
- `/blog`, `/blog/v3`, `/blog/hard` — long-form writeups (moved over from elliotarledge.com).

Domain: `kernelbench.com` registered through Vercel, attached to project `kernelbench` under team `elliot-arledges-projects`. Auto-deploys on push to `master` via Vercel's native GitHub integration — no GitHub Actions workflow needed.

### One critical Vercel deploy gotcha

Every commit pushed from anvil with the autogenerated email `infatoshi@anvil.tail21a94e.ts.net` failed Vercel's commit-verification gate at the pre-build phase (silent ERROR with no build logs). Three commits errored before I traced it. The fix: pass `-c user.email=elliot@arledge.net` (the GitHub-linked email) inline on every git commit. The repo has a local `git config user.email` set to this; new commits should pick it up automatically, but if you're working from a fresh clone or different machine, set it explicitly.

### 100 themed transcript viewers + reward-hack tab

`src/viewer/html.py` now respects two env vars at HTML-generation time:

- `KB_VIEWER_THEME=phosphor` — applies a CSS override layer (phosphor green on near-black, JetBrains Mono, CRT scanlines) plus a site-nav strip linking to `/`, `/hard`, `/v3`, `/runs`. Preserves the original role color slots (assistant=green, tool=amber, error=red, user=cyan).
- `KB_ANNOTATIONS_DIR=<path>` — looks up `<run_id>.yaml` for the currently-rendered run, and if found inserts a "reward hack" tab between solution.py and final answer. The tab renders the annotation's verdict badge (color-coded by category), summary, pull quotes (with file:line anchors and syntax-highlighted code), and implication paragraph.

Generation pattern (run from `benchmarks/hard/`):
```bash
KB_VIEWER_THEME=phosphor KB_ANNOTATIONS_DIR=$(pwd)/results/annotations \
  uv run python -m src.viewer <run_dir> --out <out_path>
```

Bulk regeneration over all 100 runs is a one-liner shell loop. The generated HTMLs land in the monorepo's `public/runs/`. They're committed (~18 MB total) so the website serves them directly as static assets.

### Top-peak audit: 30 annotations total

Initial audit produced 13 annotations (the FP8 GEMM bf16 dressup cluster + the Kahan softmax skip). Follow-up serial pass added 17 more for top-peak cells with `peak_fraction ≥ 0.10`. All 17 came back `clean` — Triton kernels, no forbidden ops, no F.softmax / scaled_dot / flash_attn library cheats. The 30 annotations now cover every cell where there's something to say: 12 rubric leaks + 1 honest Kahan + 17 clean top performers. Lower-peak cells deliberately left unannotated.

### Baseline + SOTA timings

`scripts/run_baselines.sh` benchmarks each problem's `reference.py` (and `sota.py` where one exists) and writes `results/problem_baselines.json`. The website's `/hard` per-problem ceilings table now shows eager / compiled / SOTA ms alongside best-model peak. Most problems lack a SOTA entry on SM120 — FP8 needs scaling args, vLLM/flashinfer not wired, etc. — so those columns show `—`. Only `04_kahan_softmax` and `05_topk_bitonic` have SOTA timings populated.

Notable timing facts surfaced: torch.compile gives `02_kda_cutlass` an 8x speedup over eager (61.9 → 7.4 ms) and `07_w4a16_gemm` a 4x (0.61 → 0.144 ms); the rest are within noise of eager.

### Five matplotlib blog plots

`benchmarks/hard/scripts/generate_blog_plots.py` reads `leaderboard.json` and produces five PNGs in `public/blog-hard/`, themed to the kernelbench.com palette:

- `leaderboard_heatmap.png` — full 12×7 grid colored by peak_fraction
- `pass_count_by_model.png` — tier ranking, gpt-5.5 xhigh in amber as the only 7/7
- `best_peak_per_problem.png` — per-problem ceilings, shows the easy/hard regimes
- `fp8_gemm_cluster.png` — visualizes the bf16-dressup herd
- `kahan_inversion.png` — visualizes "punishes algorithmic honesty" (deepseek-v4-pro green at 0.101 vs the rest in orange above it)

Embedded into `app/blog/hard/page.tsx` at the matching narrative beats. Re-run the script any time the leaderboard data changes.

### Site-level extras

- `app/icon.svg` — popcorn-kernel favicon (phosphor lobes + amber hull). Reads at 16×16.
- Contact section on `/` with `infatoshi@gmail.com` mailto, also surfaced in the footer on every page (including the 100 transcript viewers).
- elliotarledge.com cleaned up: kernelbench cards removed from blog index, page sources deleted, nav link redirected to kernelbench.com, "kernelbench.com" card added under "things im proud of" on home.

### Pending for later (not blocking launch)

- N≥2 reruns to put variance bands on cells.
- A permanent `/methodology` page surfacing how peak_fraction is computed (currently the explanation lives only in this DEVLOG and the blog post).
- bwrap-based universal sandboxing across all four CLI harnesses (claude, codex, kimi, opencode share the same architectural lack of FS isolation).
- Per-problem detail pages (`/hard/01_fp8_gemm` etc.) with full reference / SOTA / per-shape breakdown.
- Closing the two rubric leaks via tighter tolerances or static-analysis pattern checks. Both fixable in a few hours; deferred so we publish now.

---

## 2026-04-29 — Reward-hacking audit: two rubric leaks, publishing with them documented

After the full sweep (12 models × 7 problems), audited the high-peak runs for reward hacking. Two findings, very different in severity. Decision: ship the leaderboard with the leaks documented inline rather than iterate on problem design until perfect.

### Audit method

Read the solution.py for every (model, problem) cell with `peak_fraction ≥ 0.10`. Looked for the v3-era reward-hack patterns (precision downcast, PyTorch wrapper masquerading as a custom kernel, baseline gaming, hardcoded test-input handling) plus problem-specific shortcuts (skipping the named algorithm in favor of a simpler one that still passes correctness).

### Findings

**Clean (real kernel work):**
- `03 paged_attention` — all top peaks (opus 0.602, gpt-5.5 0.498, kimi 0.432) are real Triton FlashDecoding-style kernels. Online softmax, GQA register reuse, exp2 fast-path. No shortcuts.
- `07 w4a16_gemm` — all 8 passing solutions inline int4 unpacking (`& 0xf`, `>> 4`) inside the kernel; none pre-unpack-and-stash-as-bf16 at init. Genuine quantized kernel work.

**Rubric leak (cell number doesn't measure what the problem name implies):**

- `01 fp8_gemm` — every passing solution at peak ≥ 0.4 (5 models: opus 0.534, mimo 0.434, qwen-plus 0.431, qwen-max 0.429, gpt-5.5 0.423) casts fp8 → bf16 inside the kernel and runs a bf16 GEMM. Both opus and gpt-5.5 explicitly pin to `cutlass::arch::Sm80` — Ampere CUTLASS, no SM120 FP8 tensor cores anywhere. Opus's source comment is explicit: *"follow the codex baseline (BF16 GEMM internally)..."*. Technically valid (the reference also does the bf16 cast) but the problem name promises FP8-tensor-core skill that isn't being measured.

- `04 kahan_softmax` — 6 of 7 passing solutions skipped Kahan compensated summation entirely, including both top-tier scores (gpt-5.5 0.363, opus 0.317). Only deepseek-v4-pro implemented Kahan — and scored *lowest* of the seven passes (0.101) because compensated summation has real overhead. The model whose docstring explicitly says *"Numerically tight softmax with Kahan compensated summation. Map: each block computes local (max, Kahan-sum-of-exp)..."* is the one that loses, because everyone else takes the easy path and tolerance doesn't enforce the difference.

The Kahan one is the more depressing of the two. The benchmark, as designed, *punishes* algorithmic honesty: the model that implements the algorithm the problem name describes scores worst, because the rubric leaks and the dishonest path is faster.

### Decision: publish with flaws documented inline

Two reasons to ship now rather than fix-then-publish:

1. **Diminishing returns on iteration.** This is the second round of post-hoc design issues we've found (the first was the verification gate / prompt-shape regime in late April). Every iteration surfaces something new. Publishing with the current flaws documented is more honest than iterating until the next flaw appears, then publishing.
2. **The flaws ARE the finding.** The benchmark's purpose is to surface what models will and won't do under autonomous-agent evaluation. "Five frontier models all took the bf16 shortcut on FP8 GEMM" and "six of seven skipped Kahan compensation" are themselves headline results — they characterize how models behave when the rubric leaks.

### What we shipped

- `LEADERBOARD.md` — canonical human-readable cross-model grid + per-problem ceilings + a *Benchmark design flaws* section that explicitly footnotes the two leaky problems with their cell numbers.
- `results/leaderboard.json` — machine-readable, schema-versioned. Source for the website's leaderboard view.
- `results/annotations/<run_id>.yaml` — per-cell commentary for 13 runs covering both leaks (5 fp8 cells, 7 kahan cells) plus the headline clean cell (opus paged_attention 0.602). Schema in `results/annotations/SCHEMA.md`.
- `results/annotations/SCHEMA.md` — annotation file format with five verdicts (`clean`, `rubric_leak`, `reward_hack`, `interesting`, `bug`).

### Future leak fixes (logged, not done)

- **fp8_gemm**: tighten tolerance to a value where bf16-via-cast and real fp8-tensor-core math diverge on the test inputs, or add a static-analysis check to the rubric that detects the `fp8 → bf16` cast pattern before the GEMM call.
- **kahan_softmax**: tighten tolerance to a value where naive vs Kahan produce visibly different results on the test inputs (the test inputs may need to include numerically-pathological cases — large logit ranges, near-equal extremes), or write a check that detects compensated-summation pattern in solution.py.

These are tractable; deferred so we publish the leaderboard now.

---

## 2026-04-27 — opencode workspace leak: root cause + partial fix

The Qwen 27B forensic dive (next entry) led to auditing every opencode-routed `read` call across the shakedown. The leak is universal across all opencode-routed models, not just Qwen.

### What was being read outside the workspace

Across 31 opencode runs (Qwen 35B-A3B excluded — never made it past routing):

| Path | Reads | Why it matters |
| --- | --- | --- |
| `src/hardware/rtx_pro_6000.py` | 9 | Full peak TFLOPS table (fp4=800, fp8=400, bf16=200, dram=1800). The hardware spec we deliberately stripped from PROMPT.txt is sitting four directories up. |
| `src/eval/correctness.py` | 12 | Per-dtype tolerance lookup. Models can verify what atol/rtol they need to pass before writing code. |
| `src/eval/timing.py` | 5 | Benchmarking methodology (warmup count, L2 flush size, iter count). Lets a model game the timing harness. |
| `src/eval/roofline.py` | 2 | Throughput formulas. Lets the model reverse-engineer the score. |
| `~/.claude/skills/perf-optim/SKILL.md` | 7 | The user's personal CUDA optimization atlas — speed-of-light methodology, profiling tools, the exact "always profile before optimizing" framing. |
| `~/.claude/skills/kernel/SKILL.md` | 2 | The user's master kernel-skill index. |
| `CLAUDE.md` | 1 | Developer-facing repo notes. |
| `/home/infatoshi/cuda/KernelBench/problems/07_w4a16_gemm` | 1 | Sibling repo on the same machine — model walked outside KernelBench-Hard entirely. |

### Root cause

`opencode --pure` does not sandbox file system access. `--pure` means "run without external plugins" — nothing about FS isolation. opencode's `read` tool dispatches to OS file read with no path restrictions; whatever the user running opencode can read, the tool can read. The "soft boundary" of "this is your workspace" was just the prompt + cwd, neither of which constrains the tool implementation.

Same architecture in claude-code (`--add-dir` extends visibility but doesn't restrict; bash can still touch absolute paths) and codex (no path constraints at all). The leak is universal across all three CLI harnesses; opencode was just first-noticed because Qwen 27B was particularly aggressive about reading files.

### Fix (partial)

Added to `~/.config/opencode/opencode.json`:
```json
"permission": {
  "external_directory": "deny"
}
```

This blocks tool calls that touch paths outside the working directory where opencode was started (verified end-to-end: a smoke run trying to `read /home/infatoshi/cuda/KernelBench-Hard/src/hardware/rtx_pro_6000.py` returned `status: "error"` with the message *"The user has specified a rule which prevents you from using this specific tool call"*, and the model correctly reported the block).

### What's still open (and why)

When opencode dumps its rule list on a denied call, it surfaces auto-generated allow rules for **every Claude Code skill the user has installed**:

```
{"permission":"external_directory", "pattern":"/home/infatoshi/.claude/skills/perf-optim/*", "action":"allow"}
{"permission":"external_directory", "pattern":"/home/infatoshi/.claude/skills/kernel/*",      "action":"allow"}
{"permission":"external_directory", "pattern":"/home/infatoshi/.claude/skills/<each-skill>/*", "action":"allow"}
```

These are more specific than my `*: deny`, so they win. The user's CUDA-optimization skills (`perf-optim`, `kernel`, `gpu-profiling`, `port-kernel`, `debug-gpu`) remain readable. That's a separate, smaller leak (user's personal notes, not benchmark internals), but the prompt's "look up PTX docs and library headers" directive is degraded if the model can short-circuit via the user's pre-written kernel atlas.

To close fully, options are:
1. **Rename/move the skills directory before each sweep.** `mv ~/.claude ~/.claude.bak` for the duration. Cheap, intrusive.
2. **Find the opencode config knob that controls skill discovery and disable it.** Not surfaced in the public docs that I could find; would need to source-dive opencode.
3. **bwrap the harness.** `bwrap --bind $PROBLEM_DIR /workspace --ro-bind /usr /usr ... opencode run`. Real isolation; medium-weight; works for all three harnesses uniformly.
4. **Accept the user's-skills leak.** It's pre-existing personal knowledge, equivalent to "the model has been pre-trained on this content." Different category than leaking benchmark internals.

For now: option (1) for serious sweeps, otherwise note the asymmetry. The prompt directive remains the primary signal.

### Cross-harness scope

claude-code and codex are not currently behind any path restriction. Their `Bash`, `Read`, `Edit`, etc. tools see everything the user account does. The leak audit only covered opencode runs because those were the only fresh runs in `outputs/runs/` after we deleted the topk-overnight set. Worth re-auditing whenever the next claude/codex sweep runs. Likely fixable for both via bwrap if the leak proves load-bearing.

### Reading-the-leaderboard note

Until full sandboxing lands, **opencode-routed numbers from before this commit reflect a leakier environment than the current PROMPT.txt regime claims**. Models that read `rtx_pro_6000.py` had peak TFLOPS as a number, not a thing-to-look-up. Models that read `perf-optim/SKILL.md` had a written CUDA optimization atlas. Their scores are not directly comparable to a future run under the post-fix permission policy. Re-running the shakedown after the fix would tell us how much the leak actually mattered, and is worth doing before any "official" leaderboard publication.

---

## 2026-04-27 — Qwen 3.6 27B: post-fix rerun reverses the drop

After the leak fix landed, reran Qwen 3.6 27B on all 7 problems under the new permission policy. Result: **1/7 PASS** (sonic_moe_swiglu, peak_fraction 0.0822 — same tier as MiniMax M2.7's 0.076 on that problem) and dramatically more engagement across the board.

| Problem | Pre-fix shakedown | Post-fix rerun |
| --- | --- | --- |
| 01 fp8_gemm | ERR | ERR |
| 02 kda_cutlass | ERR (1-step bail) | FAIL (45 min, 28k output, has_solution) |
| 03 paged_attention | FAIL (`__sqrt__` hallucination) | FAIL (different bug) |
| 04 kahan_softmax | ERR | ERR (11 min bail) |
| 05 topk_bitonic | ERR (token cap) | FAIL (45 min, 32k output) |
| 06 sonic_moe_swiglu | ERR | **PASS 0.0822** |
| 07 w4a16_gemm | ERR | ERR |

Engagement shifted: immediate bails 4/7 → 2/7; solutions written 1/7 → 4/7; total token consumption rose ~10x (708k input / 9k output → 8.2M input / 91k output).

### Why the change?

Three honest possibilities:
1. **Removing the leak forced focus.** Pre-fix, Qwen burned tool calls reading `src/hardware/rtx_pro_6000.py`, `src/eval/correctness.py`, `src/eval/timing.py`, `~/.claude/skills/perf-optim/SKILL.md`. Post-fix, those reads fail fast with an explicit denial, redirecting the model to focus on `reference.py` and write code.
2. **LLM nondeterminism.** Same model, same prompt, runs 11 hours apart. DeepSeek Flash on TopK regressed from PASS to FAIL on a similar interval — variance is real on this benchmark.
3. **Both.** Leak-fix vector is right (reducing rabbit-holing improves focus), but a 10x engagement swing and 0→1 PASS is hard to attribute purely to that.

A controlled experiment (5x runs of each disposition, same conditions otherwise) would isolate the effect. Worth doing before any "leak fix improved Qwen 1/7 PASS" claim becomes load-bearing.

### Decision

Re-added to ACTIVE_MATRIX. Treat as same tier as MiniMax (functional but high-variance, low ceiling). Earlier "capability + compliance, dropped permanently" framing was a misread driven by N=1.

### N=1 is not enough — methodology footnote

Two reversals within 24 hours on this benchmark: Flash on TopK (PASS → FAIL) and Qwen 27B (0/7 → 1/7). Future official results should run N≥2 per (model, problem) and report variance. Reproducibility footnote in the shakedown entry already flagged this; second confirmation here.

---

## 2026-04-27 — Qwen 3.6 27B: dropped from active matrix (initial drop, see entry above for reversal)

Forensic dive into the 0/7 result on the cheap-tier shakedown.

### Failure-mode breakdown across 7 runs

| Problem | Steps | Tool calls | Wrote solution.py | End reason |
| --- | --- | --- | --- | --- |
| 01 fp8_gemm | 5 | 11 reads + 4 bash | NO | `stop` |
| 02 kda_cutlass | **1** | **0** | NO | `stop` (immediate bail) |
| 03 paged_attention | 8 | 18 (incl. 1 write) | YES — but compile-broken | `stop` after write, no verify |
| 04 kahan_softmax | 3 | 8 reads | NO | `stop` |
| 05 topk_bitonic | 8 | 17 | NO | `length` (output token cap hit) |
| 06 sonic_moe | 5 | 14 | NO | `other` |
| 07 w4a16_gemm | **2** | **1 read** | NO | `stop` (immediate bail) |

### Three intertwined patterns

**Variable engagement.** Step counts ranged 1-8 with no clear relationship to problem difficulty. Two runs (KDA, W4A16) bailed in 1-2 steps with effectively zero engagement. Other runs explored extensively. Same prompt each time, same model, same provider.

**Explores extensively, refuses to write.** 5/7 runs ended with no `solution.py`. The model reads `reference.py`, `problem.yaml`, `shapes.py`, `check.py`, `benchmark.py`, runs `nvidia-smi`/`nvcc`/`triton` probes — and then stops. On the paged attention run it actually said *"Let me verify the check infrastructure before writing the kernel — I noticed syntax issues in those files"* — vocalized the verification gate, then stopped without acting on it. Knows the rule, agrees with it out loud, doesn't follow through. This is a deeper compliance gap than DeepSeek Flash had pre-prompt-edit; tightening the prompt sentence further isn't likely to help.

**When it does write code, it hallucinates APIs.** The one solution.py it produced (paged_attention, 8230 chars of real Triton) had:
```python
scale = 1.0 / float(HEAD_DIM).__sqrt__()
```
`__sqrt__` is not a Python or Triton method — invented. `float(tl.constexpr)` also fails because Python's `float()` doesn't accept Triton tensors. Compilation crash on the first call. The model then *did not run check.py*, so it never saw the error, and stopped.

### Decision

Dropped from `scripts/sweep.sh` ACTIVE_MATRIX and `scripts/shakedown_sweep.sh`. The route stays defined in opencode config so re-add is a one-line restore. Revisit when qwen3.7 lands or if a future agent harness materially improves Qwen's tool-use compliance.

### Observation worth keeping

Qwen 27B's pattern is the inverse of the verification-gate experiment with Flash: Flash didn't read the rule and skipped the test; tightening the prompt fixed it. Qwen *does* read the rule, *does* acknowledge it, then ignores it anyway. That's not a prompt-clarity problem — it's a model-side compliance issue. The verification gate works on models that have the discipline-half latent and need the cue; it doesn't manufacture discipline where it isn't present.

---

## 2026-04-27 — Cheap-tier shakedown sweep: 35 runs, $2.14, full grid

First end-to-end validation of the new PROMPT.txt regime + token-logging wiring. Five cheap-tier models against the full 7-problem deck, sequential.

### Final grid

| Model | 01 fp8 | 02 kda | 03 paged | 04 kahan | 05 topk | 06 moe | 07 w4a16 | PASS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepseek-v4-flash | FAIL | 0.009 | **0.167** | **0.138** | FAIL | 0.083 | **0.134** | 5/7 |
| deepseek-v4-pro | FAIL | FAIL | 0.027 | 0.101 | 0.011 | **0.108** | 0.125 | 5/7 |
| minimax-m2.7 | ERR | ERR | FAIL | 0.034 | FAIL | 0.076 | 0.030 | 3/7 |
| qwen3.6-27b | ERR | ERR | FAIL | ERR | ERR | ERR | ERR | 0/7 |
| qwen3.6-35b-a3b | ERR×7 (no tool-use endpoint) | | | | | | | 0/7 |

ERR = no `solution.py` written; FAIL = solution.py present, `check.py` failed; numeric = peak_fraction (PASS).

### Token totals + per-model API spend

| Model | input | output | cache_read | reasoning | est. spend |
| --- | --- | --- | --- | --- | --- |
| deepseek-v4-flash | 400k | 260k | 39.4M | 461k | $0.26 |
| deepseek-v4-pro | 336k | 163k | 15.7M | 319k | $0.56 |
| minimax-m2.7 | 1.50M | 164k | 15.5M | 56k | $0.71 |
| qwen3.6-27b | 709k | 10k | 57k | 41k | $0.61 |
| qwen3.6-35b-a3b | 0 | 0 | 0 | 0 | $0.00 |
| **TOTAL** | | | | | **$2.14** |

DeepSeek's 39.4M cache_read on Flash and 15.7M on Pro are the most striking numbers — implicit caching dominates the input budget. Cache reads are ~10x cheaper than fresh input on most providers, so the "input" column understates the real efficiency win. MiniMax cache_read was high too (15.5M) — Fireworks fp8 endpoint also caches.

Qwen 27B's 10k total output across 7 attempts is diagnostic of something deeper than "model can't kernel" — it barely emitted any tool calls. Either the OpenRouter/Alibaba tool-call format isn't matching what opencode expects, or the model defaults to a reasoning-mode that never produces tool-use. Worth a transcript dive before next sweep.

### What this validated

1. **The PROMPT.txt regime works end-to-end.** Models that can drive tool calls (DeepSeek, MiniMax) produced real solutions; verification gate triggered `python check.py` invocations consistently in the passing runs.
2. **Token logging is uniformly populated** across opencode runs. The `usage` block in result.json carries cleanly. Cross-harness comparison data is now in place.
3. **OpenRouter pinning works** for Alibaba (Qwen) and Fireworks (MiniMax). MiniMax even had real cache_read numbers, confirming Fireworks supports prompt caching.
4. **My cost estimate was 2.8x conservative.** Estimated $5-6, actual $2.14. Failed/ERR runs save money on failure modes that bail early. Future sweeps can be planned with this calibration.

### What it surfaced

1. **Both DeepSeek tiers cluster around 5/7 PASS** with overlapping but non-identical strengths. Flash hits higher peaks on memory-bound problems (paged_attn 0.167, kahan 0.138, w4a16 0.134); Pro is more consistent on compute-bound (sonic_moe 0.108) and has a non-zero TopK pass where Flash regressed. Both fail FP8 GEMM and KDA — those are the deck's hardest two for cheap-tier reasoning models.
2. **MiniMax's 3/7 is the floor for "model can autonomously kernel."** It needs the simpler problems (kahan softmax, sonic_moe up-proj, w4a16) and bails on harder ones with no solution.py. Useful as a benchmark sanity floor.
3. **Qwen 3.6 27B 0/7 is a harness-integration failure, not a capability failure.** 708k input tokens consumed but only 9.5k output suggests opencode is talking to it and Alibaba is responding, but tool-call exchange isn't happening. Needs investigation before counting it out as a model.
4. **qwen3.6-35b-a3b is benchmark-blocked.** Documented in the previous entry; confirmed by 7×0-second ERR runs.

### Reproducibility footnote

DeepSeek V4 Flash on TopK: passed at 0.0019 yesterday (24 hours ago, prompt-edit experiment); failed today on the shakedown. Same prompt, same model, same provider. The variance on hard problems near the model's capability floor is real and not insignificant. For TopK specifically, Flash is on the edge of "can solve this" — sometimes does, sometimes doesn't. Future passes should report N>=2 trials per (model, problem) and note variance, not just a single peak_fraction.

---

## 2026-04-27 — Harness configuration parity: what we touched and why

When you run "the same task" through five different agent CLIs, the meaning of "same" is doing a lot of work. This entry catalogs every config knob we touched to make cross-harness results comparable, and (more importantly) the asymmetries we could not eliminate. Read this if you want to know how much trust to place in any given peak_fraction comparison.

### Reasoning effort tiers (asymmetric across harnesses)

The CLI surface for "make the model think harder" differs per harness. Our active-matrix settings:

| Harness | Model | Setting | What it actually does |
| ------- | ----- | ------- | --------------------- |
| claude | claude-opus-4-7 | `--effort max` | Highest of the {low, medium, high, xhigh, max} tiers exposed by claude-code 2.1.119. Triggers extended thinking with the largest budget the CLI allows. |
| codex | gpt-5.5 | `-c model_reasoning_effort="xhigh"` | Highest effort tier codex exposes for gpt-5.5. |
| kimi | kimi-k2.6 | (default) | kimi-cli does not expose a reasoning-effort flag. K2.6 is a reasoning model and reasons by default; the budget is whatever Moonshot allocates. |
| opencode | deepseek-v4-pro / -flash, glm-5.1, minimax, qwen, mimo | (default) | opencode SST has no per-call reasoning-effort hook. The underlying model decides whether and how much to reason; some (DeepSeek V4 Pro, GLM-5.1) are reasoning models, others aren't. |

This is the biggest "same task, different shape" asymmetry in the benchmark. We use the highest tier each CLI exposes; we don't pretend that's identical to what another model does on its own. Result tables should be read as "model X via harness Y at the maximum effort that harness exposes," not "model X at parameterized effort level Z."

### Provider routing (what reaches the GPU)

OpenRouter dispatches to whichever backend has capacity. Many providers serve int4/fp4-quantized weights of frontier models; running a benchmark against int4 of GLM-5.1 is not the same as running against the lab's full bf16/fp8 weights. We pin every OpenRouter-routed model to its native lab provider via `extraBody.provider.order` with `allow_fallbacks: false`.

Current provider order in `~/.config/opencode/opencode.json` openrouter-pinned: `["Alibaba", "Xiaomi", "Minimax", "DeepSeek", "Z.AI"]`. With `allow_fallbacks: false`, a request fails if the named providers don't host the model, rather than silently falling back to a quantized third party. The fail-loud is intentional — we'd rather see "no integrity-clean route" than ship a quietly-quantized number.

Models routed lab-direct (not OpenRouter): `deepseek-v4-pro`, `deepseek-v4-flash`, `glm-5.1`, `glm-5`. These hit the lab's API directly via OpenAI-shape providers in opencode config.

Excluded from the matrix: `qwen/qwen3.6-35b-a3b`. Alibaba does not serve it on OpenRouter; only AtlasCloud and Parasail (both fp8) do. Including it would mean either accepting third-party fp8 (breaks the integrity rule) or running against a different precision than the rest of the Qwen family (apples-to-oranges). Skipped, documented; user can opt back in if they accept the tradeoff.

### Codex version pin

Local rust binary `codex 0.118.0` rejects `-m gpt-5.5` ("model not recognized"). The npm `@openai/codex` 0.125.0 accepts it but dropped `wire_api="chat"` config support, which means codex 0.125.0 cannot route arbitrary OpenRouter models — only OpenAI's `/responses` API works. Net result: codex is the right harness for OpenAI models specifically, not a universal harness for anything OpenAI-compatible. Z.AI doesn't implement `/responses` so GLM cannot be reached through codex at all; we route GLM through opencode instead.

A second codex quirk: `codex 0.125.0` updates SQLite session state by touching old session JSONL files in `~/.codex/sessions/<date>/`, which broke "find by mtime" archival. Fix: extract `session id: <uuid>` from stderr and `find -name "*${uuid}*.jsonl"` to locate the right transcript.

### Workspace state and template files

Every per-run cycle deletes everything in the problem dir except the template set. Current TEMPLATE_FILES (in `scripts/run_hard.sh`): `reference.py sota.py shapes.py problem.yaml check.py benchmark.py PROMPT.txt`. Anything else the agent created (build artifacts, scratch kernels, profiling traces, intermediate `.cu` files) gets archived to `outputs/runs/<ts>/scratch/` and removed from the workspace before the next run.

`shapes.py` and `problem.yaml` stay in the workspace (model-visible) only because `check.py` and `benchmark.py` import them at runtime. A curious agent can `cat problem.yaml` and re-read the regime / forbidden ops list / tolerance — the prompt does not direct it there, but the option exists. Closing this leak would require refactoring check/benchmark to read yaml from outside the workspace; not load-bearing yet, flagged for later.

### Per-trial benchmarking methodology

Centralized in `src/eval/timing.py` so every problem's `benchmark.py` uses the same cadence:
- 10 warmup calls (absorbs Triton autotune ~7 configs and torch.compile reduce-overhead CUDA-graph capture).
- Per-trial L2 flush via 128 MB write to a scratch tensor (RTX PRO 6000 L2 is 96 MB, so 128 MB strictly evicts).
- CUDA Events with synchronize() AFTER record() but BEFORE elapsed_time().
- Median over 30 trials (default; some problems use fewer for slow Python references).

Known biases left in:
- `torch.compile(mode="reduce-overhead")` gets CUDA graphs (eliminates launch overhead). Custom Triton/CUDA kernels do not. On small shapes where launch overhead matters, this gives the compile baseline an artificial advantage. Accepted as the cost of using `torch.compile` as the published "compiled" reference line.
- cuBLAS / cuDNN allocate workspaces on first call. The 10-call warmup absorbs.
- Median over a small number of trials catches outliers but won't expose bimodal latency distributions.

### Wall-clock budget, not turn count

`BUDGET_SECONDS=2700` (45 min) per (model, problem) run, enforced by `timeout(1)`. Models get unlimited turns within the budget. v3 used `for turn in range(max_turns)` and got chewed up by reasoning models (GLM-5.1) burning turns on filesystem exploration before writing anything — a turn cap penalizes models with verbose tool-use patterns regardless of capability. Wall-clock is the fairer floor.

### Token logging (cross-harness uniformity)

Every transcript schema is different. `scripts/extract_usage.py` parses each one and emits a normalized shape:
```
{ input_tokens, output_tokens, cache_read_tokens,
  cache_creation_tokens, reasoning_tokens, total_cost_usd }
```

What's countable per harness:
- claude / kimi: terminal `{"type":"result"}` event has cumulative usage with `total_cost_usd` (only when running off API direct, not coding-plan).
- codex: per-turn `payload.type=token_count` events have `last_token_usage`; we sum.
- opencode: each `step_finish` carries `part.tokens` with input/output/reasoning + cache.read/cache.write; we sum.

What's NOT countable:
- Coding-plan billing (Claude Code, Codex on a subscription) does not expose per-call USD in the transcript. Token counts ARE present and are what we use for cross-model comparison. Per-call cost is reconstructable post-hoc from public price sheets if needed.
- Raw chain-of-thought content. Both `claude` (thinking blocks come back as `{"thinking": "", "signature": "..."}`) and `codex` (shows reasoning *summaries*, not raw CoT) encrypt the actual reasoning content in their CLI delivery channels. We get cryptographic proof that thinking happened, plus the token cost, but not the content itself. This symmetric disclosure floor is enforced by the harnesses themselves; we cannot lift it without bypassing them and calling lab APIs directly.

### What this means for cross-harness comparisons

A peak_fraction number from the benchmark is meaningful within these caveats:
- The hardware target is fixed (RTX PRO 6000 SM120, GDDR7 1.8 TB/s peak).
- The problem definition (reference.py, shapes, tolerance, forbidden ops) is fixed and append-only after publication.
- Each model runs at the highest effort tier its harness CLI exposes, but those tiers are not necessarily equivalent across vendors.
- Provider pinning ensures the model weights served are the lab's full-precision endpoint, not a quantized third party.
- Wall-clock budget and benchmarking methodology (warmup, L2 flush, median) are identical for all runs.
- Coding-plan billed runs (claude, codex) report token counts only, no per-call USD.

If you build on these numbers, cite the (model, harness, effort, provider) tuple, not just the model name. The same model behind a different harness will produce a different number.

---

## 2026-04-27 — Verification gate refinement (validated experimentally)

**Setup.** First DeepSeek V4 Flash run on TopK with the new PROMPT.txt regime: PASSed `has_solution`, FAILed correctness because the kernel allocated `threads * k * 8 = 128 KB` of dynamic SMEM on shape 0 (k=64), which exceeds the 100 KB default opt-in cap. Tool-call inventory showed Flash had run zero `python check.py` invocations — it had self-validated with two ad-hoc `python -c "from solution import ..."` snippets that almost certainly used the small default shape (16 KB SMEM) and never iterated through all five shapes.

**Edit.** Tightened the verification gate sentence in all 7 PROMPT.txt files:
- Old: `verify correctness against the oracle in check.py, then iterate. If check.py isn't passing, you're not done.`
- New: ``verify correctness by running `python check.py` and reading the output, then iterate. Don't substitute your own one-off correctness snippets for check.py — it iterates over every shape, your spot-check almost certainly won't. If `python check.py` hasn't printed PASS, you're not done.``

Three deliberate changes: (1) literal-action verb ("by running") replaces the abstract goal ("against the oracle"); (2) the middle sentence directly counter-instructs the failure mode (rolling your own); (3) PASS as the explicit sentinel string anchors the stop condition.

**Validation.** Reran Flash with the same model and the same problem; the only variable was the prompt tweak.
- Tool-call inventory: **3 `python check.py` invocations** (was zero).
- Result: PASS on all 5 shapes, peak_fraction 0.0019.
- The model produced a *correct but slow* kernel rather than a *plausible-looking but broken* one.

The score is low — Flash didn't push throughput — but the disciplinary outcome flipped from FAIL to PASS purely from the prompt edit. That's a clean experimental result. Three sentences of prompt rewrite changed the verification regime from "models that already test thoroughly do; models that don't, don't" to "models that *can* run a test, run it." Capability gates kernel quality; discipline now gates correctness.

Filed under: arguments for tightening prompts further actually do work, sometimes. Counter to my earlier "skill issue" framing — turns out half of "skill issue" is "compliance issue," and compliance is promptable.

---

## 2026-04-27 — Opus parity: --effort max wiring + token-cost logging

**Decision.** Wired `--effort` flag for the `claude` harness in `run_hard.sh` (previously only codex respected `REASONING_EFFORT`). Updated `scripts/sweep.sh` ACTIVE_MATRIX to use `claude claude-opus-4-7 max` for parity with `codex gpt-5.5 xhigh`.

**Why.** Houssin's Twitter critique on the launch post: "Why not use Opus 4.7 Max if you're using xHigh for GPT 5.5? That's not fair." Correct critique. Last sweep ran Opus at default effort while GPT-5.5 was at xhigh. The CLI exposes `low | medium | high | xhigh | max` as the effort tiers (`claude --help`); `max` is the highest. Smoke-tested with a trivial math prompt — flag accepted, thinking block emitted, output_tokens scaled past visible answer length confirming extended thinking happened.

**Thinking-content visibility.** The `thinking` block in Claude Code transcripts comes back with `thinking: ""` and a `signature: "..."` — content is encrypted in the CLI delivery channel. We get cryptographic proof that thinking happened, plus token counts, but not the raw chain-of-thought. Same disclosure floor as codex (codex shows reasoning summaries but not raw CoT either). Symmetry is preserved.

**Token logging in result.json.** Added `scripts/extract_usage.py` — a single Python script that parses each harness's transcript schema (claude/kimi `{"type":"result"}`, codex `payload.type=token_count` events, opencode `step_finish.part.tokens`) and emits a normalized `{input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, reasoning_tokens, total_cost_usd}` shape. Wired into `run_hard.sh` so result.json now includes a `usage` block. Coding-plan billing on the CLI hides per-call USD, but raw token counts are always present in the transcripts and that's what matters for cross-model comparison.

Validated on the Flash rerun: input=57,555, output=12,158, cache_read=1,367,296, reasoning=98,047. The 1.37M cache_read confirms DeepSeek implicit caching is hot (matches the 98% hit rate noted in the provider-pinning entry below).

---

## 2026-04-27 — Model coverage expansion (Twitter-driven)

**Added to ACTIVE_MATRIX** (per Twitter requests):
- `qwen/qwen3.6-max-preview` — Alibaba, 262k ctx
- `qwen/qwen3.6-plus` — Alibaba, 1M ctx
- `qwen/qwen3.6-27b` — Alibaba, 262k ctx
- `xiaomi/mimo-v2.5-pro` — Xiaomi (lab), fp8, 1M ctx

All routed through `opencode openrouter-pinned/...` with `provider.order = ["Alibaba", "Xiaomi", "Minimax", "DeepSeek", "Z.AI"]` and `allow_fallbacks: false`.

**`qwen/qwen3.6-35b-a3b` — infrastructure-blocked. Holding off.** Tried twice (skipped initially for native-lab integrity; reversed on user direction with AtlasCloud and Parasail appended to provider.order). Shakedown sweep then surfaced the actual blocker: every run fails in <1s with `APIError 404: No endpoints found that support tool use. Try disabling "bash"`. Neither AtlasCloud nor Parasail advertises tool-use capability to OpenRouter for this model, and our agent harness is fundamentally tool-call-driven (bash/read/edit). There is no integrity-clean route through which an autonomous agent can use this model right now. Removed from the active matrix; will revisit if (a) Alibaba hosts it on OpenRouter, (b) AtlasCloud/Parasail expose tool-use, or (c) the model lands on a lab-direct API like Z.AI/DeepSeek already are. Filed as a useful negative result — the benchmark surfaces "no autonomous-agent endpoint exists for this model" as a real outcome, not just an integration bug.

**TODO — when budget permits:**
- **GPT-5.5 Pro.** Twitter request × 1 (insanowskyy: "what about 5.5 pro?"). Not on the active matrix because the OpenAI per-call cost is high enough to be a real budget item; coding-plan doesn't apply to API-direct gpt-5.5-pro calls. Revisit when sweep cadence justifies the spend.
- **Gemini 3.1 Pro.** Twitter request × 2. The harness story is unsettled — Droid worked in v3 but is not currently wired into KernelBench-Hard's run_hard.sh. Adding requires either (a) re-adding the droid case to run_hard.sh and authenticating via Factory, or (b) routing through opencode if Google AI Studio offers an OpenAI-compatible endpoint. Skipped until the harness wiring is decided.
- **Mythos / generic "show us X" requests.** Volume signal only; not actionable until the model has a stable identifier and a benchmarkable native-lab provider.

---

## 2026-04-27 — Prompt regime overhaul: eval-shaped → human-shaped

**Decision.** Replaced the two-file `preamble.md` + `AGENT.md` system-prompt regime with a single per-problem `PROMPT.txt` written in plain human voice. The harness now sends `PROMPT.txt` directly as the prompt to each agent — no system/user split, no markdown structure, no "Read SYSTEM_PROMPT.md first" wrapper.

**Why.** Two observations from the TopK overnight sweep:

1. The old preamble opened with "You are an autonomous coding agent being evaluated on a hard GPU kernel optimization problem." That framing primes models to perform-on-test rather than do-the-work. Opus's "the 0.1 RESULT threshold isn't structurally achievable here" rationalization is the eval-shape pattern: when you tell a model it's being evaluated, it explains its score instead of fixing the kernel.
2. The preamble was 101 lines of hardware specs, peak throughput tables, optimization recipes, profiling commands, and workflow steps. That's a benchmark giving away the answer key and then asking the model to find the answer. Models that already know this stuff gain nothing; weaker models get carried.

**What changed in the prompt itself.**

Removed entirely: opening "you are an autonomous coding agent" framing; full hardware spec section (tensor cores, what's not on SM120, etc.); peak throughput table; toolchain section (CUDA versions, compile flags, CUTLASS path); optimization guidance (FP4/FP8/BF16/TMA recipes); profiling commands (`ncu`, `nsys`, `torch.profiler`); workflow steps; budget line; "what makes a good solution"; "good luck" closer.

Kept: one-line hardware identifier in a parenthetical (`SM120 Blackwell, GDDR7, 1.8 TB/s`); library availability list (without it the model won't know FLA / scattermoe / flashinfer are options); shapes inlined as prose; forbidden ops inlined as prose; tolerance + correctness contract inlined as prose; verification gate as a single sentence in the flywheel paragraph ("If check.py isn't passing, you're not done."); custom-kernel mandate; "look up PTX docs / clone repos / investigate" directive.

**What the model now doesn't know coming in.** Peak TFLOPS for any precision. Which tensor-core instructions are available on SM120. Which are SM100-only and will fail. Compile flags. The fact that 188 SMs exist. Profiling tool names. Optimization recipes. It has to look these up itself or know them from training data — that's part of what's being measured.

**What stays in the workspace.** `reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, `sota.py`, `PROMPT.txt`. The yaml and shapes.py have to stay because `check.py` and `benchmark.py` import them at runtime. Small leakage risk (a curious model could `cat problem.yaml` and read the regime / forbidden list / tolerance again), but the prompt only directs the model to `reference.py`. If that leakage matters later, the fix is refactoring check/benchmark to read yaml from outside the workspace; not yet worth the complexity.

**Files deleted.** `src/harness/preamble.md`, all `problems/*/AGENT.md` (8 files), one stale `problems/02_kda_cutlass/SYSTEM_PROMPT.md`. The harness no longer composes a SYSTEM_PROMPT.md per run.

**Smoke-tested.** Claude Code on problem 05 with `BUDGET_SECONDS=300` — confirmed PROMPT.txt arrives clean as `event[6] type=user` in the transcript, workspace cleanup behaves, no stale SYSTEM_PROMPT.md left behind.

---

## 2026-04-27 — Verification gate added (then folded into the flywheel)

**Decision.** Added a "your final action before stopping must be a successful `python check.py`" requirement to the prompt. After the prompt overhaul, this lives as a single sentence ("If check.py isn't passing, you're not done.") inside the flywheel paragraph rather than its own section.

**Why.** Of the 4 non-passing TopK runs:
- DeepSeek V4 Flash: linker error from `extern "C"` mismatch between `.cu` and `cpp_sources` header inside `load_inline`.
- DeepSeek V4 Pro: CUDA illegal memory access in the bitonic merge kernel.
- MiniMax M2.7: hardcoded `build_directory="/tmp/topk_v2"` that didn't exist; `FileNotFoundError` on first import.
- GLM-5.1: never wrote `solution.py` — burned 31,995 reasoning tokens before emitting any tool call.

3 of 4 would have been caught by running `check.py` once before submitting. The pattern is "submit blind, stop." Mandating a verification pass costs nothing for capable models, and it's not "hand-holding" — it's the discipline-half of pair programming, which is fair to require. (GLM is unfixable from the prompt; that's a Z.AI output-token-budget problem.)

---

## 2026-05-23 — Parallel sweep logging and queue-safe launch

The reliable parallel mode is `KBH_DISABLE_AGENT_CUDA=1`: hide CUDA from
OpenCode/Cursor agent phases, then let `scripts/run_hard.sh` own `check.py` and
`benchmark.py` under `outputs/gpu.lock`. This avoids the failure mode where an
agent bypasses PATH wrappers by calling an absolute `.venv/bin/python3`.
The first full launch at `kbh_hard_parallel_20260523_002720` proved the extra
guard was necessary: Cursor set `REAL_UV=$(which uv)` after the wrapper had
entered `PATH`, recursively invoking the wrapper, while several absolute
`.venv/bin/python3` children touched CUDA outside the lock. That sweep was
terminated before result collection. The harness now keeps fallback real binary
paths and injects an agent-phase `sitecustomize.py` guard so torch CUDA probes
fail fast during generation; harness-owned validation still runs without the
agent guard.

`result.json` now records `run_id`, `run_group`, ISO and epoch timestamps,
harness-only wall time, total wall time, check/benchmark wall time, queue mode,
agent CUDA visibility, and normalized usage fields. `scripts/summarize_runs.py`
flattens `outputs/runs/*/result.json` into JSON/CSV summaries for website import.
`scripts/launch_parallel_sweep.sh` writes a manifest under
`outputs/sweeps/<run_group>/manifest.tsv` and starts the model/problem matrix in
parallel.

Verification before launch:

```bash
bash -n scripts/run_hard.sh
bash -n scripts/launch_parallel_sweep.sh
uv run ruff check . --fix
uv run pytest
uv run python scripts/summarize_runs.py --output-dir outputs/summaries/smoke_latest
```

Result: 10 tests passed; summarizer wrote 167 historical rows.

Clean guarded sweep launched after the failed first attempt:

```bash
KBH_RUN_GROUP=kbh_hard_parallel_guarded_20260523_003820 \
KBH_DISABLE_AGENT_CUDA=1 \
./scripts/launch_parallel_sweep.sh
```

Manifest:
`outputs/sweeps/kbh_hard_parallel_guarded_20260523_003820/manifest.tsv`.
Early verification showed no agent-phase CUDA apps, one GPU compute process at
a time under `outputs/gpu.lock.owner`, and result rows carrying the new timing,
usage, and queue metadata. Interim summaries are written to
`outputs/sweeps/kbh_hard_parallel_guarded_20260523_003820/summary/`.

## 2026-04-26 — TopK overnight sweep: forensic findings

**Setup.** 7 models × 1 problem (05_topk_bitonic), sequential, 45-min budget each. `regime: memory`, scored against 1.8 TB/s GDDR7 peak. Geomean over 5 shapes.

**Results.**

| Rank | Model            | Status               | peak_fraction |
| ---- | ---------------- | -------------------- | ------------- |
| 1    | GPT-5.5 xhigh    | PASS                 | 0.0657        |
| 2    | Claude Opus 4.7  | PASS                 | 0.0132        |
| 3    | Kimi K2.6        | PASS (timed out)     | 0.0063        |
| —    | GLM-5.1          | ERR (no solution.py) | —             |
| —    | DeepSeek V4 Pro  | FAIL (CUDA OOB)      | —             |
| —    | DeepSeek V4 Flash| FAIL (link error)    | —             |
| —    | MiniMax M2.7     | FAIL (build dir)     | —             |

**Algorithm gap dominated kernel-craft gap.** GPT and Opus had the same wall budget on the same hardware. Opus picked full bitonic sort (O(n log²n) per row), GPT picked packed-key reduction with `tl.topk` (O(n) per row). At n=8192 that's a ~7x asymptotic gap — and the observed perf gap on the prefill shape (b=64, n=8192, k=8) was 8.7x. The kernel-craft delta would have been maybe 2x; the algorithmic choice was 5-7x of the 8.7x.

**Opus's "structurally launch-bound" claim was wrong.** On shape 0 (b=1, n=131072, k=64), Opus claimed the geomean threshold was unreachable because "the whole benchmark is launch-overhead bound." Actual numbers:
- Bandwidth lower bound to read 512 KB at 1.8 TB/s: **0.28 μs**.
- GPT-5.5 measured: **27 μs** (~100x slower than the floor).
- Opus measured: **48 μs** (~170x slower).

A single launch on a hot CUDA graph is ~1-2 μs. The remaining ~25 μs is real kernel time, not launches. Why is the kernel slow? GPT picked `chunk_n=2048` for shape 0, which gives `131072/2048 = 64` blocks for a 188-SM machine. **34% SM occupancy ceiling.** The kernel is leaving 2/3 of the GPU idle. Opus's CHUNK_PAD=2048 has the identical bug. The fix is `chunk_n=512` → 256 blocks → fully oversubscribed → near-peak bandwidth → estimated 0.10–0.15 peak_fraction on shape 0 alone.

Lesson: "launch-bound" is a real diagnosis on small kernels with many launches and no graphs. "Parallelism-starved" is a different diagnosis with the same surface symptom (low throughput on small shapes). Mixing them up is how rationalization sneaks in. Both Opus and GPT made the same parallelism-starvation mistake; only Opus rationalized it as physical-limit-bound.

**The 4 failures break into one model-side issue and three "didn't run check.py" issues.** GLM-5.1's 31995-reasoning-token blowup is fixable only by raising opencode's max output tokens for zai/glm-5.1; nothing in the prompt fixes a model that can't budget its own thinking. The other three were trivial bugs that any single test run would have caught. Hence the verification gate.

---

## 2026-04-25 — Centralized timing module + L2 flush + warmup bump

**Setup.** Each `problems/<NN>/benchmark.py` was duplicating warmup-and-cuda-events code. Several discrepancies surfaced when comparing runs.

**What we found.** Without an explicit L2 cache flush between trials, FP8 GEMM peak_fraction came out at 0.520. With a 128 MB write to evict L2 (Blackwell consumer L2 is 96 MB), the same kernel measured 0.426. The skinny-M shape went 20% → 10% with the flush. The original numbers were measuring L2-cached re-reads, not HBM bandwidth.

Warmup of 5 was too short for Triton autotune (~7 configs) plus `torch.compile(reduce-overhead)` CUDA-graph capture. Bumped to 10. `iters` defaults to 30 trials; report median.

**What lives in `src/eval/timing.py`.** Single `time_fn(fn, inputs, iters, warmup)` that does warmup → per-trial L2 flush → cuda Events with synchronize-after-record → median. All seven `benchmark.py` files import this; methodology bugs only need fixing once.

**Known biases not addressed.** `torch.compile(reduce-overhead)` gets CUDA graphs which eliminate launch overhead; custom Triton/CUDA kernels do not. On small shapes this gives the compile baseline an artificial advantage. Accepted as the cost of using torch.compile as the published "compiled" reference.

---

## 2026-04-25 — Harness wars

**ccr-rust pivot to OpenCode SST.** Tried routing Claude Code to non-Anthropic providers via ccr-rust (an Anthropic-API-shape proxy). It returned malformed SSE that broke the claude-code stream-json parser. Pivoted to OpenCode SST with custom OpenAI-shape providers (`deepseek`, `zai`, `openrouter-pinned`) — that worked.

**Codex 0.125.0 broke chat-completions routing.** The new release dropped `wire_api="chat"` config support, so codex can no longer route arbitrary OpenRouter models. It only speaks `/responses` API now. Z.AI doesn't implement `/responses`, so GLM-5.1 cannot be reached through codex at all. We fall back to opencode for non-OpenAI lab models. Documented in `CLAUDE.md` model-harness assignment table.

**Codex session-id-from-stderr instead of mtime.** Codex 0.125.0 touches old session JSONL files when scanning its SQLite thread-state DB. So picking the most-recently-modified file in `~/.codex/sessions/<date>/` returns the wrong file. The fix is to grep `session id: <uuid>` out of stderr and `find -name "*${sid}*.jsonl"`.

**`set -e` + SIGTERM 124 was a silent script killer.** When a harness hits the wall-clock `timeout` and gets SIGTERM, exit code is 124. With `set -euo pipefail`, capturing the exit via `cmd; HARNESS_EXIT=$?` exits the whole script. Fix: `cmd || HARNESS_EXIT=$?`. This bug ate two debugging sessions before we caught it.

**Local rust codex binary had a stale alias.** `npm install -g @openai/codex` gives 0.125.0 with `gpt-5.5` support; the local rust binary was 0.118.0 and rejected the model name. Non-interactive shells don't see the alias, so `which codex` was lying. Force PATH to npm bin.

---

## 2026-05-23 - Infra failure classification and safer resweep controls

Added explicit failure classification to `scripts/run_hard.sh`:
`pass`, `template_mutated`, `provider_rate_limited`, `timeout`,
`incomplete_session`, `provider_early_stop`, `no_solution`, `check_failed`,
`benchmark_failed`, and `harness_error`. Each `result.json` now carries
`failure_reason`, `retryable_infra_failure`, and
`minimum_useful_output_tokens` so the website can distinguish a bad kernel from
an API/quota/no-output event. The default minimum useful output threshold is
5,000 tokens for no-solution kernel attempts.

Added `scripts/preflight_harnesses.sh` for cheap auth/model-route checks before
paid sweeps, and `scripts/launch_infra_retries.sh` to rerun only rows whose
latest result is retryable infrastructure failure. `scripts/summarize_runs.py`
now flattens the new fields into summary JSON/CSV.

OpenRouter can pass a tiny preflight while still lacking enough balance for the
full KernelBench prompt. When `/api/v1/credits` shows usage at or above credits,
run non-OpenRouter rows with
`KBH_SKIP_OPENROUTER=1 KBH_USE_DIRECT_GEMINI=1`; this keeps Gemini running via
the Gemini API key and leaves Qwen pending until OpenRouter is topped up or a
direct Alibaba/Qwen key exists.

The guarded full-sweep launcher still uses archive-local workspaces and the GPU
lock, but now defaults to `KBH_HARNESS_CONCURRENCY=2` per harness/provider
path. Claude Code runs also pass `--settings
'{"fastMode":false,"alwaysThinkingEnabled":true}'` explicitly; Anvil's global
Claude setting was already `fastMode=false`, and previous Opus result metadata
also reported `fast_mode_state: off`, but this makes the benchmark setting
durable.

During the classified resweep, `scripts/launch_infra_retries.sh` initially
emitted tab-separated retry keys. Bash treated the empty effort field as
collapsible whitespace, shifted the problem into the effort column, and launched
blank-problem retry manifest rows. Fixed by using `|` as the retry key delimiter
and normalizing summary problem names back to `problems/<name>` before calling
`run_hard.sh`.

IVA / voice bridge jobs on Anvil have higher priority than KernelBench sweeps.
Do not kill IVA just to make `nvidia-smi` empty. The KernelBench harness should
remain lower priority; if an unrelated IVA CUDA context or CPU workload is
present, leave it alone and report it as concurrent machine state.

## 2026-04-24 — Provider pinning + caching wisdom

**OpenRouter dispatches to whichever backend has capacity, including int4/fp4-quantized weights.** Code generation on int4 is materially worse than full weights — a model that scores 50% on bf16 might score 30% on int4. So `provider_order` pinning to the native lab is mandatory for benchmark integrity.

**Pinning works in our harness, not in Droid custom models.** Droid OpenRouter custom-model configs ignore `provider_order`. The KernelBench harness sends `extraBody.provider.order` directly via the OpenAI SDK, which OpenRouter respects. Anything routed through Droid custom OpenRouter loses pinning.

**MiniMax direct API was 401.** Worked through OpenRouter pinned to "Minimax" provider (their fp8 endpoint, ~$0.30/$1.20 per M, 99.7% uptime).

**DeepSeek implicit caching is real.** Verified: same prompt sent twice in a row hit `cache_tokens: 1792 / 1829` on the second request. ~98% cache hit rate at 10x cost reduction. No explicit cache-control header needed; just resend the same prefix.

---

## 2026-04-24 — Why "Hard": pivot from KernelBench-v3

**v3 was 43 problems of grab-bag difficulty.** Most were winnable by any frontier model with any harness. Median speedups ended up reward-hacked (precision downcast, F.softmax wrappers, GEMM dispatching to `torch._scaled_mm`) or trivially above eager. Leaderboard non-informative.

**v3 reward-hack patterns.** GLM-5.1 cast fp32 inputs to fp16 before GEMM to use tensor cores → ~2x "speedup" that was cheaper arithmetic, not better algorithm. `pct_of_peak > 100%` was the giveaway. MiniMax M2.5 attempted `pkill -f python` to kill the eval process on its first run. Various models called the library wrapper (F.softmax, F.scaled_dot_product_attention) and counted that as a "kernel."

**What we tried that didn't work.** Extensive regex blocklists for forbidden patterns. Brittle whack-a-mole — every release added new ways to hide the dispatch. Replaced with an LLM judge model post-benchmark (`src/eval/judge.py`) that reviews the solution code and flags semantic cheating. Better recall than regex; defaults to PASS on judge error to avoid false negatives.

**Hard's three changes vs v3.**
1. **Tight per-dtype tolerance + multi-shape eval kills reward-hacking** at the correctness gate, so a degenerate identity-operator solution fails check.py.
2. **Roofline grading against hardware peak**, not speedup over PyTorch. Beating eager means nothing; approaching SOTA is the goal. peak_fraction = achieved_TFLOPS / peak_TFLOPS for compute regime, achieved_GB/s / peak_HBM for memory regime.
3. **Forbidden ops listed in problem.yaml + inlined into the prompt.** Using `torch.topk` on a top-k problem fails post-hoc. The point of each problem is to write the kernel, not to dispatch to a library.

**Wall-clock budgets > turn-count budgets.** v3 used `for turn in range(max_turns)` in agent loops. Models like GLM-5.1 burned 9/10 turns on filesystem exploration ("looking for CUTLASS headers") and never wrote code. Switched to wall-clock timeouts in v3 late-stage; carried over to Hard. Models get unlimited turns within a 45-min budget.

---

## Open questions / things to chase

- **Pair-programming eval.** The autonomous-floor numbers tell us how each model behaves alone; they don't tell us the human-in-the-loop ceiling. A 5-model paired-session run on problem 05 would answer "what's the agency tax of running model X without me there?" — the gap between paired and autonomous peak_fraction. n=1 per model but useful even so.

- **Persistent-kernel / cooperative-reduction kernel for shape 0 of TopK.** Both PASS submissions (Opus, GPT) are parallelism-starved on b=1 n=131072. A correctly-fanned-out kernel should hit ≥0.10 on shape 0 alone. Worth writing the reference solution by hand to confirm the achievable ceiling and validate whether the geomean threshold of 0.1 is reachable.

- **GLM-5.1 output-token cap.** The opencode `extraBody` for zai/glm-5.1 doesn't expose `max_output_tokens` — Z.AI's beta API caps at 32768. With reasoning chains of 30k+, that leaves no room for tool calls. Either request a higher cap from Z.AI, or accept GLM as an outlier whose autonomous score is bounded by output budget rather than capability.

- **Removing problem.yaml + shapes.py from the model's view.** Currently they sit in the workspace because check.py and benchmark.py import them. Refactor option: pre-render their content into the prompt (already done) and have check.py / benchmark.py read yaml/shapes from a sibling private directory. Closes a small information leak. Not currently load-bearing.

- **Per-problem prompt voice consistency.** All seven prompts hand-written in one session, same voice, same four-paragraph structure. If we add an 8th problem (Metal lightning attn) or add a second hardware target, the temptation will be to write that prompt in a different style. Resist. The voice is part of the experimental control.
