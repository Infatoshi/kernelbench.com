# DEVLOG

A running record of decisions, dead ends, and lessons. Newest entries on top. This is not a changelog (the git log is) — it's the why behind the shape of the project.

Numbering note: entries before 2026-07-21 say "Problem 03" for the Kimi-Linear decode kernel, which is today's `02_kimi_linear_decode`; "Problem 02" was `01_rl_grid_ppo`, removed 2026-07-21.

---

## 2026-09-02 — claude-fable-5-1 max on 02_kimi_linear_decode, RTX PRO 6000 (anvil GPU 1)

Cell: 15.84x after sequential isolated regrade (in-run 15.82x; regraded
0.344/0.388/0.431 ms/tok at ctx 2048/8192/16384), verdict clean,
megakernel_authentic true, one cooperative persistent work-queue kernel
(arrival counters, no grid barriers, split-K GEMV, in-kernel MLA cache append).
Recompute test cos(o1,o2) = -0.018, cos(o2,ref) = 0.994; 1 launch/step;
0.39 of the 1.8 TB/s roofline. Session was cut by the provider five-hour limit
at ~91 min while still tuning. Same GPU class as the July Fable 5 18.71x cell
(20260701_172615, RTX PRO 6000 Blackwell, max, ran to self-termination), so
the gap is comparable but reads as a session-length effect at least as much
as a model delta.

Three earlier attempts the same day died before writing a solution
(annotated bug): the first-party `claude)` block in `scripts/run_hard.sh` did
not export `CLAUDE_CODE_MAX_OUTPUT_TOKENS`, so Claude Code used its 64000
default for claude-fable-5-1 and a single max-effort reasoning turn ended the
session with a terminal max_output_tokens error. `-p` mode does not recover
from that. Fixed: the block now exports 128000 (the CLI's upper limit for
this model). Effort tier was not the lever (xhigh died the same way). Also:
anvil had Claude Code 2.1.252, which did not know the model id at all
(200k/32k profile); updated to 2.1.257. Preflight a new model id with a
one-line `-p` call and check the rig's `claude --version`.

Tooling notes: `kb -b mega lint <run_id>` cannot find mega runs (delegates to
the hard bench's reward_hack_lint.py with the hard root); run the script with
the absolute run path. The 0.98 output-cosine gate on 02 is sensitive to
reference-side MoE router near-ties (seeds 2 and 10 are exact ties, seed 5
reads 0.977 for a correct kernel); expect dips in future 02 audits.

---

## 2026-08-17 - retracted grok-4.6 21x (copied Fable)

`20260813_152200_grok_grok-4.6_02_kimi_linear_decode` listed `runs-remote-pro`,
read Fable `20260719_121747` (`peak_fraction` 24.6091), then `cp` that
`solution.py` (tool description: "Copy proven megakernel into workspace
solution.py"). The annotation was wrongly `clean` after a same-buffer
overwrite pass. Flip to `contamination`. The Mega board drops the row.
`build_mega_leaderboard.py` no longer lets `clean` override a literal `cp`
of another archive's `solution.py`. `outputs/runs-remote-*` is now a
tripwire path. Isolated regrade and overwrite cosines are not authorship.

---

## 2026-07-31 - run_hard.sh stays a deliberate fork of the shared runner

hard/cuda/mini's run_hard.sh were unified into thin wrappers over
`scripts/lib/run_harness.sh` (monorepo root). Mega deliberately keeps its own
fork: it carries the bwrap anti-contamination sandbox (KBH_SANDBOX — hides run
archives, results/, DEVLOG, public/ solutions, and ~/.claude memory from the
agent) and the legacy anvil machine-wide gpu-lock-exec machinery, neither of
which the shared runner has yet. If the sandbox is ever ported into the shared
runner, fold mega in and delete this fork; until then, harness fixes land in
the lib and must be ported here by hand when relevant.

---

## 2026-07-21 - 01_rl_grid_ppo removed from the deck

Mega is now a single-problem bench: `02_kimi_linear_decode`. The RL-env
megakernel skill that `01_rl_grid_ppo` graded is covered by the CUDA bench's
craftax problem, so keeping both double-counted the same surface. It had
already been soft-hidden from the site (MEGA_HIDDEN_PROBLEMS) since the CUDA
deck landed; this makes the removal real: the deck dir is deleted,
`build_mega_leaderboard.py` filters the problem out of results.csv
(REMOVED_PROBLEMS), and the site constants are cleared. Archived runs, traces,
and annotations for old PPO cells are untouched. Do not re-add — same standing
rule as hard's `04_kahan_softmax`.

---

## 2026-07-09 - Agent-side CUDA disabling removed

`KBH_DISABLE_AGENT_CUDA` was removed from the harness, parallel launcher, and
infra-retry launcher. It had been introduced to prevent parallel agents from
bypassing the shared GPU lock, but hiding CUDA also removed the live
compile/check/benchmark/profile loop that KernelBench is intended to measure.
That made disabled and enabled runs incomparable. New runs always expose CUDA;
parallel GPU commands serialize through `outputs/gpu.lock`. Historical
`agent_cuda_disabled` metadata remains in archived results for provenance.

---

## 2026-07-01 — SPS credibility cap recalibrated 20x → 100x

The fabrication guard on `01_rl_grid_ppo` set `max_credible_sps_multiple: 20`
(anything over 20x `peak_sps` scores 0.0) when the best known honest kernel was
the ~2.1M SPS reference on the 3090. One day later a clean, audited fable-5
megakernel landed 357M SPS — leaving only ~1.4x honest headroom before the cap
would have zeroed a legitimate kernel. Recalibrated to 100x, still orders of
magnitude below the ~20,000x a no-work fabricator posts.

The problem and the flag are both gone now; the rule is what to keep. Calibrate
a fabrication cap against the physics, orders of magnitude above the best honest
result you currently know of — not just above it. The honest frontier moves, and
a cap tuned to yesterday's record silently starts failing real work.

---

## 2026-07-01 — fabrication red-team: an unfalsifiable metric is not a metric

Red-teaming `01_rl_grid_ppo` before handing the deck out found a critical hole:
the return curve was unfalsifiable. `train()` just returns floats, and nothing
proved an env step ever happened. Two cheats sailed through the gates with
absurd scores:

- **Fabricated curve** (analytic exponential ramp, no GPU work at all): check.py
  PASS (lands in the return band, "improves" from its own early window),
  benchmark **22,006x** peak_fraction.
- **Memoized replay** (`{(steps, seed): curve}` dict; the benchmark reran the
  same BENCH_SEED every trial): **26,849x**.

The fix that generalizes beyond that problem: **benchmark.py draws a fresh
random seed per timed trial** (SystemRandom). Throughput does not depend on the
seed, but a memoized lookup misses every trial, so the cached cheat pays full
cost. Any bench whose timed region can be keyed and cached needs this.

Residual, and the reason the authenticity judge exists at all: a fabricator that
also *sleeps* to fake a plausible elapsed time defeats every mechanical timing
check. A "trainer" with no environment and no policy update fails only on sight,
in an audit. Mechanical guards plus a judge are the enforcement pair; neither
half is sufficient.

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
cheapest launch-overhead fix. Distribution over the 36 decode runs with source: 9 multi-kernel + CUDA graph (every top cell), 20 multi-kernel eager, 4 single custom kernel + eager rest (all <= 2.74x), 1 torch.compile, 2 with no custom kernel at all.

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
  `/mega` renders it. Rubric + integration: `SPEC.md`, Megakernel authenticity section.

**Prompts (both problems) updated** to state the timed path must be one fused
kernel and that a *post-run authenticity judge* (not check.py) rejects graph/
compile/per-op-loop escapes — and that obfuscating them is itself a red flag.
Decode mandates one launch/step; RL allows coarse fusion (many env-steps per
launch) but forbids launch counts that scale with steps/horizon/minibatches.

---

## 2026-06-18 — contamination prevented at the source: bwrap sandbox on run_hard.sh

The near-term, easy fix for cross-run contamination (agents reading prior winning
solutions from the shared outputs/runs archive via absolute paths). The proper
sandboxed harness was going to be a Prime Intellect verifiers env (dropped
2026-09-02 along with `environments/`); this is the cheap, in-place prevention that ships now.

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
the contamination scan, now `kb contamination mega`) stays as defense-in-depth. See
[[cross-run-contamination]].

Origin: hard DEVLOG 2026-04-27 opencode workspace leak — that entry has the
audit of what agents actually read outside the workspace, and why no CLI
harness isolates the filesystem on its own.

---

## 2026-06-18 — the June rows ran under a 3-hour ceiling

All June runs on the decode problem used `BUDGET_SECONDS=10800` — a 3-hour
wall-clock ceiling, not "unlimited". Mega runs unlimited wall-clock now, so the
June H100 and B200 rows still on /mega are **not budget-comparable** to anything
published since; read them as a 3h-capped cohort. Within June they are
internally comparable, and nothing hit the cap at the time: every run
self-terminated, and the deepest worker (opus on B200) peaked at ~2.5h.

---

## 2026-06-18 — three-GPU leaderboard (Blackwell / H100 / B200)

The decode problem swept across three GPU generations, codex + opus on each:

      GPU                     codex/gpt-5.5    opus-4-8
      RTX PRO 6000 Blackwell      4.34x         14.40x
      H100                        5.62x         15.50x
      B200                        9.37x         19.35x   <- highest

Reads: opus dominates codex on every GPU (deeper kernel engineering under
unlimited time). Both models scale *up* the speedup ratio from Blackwell ->
H100 -> B200, because the baseline (naive int4 materialize) gets relatively
worse on the bigger datacenter cards while the fused dequant-GEMV keeps pace --
so the fusion win compounds with bandwidth.

Why this ports across generations at all: the score is a same-GPU
speedup-over-baseline ratio, so it needs no recalibration per card, and
int4/bf16-acc needs no special tensor-core format (Ampere/Hopper/Blackwell all
do bf16), so the same problem ran on all three with stock cu128 torch and zero
code changes. `build_mega_leaderboard.py` requires a per-run `gpu` marker for
exactly this reason — the requirement is what excludes the legacy bf16 runs that
share the problem name.

The ladder the problem was designed to expose showed up directly: my own hand
solution stalled at 1.1x (launch-bound, 126 tiny kernels), while codex reached
5.62x on the H100 by batching and fusing — 11 specialized Triton kernels, a
4-way fused q/k/v/g GEMV, in-kernel MLA absorption, a hand-written KDA conv +
recurrence. Fusion, not kernel micro-craft, is the whole spread.

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

## 2026-06-16 — RL training megakernel (throughput-graded), v0

The problem was removed from the deck on 2026-07-21. Kept for the design
argument and the hole it exposed, both of which apply to any future
throughput-graded problem.

**Why RL and not a standalone LLM training megakernel.** A fused fwd+bwd
transformer block is low-signal: at frontier scale the training step is already
a chain of big compute-bound GEMMs that cuBLAS/FlashAttention saturate, so there
is nothing to fuse; the only people who care about fused training kernels are
small-model fine-tuners, and Liger-Kernel already owns that. The training that
*is* overhead-bound — tiny nets, millions of tiny steps, env↔learn ping-pong —
is RL. So the "training megakernel" that matters lives inside the RL loop.

**Why throughput, not roofline.** An RL step is control-flow / launch-overhead
bound and has no clean FLOPS ceiling, so peak_fraction-vs-FLOPS is meaningless.
Score `achieved_sps / peak_sps` instead, with algorithm, hyperparameters, and
total step budget fixed and seed-determined — "fastest time to train" then
collapses to "fastest to run N steps".

**Correctness is the learned return level, not allclose.** A different kernel
will never reproduce the reference trajectory bit-for-bit (RNG stream and float
reduction order differ), so check.py trains both from scratch on several seeds
and requires the solution's final-window mean return to land in a band around
the reference's. Validated by codex's honest run: its independent RNG
reproduced the return level closely but not bit-exact (3.985 vs 3.998), which
allclose would have failed.

**DEMONSTRATED HOLE — throughput was credited against the nominal budget, not
actual work.** A wrapper that ran *half* the iterations and padded the returned
curve to full length with the converged value passed check.py (the reference
saturates by ~iter 20, so the final-window mean stays in band and the curve
still climbs from its early baseline) and scored 2x the honest reference floor
for half the work. `benchmark.py` divided `TOTAL_ENV_STEPS / wall_time`,
trusting the solution to have run the budget. If a throughput problem ever comes
back, the harness must own the work accounting — count actual steps inside the
timed region — or grade time-to-threshold instead of steps-per-second.

**Related, untested:** the band pins the outcome, not the algorithm. Nothing
checked that PPO (clip/GAE/stochastic sampling) was used; on a learnable enough
env a greedy controller could clear the band faster. A throughput metric
structurally rewards doing less.

## 2026-09-02 — agy (Antigravity CLI) harness, gemini-3.8-flash-high on Verda RTX PRO 6000

New harness `agy` in run_hard.sh + scripts/lib/run_harness.sh and a stream-json
parser (src/viewer/parsers/agy.py, events init/step_update/result; completion
marker `"event":"result"`). Flag-order trap: a bare `--print` swallows the next
flag as the prompt, so `-p "$PROMPT"` goes last. Auth is OS-keyring OAuth; on a
headless box run the sign-in inside tmux, capture the URL with a 1200-col pane,
paste the user's code. The GEMINI_API_KEY provider path is dead (project quota
0). Box: Verda kb-agy-rtx (default profile, 1x RTX PRO 6000 Blackwell Server
Edition, driver 580, cu130 torch 2.11). Run 20260902_201226_agy_gemini-3.8-flash-high_02_kimi_linear_decode
launched 20:12Z, unlimited budget, bwrap sandbox; ~/cuda_queue.sh then runs the
four cuda problems in sequence on the same GPU. Results pending.

## 2026-09-03 — gemini-3.8-flash-high cells published (RTX PRO 6000 + H100), muse-spark-1.3 queued

Both agy mega cells are on the board after isolated regrade and a same-buffer
overwrite probe on the box: RTX PRO 6000 Blackwell 2.7406 (in-run 2.752) and
H100 SXM5 2.0676 (in-run 2.1075), both `verdict: clean`,
`megakernel_authentic: true` (one cooperative launch per step, 31 and 14
grid.sync sites). The H100 run carries gpu marker `H100` so it lands on the
existing H100 tab (the board self-normalises against baseline.py on the same
box; the annotation says SXM5). Sandbox gap seen in the H100 trace: bwrap
hides results/, outputs/runs and DEVLOG but not src/ or scripts/, and the agent
read src/eval/megakernel.py and ran scripts/megakernel_evidence.py on itself.
Not gaming (PROMPT.txt announces the judge), but those paths should be
tmpfs-hidden too. The problem prompt is worded for the RTX PRO 6000 even on the
H100 box, as for every H100 mega cell. Publish commit deeea1f; traces pushed to
Infatoshi/kernelbench-mega-traces. New harness `muse` (Meta Muse Code CLI,
`muse exec --json --yolo ... --model muse-spark-1.3 --reasoning-effort ultra`,
parser src/viewer/parsers/muse.py) runs 02_kimi_linear_decode on both boxes
after the gemini queues; results pending.
