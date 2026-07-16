# KernelBench-CUDA — DEVLOG

## 2026-07-16 — Pre-debut deck repairs: torch 2.13 init fixes, numeric stress, 03 long-ctx

Caught by the Grok 4.5 cell audits before the first publish (legal because the
deck is still unpublished):

1. **torch 2.13 rejects CPU generators on CUDA tensors.** Two template sites
   hard-crashed independent of any solution: `04 reference.reset_parameters`
   (check.py calls it after `.to(device)`) and `03 check.py _reinit`. Grok's 04
   run tripped `template_mutated` by fixing the first with a bit-exact
   CPU-draw-then-`copy_` hunk (audit verified `torch.equal` against the
   original stream); Grok's 03 solution monkey-patched
   `torch.nn.init.normal_` at import to survive the second (also bitwise
   faithful). Both templates now carry the CPU-draw-then-copy pattern at the
   source level — identical generator streams — so future solutions need
   neither workaround. The 04 run was invalidated by the guard despite clean
   intent (verdict `interesting`, not reward hack); rerun launched on the
   fixed deck. Post-hoc, its solution passes the full strict battery and
   benchmarks geomean peak_fraction 0.3728.
2. **Numeric stress wired into the deck.** `_CASES` now covers 01 (hidden
   ×1e-2 / ×8), 02 (qkv ×1e-2 / ×8), 04 (obs+state ×1e-2 / ×8, policy_forward
   section only — env_step/run synthesize inputs internally and are
   position-exact). 03 gets NO scale cases by construction: `run()`
   synthesizes inputs from the seed (nothing external to scale) and RMSNorm
   makes the stack weight-scale invariant, so input/state scaling cannot bite.
3. **Two tolerances calibrated against measured noise, not guesses.** 04
   `small_obs_state`: hard's `_TINY_FP32` atol 1e-7 is below the fp32
   machine-epsilon floor for this pipeline — the audited kernel measures
   4.6e-7 reorder noise (44/196608 bad at atol 1e-7), so the case uses
   atol 1e-6 (≈3x margin, still ~100x tighter than a wrong-gate signature).
   01 `large_hidden`: at 8x hidden, output magnitudes reach ~370 (mean ~57)
   and even an idealized bf16 pipeline (fp32 matmuls, bf16 silu*up
   intermediate) measures max_abs=2.0. The final case is atol 1.5 / rtol 5e-2:
   the smallest point passing the audited kernel at every check variant
   (min margin +0.49; floor sim +1.03) while wrong/cached kernels diverge by
   O(mean magnitude) ~57. Floor/scan harnesses: `backfill/` scratch
   (ephemeral). **Gotcha worth remembering:** the pass predicate is
   `torch.allclose(ref, sol)` whose rtol multiplies |sol| (asymmetric), but
   check.py's diagnostic `bad=` count uses |ref| — a scan built on the
   diagnostic semantics undercounts and picks atol 1.0, which then fails the
   same kernel at seed 123 by −0.006. Calibrate against the actual predicate,
   across ALL seeds the check runs (42/123/456), not just seed 42: the failing
   element moved between seeds.
4. **03 hardening is a long-ctx spot check.** check.py used to top out at
   ctx 512, so numerics at 2k+ were verified by no one (audit finding on the
   passing Grok cell). Measured on that kernel: max_abs is a CONSTANT 0.0625
   at both ctx 2048 and ctx 8192 (a per-layer bf16 rounding constant, not
   accumulation), reference wall time ~13s at 8k. check.py now runs one
   (seed 42, ctx 8192, decode 16) comparison at the same 0.08 tolerance.
5. **04 peak_sps stays 150M**, derivation now documented in problem.yaml: the
   first audited fused CUDA baseline sustains ~50-61M sps (pf 0.33-0.41), so
   150M is an aspirational roofline-proxy ceiling, not a best-known-kernel
   number. Do not fit the ceiling to the kernel.

All published cells are re-validated against this final surface before the
board debuts (backfill checks run the archived solutions through the current
check.py + numeric_stress.py; 01/02/03/04 PASS, with 04's fresh rerun pending).

## 2026-07-15 — Latency-anchored relative scoring (standing metric decision)

Peak fraction is the wrong headline for cells whose roofline ceiling is
structurally unreadable. Two live examples: hard's `05_topk_bitonic`, where the
~0.02 ceiling is launch-overhead-bound for EVERY model, and cuda's
`02_deepseek_nsa`, where the dense-equivalent FLOPs baseline can never be
attained by a correct *sparse* kernel (Grok 4.5's first clean pass scored
0.0177 and "0.0177" tells a reader nothing). For those, grade and display on
**milliseconds**, not peak fraction.

The design (user decision 2026-07-15; implement per this when wiring publish):

1. **ms per shape stays the ground truth.** result.json already records
   shape-by-shape times; the leaderboard builder must carry per-shape ms for
   the solution variant so every displayed number is reproducible from the
   archive.
2. **Persisted headline = geomean speedup vs a FROZEN anchor.** Anchor is the
   deck's eager torch reference (`reference.Model`), timed per shape at deck
   publication and committed alongside the problem (eager_ms per shape).
   Score per shape = eager_ms / solution_ms; problem score = geomean over the
   shape sweep. The anchor is frozen at publication with the deck, so a
   published cell's score never changes afterward — same frozen-board rule as
   prompts.
3. **Best..worst linear span is a PRESENTATION layer only.** On the site, also
   show the board-relative position: with `t` a cell's geomean ms,
   scaled = (worst - t) / (worst - best) across published models
   (worst -> 0, best -> 1; degenerate span 0 -> all 1). Computed at site build
   from the current board, NEVER persisted into leaderboard.json. When a new
   best lands, the span shift is a site render change, not a re-grade of
   historical cells — the same rule that keeps labs trusting the board.
4. **peak_fraction is demoted to context**, shown as a secondary column
   ("roofline attainability"), never the sort key on structurally unreachable
   ceilings. It stays the headline where the ceiling is real (MoE, GEMM).
5. **Cross-bench reach: presentation only.** hard's topk cell gets the same
   ms + span DISPLAY treatment; hard and mega graders, prompts, and
   leaderboard.json schemas stay frozen. No hard/mega cell is re-graded.

## 2026-07-15 — Why a third single-GPU bench (and why not touch Hard/Mega)

Hard and Mega are live boards that labs already care about. Changing a prompt,
tightening a forbidden list, or swapping "CUDA or Triton" for "CUDA only"
would re-grade historical cells and look like moving the goalposts. So CUDA
is a **new, isolated deck** with the same harness DNA, not a Hard fork of the
published problems.

### Thesis

We want an objective read on **CUDA kernel writing**, including:

1. Problems that map cleanly to CUDA but are *annoying* in raw CUDA (reductions,
   online softmax, shared-memory layouts) while being *easy* in Triton.
2. Optional megakernel / sim paths (grid env + multi-layer MinGRU policy) where
   fusion is allowed, not mandatory — grade sustained steps/sec and see what
   strategy the model picks.
3. Explicit sidecars for **instruction following**: did the model obey the CUDA
   mandate, or did it sneak back to `@triton.jit` / a DSL / a pure torch op?

Hard already *allows* Triton (and many winning cells are Triton). That is fine
for "best kernel on the metal." It is the wrong axis for "can you write CUDA."

### v0 deck (retired easy ops)

- `01_rmsnorm_residual` / `02_online_softmax` — Grok 4.5 smoked them clean
  (real CUDA, language gate pass). User decision 2026-07-16: **drop them**.
  Too tutorial / Triton-blog-post shaped for a "super hard CUDA" board.
  Annotations kept; problem dirs removed.

### v1 deck (perfect stack, 2026-07-16)

- `01_glm52_fused_moe` — GLM-5.2 MoE (E=256, top_k=8, 1 shared) fused gate|up pack; no Mixtral.
- `02_deepseek_nsa` — NSA-inspired block top-n + sliding window + sparse attn.
- `03_megaqwen_decode` — Qwen3-0.6B geometry (4-layer slice); improve Infatoshi/MegaQwen.
- `04_grid_mingru_sps` — RL sim SPS; fusion optional.

Deck is frozen at four. Spec-decode tree attention was floated as a fifth and
rejected (user decision 2026-07-15); do not re-pitch it.

Smoke 2026-07-16: all four references run on PRO 6000; MoE `check.py` PASS with
minimal CUDA silu_mul solution (language gate `cuda_raw`); Triton gate fails as
designed.

Shape discipline (Hard FP8 technique): MoE/NSA include misaligned T/S (4127,
8191, …) and serving tails (T=1); score = geomean. MegaQwen: prefill untimed,
decode-only timed at ctx ∈ {2k,8k,32k,128k}; pure numeric last_hidden (no tokens).

### Language gate

`src/eval/cuda_language.py` is the hard fail. Numeric PASS with Triton still
fails the problem. Report goes to `cuda_language.json` for future leaderboard
columns (`framework`, `triton_cheat`).

### Harness

Rsynced from Hard (scripts + src + kbh CLI). Package renamed
`kernelbench-cuda`. Do not edit Hard/Mega prompts from this workstream.

### First smoke models

Grok 4.5 (fast agent) on RTX PRO 6000 for structure validation, then the
planned matrix: Fable 5, GPT-5.6 Sol xhigh, Kimi, GLM-5.2, Opus 4.8, Grok 4.5.

Launched 2026-07-15 ~18:43 (Anvil), container mode, unlimited budget; all
three finished (~37–53 min). Audits in `results/annotations/`:

| run | verdict | notes |
| --- | --- | --- |
| `...01_rmsnorm_residual` | clean | real CUDA; geomean 0.33 (skinny-row drag) |
| `...02_online_softmax` | clean | real CUDA; pf>1 = one-pass bytes formula, not a hack |
| `...03_grid_mingru_sps` | clean | CUDA + cuBLAS GEMMs + CUDAGraph; peak_sps raised to 150M |

v1 deck replaces 01/02 with grouped GEMM + flash attn + MoE scatter; keeps 03.

### Explicit non-goals this session

- Do not modify `benchmarks/hard` or `benchmarks/mega` problem surfaces.
- Do not claim a published CUDA board until cells are audited.
- Full Craftax classic parity is deferred; 03 is the SPS / MinGRU probe.
