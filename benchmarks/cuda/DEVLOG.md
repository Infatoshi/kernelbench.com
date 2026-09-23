# KernelBench-CUDA — DEVLOG

## 2026-09-22 — Opus 5.5 CUDA sweep, audit, and resume failures

Four Opus 5.5 `[xhigh]` cells on tetra were correct in sequential isolated
regrades on RTX PRO 6000: 01 Fused MoE 0.1036, 02 Native Sparse Attention
1.0957 (0.090 ms geomean across six shapes), 03 MegaQwen Decode 0.0739,
04 Grid + MinGRU 0.7938. The four annotation YAMLs are the source for
verdicts and details. 01 is clean; 02-04 are interesting because the operator
resumed the same Claude sessions twice after GPU drops. Two copies overlapped
in each workspace. 03's copies explicitly exchanged profiling findings;
04's second copy touched scratch files but reported leaving the final
solution untouched. Do not call the 02-04 development traces one agent's
hill climb. Their final kernels are genuine and their isolated regrades are
usable, with the operator overlap disclosed.

The `template_mutated` flags on 02-04 were harness false positives. Resume
mode copied `repo/src` into an existing `repo/src` and likewise copied into
an existing `trusted_src`, creating nested `src/src`. The mutation logs name
only those extra directories. The seven graded problem files, the top-level
trusted source, and the archived final solutions matched their canonical
copies byte for byte. The harness now preserves the existing workspace src
and original trusted snapshot on resume, avoiding the nesting while retaining
evidence of any pre-interruption edits. The original `result.json` flags remain
as raw history; the annotations explain and override them for publication.

Same-buffer overwrite probes on a quiet tetra GPU 0 passed the checker gate
for all four final kernels: cos(out1,out2) was -0.0141, -0.4529, 0.4422 and
-0.9821 for 01-04; cos(reference,solution) was at least 0.99998. In-place
weight changes also invalidated 03's packed weights and 04's prepared weights.
The first probe spent about 89 seconds compiling its saved CUDA extension;
the other three completed in about 8, 10 and 3 seconds with their cached
builds. Future audits should reuse the run's extension cache.

The check-only NSA restamp under the revised RTX check finished 15 PASS,
4 FAIL across 19 archives. All four FAIL cells were already incorrect before
this pass, so no previously passing score was newly withdrawn. Only their
46,371 bytes of updated `result.json` files were pulled from 11.18 GB of
remote archives. The four new Opus archives were also pulled thin; their
combined full size on tetra was 4.54 GB. Native Claude sessions and stream
transcripts were retained for the audit and HF export. The CUDA trace
converter now selects the complete native session across resumes and removes
private home/config/environment tool calls together with their paired results
before redaction. Four exact staged exports had zero redaction findings, were
uploaded in HF dataset commit `e224d2df9f46de5af45d421bdb94fd8b63dbad95`,
then downloaded and SHA256-matched. `kb publish cuda` and its site gates passed
locally; the site now presents Native Sparse Attention in measured milliseconds
because its dense-equivalent roofline is not a useful headline.

Live release: `654b3cc` on `origin/master`; GitHub's Vercel status reported
success. The live `/cuda` board shows all four audited Opus cells, all four
served kernel files SHA256-match `public/runs/`, and all four HF trace blob
pages return 200. The three-part CUDA post draft and inspected dark-mode
images are in `media/posts/audited/opus55-cuda/` (`00_post.txt`, `01.png` to
`03.png`). It is unsent and still has `FILL-IN` for Elliot's paragraph. The
post names the overlapping resumes; it does not present 02-04 as one-agent
hill climbs.
At 2026-09-22 22:07 MDT, the same three-post thread was saved in Chrome as an
unsent draft for `@elliotarledge`. The X Drafts list showed the opener with
"2 more posts" and three uploaded images (results, latency, design diagram).
`FILL-IN` remains in post 1 for Elliot's paragraph. No CUDA post was sent.

## 2026-09-22 — 02_deepseek_nsa: large_qkv dropped from the RTX PRO 6000 check

The widened check (#11) withdrew four independent kernels on 02 (Grok 4.5, Grok 4.6,
Fable 5.1 at 1.0627, GPT-6 Astra) with byte-identical stats (126 bad, max 28.75) at
(1,8,8191,128) under `large_qkv`. An oracle written from PROMPT.txt, not from
reference.py, matched the reference at (1,1,2048,64) and (1,1,8191,128) and agreed
with the four kernels, not the reference, at the failing config. Every bad element
is one row, (b=0,h=6,t=4835): key blocks 49 and 35 tie for the 8th top-k slot with
fp64 importances 9.014908865 vs 9.014908415, a gap of 0.47 fp32 ULP. The reference's
per-token mat-vec picks 49, any batched GEMM over a query tile picks 35, and Grok
4.7's block-sum route makes them bit-identical and resolves on its tie-break. x8 on
q and k makes logits 64x and the softmax a hardmax, so the swapped block flips the
row from +17.25 to -11.50. At nominal scale the same row diverges by 0.082 inside
tolerance. The spec fixes no accumulation order, so neither side is wrong; the case
was a selection-tie detector, not a numerics test. check.py now skips `large_qkv`
(problems-rtxpro6000 only; the H100 decks are untouched). This changes the graded
surface for 02 only, so every published RTX 02 cell needs a check-only restamp.
The DeepSeek V4 Flash 02 withdrawal (96,577 bad elements past row 2007) is a real
failure and is not affected.

## 2026-09-17 — check 01/02 at the shapes they are graded on (deck correction, cells stale)

**This changes `check.py` on a published deck, so every cell below is stale
until re-checked. The numbers on the board for 01 and 02 were produced under
the old, narrower check and must be re-validated or withdrawn.**

A review of the grading surface found that neither problem verified the shapes
it scores:

- `01_glm52_fused_moe` capped the check at `T = min(T, 256)` while `benchmark.py`
  times T = 4096, 4127, 1, 8192, 512, 1000. The tile-config branches the kernel
  uses at prefill sizes never ran under `check.py`.
- `02_deepseek_nsa` hardcoded two check shapes at S = 256/384, both D = 64,
  while the graded sweep runs S to 8192 and includes `{S: 8191, D: 128}`. D = 128
  was never verified anywhere, and at those short lengths every block is
  selected, so the block-selection path that *is* the operation at long context
  was never exercised.
- No `benchmark.py` in this deck (or hard's) calls `allclose` or
  `check_correctness` — verified: 0 hits across 10 files. So a graded number
  proved "correct at T<=256" and "fast at T=8192" as two separate facts, with
  nothing tying them together. A shape-conditional fast path is exactly the
  branch `reward_hack_lint.py`'s regexes cannot see.

Both checks now iterate the graded `shapes.SHAPES`:
- 01: all 6 shapes at their own T, 3 seeds each (42/123/456), as before.
- 02: all 6 shapes, one seed (42). One shape at one seed, not six at two, is
  the same trade 03's long-ctx spot check made; the reason here is cost, not
  correctness. The reference loops `B*H*S` tokens in Python with a per-token
  block loop and sort — 310,632 iterations across the six shapes, ~250s at a
  pessimistic 800us/iteration against the 1800s `KBH_CHECK_TIMEOUT_SECONDS`
  budget. If a box measures it over budget, drop to the three longest shapes
  rather than reinstating a short-shape cap.

Affordability for 01 was measured, not assumed: dense-equivalent work at
T = 8192 is ~3.7 TFLOP and the reference is fp32, so the cap was a leftover
from an earlier, slower reference rather than a live constraint.

**Backfill required.** 15 published 01 cells and 14 published 02 cells were
scored under the old check. Re-run each archived `solution.py` through the
current `check.py` in check-only mode, on an idle RTX PRO 6000 (the hardware
the numbers were produced on):

```
cd benchmarks/cuda
KBH_REGRADE_CHECK_ONLY=1 KBH_REGRADE_DECK=problems-rtxpro6000 \
    scripts/regrade_sequential.sh $(python3 ../../scripts/check_publish_gates.py --list-stale cuda)
```

`regrade_sequential.sh` restores `reference.py sota.py shapes.py problem.yaml
check.py benchmark.py PROMPT.txt` plus `src/` and the locked project from the
canonical deck, so the corrected check applies and any agent edit to the graded
surface is reverted and reported. `KBH_REGRADE_CHECK_ONLY` matters: without it
the regrade also replays `benchmark.py` and overwrites `peak_fraction` from the
fresh timing, so a "re-check" on any other box or thermal state would silently
re-rank the board. In check-only mode the published timing and `benchmark.log`
are kept, `graded_surface_sha` is stamped, and a cell that now FAILs has its
`peak_fraction` voided: it is withdrawn, not re-run, and gets an annotation
explaining which shape broke it. `--list-stale` enumerates exactly the cells
publish gate E refuses, so the same command also stamps the cells whose check
did not change.

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

## 2026-09-03 — gemini-3.8-flash-high on RTX PRO 6000 (published) and H100 SXM (held)

agy (Antigravity CLI) cells, all isolated-regraded and overwrite-probed on the
box. RTX PRO 6000 board: 02_deepseek_nsa 0.0958 clean, 03_megaqwen_decode 0.0426
clean, 04_grid_mingru_sps 0.3637 interesting (exact-fp32 path only for the
num_envs <= 256 shape check.py runs, TF32 on graded shapes, but the strict
oracle matches positions/rewards/logits at every graded shape and seed, same
bar as the qwen3.8-max 0.2848 cell). 01_glm52_fused_moe 0.0918 is
`verdict: contamination` and unpublished: the agent grepped every prior
peak_fraction in results/annotations for the problem and opened the 0.2787
top-scorer's annotation before coding. Root cause: scripts/lib/run_harness.sh
has no bwrap sandbox (mega's run_hard.sh does), and `kb contamination` only
looks for outputs/runs references, so annotation reads pass it. Mitigation on
the boxes was moving results/ and DEVLOG.md out of the tree and pulled run dirs
out of outputs/runs before the later cells; repo fix pending (port the KBH_SBX
block, extend kb contamination to results/annotations reads). Publish commit
deeea1f (manifest published_runs.json += the three clean run ids); traces on
Infatoshi/kernelbench-cuda-traces. H100 SXM5 cells (deck problems-h100sxm,
archives in outputs/runs-h100sxm, gitignored): 01 0.0805 contamination (same
annotation reads), 02 0.0417, 03 0.049, 04 0.3645; there is no H100 SXM cuda
board, so these stay unpublished until one is decided. Bench-level fixes noted:
check.py should exercise a long-ctx shape for 03 (both gemini cells fill the KV
cache with noise above 8192, like the deepseek-v4-pro cell) and a graded-shape
fidelity check for 04.
