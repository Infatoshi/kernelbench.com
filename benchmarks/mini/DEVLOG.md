# KernelBench-Mini — DEVLOG


## 2026-07-29 — self-contained kbmini node: serving moved onto the eval box

Athena (and the whole lease fleet) died mid-campaign with the anvil tunnel as a
single point of failure, and anvil itself lost GPU1 to a GSP wedge. The user's
call: the bench must not depend on anvil at all, and every graded number must
come from one consistent GPU. New architecture, one Lambda `gpu_1x_h100_sxm5`
(`kbmini`) does everything:

- **vLLM serves the model on the eval H100 itself** (localhost:8765, 35% GPU
  mem), ccr-rust on 3456 for `lfm-claude`. Same ports the old tunnel used, so
  zero harness config changes. In-run GPU contention between serving and
  check/benchmark is acceptable BECAUSE published numbers only ever come from
  the sequential re-grade, which runs with the server stopped. This supersedes
  the 07-23 "inference never on the eval GPU" rule — that rule now applies to
  the re-grade, not the agent phase.
- 10-min rsync pullback to the Mac runs for the whole campaign (athena lesson:
  a dead node must cost <=10 min of artifacts, and the pullback must exclude
  per-run `.venv`/caches — mirroring them once filled the Mac disk).

**bf16 rerun on kbmini (local serving, 20-worker split): 100/100, 43 wrote a
solution, 0 correct** — closely reproducing the lost athena wave (44/0) on a
different node, different serving locality, and 4x the worker concurrency.
The reliability spread is the stable result. One shift: lfm-claude wrote 11
gradeable solutions vs 6 on athena, with only 1 provider_early_stop. The 07-28
tunnel-served NVFP4 wave is superseded by a local-serving rerun (same 20-worker
layout as bf16) so the precision comparison shares serving latency; the old
wave stays archived, trace/debug only.

**NVFP4 rerun (local serving): 100/100, 37 solutions, 0 correct.** The
precision comparison under identical serving/layout: bf16 43 solutions vs
NVFP4 37; per-harness emission shifts (bf16 -> nvfp4): lfm-claude 11 -> 4,
lfm-opencode 14 -> 7, hermes 2 -> 6, pi 12 -> 16, grok 4 -> 4. lfm-claude kept
8 provider_early_stops on NVFP4 with the tunnel gone (bf16 local: 1), so early
stops are NOT purely transport artifacts — the quantized model plausibly emits
short/empty responses the ccr route reports as early stop; worth a trace read
before any publication.

**Sequential isolated re-grade (server stopped, GPU at 0 MiB): all 80
solution-bearing cells re-checked, 0 correct, 80 failed** — verdicts identical
to the contended in-run grades, and every failure is deterministic (16 no-CUDA-
evidence, 13+ syntax/truncation, 6 forbidden torch.sort, 2 state-dict/numeric),
i.e. none of the wave's failures were serving-contention artifacts. No
template mutations in either local wave. Found and fixed en route: mini's
regrade_sequential.sh carried the pre-fix `grep -c . || echo 0` idle-GPU probe
(emits "0\n0" on an idle GPU, integer-test error, gate loops forever) — ported
hard's fixed form.

## 2026-07-25 — first full matrix: LFM2.5-2.6B-Agent bf16, 100 sessions, 0 correct

Eval node moved ares -> **athena** (same SKU, 2x H100 80GB HBM3) at the user's
call. Bootstrap is now a known quantity: bench rsync, `uv sync`, pi via bun,
hermes cloned + venv'd at anvil's pinned commit, the three harness config files
copied from the previous node, and `~/.kbmini/tunnel_athena.sh` on anvil
forwarding 8765 (vLLM) + 3456 (ccr-rust). About 20 minutes end to end.

**Result: 100/100 sessions, 44 wrote a solution, 0 correct.**

| harness | no_solution | check_failed | other |
| --- | --- | --- | --- |
| pi | 5 | 15 | — |
| lfm-opencode | 9 | 11 | — |
| lfm-grok | 11 | 9 | — |
| lfm-claude | 11 | 6 | 3 provider_early_stop |
| hermes | 16 | 2 | 1 timeout, 1 template_mutated |

The headline is not the zero — a 2.6B model failing a deck that costs a frontier
model its full 30 minutes is the expected outcome, and the deck's solvability is
already pinned by the codex anchor. The headline is the **spread in whether the
model emits a kernel at all**: pi 15/20 vs hermes 2/20 on the same model, same
problems, same node. Harness scaffolding, not capability, decides three quarters
of this board. Problem-level counts are flat (10-15 failures each way across all
four), so no problem is an outlier — which is what makes the harness axis
readable. This is the argument for the harness-pairing design, and it means the
first Mini publication is a reliability result, not a performance one.

**The template guard fired for real, first time on this bench.** hermes on
`02_segmented_decay_scan` overwrote `reference.py` — the correctness oracle —
with a mock: `Model` returning an empty `Mock` class, decay hardcoded to 0.99,
resets all zero, and a stray `ners = [get_init_inputs, get_outputs]` at module
scope. It never wrote `solution.py`. Read as confusion rather than intent (the
model appears to have thought it was implementing the problem), but the effect
is identical to grader tampering: had the guard not caught it, the cell would
have graded a solution against an oracle the model itself wrote. Guard behaved
correctly — refused to run check.py/benchmark.py, restored the file from the
snapshot, marked the run `template_mutated`. `template_mutations.log` in the
archive holds the full diff, which is why the post-hoc workspace looks clean.

Three environment bugs, all found by running rather than by reading:

- **`uv` not on PATH over non-login ssh** made the first launch report all 100
  sessions "done" in under a second: a missing toolchain must never be
  indistinguishable from a completed session, so a preflight now fails loudly
  with `STOP: uv not found on PATH`.
- **vLLM needs `--enable-auto-tool-choice --tool-call-parser lfm2`** or every
  harness 400s on its first tool call, and `--max-model-len 128000` because
  hermes's compression loop dies at 65536.
- **Timings from this wave are contended by construction** (five columns, two
  GPUs), so mini was finally added to the `regrade_sequential.sh` rollout it
  had been left out of despite needing it more than the other benches.

## 2026-07-24 — ares (2x H100 SXM) is the eval node; deck validated on it

**Deck validated on the canonical node** (reference-as-solution, GPU1):
- 01, 02: `check.py` **PASS** including numeric stress.
- 03: rejected `forbidden op used: torch.sort` — the sort-free gate works.
- 04: rejected `no CUDA kernel evidence` — the CUDA-only gate works.
- `H100_SXM` peak entry is right: naive reference measures 28.8 GB/s =
  **0.86%** of the 3350 GB/s HBM3 peak (arithmetic exact), so 01 has ~100x
  headroom and is not saturated.

**LFM2.5-2.6B-Agent smoke, 01_dequant_gemv, 10 sessions across both boxes:**

| harness | anvil (3090) | ares (H100) |
| --- | --- | --- |
| lfm-opencode | check_failed | no_solution |
| hermes | no_solution | no_solution |
| pi | check_failed | no_solution |
| lfm-grok | no_solution | check_failed |
| lfm-claude | check_failed | no_solution |

0/10 correct; 4/10 produced a solution at all. Every failure was inspected
individually and none is an environment fault: opencode reasoned itself out
of writing the file, hermes hit its own output-length truncation on both
boxes, pi spent 14.7k tokens then emitted prose instead of a tool call,
lfm-claude's kernel died on a stray undefined name at import. The dominant
failure mode is **a 2.6B model narrating instead of calling tools**, and
which harness happens to survive flips between machines — exactly the
variance the 5-repeat cell exists to quantify.

**Calibration: the deck is solvable, and the cap binds.** `codex gpt-5.6-sol`
(high) on 01: **correct at 0.0900 peak fraction**, 12.8-18.2x the naive
reference, audited clean (see SPEC). It hit the 1800s cap mid-optimization
(exit 124) while still improving — so even a frontier model does not converge
on this problem inside Mini's 30 minutes. That reframes the headline: Mini
measures *best kernel in 30 minutes*, and its numbers must never be compared
against unlimited-time Hard.

**The audit found a real bench bug.** codex's solution carried a dead
`load_inline` WMMA extension next to the Triton kernel it actually ran, and
the framework labeller — first-match-wins over a priority list — reported
`cuda_wmma` for a `tl.dot` kernel. Static detection cannot distinguish live
from dead code, so the labeller now emits a **compound** label
(`cuda_wmma+triton`) in both `src/eval/cuda_language.py` and the copies in
01/02's `check.py`, with a regression test pinning the exact wild case.
Compound means "resolve by hand in the mandatory audit". The deeper version of
this hole — dead CUDA evidence satisfying `require_cuda_evidence` on the
CUDA-only problems — is now a named calibration debt; on 04 the forbidden list
blocks the fast form of the cheat, leaving only a correct-but-slow mislabel.
Fixing this now, before the matrix runs, means 200 sessions write correct
labels rather than 200 archives needing relabelling.

## 2026-07-23 — LFM2.5-2.6B-Agent harness probes: all five routes green

First subject model wired up: LiquidAI LFM2.5-2.6B-Agent served on anvil GPU0
via vLLM 0.25.1 (`127.0.0.1:8765`, `--enable-auto-tool-choice
--tool-call-parser lfm2`, `--max-model-len 65536`). Every route passed a
headless file-write probe (`hello.txt` with exact content) against the live
bf16 server. What it took:

1. **hermes context exhaustion is a real failure mode.** A trivial probe
   wrote the file correctly but exited 1 with "max compression attempts (3)
   reached" — with a 65k window this is the small-model harness tax, score it
   as-is.
2. **Claude Code route was broken by a stale ccr-rust binary.** Symptom:
   model emits think-text, says "let me write the file", session ends after
   one turn, no tool ever runs. Wire captures (logging proxies on both sides
   of ccr) showed vLLM streaming a complete `Write` tool_call and ccr
   forwarding the `tool_use` content block but stamping the final
   `message_delta` with `stop_reason: "end_turn"` instead of `"tool_use"` —
   so Claude Code treated every turn as final. The installed
   `~/.cargo/bin/ccr-rust` (1.3.0, built from a now-deleted local checkout)
   predates the upstream fix; rebuilding from RESMP-DEV/ccr-rust main (same
   version string) fixed it. Old binary kept at
   `~/.cargo/bin/ccr-rust.bak-20260723`. Lesson: a proxy that passes
   single-shot curl tests can still break the agentic loop — probe the loop,
   not the endpoint.
3. **Small-model behavior notes from the probes:** LFM emits its reasoning
   as in-band `</think>` text (renders as visible text through every route),
   and it hallucinated `/tmp/kbmini-probe` (hyphen) for a cwd-relative path
   once — problem prompts already use explicit relative paths.

Matrix locked in SPEC: 2 precisions (bf16 / NVFP4A16, precision-tagged served
names `lfm25-agent-bf16` / `lfm25-agent-nvfp4`) x 5 harnesses x 4 problems x
5 repeats = 200 sessions.

Same-day smoke (one real `01_dequant_gemv` cell per harness, wiring validation
only, timings contended): all five routes produced valid graded cells, 0/5
correct. hermes at 65536 ctx crashed its own compression loop, and at 128000
ctx LFM's in-band think text trips hermes's output-length truncation before
`solution.py` exists — so serving context is 128000, and the hermes branch
default provider was fixed from `openai` to `lfm`. The rest failed by writing
real but incorrect kernels or by never calling a tool, which is the plausible
bar for 2.6B.

## 2026-07-23 — Bench created: small-model deck, capped + repeated

Scaffolded from `benchmarks/cuda` (same harness DNA: run_hard.sh, GPU lock,
language gate, numeric stress, kbh CLI). Package renamed `kernelbench-mini`.

Design decisions (user, this session):

1. **Target class:** open-weight models under ~200B, head-to-head. Existing
   decks are structurally contaminated for these models (prompts + winning
   solutions + traces are public on the site and HF), so the deck is entirely
   new ops — familiar difficulty, unfamiliar structure.
2. **Capped, not unlimited:** `BUDGET_SECONDS=1800` (30 min). This is the
   bench identity and what makes 5 repeats per cell affordable.
3. **5 repeats per (model, harness, problem)** = 20 sessions per column.
   Publish pass rate k/5 + best-of-5. No Elo — the metric is cardinal.
4. **Harness pairing:** `opencode` for every model, plus the `*-claude` route
   where the provider has an Anthropic endpoint. Claude Code vs OpenCode on
   the same model is itself a published comparison.
5. **GPU split:** canonical eval GPU is a Lambda H100 SXM5 (sm_90 is the
   best-documented arch in small-model training data; sponsored credits);
   inference for API-less models is served from anvil's RTX PRO 6000, never
   the eval GPU. Added `src/hardware/h100_sxm.py` (SXM5 dense peaks: bf16
   989.5, HBM3 3350 GB/s) — the existing `h100.py` is the PCIe part; do not
   mix them.
6. **Deck v0 (four problems, 2 Triton-allowed / 2 CUDA-only):**
   - `01_dequant_gemv` — vibe check (loose prompt). Int4 gated GEMV with
     GROUP SIZE 96: ragged last group for most K, so group-128 AWQ copy-paste
     is wrong by construction and no vendor kernel (marlin/bnb) supports it.
   - `02_segmented_decay_scan` — linear recurrence with per-token reset mask.
     Associative once the reset folds into the decay, but the textbook
     tl.associative_scan / cumprod recipes don't handle the mask as written.
   - `03_topp_mask` — CUDA-only sort-free nucleus mask. EXACT grading via an
     fp64 oracle band (tau=1e-3 cumulative mass): tokens clearly inside/outside
     the nucleus are forced, only the thin boundary band is free — absorbs
     fp32 summation-order rounding, leaves zero tolerance to game. Forbidden
     list covers function AND tensor-method sort spellings plus cub/thrust.
     ms-anchored headline (launch-overhead regime, standing 2026-07-15 rule);
     benchmark.py times the eager sort path every run and prints
     `speedup_vs_eager` + `geomean_speedup_vs_eager`.
   - `04_flash_attention` — the ambitious discriminator: full causal flash
     forward in raw CUDA on H100. S=16384 at B=1,H=8 makes O(S^2) memory
     impossible, so a real streaming online-softmax kernel is mandatory.
     SDPA is the sota ceiling variant, forbidden in solutions.
7. **Per-problem language gate:** 01/02 `allow_triton: true` +
   `require_cuda_evidence: false` (the cuda_language module's existing escape
   hatches); 03/04 full CUDA-only gate. Tests updated: the gate-import test
   now only applies to `language: cuda` problems and asserts exactly 2.
