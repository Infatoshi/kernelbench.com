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
- ccr-rust's source is deleted everywhere (anvil binary only survives; built
  against glibc 2.43, node has 2.35). It runs under anvil's shipped loader:
  `ld-linux-x86-64.so.2 --library-path ~/.kbmini/ccrlibs ccr-rust`. Rebuild
  properly if ccr ever needs changes.
- Lambda image gotcha: no `ninja` — vLLM's KV-cache init shells out to it and
  the engine core dies with FileNotFoundError. `apt install ninja-build`.
- vLLM API drift: `WeightsMapper(orig_to_new_renamings=...)` kwarg no longer
  exists (renamed singular). `serve_nvfp4.py` now pops `.w1`/`.w3` from
  `orig_to_new_stacked` in place instead of constructing a new mapper.
- 10-min rsync pullback to the Mac runs for the whole campaign (athena lesson:
  a dead node must cost <=10 min of artifacts, and the pullback must exclude
  per-run `.venv`/caches — mirroring them once filled the Mac disk).

**bf16 rerun on kbmini (local serving, 20-worker split): 100/100, 43 wrote a
solution, 0 correct** — closely reproducing the lost athena wave (44/0) on a
different node, different serving locality, and 4x the worker concurrency.
The reliability spread is the stable result. One shift: lfm-claude wrote 11
gradeable solutions vs 6 on athena, its provider_early_stops gone — those were
tunnel/ccr artifacts, not model behavior. The 07-28 tunnel-served NVFP4 wave is
superseded by a local-serving rerun (same 20-worker layout as bf16) so the
precision comparison shares serving latency; the old wave stays archived,
trace/debug only.

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

- **`uv` not on PATH over non-login ssh.** The first launch reported all 100
  sessions "done" in under a second. `run_hard.sh` did `REAL_UV="$(command -v
  uv)"` and died on the empty result, which the sweep loop logged as a finished
  run. A missing toolchain must never be indistinguishable from a completed
  session: there is now an explicit preflight that exports `~/.local/bin` and
  fails with `STOP: uv not found on PATH`.
- **vLLM needs `--enable-auto-tool-choice --tool-call-parser lfm2`.** Without
  them every harness 400s on its first tool call. The server also has to be
  re-served at `--max-model-len 128000`; hermes hard-requires >=64k and its
  compression loop dies at 65536.
- **Timings from this wave are contended by construction** (five columns, two
  GPUs, per-GPU lock dirs). mini had been left out of the `regrade_sequential.sh`
  rollout despite needing it more than the other benches; copied in.

## 2026-07-24 — ares (2x H100 SXM) is the eval node; deck validated on it

Lambda is no longer the plan: **ares** (`ssh ares`, 2x H100 80GB HBM3,
driver 580, 5.2T free) is already-rented capacity and its GPUs are the exact
SXM part the deck declares (`hardware: [H100_SXM]`). No metered node to
forget about.

Node bootstrap: uv, **Node 22** (ares shipped none, which also unblocked its
pre-installed codex), hermes 0.19.0 (clone + venv, matching anvil), pi 0.73.1.
claude 2.1.218 / grok 0.2.106 / opencode 1.17.8 were already present.

**Inference stays on anvil** — the eval GPU must never host the model. anvil
serves LFM and reverse-tunnels two ports into ares:
`~/.kbmini/tunnel_ares.sh` (self-healing retry loop) forwards **8765** (vLLM,
OpenAI-compatible) and **3456** (ccr-rust, Anthropic shim for Claude Code).
Tunnelling ccr rather than installing it on ares sidesteps a glibc mismatch
(anvil 2.43 vs ares 2.35 — an anvil-built binary will not run there).
Verified under 5-way concurrency, which is the load the matrix produces.

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

Matrix launcher added (`scripts/launch_matrix.sh`): one worker per harness
column (never problem-major — head-of-line blocking), workers pinned
round-robin across GPUs, **each GPU gets its own `KBH_GPU_LOCK_DIR`** so
compile/check/benchmark serialize per GPU while different GPUs run truly
concurrently. Its timings are contended and not publishable; the sequential
re-grade rule still applies.

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

Operational notes: an archive costs ~4.7G once a solution triggers the graded
`uv run` (an archive-local `.venv` of torch+cu130) — reproducible from the
preserved `pyproject.toml`/`uv.lock`, so reapable after publish. On ares only
**codex** has live credentials; Claude Code's OAuth expired there and grok
401s (grok auth does not survive being copied between machines — needs a
per-box device login or `XAI_API_KEY`). None of that blocks the LFM matrix,
which routes entirely through the tunnelled local endpoint.

## 2026-07-23 — LFM2.5-2.6B-Agent harness probes: all five routes green

First subject model wired up: LiquidAI LFM2.5-2.6B-Agent served on anvil GPU0
via vLLM 0.25.1 (`127.0.0.1:8765`, `--enable-auto-tool-choice
--tool-call-parser lfm2`, `--max-model-len 65536`). Every route passed a
headless file-write probe (`hello.txt` with exact content) against the live
bf16 server. What it took:

1. **Serving context raised 8192 -> 65536.** hermes refuses to start below
   64k context, and Claude Code's default request shape assumes big budgets.
   The model supports 128k positions, so this is in-spec; the throughput
   runbook's 8192 was a benchmarking choice, not a model limit.
2. **pi hangs with sessions.** `pi --mode json -p` (defaults) times out with
   zero output; `--no-session` fixes it in both text and json modes. The
   `pi)` branch now passes `--mode json --no-session`.
3. **hermes context exhaustion is a real failure mode.** A trivial probe
   wrote the file correctly but exited 1 with "max compression attempts (3)
   reached" — with a 65k window this is the small-model harness tax, score it
   as-is.
4. **Claude Code route was broken by a stale ccr-rust binary.** Symptom:
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
5. **Two cosmetic vLLM 400s through ccr remain, both harmless:** Claude
   Code's session-title side request sends `tools: []` (vLLM rejects empty
   arrays; the request is non-essential), and before the `maxtoken` clamp
   Claude Code asked for 64000 output tokens (`CLAUDE_CODE_MAX_OUTPUT_TOKENS`
   is ignored by CC 2.1.218). Fixed by adding
   `["maxtoken", {"max_tokens": 8192}]` to the ccr transformer chain
   (`scripts/ccr-lfm.config.json`); the previously listed "openai"
   transformer does not exist in ccr-rust and was silently skipped.
6. **Small-model behavior notes from the probes:** LFM emits its reasoning
   as in-band `</think>` text (renders as visible text through every route),
   and it hallucinated `/tmp/kbmini-probe` (hyphen) for a cwd-relative path
   once — problem prompts already use explicit relative paths.
7. **Grok Build** needed only the documented `[model."<id>"]` config.toml
   block (`api_backend = "chat_completions"`); worked first try, 2 turns.

Matrix locked in SPEC: 2 precisions (bf16 / NVFP4A16, precision-tagged served
names `lfm25-agent-bf16` / `lfm25-agent-nvfp4`) x 5 harnesses x 4 problems x
5 repeats = 200 sessions.

Same-day smoke (one real `01_dequant_gemv` cell per harness, bf16 served on
anvil GPU1/3090 after Laguna took GPU0; wiring validation only, timings
contended): all five routes produced valid graded cells. lfm-opencode, pi,
and lfm-claude wrote real (incorrect) solutions — check_failed; lfm-grok ran
18 turns then ended by asking a clarifying question (no_solution); hermes at
65536 ctx crashed its own compression loop ("max compression attempts
reached"), at 128000 ctx it completes but LFM's in-band think text trips
hermes's output-length truncation before solution.py exists (no_solution).
Serving context is therefore 128000, and the hermes branch default provider
was fixed from `openai` to `lfm`. 0/5 correct — plausible bar for 2.6B;
repeats will quantify.

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

Validation this session: `uv run pytest` 56 passed; ruff clean on all new code
(19 pre-existing errors in inherited `build_v2_leaderboard.py` /
`reward_hack_lint.py`, identical in benchmarks/cuda — not touched). CPU
semantics smokes: 01 ragged-group round trip, 02 reference loop vs brute-force
per-element recurrence, 03 reference mask satisfies its own fp64 oracle bands
(incl. flat-logits stress) + nucleus mass >= p + top-1 always kept, 04
reference matches SDPA within 3e-2.

Open before first publish (see SPEC "Calibration debts"): stress-atol
calibration on real kernels, TAU validation for 03, frozen eager_ms anchor for
03, and a cheap-model smoke of all four problems end-to-end through the
harness on a Lambda H100.
