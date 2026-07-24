# KernelBench-Mini — DEVLOG

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
