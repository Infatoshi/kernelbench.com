# KernelBench-Mini — SPEC

Machine location: canonical monorepo on the Mac at
`~/dev/sites/kernelbench.com/benchmarks/mini`. Deck development smokes run on
anvil's RTX PRO 6000; **the canonical graded GPU is a Lambda H100 SXM5**
(`gpu_1x_h100_sxm5`), provisioned per sweep via `kb lambda` and torn down after.

## Thesis

Rank **small open-weight models (< 200B params)** against each other on kernel
writing, on a **fresh, unpublished deck** they cannot have been post-trained on.
Hard/Mega/CUDA prompts, winning solutions, and full traces are public (site +
HF), so those decks are structurally contaminated for exactly this model class.
Mini's deck is new ops with a structural twist each: familiar difficulty,
unfamiliar shape — the memorized tutorial kernel is wrong by construction.

Three deltas define Mini against Hard:

1. **Capped sessions.** 30 minutes wall-clock per agent session
   (`BUDGET_SECONDS=1800` in `scripts/run_hard.sh`), not unlimited. Small
   models loop; the cap is part of the bench identity and is what makes
   repeats affordable.
2. **5 repeats per cell.** The unit of publication is
   (model, harness, problem) x 5 independent sessions. Score two axes:
   **pass rate k/5** (reliability — where small models actually differentiate)
   and **best-of-5 performance** (capability). No pairwise/Elo machinery: the
   metric is cardinal, repeats give the spread.
3. **Two harnesses per model** where routes exist: `opencode`
   (OpenAI-compatible, covers everything) and a `*-claude` route when the
   provider has an Anthropic skin. Same model under both is itself a published
   comparison; never mix harnesses inside a cell.

## Architecture

- **Eval GPU:** one Lambda H100 SXM5 per sweep. Sessions overlap at moderate
  concurrency; GPU commands serialize through `outputs/gpu.lock`.
- **Inference:** provider APIs where they exist; models without an API are
  served from anvil's RTX PRO 6000 (96 GB, vLLM, OpenAI-compatible) — never on
  the eval GPU, so kernel timings stay clean.
- **Publish-grade numbers** come from the standing mandatory sequential
  isolated re-benchmark (2026-07-19 rule): rerun each cell's best solution
  through check.py + benchmark.py alone on the quiet canonical node.

## Scoring

- 01 (memory), 02 (memory), 04 (compute): roofline `peak_fraction` vs H100
  SXM peaks (`src/hardware/h100_sxm.py` — SXM numbers, not the PCIe part in
  `h100.py`).
- 03: **ms-anchored** per the standing 2026-07-15 metric rule — headline is
  geomean speedup vs the eager sort-based reference; the `eager_ms` anchor per
  shape is frozen at deck publication; peak_fraction is context only.
- Per-model headline: correctness rate over all 20 runs, plus geomean of
  best-of-5 scores across problems. Token/cost columns from
  `scripts/summarize_runs.py` are a secondary axis ($ per passing kernel).

## The deck (frozen at four once published)

| NN | problem | language | regime | twist |
| --- | --- | --- | --- | --- |
| 01 | `01_dequant_gemv` | Triton allowed | memory | int4 gated GEMV, **group size 96** (ragged last group; no group-128 copy-paste, no vendor kernel path) |
| 02 | `02_segmented_decay_scan` | Triton allowed | memory | decay scan with **per-token reset mask** (textbook associative_scan / cumprod recipes don't apply as written) |
| 03 | `03_topp_mask` | CUDA-only | ms-anchored | **sort-free** nucleus mask; exact-integer output graded by an fp64 oracle band (tau=1e-3 mass) — no tolerance to game |
| 04 | `04_flash_attention` | CUDA-only | compute | full causal flash forward; S=16384 makes O(S^2) memory impossible; SDPA / flash-attn / Triton forbidden |

01 is deliberately the **vibe check**: its PROMPT is loose (points at files,
"show me what you do with it") where 02-04 use the structured house prompt.
The pairing measures the same model with and without full specification.

- 01/02 set `allow_triton: true, require_cuda_evidence: false` — DSLs still
  fail; framework is recorded in `framework.txt`.
- 03/04 set `language: cuda` and run `src/eval/cuda_language.py` exactly as
  KernelBench-CUDA does.

## Anti-reward-hack surface (all inherited, all mandatory)

Numeric stress cases per problem (`src/eval/numeric_stress.py`), `kb lint`
tripwire, **manual solution+trace audit per published cell** (annotation YAML),
contamination tripwire before publish, template-mutation guard. 03's exact
oracle band and 02's linear-in-x semantics shrink the tolerance-gaming surface
by design. A fresh Lambda node per sweep also shrinks cross-run contamination:
the archive on the node holds only that sweep's runs.

Audit note for 03: the launch-overhead regime plus an exact output makes
"cache the mask/threshold" the obvious cheat; the flat/peaky logits stress
cases break cached thresholds, but the per-cell audit must still empirically
overwrite the input buffer and confirm the mask changes.

## Running

```bash
# one cell (one session):
uv run kbh run opencode <model> problems-h100/01_dequant_gemv
# one full column (4 problems x 5 repeats, sequential):
./scripts/sweep_mini.sh opencode <model>
./scripts/sweep_mini.sh <provider>-claude <model>
# launch one sweep_mini.sh per model to parallelize (per-harness workers).
```

Pre-publish checklist: sequential re-benchmark on the canonical node, per-cell
audits, `kb lint`, contamination check, redaction, then publish to `/mini`.

## First subject: LFM2.5-2.6B-Agent, 5 harnesses x 2 precisions

The inaugural matrix is one model, LiquidAI LFM2.5-2.6B-Agent, served locally
on anvil GPU0 (RTX PRO 6000) via vLLM 0.25.1 at `127.0.0.1:8765`, driven
through five agent harnesses at two weight precisions:

- Precisions (one server at a time; restart between; `nvidia-smi` first):
  - bf16: `~/dev/liquidai/LFM2.5-2.6B-Agent`, served-model-name
    `lfm25-agent-bf16`
  - NVFP4A16: `~/dev/liquidai/LFM2.5-2.6B-Agent-NVFP4A16`, served-model-name
    `lfm25-agent-nvfp4` — must launch via the patched `serve_nvfp4.py`
    entrypoint (plain `vllm serve` breaks the fused w13 weight names)
  - Serve with `--max-model-len 128000` (not the throughput runbook's 8192):
    hermes hard-requires >=64k context and its compression loop crashed the
    session at 65536 ("max compression attempts reached"); at 128000 it runs
    to completion. 128000 is the model's `max_position_embeddings`.
- Harness routes (all five verified 2026-07-23 against the live bf16 server):
  - `lfm-claude` — Claude Code -> ccr-rust (port 3456,
    `scripts/ccr-lfm.config.json`) -> vLLM. Start ccr before the sweep.
  - `lfm-opencode` — archive-local `opencode.json` provider block.
  - `hermes` — Nous Hermes agent CLI, named `lfm` provider in
    `~/.hermes/config.yaml`.
  - `pi` — badlogic pi coding agent, provider in `~/.pi/agent/models.json`;
    `--no-session` is mandatory (session persistence hangs headless).
  - `lfm-grok` — Grok Build CLI, `[model."<id>"]` block in
    `~/.grok/config.toml` with `api_backend = "chat_completions"`.
- Full matrix: 2 precisions x 5 harnesses x 4 problems x 5 repeats =
  200 sessions. Precision is encoded in the served model name, so every run
  archive self-describes its weight format.
- The eval GPU (Lambda H100) never hosts inference; agents on the eval node
  reach the anvil server via a reverse tunnel
  (`ssh -N -R 8765:127.0.0.1:8765 ubuntu@<lambda-ip>` from anvil).

## Calibration debts (must clear before the deck freezes)

- Numeric-stress atols for 01/02/04 are engineering estimates; calibrate
  against a passing kernel on the canonical H100, on the actual allclose
  predicate, across all check seeds (cuda DEVLOG 2026-07-16 lesson).
- 03 TAU=1e-3 must be validated against a real sort-free fp32 kernel's
  boundary noise (especially under `flat_logits`).
- Freeze 03's `eager_ms` anchor per shape on the canonical node at publication.
