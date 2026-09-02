# KernelBench-Mini — SPEC

Machine location: canonical monorepo on the Mac at
`~/dev/sites/kernelbench.com/benchmarks/mini`. **Canonical graded GPU is an
H100 SXM** (hardware key `H100_SXM` — SXM peaks in `src/hardware/h100_sxm.py`,
not the PCIe `H100` entry). Provision per sweep via `kb lambda` as a
`gpu_1x_h100_sxm5` (campaign name `kbmini` is conventional) and tear down when
done. Deck smokes may use other boxes; publish-grade numbers do not.

Status: **WIP, deck unpublished.** Keep Mini out of public posts until debut.
Calibration debts below must clear before the deck freezes.

## Thesis

Rank **small open-weight models (< 200B params)** against each other on kernel
writing, on a **fresh, unpublished deck** they cannot have been post-trained on.
Hard/Mega/CUDA prompts, winning solutions, and full traces are public (site +
HF), so those decks are structurally contaminated for exactly this model class.
Mini's deck is new ops with a structural twist each: familiar difficulty,
unfamiliar shape — the memorized tutorial kernel is wrong by construction.

Three deltas define Mini against Hard:

1. **Capped sessions.** 30 minutes wall-clock per agent session
   (`BUDGET_SECONDS=1800` via `KB_BUDGET_SECONDS_DEFAULT` in
   `scripts/run_hard.sh`), not unlimited. Small models loop; the cap is part of
   the bench identity and is what makes repeats affordable.
2. **5 repeats per cell.** The unit of publication is
   (model, harness, problem) x 5 independent sessions. Score two axes:
   **pass rate k/5** (reliability — where small models actually differentiate)
   and **best-of-5 performance** (capability). No pairwise/Elo machinery: the
   metric is cardinal, repeats give the spread.
3. **Harness pairing where routes exist.** Default published comparison is
   `opencode` (OpenAI-compatible) vs a `*-claude` Anthropic-skin route when the
   provider has one. Local-vLLM subjects (no public API) use the five LFM
   harnesses below instead — same model under multiple harnesses is itself a
   published comparison; never mix harnesses inside a cell.

Eligibility is machine-readable in `roster.yaml` (`sub_200b_open_weight`,
advisory until debut). One unranked frontier cell may run as a calibration
anchor only.

## Architecture (current — 2026-07-29 onward)

**One self-contained H100 SXM node does everything.** Do not rebuild the
retired anvil→tunnel→athena layout (see DEVLOG history only).

| Phase | Where | Rule |
| --- | --- | --- |
| Agent sessions | eval H100 | Agents write kernels; GPU lock serializes compile/check/benchmark |
| Inference (API-less models) | **same** eval H100 | vLLM on `127.0.0.1:8765` (~35% GPU mem); ccr-rust on `3456` for `lfm-claude` |
| Inference (API models) | provider APIs | No local vLLM; node is eval-only |
| **Publish-grade grade** | same H100, **quiet** | vLLM/ccr **stopped**; sequential `regrade_sequential.sh`; GPU at 0 MiB compute |

In-run timings during a parallel wave are contended by construction (serving +
many workers). They are archive/debug only. **Published peak_fraction / ms /
speedup must come from the isolated sequential re-grade** (standing 2026-07-19
rule).

Lambda worker: `KB_LAMBDA_BENCH=mini` (problems root defaults appropriately via
worker + `kb -b mini`). Archives live under `benchmarks/mini/outputs/runs/`
(or `outputs/runs-lambda-<name>/` after pull). Never leave Mini artifacts only
on a rented node.

## Scoring

- 01 (memory), 02 (memory), 04 (compute): roofline `peak_fraction` vs H100 SXM
  peaks.
- 03: **ms-anchored** (standing 2026-07-15 metric rule) — headline is geomean
  speedup vs the eager sort-based reference; the `eager_ms` anchor per shape is
  frozen at deck publication; peak_fraction is context only.
- Per-model headline: correctness rate over all runs in the matrix, plus
  geomean of best-of-5 scores across problems. Token/cost from
  `scripts/summarize_runs.py` is a secondary axis ($ per passing kernel).
- Mini's ceiling is **"best kernel in 30 minutes"**, not "best possible
  kernel." Never compare Mini peak_fraction to unlimited Hard/Mega cells as if
  they shared a protocol.

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
by design.

Audit note for 03: the launch-overhead regime plus an exact output makes
"cache the mask/threshold" the obvious cheat; the flat/peaky logits stress
cases break cached thresholds, but the per-cell audit must still empirically
overwrite the input buffer and confirm the mask changes.

## Which script to use (and which to ignore)

| Use | Script | When |
| --- | --- | --- |
| **Default column** | `./scripts/sweep_mini.sh <harness> <model> [effort]` | One (harness, model): 4 problems x 5 repeats, sequential |
| **Local-vLLM full matrix** | `./scripts/launch_matrix.sh <served-model-name>` | All five LFM harnesses in parallel on one node |
| **One session** | `./scripts/run_hard.sh <harness> <model> problems-h100/<prob> [effort]` or `kb -b mini run ...` | Debug / smoke |
| **Clean numbers** | `./scripts/regrade_sequential.sh <run_dir> ...` | After every wave, server stopped |
| **Leaderboard build** | `./scripts/publish_v2.sh` or `kb publish mini` | Only when publishing (deck still WIP) |

**Do not use for Mini campaigns** unless you are deliberately debugging legacy
hard-shaped matrices: `sweep.sh`, `launch_parallel_sweep.sh`,
`sweep_deck.sh`, `sweep_campaign.sh`. They are hard-bench copies and wrong
defaults for the 5-repeat Mini cell.

Harness transport reference: repo-root `docs/HARNESSES.md` (includes
`lfm-*` / `hermes` / `pi`). Env vars: `docs/ENV.md` (`KBMINI_*`, `KBH_*`).

## Operator runbook

Do these in order. Act without asking once the model/harness identity is known.

### A. API model (OpenRouter / native CLI / `*-claude` provider)

1. **Preflight keys** in `~/.env_vars` for the harness (`docs/HARNESSES.md`).
2. **Node:** `KB_LAMBDA_BENCH=mini kb lambda up kbmini gpu_1x_h100_sxm5` (or
   equivalent H100 SXM). `kb lambda sync kbmini` + bootstrap (uv, torch cu
   matching driver, agent CLIs). Confirm `torch.cuda.is_available()` — not
   merely that `nvcc` exists.
3. **Smoke:** one `run_hard.sh` cell on `problems-h100/01_dequant_gemv`.
4. **Sweep:** one `./scripts/sweep_mini.sh <harness> <model> [effort]` per
   column. Parallelize across columns (per-harness workers), never a
   problem-major loop.
5. **Pull:** `kb lambda pull kbmini` into
   `benchmarks/mini/outputs/runs-lambda-kbmini/` (venvs excluded).
6. **Re-grade** on a quiet GPU (no other CUDA jobs):  
   `./scripts/regrade_sequential.sh outputs/runs-lambda-kbmini/<run_id> ...`  
   (or the merged runs dir you publish from).
7. **Audit** every solution-bearing cell (solution.py + trace →
   `results/annotations/<run_id>.yaml`). `kb -b mini lint`,  
   `kb contamination mini`.
8. **Publish** only at debut time: `kb publish mini`, redaction, then wire the
   Mini section into homepage `HomeDecks` on `/` (scroll category alongside
   Mega/CUDA/Hard — never a dedicated `/mini` page). Until debut: archives +
   DEVLOG only; no public posts of the deck.
9. **Teardown:** `kb lambda down kbmini` and confirm `kb lambda ls`.

### B. Local-vLLM model (no public API) — LFM-class

Same as A, plus serving on the eval node:

1. Copy weights onto the node (or from a known path). Example subject:
   LiquidAI LFM2.5-2.6B-Agent.
2. **Install ninja** on Lambda images (`apt install ninja-build`) — vLLM KV
   init shells out to it.
3. **Serve one precision at a time** on `127.0.0.1:8765`,
   `--gpu-memory-utilization` ~0.35, `--max-model-len 128000` (hermes needs
   ≥64k; 128k is the model max), and for LFM tool calls:
   `--enable-auto-tool-choice --tool-call-parser lfm2`.
   - **bf16:** plain `vllm serve <path> --served-model-name lfm25-agent-bf16 ...`
   - **NVFP4A16:** **must** use `scripts/serve_nvfp4.py` (not plain vLLM) —
     fused `.w13` checkpoints break default w1/w3 stack rewrites. Served name
     `lfm25-agent-nvfp4`.
4. **ccr-rust** on `3456` before any `lfm-claude` column (`scripts/ccr-lfm.config.json`).
   On glibc-older Lambda images the anvil binary may need the shipped loader +
   `~/.kbmini/ccrlibs` (DEVLOG 2026-07-29).
5. **Matrix:**  
   `KBMINI_GPUS="0" KBMINI_SPLIT_BY_PROBLEM=1 ./scripts/launch_matrix.sh lfm25-agent-bf16`  
   Default harnesses: `lfm-opencode hermes pi lfm-grok lfm-claude`.  
   Precision is the served model name — archives self-describe bf16 vs nvfp4.
6. **Pull continuously** during long waves (rsync exclude `.venv`/caches) so a
   dead node costs ≤10 minutes of artifacts.
7. **Stop vLLM and ccr** before re-grade. Confirm GPU compute apps empty.
8. Then A.6–A.9.

Local harness env defaults: `KBMINI_BASE_URL=http://127.0.0.1:8765/v1`,
`KBMINI_API_KEY=local` (see `docs/ENV.md`).

### First subject (completed campaign — not a second SOP)

LFM2.5-2.6B-Agent x 2 precisions (bf16, NVFP4A16) x 5 local harnesses x 4
problems x 5 repeats = **200 sessions** on self-contained `kbmini`. Result:
**0/200 correct**; solution emission ~43 (bf16) / 37 (NVFP4); harness spread
dominates precision. Full write-up: DEVLOG 2026-07-29 / 2026-07-25. Do not
re-run this matrix unless the protocol or deck changes.

## Deck calibration anchor

**Status: MISSING ON DISK — rerun required before Mini debuts.**

Historically (2026-07-24 on the then-canonical node): `codex gpt-5.6-sol`
(effort high) on `01_dequant_gemv` graded **correct, peak_fraction 0.0900**,
audited clean (archive id `20260724_221725_codex_gpt-5.6-sol_01_dequant_gemv`).
That archive lived on lease nodes that died; it is **not** under
`outputs/runs/` or `outputs/runs-lambda-kbmini/` on the Mac.

What the anchor must re-pin when re-run on H100 SXM under the current 1800s
cap:

- The deck is **solvable as specified** — a small model scoring zero is a
  capability signal, not a broken problem.
- The 1800s cap **binds even a frontier model** (prior run hit the cap mid-
  optimization still climbing). Headline language stays "best in 30 minutes."

Rerun recipe: quiet H100 SXM, no local vLLM,  
`./scripts/run_hard.sh codex gpt-5.6-sol problems-h100/01_dequant_gemv high`  
(or current codex model slug), then sequential re-grade if the session shared
the box, full audit → `results/annotations/<run_id>.yaml`, restore the archive
into `outputs/runs/`, update this section with the new run id and metrics.

## Calibration debts (must clear before the deck freezes)

- Numeric-stress atols for 01/02/04 are engineering estimates; calibrate
  against a passing kernel on the canonical H100, on the actual allclose
  predicate, across all check seeds (cuda DEVLOG 2026-07-16 lesson).
- 03 TAU=1e-3 must be validated against a real sort-free fp32 kernel's
  boundary noise (especially under `flat_logits`).
- Freeze 03's `eager_ms` anchor per shape on the canonical node at publication.
- **Dead-code language evidence.** Framework detection is static, so it cannot
  tell a live kernel from an unused one. The labeller now emits a compound
  label (`cuda_wmma+triton`) instead of crowning the highest-priority match,
  which makes the ambiguity visible — but on the CUDA-gated problems (03/04) a
  hand-rolled pure-PyTorch solution carrying a dead `load_inline` block would
  still satisfy `require_cuda_evidence`. On 04 the forbidden list (SDPA,
  flash_attn, flashinfer, xformers) blocks the fast version of that cheat, so
  the residue is a correct-but-slow cell mislabelled as CUDA. Before freeze,
  make the evidence check runtime-based (did an extension actually compile /
  did a `.cu` get built during the session) rather than regex-only.

## Pre-debut checklist

- [ ] Codex (or equivalent frontier) anchor re-run on H100 SXM; archive on Mac
- [ ] Calibration debts above cleared or explicitly waived in DEVLOG
- [ ] Every published cell: sequential re-grade + manual audit YAML
- [ ] `kb -b mini lint`, `kb contamination mini`, redaction scan
- [ ] `roster.yaml` gate enforced at publish if still desired
- [ ] `kb publish mini` feeds homepage HomeDecks (scroll category on `/`, like Mega/Hard/CUDA — **no** `kernelbench.com/mini` route)
- [ ] Deck freeze: no PROMPT/reference/check/benchmark edits after publish
