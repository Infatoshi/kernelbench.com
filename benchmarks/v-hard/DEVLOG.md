# DEVLOG

A running record of decisions, dead ends, and lessons. Newest entries on top. This is not a changelog (the git log is) — it's the why behind the shape of the project.

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
