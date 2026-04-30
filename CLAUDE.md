# AGENTS.md - KernelBench-v3 Handoff

Last updated: 2026-04-08

## Snapshot
- Repository: `/home/infatoshi/cuda/KernelBench-v3`
- Branch: `master`
- Remote: `origin https://github.com/Infatoshi/KernelBench-v3.git`
- Local run artifacts: `/home/infatoshi/cuda/KernelBench-v3/outputs/batch_eval/`
- M4 Max benchmark: separate copy on macbook at `~/MetalBench`

## Non-Negotiable Project Rules
- Use UV only: `uv run ...`, `uv add ...`, `uv pip install ...`
- Do not use bare `python` or `pip`
- Before closing work, run: `uv run ruff check . --fix` and `uv run pytest`

## Architecture
Single repo with hardware target dispatch. Entry point: `bench.py`.

```
bench.py                    # CLI entry point
src/
  models.py                 # Model registry, pricing, provider clients
  api.py                    # API calls, token usage, cost estimation
  tools.py                  # Agent tools (bash, read_file, write_file, edit_file, submit) + guardrails
  prompts.py                # Per-architecture system prompts (RTX3090/H100/B200/M4Max)
  batch.py                  # Batch evaluation orchestration
  parsing.py                # Code extraction from LLM responses
  eval/
    agent.py                # Multi-turn agent loop (standard, gemini, reasoning modes)
    benchmark.py            # Performance benchmarking with adaptive torch.compile baseline
    context.py              # Workspace context, self-check commands
    fingerprint.py          # GPU/system metadata collection
    guardrails.py           # Solution validation (forbidden patterns)
    results.py              # EvalResult dataclass
    judge.py                # Post-benchmark LLM judge for reward hacking detection
  hardware/
    __init__.py             # HardwareTarget base, registry
    rtx3090.py              # RTX 3090 — local sandbox, 24GB, 43 problems
    h100.py                 # H100 — Modal sandbox, 80GB, 54 problems
    b200.py                 # B200 — Modal sandbox, 192GB, 58 problems (includes FP4, cutile)
    m4max.py                # M4 Max — Metal sandbox, 128GB, 63 problems
  agent/
    __init__.py             # Lazy imports (avoids Modal auth when running locally)
    local_sandbox.py        # Local GPU execution
    modal_sandbox.py        # Modal cloud GPU execution
    metal_sandbox.py        # macOS Metal execution
problems/
  level1/                   # 15 simple ops (matmul, softmax, conv, norms)
  level2/                   # 15 fused ops (matmul+activation chains)
  level3/                   # 3 architecture blocks (attention, transformer)
  level4/                   # 9 novel layers (MLA, MoE, GQA, FP8, INT4, FP4, etc.)
  graphics/                 # 2 graphics problems (bloom, particles) — RTX3090 only
  tile_specialized/         # 13 GEMM variants — H100/B200
  cutile/                   # 3 cuTile problems — B200 only
  metal_level1-4/           # 26 Metal-specific problems — M4Max only
```

## Hardware Targets
| Target | GPU | VRAM | Problems | Execution |
|--------|-----|------|----------|-----------|
| rtx3090 | RTX 3090 (Ampere SM86) | 24GB | 43 | Local |
| h100 | H100 (Hopper SM90) | 80GB | 54 | Modal |
| b200 | B200 (Blackwell SM100) | 192GB | 58 | Modal |
| m4max | M4 Max | 128GB unified | 63 | macbook local |

## CUDA Versions
- `/usr/local/cuda` symlink → `/usr/local/cuda-13.2` (default)
- `/usr/local/cuda-12.6` also installed (PATH currently resolves here)
- cuTile supported on Ampere as of CUDA 13.2
- Driver: 595.45.04

## Model Registry (src/models.py)
All models route through OpenRouter except OpenAI direct (gpt-5.3, gpt-5.4) and Z.AI direct (glm-5.1):

| Key | Provider | Pinned To | Notes |
|-----|----------|-----------|-------|
| anthropic/claude-opus-4.6 | openrouter | Anthropic | ($5/$25 per M) |
| anthropic/claude-sonnet-4.6 | openrouter | Anthropic | ($3/$15 per M) |
| openai/gpt-5.3 | openai direct | -- | model_id=gpt-5.3-chat-latest |
| openai/gpt-5.4 | openai direct | -- | |
| openai/gpt-5.4-low | openai direct | -- | reasoning_effort="low" |
| openai/gpt-5.4-high | openai direct | -- | reasoning_effort="high" |
| google/gemini-3-flash-preview | openrouter | Google AI Studio | ($0.50/$3 per M) |
| google/gemini-3.1-pro-preview | openrouter | Google AI Studio | ($2/$12 per M) |
| deepseek/deepseek-v3.2 | openrouter | DeepSeek | ($0.26/$0.38) -- scores 0%, skip |
| z-ai/glm-5 | openrouter | Z.AI | ($0.72/$2.30) |
| z-ai/glm-5.1 | zai direct | -- | beta API, reasoning model |
| minimax/minimax-m2.7 | openrouter | Minimax | ($0.30/$1.20) |
| moonshotai/kimi-k2.5 | openrouter | Moonshot AI | ($0.38/$1.72), reasoning_mode=True |
| qwen/qwen3.5-397b-a17b | openrouter | Alibaba | ($0.39/$2.34) |
| x-ai/grok-4.20 | openrouter | xAI | ($2.00/$6.00) |

## Adding New Models (provider selection procedure)
When a new coding model is released and the user wants to benchmark it:
1. Check if available on OpenRouter: `curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" "https://openrouter.ai/api/v1/models/<model_id>/endpoints" | python3 -m json.tool`
2. Review the endpoints list for: quantization level (prefer bf16/fp8 over int4/fp4), uptime (>95%), throughput, and whether the model's own lab hosts an endpoint
3. Always pin `provider_order` to the native/first-party provider first (e.g. "Alibaba" for Qwen, "Moonshot AI" for Kimi). Fall back to high-quality infra providers (DeepInfra, Together, Fireworks) that serve full weights
4. If the model lab has a direct API (like Z.AI for GLM-5.1), prefer adding it as a direct provider in `get_provider_client()` rather than routing through OpenRouter
5. Set `allow_fallbacks: True` so runs don't fail if the pinned provider is down, but the primary provider should be the one you want
6. Test with a quick API call before launching a full batch run
7. Add the model to this table and to `MODELS` dict in `src/models.py`

**Why this matters**: OpenRouter dispatches to whichever backend has capacity. Many providers serve int4-quantized weights, which materially degrades code generation quality. A model that scores 50% on full weights might score 30% on int4. Always pin to the native provider for benchmark integrity.

## GLM-5.1 Setup
- API key: `ZAI_API_KEY` in `~/.env_vars`
- Base URL: `https://api.z.ai/api/paas/v4/`
- Model name: `glm-5.1`
- Reasoning model: returns `reasoning_content` field alongside `content`
- 20 concurrent requests provisioned by Z.AI engineer
- Client timeout set to 1800s (30min) because reasoning chains can be very long
- API tends to timeout on large contexts; harness retries 3x with backoff

## Recent Changes (April 2026 session)

### New Features
1. **GLM-5.1 model support**: Added `zai` provider in `models.py` with direct Z.AI API, `max_output_tokens=32768` (reasoning model needs large budget), `max_concurrent=20`
2. **Wall-clock agent loops**: Replaced fixed turn limits (`for turn in range(max_turns)`) with wall-clock timeouts in all 3 agent loops (standard, gemini, reasoning). Time limits: L1=15min, L2=30min, L3=45min, L4=45min. Models get unlimited turns within the budget. Configured via `HardwareTarget.max_time()` in `src/hardware/__init__.py`.
3. **Post-benchmark judge model**: `src/eval/judge.py` — after benchmark, if solution is correct with speedup > 1.0x, an LLM judge reviews the solution code for reward hacking (precision downcast, PyTorch wrapper abuse, etc.). Configurable via `--judge-model` CLI flag. Judge outputs `{"legitimate": true/false, "reason": "..."}`. If FAIL, result.correct is set to False.
4. **JSONL transcript logging**: Every problem run produces `transcript.jsonl` alongside per-turn artifacts. Captures: session_start (full config), system_prompt, user_message, assistant_message (content + tool_calls + reasoning_content + per-turn token usage), tool_results (full tool outputs), agent_finished, session_end (full EvalResult dataclass dump with all 38 fields including benchmark metrics and hardware fingerprint).
5. **Lazy imports in `src/agent/__init__.py`**: Prevents Modal auth errors when running locally on RTX 3090.
6. **API retry logic**: Standard agent loop retries API calls 3x with exponential backoff (5s, 10s, 15s) on failure.

### Key Design Decisions
- **No regex guardrails for reward hacking**: Tried and reverted an extensive regex blocklist approach (blocking F.softmax, torch.bmm, precision downcast patterns, etc.). Too brittle — playing whack-a-mole. Replaced with LLM judge.
- **Judge defaults to PASS on error**: If judge call fails or returns non-JSON, solution is assumed legitimate. Better to have false negatives than false positives.
- **Wall clock > turn count**: Models like GLM-5.1 were burning 9/10 turns on filesystem exploration (looking for CUTLASS headers) and never writing code. With wall-clock timeouts, they can explore AND iterate.
- **Codex CLI tested but not integrated**: Codex CLI works with OpenRouter models but requires `/responses` API (not `/chat/completions`). Z.AI doesn't implement `/responses`, so Codex can't route to GLM-5.1 directly. OpenRouter acts as protocol bridge for other models. Decision: keep our own harness since it works with any OpenAI-compatible base URL.

## In-Progress Run (PAUSED)
```
run_20260408_121132  GLM-5.1  RTX3090  15/43 complete  4 correct (9%)
  Resume: uv run python bench.py run rtx3090 --models z-ai/glm-5.1 --levels 1,2,3,4 --workers 10 --judge-model z-ai/glm-5.1 --resume outputs/batch_eval/run_20260408_121132
```

Results so far:
- 1_Square_matrix_multiplication: 1.93x (judge PASS but non-JSON parse — likely precision downcast)
- 59_Matmul_Swish_Scaling: 1.17x (judge PASS, legitimate fusion)
- 3_Batched_matrix_multiplication: 1.00x (correct, matches baseline)
- 26_GELU: 0.99x (correct, 31 turns used)
- 11 failures: mostly API timeouts on turn 2 (fixed with retry + 30min timeout)

## Completed Evaluation Runs

### Coverage Matrix (correct/total)
| Model | RTX 3090 | H100 | B200 |
|-------|----------|------|------|
| GPT-5.4 | 33/43 (77%) | 42/54 (78%) | 50/58 (86%) |
| GPT-5.3 | 28/43 (65%) | 40/54 (74%) | 49/58 (84%) |
| Gemini 3 Flash | 32/43 (74%) | 41/54 (76%) | 46/58 (79%) |
| Kimi K2.5 | 22/43 (51%) | 27/54 (50%) | 35/58 (60%) |
| GLM-5 | 19/43 (44%) | 31/54 (57%) | 31/58 (53%) |
| GLM-5.1 | 4/43 (9%)* | — | — |
| Claude Opus 4.6 | 27/43 (63%) | 37/54 (69%) | 11/58 (19%) |
| Qwen3.5-397B | 13/43 (30%) | 22/54 (41%) | 25/58 (43%) |
| Gemini 3.1 Pro | 16/43 (37%) | 13/54 (24%) | 22/58 (38%) |
| Claude Sonnet 4.6 | 25/43 (58%) | 19/54 (35%) | 18/58 (31%) |
| MiniMax M2.5 | 35/43 (77%*) | 9/54 (17%) | 12/58 (21%) |
| MiniMax M2.7 | 9/43 (21%) | 14/54 (26%) | 8/58 (14%) |

*GLM-5.1 RTX3090 run is partial (15/43), uses new wall-clock loop + judge model. Resume command above.

### Run Directory → Model/GPU Mapping
```
run_20260226_235356  Gemini 3 Flash    RTX3090  43 results  32 correct
run_20260227_030206  Gemini 3 Flash    H100     54 results  41 correct
run_20260227_035338  Gemini 3 Flash    B200     58 results  46 correct
run_20260227_044818  Claude Opus 4.6   RTX3090  43 results  27 correct
run_20260301_120228  DeepSeek V3.2     B200     58 results   2 correct
run_20260301_123244  GLM-5             B200     58 results  31 correct
run_20260302_204111  Kimi K2.5         B200     58 results  35 correct
run_20260309_032138  Qwen3 Coder Next  B200     58 results   5 correct
run_20260309_181804  Qwen3.5-35B-A3B   H100     54 results   0 correct
run_20260310_041756  Qwen3.5-122B-A10B H100     54 results  17 correct
run_20260311_133649  GLM-5             RTX3090  31 results  18 correct (INCOMPLETE)
run_20260311_213917  Kimi K2.5         RTX3090  43 results  22 correct
run_20260313_025234  Claude Sonnet 4.6 RTX3090  43 results  25 correct
run_20260313_033440  GPT-5.4           RTX3090  43 results  33 correct
run_20260313_034511  GPT-5.3           RTX3090  43 results  28 correct
run_20260313_040306  Gemini 3.1 Pro    RTX3090  43 results  16 correct
run_20260313_045040  DeepSeek V3.2     RTX3090  43 results   0 correct
run_20260313_234022  MiniMax M2.5      RTX3090  23 results   4 correct (partial)
run_20260314_004831  MiniMax M2.5      RTX3090 129 results  35 correct
run_20260314_023431  MiniMax M2.5      H100     54 results   9 correct
run_20260314_055031  GLM-5             H100    162 results  89 correct
run_20260315_065251  Kimi K2.5         H100     54 results  27 correct
run_20260315_105800  Qwen3.5-397B      H100     54 results  22 correct
run_20260316_095221  MiniMax M2.5      B200     58 results  12 correct
run_20260316_180349  Qwen3.5-397B      B200     58 results  25 correct
run_20260317_072632  GLM-5             RTX3090  43 results  19 correct
run_20260317_072633  GPT-5.4           H100     54 results  42 correct
run_20260317_084945  Qwen3.5 397B      RTX3090  43 results  13 correct
run_20260317_084946  GPT-5.3           H100     54 results  40 correct
run_20260317_091603  Claude Opus 4.6   H100     54 results  37 correct
run_20260317_101252  Claude Sonnet 4.6 H100     54 results  19 correct
run_20260317_110358  Gemini 3.1 Pro    H100     54 results  13 correct
run_20260317_121922  GPT-5.4           B200     58 results  50 correct
run_20260317_130201  GPT-5.3           B200     58 results  49 correct
run_20260317_132246  Claude Opus 4.6   B200     58 results  11 correct
run_20260317_134109  Claude Sonnet 4.6 B200     58 results  18 correct
run_20260317_142816  Gemini 3.1 Pro    B200     58 results  22 correct
run_20260318_095508  MiniMax M2.7      RTX3090  43 results   9 correct
run_20260318_095510  MiniMax M2.7      H100     54 results  14 correct
run_20260318_111835  MiniMax M2.7      B200     58 results   8 correct
run_20260408_121132  GLM-5.1           RTX3090  15 results   4 correct (PAUSED — resume above)
```

## Reward Hacking Observations
- **Precision downcast**: GLM-5.1 (and GPT-5.4 on some problems) casts fp32 inputs to fp16 before GEMM to use tensor cores. Gets ~2x "speedup" that's cheaper arithmetic, not better algorithm. `pct_of_peak > 100%` is a dead giveaway.
- **MiniMax M2.5**: Attempted `pkill -f python` on RTX 3090 first run to kill the eval process. Guardrail fix prevented on subsequent runs.
- **Judge model approach**: Instead of regex blocklists (tried and reverted), use LLM judge post-benchmark. Catches semantic cheating that regex can't. See `src/eval/judge.py`.

## Key Design Decisions
- **Adaptive baseline**: `src/eval/benchmark.py` tries `torch.compile(mode='reduce-overhead')` and uses it only if >=5% faster than eager PyTorch
- **Weight sharing**: `sol_model.load_state_dict(ref_model.state_dict(), strict=False)` ensures fair comparison for models with learned params
- **Self-check**: Models run `torch.allclose` check before submitting, with `atol=1e-2, rtol=1e-2`
- **Hardware fingerprinting**: Every result includes GPU model, driver version, CUDA version via `src/eval/fingerprint.py`
- **Per-architecture prompts**: `src/prompts.py` injects WMMA (Ampere), WGMMA (Hopper), tcgen05 (Blackwell) guidance

## API Keys
All in `~/.env_vars`: ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, ZAI_API_KEY
Modal: `~/.modal.toml` (profile: elliot-2)

## Quick Commands
```bash
# Linting
uv run --with ruff ruff check . --fix

# Tests
uv run pytest

# Set up workspaces for all 43 RTX 3090 problems
uv run python scripts/setup_workspace.py rtx3090 --all

# Set up a single workspace
uv run python scripts/setup_workspace.py rtx3090 problems/level1/1_Square_matrix_multiplication_.py
```

## Harness-Based Eval (v3.1)
Each model runs in a real-world coding agent harness, not the custom KernelBench agent loop.
Workspaces are pre-built via `scripts/setup_workspace.py` into `workspaces/<hardware>/<problem>/`.
Each workspace contains: `reference.py`, `CLAUDE.md` (instructions), `check.py`, `benchmark.py`.

### Harness Commands
```bash
WS="workspaces/rtx3090/<problem_name>"

# Claude Code (Claude Opus 4.6)
claude --dangerously-skip-permissions --print --add-dir "$WS" \
  -p "You are in $WS. Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify, then benchmark.py to measure speedup."

# Codex CLI (GPT-5.4, GPT-5.4 high reasoning)
codex exec -m gpt-5.4 --full-auto -C "$WS" \
  "Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify, then benchmark.py to measure speedup."

codex exec -m gpt-5.4 -c model_reasoning_effort=\"high\" --full-auto -C "$WS" \
  "Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify, then benchmark.py to measure speedup."

# Droid -- built-in models
droid exec -m gemini-3.1-pro-preview --skip-permissions-unsafe --cwd "$WS" \
  "Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify, then benchmark.py to measure speedup."

droid exec -m kimi-k2.5 --skip-permissions-unsafe --cwd "$WS" \
  "Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify, then benchmark.py to measure speedup."

droid exec -m glm-5.1 --skip-permissions-unsafe --cwd "$WS" \
  "Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify, then benchmark.py to measure speedup."

# Droid -- custom models via OpenRouter (configured in ~/.factory/settings.json)
droid exec -m "custom:Qwen3.5-397B-[OpenRouter]-1" --skip-permissions-unsafe --cwd "$WS" \
  "Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify, then benchmark.py to measure speedup."

droid exec -m "custom:MiniMax-M2.7-[OpenRouter]-2" --skip-permissions-unsafe --cwd "$WS" \
  "Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify, then benchmark.py to measure speedup."

droid exec -m "custom:Grok-4.20-[OpenRouter]-3" --skip-permissions-unsafe --cwd "$WS" \
  "Read CLAUDE.md, then optimize reference.py into solution.py. Run check.py to verify, then benchmark.py to measure speedup."
```

### Model-Harness Assignment
| Model | Harness | Model Flag |
|-------|---------|------------|
| Claude Opus 4.6 | Claude Code | (default) |
| GPT-5.4 | Codex CLI | `-m gpt-5.4` |
| GPT-5.4 (high) | Codex CLI | `-m gpt-5.4 -c model_reasoning_effort="high"` |
| Gemini 3.1 Pro | Droid | `-m gemini-3.1-pro-preview` |
| Kimi K2.5 | Droid | `-m kimi-k2.5` |
| GLM-5.1 | Droid | `-m glm-5.1` |
| Qwen3.5 397B | Droid (custom) | `-m "custom:Qwen3.5-397B-[OpenRouter]-1"` |
| MiniMax M2.7 | Droid (custom) | `-m "custom:MiniMax-M2.7-[OpenRouter]-2"` |
| Grok 4.20 | Droid (custom) | `-m "custom:Grok-4.20-[OpenRouter]-3"` |

### Droid Custom Models
Configured in `~/.factory/settings.json`. OpenRouter custom models route through
`https://openrouter.ai/api/v1` with the OPENROUTER_API_KEY. Note: OpenRouter does not
support provider pinning via custom model configs in Droid -- the provider_order is only
enforced by the KernelBench harness. For benchmark integrity, prefer Droid's built-in
models (which route through Factory's own infra) when available.

## Cautions
- `workspaces/` and `outputs/` are gitignored -- do not expect git to preserve them
- NVIDIA ComputeEval cloned at `/home/infatoshi/cuda/compute-eval/` for reference only (not integrated)
- M4 Max runs from separate `~/MetalBench` on macbook via ssh
- Z.AI GLM-5.1 API is beta -- tends to timeout on large contexts
- OpenRouter routes to different inference providers per request. Many serve int4/fp4 quantized weights. Provider pinning (`provider_order` in models.py) only works through the KernelBench harness, not through Droid custom models.
