# KernelBench-Hard

Surgical GPU kernel benchmark. 6 active CUDA problems, frontier coding agents, roofline-based metric (achieved TFLOPS or GB/s vs hardware peak). Link-don't-spoil problem briefs: agents receive repo/paper URLs, not source snippets.

Sibling project to [KernelBench-v3](https://github.com/Infatoshi/KernelBench-v3) (volume-oriented; local open-weight models). Hard is for frontier-model harnesses on a small, high-signal deck.

## PR policy

This repository is published for transparency: it documents the exact prompts, harnesses, traces, kernels, and scoring code I use to evaluate models. It is not an open benchmark track, and I am not accepting PRs that change the problems, hardware target, scoring, prompts, or results. Issues and forks are welcome for discussion or independent experiments, but the canonical repo stays fixed so the published comparisons remain reproducible.

## Problem deck

| # | Problem | Hardware | What it tests |
|---|---------|----------|---------------|
| 01 | FP8 e4m3 GEMM (off-alignment shapes) | RTX PRO 6000 (SM120) | Tensor-core GEMM, epilogue fusion |
| 02 | KDA (Kimi Delta Attention) via CUTLASS CuTe | RTX PRO 6000 | Novel attention from paper, CUTLASS 4.x |
| 03 | Paged Attention decode | RTX PRO 6000 | Indirect indexing, pointer chasing |
| 05 | TopK with bitonic sort | RTX PRO 6000 | Small-output, comparator networks |
| 06 | Sonic-MoE up-projection: grouped GEMM + fused SwiGLU | RTX PRO 6000 | Megakernel, load balancing, variable-length |
| 07 | W4A16 weight-only GEMM (AWQ/GPTQ-style) | RTX PRO 6000 | Bit unpack, quantization, memory-bound decode |
| ~~08~~ | ~~Lightning Attention step (decode) — M4 Max, Metal~~ | _deferred_ | _M4 Max track is on the TODO list, not prioritized for the first sweep_ |

## Hardware

- **RTX PRO 6000 Blackwell Workstation** (SM120, 96GB GDDR7, 1.8 TB/s, ~200 BF16 / ~400 FP8 / ~800 FP4 TFLOPS dense)
- **M4 Max** (Metal 3, unified memory) — for problem 08 only

Required: CUDA 13.x (symlink `/usr/local/cuda-13`), torch 2.11+cu130, Python 3.11+.

## Active model matrix

One harness per model, each pinned to the highest-fidelity native endpoint.

| Model | Harness | Route |
|-------|---------|-------|
| Claude Opus 4.7 | `claude` | Anthropic direct |
| GPT-5.5 xhigh | `codex` (`-c model_reasoning_effort="xhigh"`) | OpenAI direct (npm `@openai/codex`) |
| Kimi K2.6 | `kimi` | Moonshot direct (api.moonshot.cn) |
| GLM-5.1 | `opencode zai/glm-5.1` | Z.AI direct (api.z.ai) |
| GLM-5.1 | `zai-claude glm-5.1` | Z.AI Anthropic-compatible endpoint (`api.z.ai/api/anthropic`) |
| GLM-5.1 | `droid custom:GLM-5.1-[Z.AI-Coding-Plan]-0` | Z.AI OpenAI-compatible coding endpoint via Factory |
| Minimax M2.7 | `opencode openrouter-pinned/minimax/minimax-m2.7` | OpenRouter pinned to Minimax lab (fp8) |
| DeepSeek V4 Pro | `opencode deepseek/deepseek-v4-pro` | DeepSeek direct (api.deepseek.com) |
| DeepSeek V4 Flash | `opencode deepseek/deepseek-v4-flash` | DeepSeek direct (api.deepseek.com) |
| Nemotron 3 Ultra | `opencode-nemotron nvidia/nemotron-3-ultra-550b-a55b` | OpenCode via OpenRouter pinned to DeepInfra (bf16) |
| Nemotron 3 Ultra diagnostic | `nvcf-nemotron nemotron-3-ultra` | NVIDIA NVCF shared function via local OpenAI-compatible proxy |

Each configured sweep runs its selected model matrix across the 6 active CUDA problems. Generations are unlimited-time (each run goes until the model decides it is done, under a large `BUDGET_SECONDS` wall-clock ceiling), so per-run GPU time varies; budget by the ceiling you set.

Nemotron 3 Ultra is scored through `opencode-nemotron`, not Claude Code or Droid. OpenCode is the least distorted route because it speaks OpenAI-compatible APIs directly; Claude Code needs an Anthropic-compatible router layer for this model, and Droid is not the native endpoint for this provider. The `opencode-nemotron` harness writes an archive-local OpenCode config for each run, pins OpenRouter to DeepInfra with `allow_fallbacks=false`, and requires `OPENROUTER_API_KEY`. The NVCF route is kept only for diagnostics because the Ultra function was observed degrading/504ing.

> **Codex binary note.** The harness uses the npm-distributed `@openai/codex` (currently `0.125.0`) installed at `~/.local/node-*/bin/codex`. The shell has `codex` aliased to a locally-built rust binary, but aliases don't expand inside non-interactive scripts, so the harness picks up the npm version automatically via PATH. To upgrade in the future: `npm install -g @openai/codex`.

## Deferred / upcoming

- **Gemini 3.1 Pro** (via `gemini-cli`) — low community interest; adding once bandwidth clears.
- Other models: _to be populated — Grok, Qwen, Sonnet 4.6, etc._

## Quick start

```bash
# Install (uv only)
uv sync

# Apply torch inductor CSE typing hotfix (required for torch 2.11.0 compile baseline)
./scripts/patch_torch.sh

# Run a single problem through a single harness
uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/01_fp8_gemm

# Preferred Nemotron route, with OPENROUTER_API_KEY in the environment
uv run kbh run opencode-nemotron nvidia/nemotron-3-ultra-550b-a55b problems-rtxpro6000/01_fp8_gemm

# Targeted Nemotron route preflight, without running the rest of the matrix
KBH_USE_OPENROUTER_NEMOTRON=1 KBH_PREFLIGHT_ONLY=opencode_nemotron_ultra ./scripts/preflight_harnesses.sh

# Include Nemotron in broad sweeps/preflight
KBH_USE_OPENROUTER_NEMOTRON=1 ./scripts/sweep.sh

# Diagnostic Nemotron/NVCF route, with NGC_API_KEY in the environment
uv run kbh run nvcf-nemotron nemotron-3-ultra problems-rtxpro6000/01_fp8_gemm

# Full sweep (active matrix × all 6 active CUDA problems)
./scripts/sweep.sh

# Plot roofline for a completed run
uv run python scripts/roofline_plot.py outputs/runs/<run_dir>
```

## Viewing transcripts in a browser

Each run produces a `transcript.jsonl` (or `codex_session.jsonl` for Codex). The viewer renders one of these as a self-contained HTML page with collapsible reasoning, tool calls, unified diffs for file writes, syntax-highlighted artifacts, and (for Claude Code) collapsed subagent dropdowns.

### Generate the HTML

```bash
# Auto-detects the harness format
uv run python -m src.viewer outputs/runs/<run_dir>

# Specify a transcript file explicitly
uv run python -m src.viewer outputs/runs/<run_dir> --transcript codex_session.jsonl

# Open in a local browser (works only if you're on the GPU box itself)
uv run python -m src.viewer outputs/runs/<run_dir> --open
```

Output: `outputs/runs/<run_dir>/index.html` — a single self-contained file (CSS embedded, Prism.js loaded from CDN for syntax highlighting).

### Viewing remotely from a Mac (recommended)

The GPU machine is headless; view the generated HTML over an SSH tunnel from your laptop.

**Step 1 — On the GPU server**, start a static HTTP server pointed at the runs directory:

```bash
./scripts/serve.sh outputs/runs 8000
# or for a specific subdir:
./scripts/serve.sh outputs/runs/20260424_193033_claude_claude-opus-4-7_01_fp8_gemm 8000
```

The script wraps `python3 -m http.server` and prints the URL.

**Step 2 — On your Mac**, open a separate terminal and forward the port:

```bash
ssh -N -L 8000:localhost:8000 anvil
```

Replace `anvil` with whatever SSH alias points at the GPU box. The `-N` flag makes ssh do nothing but hold the tunnel open.

**Step 3 — Open the browser on Mac:**

```
http://localhost:8000
```

You'll see a directory listing if you served `outputs/runs/`. Click any run folder to load its `index.html`. To stop, Ctrl-C the ssh process on Mac (closes the tunnel) and Ctrl-C the http.server on Linux.

### Port collisions

If port 8000 is taken, pick another:

```bash
# server
./scripts/serve.sh outputs/runs 9123
# mac
ssh -N -L 9123:localhost:9123 anvil
# browser: http://localhost:9123
```

### Persistent serving (multiple runs over time)

For long-running viewing across many sweeps, run the server in the background with `nohup`:

```bash
nohup ./scripts/serve.sh outputs/runs 8000 > /tmp/viewer.log 2>&1 &
```

Then keep an SSH tunnel up on Mac as needed. New runs in `outputs/runs/` appear immediately — refresh the directory listing in your browser.

### What the page shows

For each run, the HTML includes:

- **Header**: harness, model, session id, working directory
- **Summary cards**: turn count, tool call count, total events, input/output/cache tokens, elapsed time
- **Tab bar** (auto-shown when artifacts exist): `solution.py` (syntax-highlighted), `final answer`, `benchmark.log`, `check.log`, `result.json`
- **Timeline**: every event in chronological order
  - User / assistant / tool / system events, color-coded
  - Reasoning blocks collapsed by default (click to expand)
  - Tool calls show file path next to the tool name; if it's a file write/edit, an inline unified diff with `+`/`-` colored highlighting opens by default
  - Tool args block strips out long content fields when a diff is already shown
  - Tool results pretty-printed, with JSON-encoded outputs flattened so embedded newlines render as line breaks
  - Per-turn token usage badges
- **Subagents** (Claude Code Agent tool): the entire subagent run collapses into a single dropdown summarizing `subagent_type · N events · M tools · prompt-preview`. Click to expand and see the nested timeline.

## Design principles

See [SPEC.md](./SPEC.md). The short version:

- **Roofline, not speedup ratio.** Score is achieved throughput as a fraction of hardware peak. PyTorch eager and the SOTA reference (sonic-moe, flashinfer, marlin, cudnn) are reference lines on the plot, not the grading denominator.
- **Per-dtype tolerance.** fp32 atol/rtol 1e-4; fp16/bf16 1e-2. Closes the "identity kernel passes" class of cheats.
- **Multi-shape eval.** Each problem has 3-5 canonical shapes. Score is geomean over shapes. Rewards general kernels, penalizes hyperspecialization.
- **Link, don't spoil.** `AGENT.md` per problem gives repo URLs and paper links. Agents navigate, grep, and read source themselves. Navigation skill is part of the evaluation.
- **Algorithmic FLOPS.** For sparse/conditional kernels (MoE, paged attn), FLOPS is the dense-equivalent; agents can't skip work and call it optimization.
- **No custom tools, no MCP.** Each harness uses its native shell/read/write. We measure the harness as shipped.

## License

TBD (probably Apache 2.0 to match kernel-space conventions).
