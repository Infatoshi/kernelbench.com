# KernelBench-CUDA — DEVLOG

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
