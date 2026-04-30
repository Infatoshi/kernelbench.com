# KernelBench-Hard: Design Specification

Last updated: 2026-04-27.

## Purpose

A small, hand-curated GPU kernel benchmark where frontier coding agents attempt to beat the state-of-the-art kernel on a specific operation on specific hardware. Unlike KernelBench-v3, the goal is not breadth or quantity — it's to produce a few genuinely-hard traces that reveal how each (model, harness) pair approaches kernel engineering.

## Why "Hard"

v3 was 43 problems of grab-bag difficulty. Most were winnable by any frontier model with any harness. Median speedups ended up reward-hacked or trivially-above-eager, which made the leaderboard non-informative. Hard has ~8 problems where:

1. A reward-hacked solution fails correctness (tight atol, multi-shape eval, SOTA comparison).
2. Eager PyTorch is not the baseline — SOTA references are (sonic-moe, flashinfer, marlin, Tri Dao's attention kits). Beating PyTorch means nothing; approaching SOTA is the goal.
3. The problem requires reading source code / papers that the agent must navigate to. No spoon-feeding.

## Non-goals

- No public leaderboard. We publish roofline plots and kernels, not a number that models compete on.
- No "portable kernel" judging. SM120-specific optimizations are expected and good.
- No multi-turn/multi-day agent loops. 45 minutes wall-clock, one shot.

## Metric

### Primary: fraction of hardware peak

**Compute-bound problems** (GEMM, attention, MoE):
```
score = achieved_TFLOPS / peak_TFLOPS_for_precision
```
where `peak_TFLOPS_for_precision` is looked up from `src/hardware/rtx_pro_6000.py` (e.g., ~200 BF16, ~400 FP8, ~800 FP4 TFLOPS dense on SM120).

**Memory-bound problems** (softmax, layernorm, topk, W4A16 decode):
```
score = achieved_GB/s / peak_HBM_bandwidth
```
where `peak_HBM_bandwidth = 1800` GB/s for RTX PRO 6000 GDDR7.

**Which regime a problem is in** is decided by its arithmetic intensity and declared in `problem.yaml` as `regime: compute` or `regime: memory`.

### Shape sweep

Each problem has 3-5 canonical shapes declared in `shapes.py`. Score is the geometric mean over shapes. This penalizes kernels that hyperspecialize to the single "canonical" shape and crater on adjacent ones.

### Reference lines

`benchmark.py` also reports:
- PyTorch eager throughput (floor)
- `torch.compile(mode="reduce-overhead")` throughput (if compile succeeds)
- SOTA throughput from `sota.py` (ceiling we're chasing)

These are plotted but not graded. The agent's kernel is graded purely on absolute throughput.

## Correctness

- Per-dtype tolerance: fp32 `atol=1e-4, rtol=1e-4`; fp16/bf16 `atol=1e-2, rtol=1e-2`; fp8 `atol=0.1, rtol=0.1` (permissive, noise-dominated).
- 3 seeds per shape (42, 123, 456). All must pass.
- No `strict=False` state_dict loading in `check.py` — missing parameters fail correctness.
- NaN or inf in output is an automatic fail.

## Algorithmic FLOPS rule

For kernels that can skip work (sparse MoE routing, attention with causal masking, early-exit top-k):
- FLOPS is counted as the **dense equivalent**.
- Agents cannot skip work and claim higher TFLOPS.
- Example: MoE with 8 experts, top-2 routing, 32k tokens is scored as if every token × every expert. The agent's job is to be efficient at the dense-equivalent work, not to be a Pareto-frontier algorithm designer.

## Prompt design

Each `problems/<X>/PROMPT.txt` is a single human-voice query that the harness sends directly to the agent under test. No system/user split, no preamble file, no markdown structure. The voice is "engineer assigning a real task to a coding partner" — not "evaluator instructing a test subject."

Each prompt contains:
- Hardware identifier in a parenthetical on the first line (e.g., "RTX PRO 6000 (SM120 Blackwell, GDDR7, 1.8 TB/s)").
- Pointers to `reference.py` and `solution.py`, plus an explicit allowance to make a mess in the directory as long as the final answer lands in `solution.py`.
- Op semantics + correctness tolerance + every shape inlined as prose.
- Custom-kernel mandate + forbidden ops listed by name + a "look it up yourself" directive (PTX docs, CUTLASS sources, library headers).
- A flywheel sentence: implement, profile (ncu/nsys/torch.profiler), time it with benchmark.py, verify against check.py, iterate. Ends with "If check.py isn't passing, you're not done. Take as long as you need to actually push the number up."

What is **not** in any prompt:
- Peak throughput numbers, instruction recipes, or microarchitecture spec tables.
- Optimization guidance ("use mma.sync block-scaled variants").
- A wall-clock budget (the harness enforces timeout; the model has no concept of time).
- "You are being evaluated" framing or "good luck" closers.

Agents must navigate to references themselves (`git clone`, `curl`, `gh api`, read source, grep). Navigation is part of what we measure. The prompt names libraries by reputation only ("look up CUTLASS or sonic-moe or quack-kernels"); it does not URL-link them.

## Harness and tools

Every agent uses the **native tools** of its CLI. No MCP injections, no custom tool wrappers. We measure the harness as it ships.

Harnesses active in the current sweep: see README.md model matrix.

## Sandbox

Local execution on the CUDA host. Agent runs as the user in `problems/<X>/`. The directory is disposable — pre-run snapshot, post-run archive. No real isolation.

## Reward-hack resistance

- Tight per-dtype tolerance kills the "identity operator passes" class.
- Multi-shape eval kills single-shape hyperspecialization.
- SOTA reference as ceiling (not PyTorch as floor) makes dispatching to `F.sdpa` obviously pointless — the score is "how close to the best hand-tuned kernel," which SDPA itself often isn't.
- `problem.yaml` declares the forbidden function list (e.g., MoE forbids `torch.nn.functional.linear`; FP8 GEMM forbids `torch._scaled_mm`). Violations fail post-hoc in `check.py`.

## Adding a new problem

Checklist (see `CLAUDE.md` for the full procedure):
1. Create `problems/<NN>_<name>/` directory.
2. Write `reference.py` (naive PyTorch, for correctness).
3. Write `sota.py` (library call that produces the hardware-ceiling numbers).
4. Write `shapes.py` (canonical shape list, 3-5 entries).
5. Write `problem.yaml` with metadata (regime, flops/bytes formulas, tolerance, forbidden ops, SOTA dep).
6. Copy `check.py` and `benchmark.py` templates from an existing problem; parameterize for new inputs.
7. Write `PROMPT.txt` matching the voice and structure of the existing seven (see `## Prompt design` above).
8. Run `./scripts/run_hard.sh claude claude-opus-4-7 problems/<NN>_<name>` as a smoke test.
