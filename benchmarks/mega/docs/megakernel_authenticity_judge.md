# Megakernel authenticity — judge gate + advisory tripwires

**Status:** v2.1 (2026-07-01). Replaces the v2 substring ban on
`torch.compile` / `CUDAGraph`.

## Why this exists

Mega problems are *megakernel* benchmarks: the scored path must be a genuine
**fused** kernel, not a launch-overhead workaround. The two ways a submission
fakes this:

1. **Hidden launches** — a Python loop of many small kernels wrapped in a
   `torch.cuda.CUDAGraph` or `torch.compile`d region so the *replay* is one
   launch but nothing is actually fused on-chip. (This is exactly how the Opus
   `02_kimi_linear_decode` 19.35x cell got its number — see the audit.)
2. **No kernel at all** — plain eager PyTorch with a docstring claiming fusion.

## Why not just forbid the strings

We tried (`forbidden: [torch.compile, CUDAGraph, ...]` scanned as substrings).
The red-team battery in `tests/test_megakernel_evidence.py` (cases A1–A7) proved
a substring gate is **both**:

- **Leaky** — `getattr(torch.cuda, "CUDAGra"+"ph")` (A5) and
  `importlib.import_module` runtime codegen (A6) carry no literal banned token,
  so they slip straight through.
- **Brittle** — an honest solution whose comment says *"no torch.compile, no
  CUDA graphs"* (A7) gets **false-failed** on its own disclaimer.

A substring gate therefore punishes honesty and rewards obfuscation — the worst
possible split. So authenticity is decided by an **LLM judge reading the code**,
fed **deterministic advisory evidence**.

## The design

### Bright line (hard fail, in `check.py`)

Importing a prebuilt library (`transformers`, `vllm`, `marlin`, `pufferlib`, …)
is unambiguous and stays a hard fail — but matched by **AST import statements**,
not raw substring, and recursively across `solution.py` + every local module it
imports. Naming a lib in a comment no longer fails; `importlib`-obfuscated
imports are caught by the judge instead (and flagged by the codegen tripwire).

### Advisory tripwires (deterministic, never auto-reject)

`src/eval/megakernel.py` (CLI: `scripts/megakernel_evidence.py`) extracts, over
the recursive static source:

| signal | how |
| --- | --- |
| `kernel_count` | `@triton.jit`, `load_inline`, `__global__ void` |
| `graph` | `CUDAGraph` / `cuda.graph` / `graph.replay` / `make_graphed_callables`, on **code with string+comment contents stripped** (so a disclaimer comment does not trip it) |
| `compile` | `torch.compile` / `_dynamo`, same stripped-code view |
| `codegen` | `exec`/`eval`/`compile`, `importlib.import_module`, or writing a `.py`/`.cu`/`.so` file |
| `obfuscation` | `getattr(x, "a"+"b")`, or any string-concat that folds into a banned token (`"CUDAGra"+"ph"`) — AST-level, survives string stripping |

`check.py` also writes `framework.txt` (`eager`/`triton`/`cuda_raw`/`cudagraph`/
`compile`/`ptx`) as a coarse label.

### Judge gate (decides authenticity)

The **mandatory pre-publish audit** (see `AGENTS.md` — the same subagent that
audits for reward-hacking) now also renders the judge prompt from
`render_judge_prompt(...)` and returns a verdict. The judge reasons from the
**code**, treating the tripwires as hints and comments/docstrings as untrusted.
Red-team result: the judge correctly PASSed A1, FAILed A2–A6 (incl. the
obfuscated/codegen evasions the substring scan missed), and FAILed A7 (eager,
ignoring the lying docstring).

Record the verdict in `results/annotations/<run_id>.yaml`:

```yaml
megakernel_authentic: false   # true | false ; omit/None = not yet judged
authenticity_reason: >
  Timed path replays a torch.cuda.CUDAGraph over ~12 per-op kernels; no single
  fused kernel. Cuts launch overhead without fusing on-chip dataflow.
```

`scripts/build_mega_leaderboard.py` **excludes** any run whose annotation has
`megakernel_authentic: false` (alongside the contamination exclusion). `None`
(unjudged) is kept — judge before you publish.

## How to run it

```bash
# evidence bundle (JSON) for one run
uv run python scripts/megakernel_evidence.py outputs/runs/<run_dir>

# + rendered judge prompt to hand to the audit subagent
uv run python scripts/megakernel_evidence.py outputs/runs/<run_dir> --prompt --problem 02_kimi_linear_decode

# regression test for the deterministic evidence layer (the red-team battery)
uv run pytest tests/test_megakernel_evidence.py
```

## Known implication

Under v2.1 the **published Opus 19.35x `02_kimi_linear_decode` cell is a
CUDA-graph solution** and is *not* an authentic megakernel. It has not been
re-judged/annotated here yet; when the mega board is (re)published, that cell
must get `megakernel_authentic: false` (or the run re-done as a true single
fused kernel). Mega is not on the public site yet, so nothing user-facing is
wrong today — but do not re-publish the graph cell as a megakernel headline.
