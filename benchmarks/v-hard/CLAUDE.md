# KernelBench-Hard — Developer Instructions

Last updated: 2026-04-27.

This file is for **coding agents editing the repo** (you, via Claude Code). Do not confuse with `problems/<X>/PROMPT.txt` — those are the human-voice queries fed to agents _under test_.

For the journey behind the current design, read [DEVLOG.md](./DEVLOG.md).

## What this repo is

Small kernel benchmark. Frontier coding agents are given URLs to SOTA implementations (sonic-moe, flashinfer, marlin) and asked to write a competitive kernel on RTX PRO 6000 Blackwell (SM120) in 45 minutes. Roofline-graded. Published artifact is the best kernel per (problem × model × harness), plus the agent trace.

See [SPEC.md](./SPEC.md) for methodology. See [README.md](./README.md) for the model matrix and quick start.

## Non-negotiable rules

- **uv only.** No bare `python`, no `pip`. Use `uv run ...`, `uv add ...`, `uv pip install ...`.
- **Before committing:** `uv run ruff check . --fix && uv run pytest`.
- **Never edit `problems/*/solution.py`**. Those files are agent output; they're gitignored for a reason. If you need to inspect one, read it from `outputs/runs/<run>/<problem>/solution.py`.
- **Never modify `problems/*/reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, or `PROMPT.txt`** once a sweep has been published. Those define the benchmark — changing them invalidates prior results.
- **torch.compile fix.** torch 2.11.0+cu130 has a broken inductor CSE typing annotation that breaks the compile baseline. Run `./scripts/patch_torch.sh` after every `uv sync`.

## Repo layout

```
KernelBench-Hard/
├── README.md, SPEC.md         project docs
├── CLAUDE.md, AGENTS.md, .cursorrules   dev-facing (this file)
├── pyproject.toml             uv project
├── problems/                  the deck (append-only after release)
│   └── NN_name/
│       ├── reference.py       naive PyTorch, for correctness
│       ├── sota.py            library call for the ceiling number
│       ├── shapes.py          canonical shape list (read by check.py / benchmark.py)
│       ├── problem.yaml       metadata (flops, bytes, tolerance, forbidden ops)
│       ├── check.py           correctness runner (per-dtype atol)
│       ├── benchmark.py       roofline measurement: eager, compiled, sota, solution
│       ├── PROMPT.txt         human-voice query sent to the agent under test
│       └── solution.py        agent output (gitignored)
├── src/
│   ├── harness/               claude.py, codex.py, kimi.py, ccr_router.py
│   ├── eval/                  correctness.py, roofline.py, shapes.py, report.py
│   ├── hardware/              rtx_pro_6000.py, m4_max.py — peak lookup
│   └── sandbox/               local.py, metal.py
├── scripts/
│   ├── run_hard.sh            fire one (harness, model, problem)
│   ├── sweep.sh               full active matrix
│   ├── setup_problem.py       install SOTA deps for a problem
│   ├── roofline_plot.py       post-hoc plot from run artifacts
│   └── patch_torch.sh         torch 2.11 inductor CSE typing hotfix
├── outputs/runs/              per-run archival (gitignored)
└── docs/                      design notes, reward-hack case studies
```

## Adding a new problem

1. Pick the next NN (zero-padded). Don't reuse numbers.
2. Create `problems/NN_name/`.
3. Required files (order matters — write them in this order so you can sanity-check each):
   - `reference.py` — shortest naive PyTorch that produces the right answer. No optimization tricks. This is the correctness oracle.
   - `shapes.py` — 3 to 5 canonical shapes as a list of dicts. Include at least one "off-alignment" shape (e.g., K not multiple of 128 for GEMM).
   - `problem.yaml` — metadata. See `problems/01_fp8_gemm/problem.yaml` as the canonical example.
   - `sota.py` — wrap the library function that defines the ceiling. If no library supports SM120 yet, leave a stub and document the H100 paper number in a comment.
   - `check.py` — copy from 01_fp8_gemm, change the import line for `reference` and `shapes`.
   - `benchmark.py` — copy from 01_fp8_gemm, change the throughput formula to match `problem.yaml.flops_formula` / `bytes_formula`.
   - `PROMPT.txt` — single cohesive human-voice query. Match the structure of the existing seven: hardware in parenthetical on first line, file roles + "make a mess" allowance, op semantics + tolerance + every shape inlined as prose, custom-kernel mandate + forbidden ops list spelled out + suggested implementation paths + "look it up yourself" directive, flywheel sentence ending with "Take as long as you need to actually push the number up." Do not include peak throughput numbers, optimization recipes, or "you are being evaluated" framing.
4. Smoke-test: `./scripts/run_hard.sh claude claude-opus-4-7 problems/NN_name` on a cheap model first. Verify `check.py` runs, `benchmark.py` runs, result.json is sane.
5. Once you're happy, run the full model matrix sweep.

## Running a sweep

```bash
# Single (harness, model, problem)
./scripts/run_hard.sh claude claude-opus-4-7 problems/01_fp8_gemm

# Full active matrix on one problem
for model_harness in "claude claude-opus-4-7" "codex gpt-5.5 xhigh" "kimi kimi-k2.6"; do
    read -r HARNESS MODEL <<< "$model_harness"
    ./scripts/run_hard.sh "$HARNESS" "$MODEL" problems/01_fp8_gemm
done

# Everything (this is what sweep.sh does)
./scripts/sweep.sh
```

## Interpreting results

- `outputs/runs/<ts>_<harness>_<model>_<problem>/result.json` — scalar metrics (correct, achieved_tflops, peak_fraction, shape-by-shape times)
- `outputs/runs/<ts>_..../transcript.jsonl` — full agent trace
- `outputs/runs/<ts>_..../solution.py` — final agent-written kernel
- `outputs/runs/<ts>_..../roofline.png` — visual: peak line, eager/compile/SOTA/solution points per shape

Run `uv run python scripts/roofline_plot.py outputs/runs/<ts>_...` to (re)generate the plot.

## Testing

Keep `tests/` minimal. We test:
- `src/hardware/` peak-value lookup
- `src/eval/roofline.py` throughput math
- `src/eval/correctness.py` per-dtype tolerance enforcement

We do **not** test problem files directly — those are validated by running a real agent against them.

```bash
uv run pytest
```

## When a sweep fails

Most likely causes:
1. **torch.compile CSE crash** — run `./scripts/patch_torch.sh`.
2. **CUDA_HOME pointing at 12.8** — harness script already sets `CUDA_HOME=/usr/local/cuda-13`; make sure you sourced it.
3. **`sota.py` import fails** — the SOTA dep isn't installed. Check `problem.yaml` for the pinned version; install with `uv pip install <spec>`.
4. **Agent CLI not authenticated** — `claude`, `codex`, `kimi` each need their own auth. Check `~/.env_vars` and each CLI's `info` / `whoami` command.
5. **Agent ran out of budget before writing anything** — increase the `timeout 2700` in run_hard.sh or accept this as a failure mode worth recording.
