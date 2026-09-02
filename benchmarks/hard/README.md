# KernelBench-Hard

Per-op GPU kernel bench: frontier coding agents write competitive CUDA or
Triton kernels, unlimited wall-clock, roofline-graded, reward-hack audited.
Default deck is `problems-rtxpro6000/` (RTX PRO 6000); variants exist as
`problems-h100/` and `problems-b200/`.

| NN | problem | note |
| -- | ------- | ---- |
| 01 | `01_fp8_gemm` | FP8 e4m3 GEMM, off-alignment shapes |
| 02 | `02_kda_cutlass` | Kimi Delta Attention via CUTLASS CuTe |
| 03 | `03_paged_attention` | paged attention decode |
| 05 | `05_topk_bitonic` | top-k via bitonic sort (ms-anchored) |
| 06 | `06_sonic_moe_swiglu` | Sonic-MoE grouped GEMM + SwiGLU |
| 07 | `07_w4a16_gemm` | W4A16 weight-only quantized GEMM |

```bash
uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/01_fp8_gemm
```

See `SPEC.md`, `DEVLOG.md`, and repo-root `AGENTS.md`.

## Developer guide (hard, cuda, mini; mega deltas noted)

The benchmark: frontier coding agents get URLs to SOTA implementations (sonic-moe, flashinfer, marlin) and are asked to write a competitive kernel on the deck's GPU in one autonomous session with unlimited time. Roofline-graded. The published artifact is the best kernel per (problem x model x harness) plus the agent trace. `problems/<X>/PROMPT.txt` is the human-voice query fed to the agent under test, not operator documentation. Methodology: `SPEC.md`. History: `DEVLOG.md`.

### Layout

```
benchmarks/<bench>/
├── README.md, SPEC.md, DEVLOG.md
├── pyproject.toml             uv project
├── problems*/                 the deck (append-only after release); hard has problems-rtxpro6000 (default), -h100, -b200
│   └── NN_name/
│       ├── reference.py       naive PyTorch, the correctness oracle
│       ├── sota.py            library call for the ceiling number
│       ├── shapes.py          canonical shape list (read by check.py / benchmark.py)
│       ├── problem.yaml       metadata (flops, bytes, tolerance, forbidden ops, hardware key)
│       ├── check.py           correctness runner (per-dtype atol + numeric stress)
│       ├── benchmark.py       roofline measurement: solution first; optional eager/compiled/sota
│       ├── PROMPT.txt         query sent to the agent under test
│       └── solution.py        agent output (gitignored; never edit)
├── src/
│   ├── kbh/                   the `kbh` CLI
│   ├── harness/               classification.py (failure/usage classification)
│   ├── eval/                  correctness.py, numeric_stress.py, roofline.py, shapes.py, timing.py, report.py, cuda_language.py (cuda bench)
│   ├── hardware/              peak lookup per GPU + identify.py (nvidia-smi name -> key)
│   └── viewer/                transcript parsers + HTML rendering
├── scripts/
│   ├── run_hard.sh            thin identity wrapper over repo-root scripts/lib/run_harness.sh (edit the lib; mega keeps a deliberate fork with its bwrap sandbox)
│   ├── sweep.sh               full active matrix
│   ├── setup_problem.py       install SOTA deps for a problem
│   ├── roofline_plot.py       post-hoc plot from run artifacts
│   └── patch_torch.sh         torch 2.11 inductor CSE typing hotfix
└── outputs/runs/              per-run archives (gitignored)
```

Shared files listed in `kbtool/tests/test_repo_consistency.py` must stay byte-identical across hard/cuda/mini/mega; a deliberate fork needs a DEVLOG entry and removal from that list.

### Adding a problem

1. Next NN, zero-padded; never reuse a number (04 is retired on hard, 01 on mega).
2. Create `problems-rtxpro6000/NN_name/` (hard) or `problems/NN_name/` (mega).
3. Write in this order so each file can be sanity-checked: `reference.py` (shortest naive PyTorch, no tricks), `shapes.py` (3 to 5 shapes, at least one off-alignment, e.g. K not a multiple of 128), `problem.yaml` (copy `01_fp8_gemm` or mega `02_kimi_linear_decode`), `sota.py` (library ceiling; stub with the H100 paper number in a comment if nothing supports the GPU yet), `check.py` and `benchmark.py` (copy the closest problem; throughput formula must match `problem.yaml` flops/bytes formulas), `PROMPT.txt` (one cohesive human-voice query matching the existing structure: hardware in a parenthetical on line one, file roles and the "make a mess" allowance, op semantics, tolerance, every shape inlined as prose, custom-kernel mandate with the forbidden ops spelled out, suggested paths, "look it up yourself", and the closing flywheel sentence ending "Take as long as you need to actually push the number up." No peak numbers, recipes, or "you are being evaluated" framing).
4. Smoke on a cheap model: `uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/NN_name` (mega: `./scripts/run_hard.sh claude claude-opus-4-7 problems/NN_name`). Confirm `check.py` and `benchmark.py` run and `result.json` is sane.
5. Then the full matrix.

### Running

```bash
uv run kbh run claude claude-opus-4-7 problems-rtxpro6000/01_fp8_gemm    # one cell
for mh in "claude claude-opus-4-7" "codex gpt-5.5 xhigh"; do              # matrix on one problem
    read -r HARNESS MODEL EFFORT <<< "$mh"
    uv run kbh run "$HARNESS" "$MODEL" problems-rtxpro6000/01_fp8_gemm $EFFORT
done
./scripts/sweep.sh                                                        # everything
```

Mega is driven from inside `benchmarks/mega/` with `./scripts/run_hard.sh <harness> <model> problems/02_kimi_linear_decode` and `./scripts/sweep.sh`, not `kb`/`kbh`. Before a sweep: `nvidia-smi`, and `./scripts/patch_torch.sh` after every `uv sync` (torch pins per bench in `docs/TORCH.md`).

### Correctness

`check.py` validates nominal canonical shapes and seeds, then reruns them under problem-specific numeric stress from `src/eval/numeric_stress.py` (rescaled activations or weights; no hidden shapes). Integer outputs are exact; floating outputs use explicit per-dtype tolerances and report max abs/rel error, bad element count, worst index, and tolerance on failure. This catches zero-output, cached-nominal, and loose-tolerance cheats. `KBH_NUMERIC_STRESS=0` is for local debugging only, never official checks or backfills. `benchmark.py` does not import numeric stress and times the canonical deck only, so scores stay comparable.

### Results

`outputs/runs/<ts>_<harness>_<model>_<problem>/` holds `result.json` (correct, achieved_tflops, peak_fraction, per-shape times, gpu stamp, failure_reason), `transcript.jsonl`, `solution.py`, `roofline.png`. Regenerate the plot with `uv run python scripts/roofline_plot.py outputs/runs/<run>`. Archives are thin: `repo/.venv` is stripped after scoring (`scripts/lib/strip_run_venv.sh`; `KBH_KEEP_RUN_VENV=1` to keep; `--tree benchmarks/<bench>/outputs` reclaims old fat archives). Recreate an env with `uv run` from the archived `uv.lock`.

### Tests

`uv run pytest` covers `src/hardware/` peak lookup, `src/eval/roofline.py`, `src/eval/correctness.py`, and `src/eval/numeric_stress.py` against the classic cheats. Problem files are validated by running a real agent or a disposable smoke workspace, not by unit tests. Repo-wide guards: `uv run --project kbtool pytest kbtool/tests/`.

### When a sweep fails

1. torch.compile CSE crash: `./scripts/patch_torch.sh`.
2. `CUDA_HOME` at 12.8: the runner sets `/usr/local/cuda-13`; make sure it was sourced.
3. `sota.py` import fails: install the pinned dep from `problem.yaml` with `uv pip install <spec>`.
4. Agent CLI not authenticated: each CLI has its own login; check `~/.env_vars` and the CLI's `whoami`.
5. No solution written: runs are unlimited-time, so this is a real failure (early stop, provider error), not a budget cutoff. Record it; `retryable_infra_failure` marks the retryable ones.
