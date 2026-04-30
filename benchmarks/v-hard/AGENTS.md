# KernelBench-Hard — Developer Instructions (codex / droid)

This is the codex / droid / cursor-agent equivalent of `CLAUDE.md`. Content is identical; format is plain markdown for any CLI.

See [`CLAUDE.md`](./CLAUDE.md) for the canonical version. All rules there apply.

Summary of the non-negotiables:

- **uv only.** `uv run ...`, `uv add ...`, `uv pip install ...`. Never `pip` or bare `python`.
- **Before committing:** `uv run ruff check . --fix && uv run pytest`.
- **Do not edit `problems/*/solution.py`** — those are agent output.
- **Do not modify `reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, or `PROMPT.txt`** of an already-published problem.
- **Apply the torch 2.11 inductor CSE hotfix** via `./scripts/patch_torch.sh` after any `uv sync`.

## Quick actions

```bash
uv sync
./scripts/patch_torch.sh
./scripts/run_hard.sh claude claude-opus-4-7 problems/01_fp8_gemm
```

## Repo layout and adding a new problem

See `CLAUDE.md` — everything there is authoritative.
