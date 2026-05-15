# KernelBench-Hard — Developer Instructions (codex / droid)

This is the codex / droid / cursor-agent equivalent of `CLAUDE.md`. Content is identical; format is plain markdown for any CLI.

See [`CLAUDE.md`](./CLAUDE.md) for the canonical version. All rules there apply.

Summary of the non-negotiables:

- **uv only.** `uv run ...`, `uv add ...`, `uv pip install ...`. Never `pip` or bare `python`.
- **Before committing:** `uv run ruff check . --fix && uv run pytest`.
- **Do not edit `problems/*/solution.py`** — those are agent output.
- **Do not modify `reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, or `PROMPT.txt`** of an already-published problem; `scripts/run_hard.sh` invalidates and restores runs that change them.
- **Apply the torch 2.11 inductor CSE hotfix** via `./scripts/patch_torch.sh` after any `uv sync`.
- **Z.ai GLM-5.1 Claude Code reruns:** use `zai-claude`; `scripts/run_hard.sh` sets `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, `CLAUDE_CODE_MAX_RETRIES=1000000`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`, and routes all Claude Code aliases, including Haiku / Explore / subagents, to `glm-5.1`.

## Quick actions

```bash
uv sync
./scripts/patch_torch.sh
./scripts/run_hard.sh claude claude-opus-4-7 problems/01_fp8_gemm
```

## Repo layout and adding a new problem

See `CLAUDE.md` — everything there is authoritative.
