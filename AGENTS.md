# kernelbench.com — coding-agent instructions (codex / droid / cursor / etc.)

This is the codex-style equivalent of `CLAUDE.md`. Read [`CLAUDE.md`](./CLAUDE.md) for the canonical version — same content, slightly more verbose. Per-suite rules live in `benchmarks/hard/AGENTS.md` and `benchmarks/hard/CLAUDE.md`.

## TL;DR for an agent picking this up cold

1. **Read [`benchmarks/hard/DEVLOG.md`](./benchmarks/hard/DEVLOG.md) first** — newest entry on top is the launch-prep summary, single best place to catch up.
2. **Vercel deploy gate**: every commit must be authored as `elliot@arledge.net`. Other emails fail Vercel's commit-verification at the pre-build phase silently. The repo's local `git config user.email` is already set; plain `git commit` works.
3. **Site reads benchmark data from disk at build time** (`lib/data.ts` → `benchmarks/hard/results/*.json`). No HTTP fetch. To update what the site shows, commit the data file.
4. **Auto-deploy on push to `master`** via Vercel's native GitHub integration. No GitHub Actions workflow.
5. **uv only inside `benchmarks/hard/`**. No bare `python`, no `pip`.
6. **Don't edit benchmark problem definitions** (`problems/*/{reference,check,benchmark}.py`, `problem.yaml`, `shapes.py`, `PROMPT.txt`) once a sweep is published.

## Layout summary

- Website at root (Next.js 16, Tailwind v4, JetBrains Mono, phosphor green on near-black).
- `benchmarks/hard/` — KernelBench-Hard suite (subtree of `Infatoshi/KernelBench-Hard`).
- `benchmarks/v3/` — KernelBench v3 suite (subtree of `Infatoshi/KernelBench-v3`).
- `public/runs/<run_id>.html` — 100 themed transcript viewers, committed.
- `public/blog-hard/*.png` — 5 matplotlib plots embedded in the v-Hard writeup.
- `public/data/v3/` — v3 results.csv + per-cell solution.py / reference.py.

## Where decisions are logged

- `benchmarks/hard/DEVLOG.md` — chronological record of what was tried, what worked, what didn't. The 2026-04-30 entry summarizes everything since 2026-04-29 launch prep.
- `benchmarks/hard/SPEC.md` — methodology spec.
- `benchmarks/hard/results/annotations/SCHEMA.md` — per-cell annotation file format.

When you finish substantive work, add to `benchmarks/hard/DEVLOG.md`. When you change how to operate the repo, update `CLAUDE.md` and `AGENTS.md`.
