# kernelbench.com

GPU kernel engineering benchmarks for autonomous LLM coding agents.

This is the canonical monorepo: it ships both the public website and the benchmark suites it visualizes. The website reads benchmark data straight from `benchmarks/*/results/` at build time — what's on disk is what's on the site.

## Layout

```
.
├── app/                    Next.js website
├── lib/data.ts             Reads benchmark data from benchmarks/ at build time
├── public/                 Website static assets
├── benchmarks/
│   ├── hard/             Latest (2026-04). Single Blackwell, 6 problems, 10 models. Live on /hard.
│   │   ├── SPEC.md         Design + methodology.
│   │   ├── DEVLOG.md       Decisions, dead ends, lessons.
│   │   ├── LEADERBOARD.md  Human-readable cross-model grid + rubric-leak footnotes.
│   │   ├── results/
│   │   │   ├── leaderboard.json    Schema-versioned, machine-readable (drives the site).
│   │   │   └── annotations/        Per-cell YAML commentary (clean / rubric_leak / etc.).
│   │   ├── problems/       Problem definitions (reference.py, check.py, benchmark.py, …).
│   │   ├── src/            Eval infrastructure (timing, correctness, hardware ceilings).
│   │   ├── scripts/        Sweep orchestration.
│   │   └── tests/
│   ├── mega/              Megakernel bench (2 problems). Same shape as hard/; not yet published to the site.
│   └── v3/                 Archive (2026-02). RTX 3090 + H100 + B200, 43-58 problems/GPU, 10 models.
│       ├── AGENTS.md, README.md, SPEC.md   (archive keeps its own agent doc)
│       ├── docs/           Overflow reference (problem inventory, reward-hack cases, v4 briefs).
│       ├── problems/, src/, scripts/, tests/, …
├── environments/           Prime Intellect `verifiers` mirrors of the benches (kernel_hard / kernel_mega / kernel_v3).
├── AGENTS.md               Single operator guide for the whole repo + both active benches (CLAUDE.md → symlink).
└── README.md               (this file)
```

Doc convention: agent instructions for the website and both active benches (hard + mega) are consolidated into one top-level `AGENTS.md` (`CLAUDE.md` is a symlink to it), so Claude Code and Codex read the same file and there's no confusion about which bench you're editing. Per-bench `README.md` (humans), `SPEC.md` (design), `DEVLOG.md` (running record), and `LEADERBOARD.md` (grid) stay in each bench dir; only the `benchmarks/v3/` archive keeps its own `AGENTS.md`. The website surfaces the visualization-ready slice (leaderboard, per-problem ceilings, annotations); the benchmark subdirs hold the full machinery so the work can be reproduced or extended.

## How the site reads data

`lib/data.ts` reads `benchmarks/hard/results/leaderboard.json` and the YAML annotations directly from the filesystem at build time. No network fetch, no HTTP cache — Next.js bakes the data into the page during `next build`. To update what the site shows: change the file under `benchmarks/`, push, Vercel rebuilds.

## Versions

| version | date | hardware | problems | models | live page |
| --- | --- | --- | --- | --- | --- |
| **hard** | 2026-04 | RTX PRO 6000 Blackwell (sm_120) | 6 | 10 | [/hard](https://kernelbench.com/hard) |
| v3 | 2026-02 | RTX 3090 + H100 + B200 | 43-58 per GPU | 10 | [/v3](https://kernelbench.com/v3) |

## Running the website locally

```bash
npm install
npm run dev
```

## Deploying

Vercel native GitHub integration. Every push to `master` auto-deploys. No CI workflow required.

## Source / mirrors

The standalone `Infatoshi/KernelBench-Hard` and `Infatoshi/KernelBench-v3` GitHub repos still exist and are mirror images of `benchmarks/hard/` and `benchmarks/v3/` here. This monorepo is the new canonical home.
