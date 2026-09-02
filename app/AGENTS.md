# kernelbench.com — the website

Next.js 16 + Tailwind, package manager **bun** (`bun.lock`). Vercel builds on push from the Mac checkout; `kb deploy "<msg>"` is publish + commit + push. Commit email must be `elliot@arledge.net` or Vercel silently fails the build.

```bash
bun install
bun dev          # http://localhost:3000
bun run build
```

## Data flow

Site data is baked at build time by `app/_lib/data.ts` from `benchmarks/*/results/` (hard/cuda: `results/leaderboard.json`; mega: `public/data/mega/results.csv`), the annotation YAMLs (`results/annotations/*.yaml`, tiny YAML subset parser in `data.ts`, schema in `benchmarks/hard/AGENTS.md`), and `public/data/models.json`. `kb publish <bench>` regenerates all of these plus the redacted `public/runs/*_solution.py.txt` kernels (mega: `public/data/mega/code/`); never hand-edit them. `git add` new annotation YAMLs before publishing: models.json only joins annotations that git tracks, and an untracked one ships the cell as `unaudited`, which the homepage column chart drops.

Transcripts go to HF (`Infatoshi/kernelbench-<bench>-traces`) via `kb push-runs <bench>`; each run page links its trace. HF `/blob/` and `/resolve/` must both return 200 before a link ships.

## Adding a model or a lab

- New model: `LIVE_MODEL_SLUGS` in `app/_lib/models.server.ts` (homepage and /models roster), `MODEL_NAMES` and `SHORT_NAMES` in `app/_lib/charts.ts` (display name, chart label). Bench and problem labels are `BENCH_LABELS` / `PROBLEM_LABELS` there; GPU tabs are `HOME_GPU_TABS` in `app/_lib/models.ts`.
- New lab: `LAB_BRANDS` in `app/_lib/models.ts` plus `public/logos/labs/<lab>.svg`.
- Mini is a homepage `HomeDecks` scroll category on `/` when it debuts, not a `/mini` route. Multi is unpublished.
- Old external links land on bench sections via the redirects noted at the top of `app/{hard,cuda,multi}/page.tsx`; keep them working.

## Before `kb deploy`

Skim the homepage chart and `/hard` (dark mode, look at the actual render). Run the redaction scan (`media/AGENTS.md`) before any `public/runs` commit. Site palette tokens live in `app/globals.css`; `media/kbh_theme.py` copies them, so a palette change updates both.
