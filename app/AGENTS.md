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

- New model, four tables, all enforced by `scripts/check_publish_gates.py` at `kb publish` / `kb deploy`: `LIVE_MODEL_SLUGS` in `app/_lib/models.server.ts` (homepage and /models roster; `RETIRED_MODEL_SLUGS` there for models kept off it), `MODEL_NAMES` in `scripts/build_model_index.py` (display name in `models.json`), and `MODEL_NAMES`, `SHORT_NAMES`, `LIVE_MODEL_IDS` in `app/_lib/charts.ts` keyed by every board `model` id the model ships under (bare slug and provider-prefixed both occur, e.g. `qwen3.8-max` and `qwen/qwen3.8-max`). Bench and problem labels are `BENCH_LABELS` / `PROBLEM_LABELS` there; GPU tabs are `HOME_GPU_TABS` in `app/_lib/models.ts`.
- New lab: `LAB_BRANDS` in `app/_lib/models.ts` plus `public/logos/labs/<lab>.svg`.
- Mini is a homepage `HomeDecks` scroll category on `/` when it debuts, not a `/mini` route. Multi is unpublished.
- Old external links land on bench sections via the redirects noted at the top of `app/{hard,cuda,multi}/page.tsx`; keep them working.

## Before `kb deploy`

Skim the homepage chart and `/hard` (dark mode, look at the actual render). Run the redaction scan (`media/AGENTS.md`) before any `public/runs` commit. Site palette tokens live in `app/globals.css`; `media/kbh_theme.py` copies them, so a palette change updates both.

## After every finished run

In this order. The publish gates check steps 2, 4, 5 and 6; nothing checks 1, 3 or 7 for you.

1. Pull the archive to `benchmarks/<bench>/outputs/runs/<run_id>` and regrade in isolation (rules in root `AGENTS.md`).
2. Mega only: write the GPU label to `outputs/runs/<run_id>/gpu` (`RTX PRO 6000 Blackwell`, `H100`, `B200`); `build_mega_leaderboard.py` drops rows without it.
3. Write and `git add` the audit YAML in `results/annotations/`.
4. Cuda RTX only: append the run_id to `benchmarks/cuda/results/published_runs.json` `run_ids` (or to its `excluded` map with a reason); the RTX cuda board publishes only what is listed.
5. New model: add its slug to `LIVE_MODEL_SLUGS` (`app/_lib/models.server.ts`), its display name to `MODEL_NAMES` in `scripts/build_model_index.py`, and every board model id it ships under (bare and provider-prefixed, as they appear as `model` in `public/data/catalog.json`) to `MODEL_NAMES`, `SHORT_NAMES` and `LIVE_MODEL_IDS` in `app/_lib/charts.ts`. A model deliberately off the homepage goes in `RETIRED_MODEL_SLUGS` instead. New lab: `app/AGENTS.md`.
6. `kb publish <bench>` (add `--push` or run `kb push-runs <bench>` for the HF transcripts). If the gate fails, fix what it names and rerun; the commit and deploy do not happen until it passes.
7. `kb deploy`, then open the live homepage and confirm the model is in the roster and the new cell is on the board for that GPU tab. The push is on `master` and Vercel builds it; the commit email must be `elliot@arledge.net` or nothing deploys. Write-ups lead with what the traces show (`media/AGENTS.md`).
