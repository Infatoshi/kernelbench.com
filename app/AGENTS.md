# kernelbench.com — the website

Next.js 16 + Tailwind, package manager **bun** (`bun.lock`). Vercel builds on push from the Mac checkout; `kb deploy "<msg>"` is publish + commit + push. Commit email must be `elliot@arledge.net` or Vercel silently fails the build.

```bash
bun install
bun dev          # http://localhost:3000
bun run build
```

## Single-page homepage (2026-11-23 redesign)

`app/page.tsx` renders the entire active site on one page (`app/_components/kernelbench-board.tsx`, data built at build time by `app/_lib/sketchpad.server.ts`, vector engineer's notepad sketches in `app/_lib/sketches.ts`, NVIDIA dark graph-paper CSS in `app/sketchpad.css`). Top-left masthead is plain `kernelbench.com` with no logo or multi-page nav (`app/layout.tsx`). Active GPU tabs are `RTX PRO 6000` and `H100 PCIe` (B200 excluded from board tabs, preserved in audit logs). Hovering (or tapping on mobile) any bar row opens an interactive popover directly under the pointer (`Audit report`, `Kernel solution`, `Agent trace`, plus CUDA 3-problem switcher pills); the full per-model audit log, per-shape breakdowns, and redacted kernels live in the slide-over drawer. `Method`, `Older & retired models`, and `Hard archive` are collapsed bottom disclosures. Legacy deep-link routes (`/models`, `/models/[slug]`, `/runs`, `/runs/[gpu]/[rid]`, `/hard`, `/code`) stay live for external links.

## Data flow

Site data is baked at build time by `app/_lib/data.ts` from `benchmarks/*/results/` (hard/cuda: `results/leaderboard.json`; mega: `public/data/mega/results.csv`), the annotation YAMLs (`results/annotations/*.yaml`, tiny YAML subset parser in `data.ts`, schema in `benchmarks/hard/AGENTS.md`), and `public/data/models.json`. `kb publish <bench>` regenerates all of these plus the redacted `public/runs/*_solution.py.txt` kernels (mega: `public/data/mega/code/`); never hand-edit them. `git add` new annotation YAMLs before publishing: models.json only joins annotations that git tracks, and an untracked one ships the cell as `unaudited`, which the homepage column chart drops.

Transcripts go to HF (`Infatoshi/kernelbench-<bench>-traces`) via `kb push-runs <bench>`; each run page links its trace. HF `/blob/` and `/resolve/` must both return 200 before a link ships.

## Adding a model or a lab

- New model, four tables, all enforced by `scripts/check_publish_gates.py` at `kb publish` / `kb deploy`: `LIVE_MODEL_SLUGS` in `app/_lib/models.server.ts` (homepage and /models roster; `RETIRED_MODEL_SLUGS` there for models kept off it), `MODEL_NAMES` in `scripts/build_model_index.py` (display name in `models.json`), and `MODEL_NAMES`, `SHORT_NAMES`, `LIVE_MODEL_IDS` in `app/_lib/charts.ts` keyed by every board `model` id the model ships under (bare slug and provider-prefixed both occur, e.g. `qwen3.8-max` and `qwen/qwen3.8-max`). Bench and problem labels are `BENCH_LABELS` / `PROBLEM_LABELS` there; GPU tabs are `HOME_GPU_TABS` in `app/_lib/models.ts`.
- New lab: `LAB_BRANDS` in `app/_lib/models.ts` plus `public/logos/labs/<lab>.svg`.
- Mini is a homepage `HomeDecks` scroll category on `/` when it debuts, not a `/mini` route. Multi is unpublished. Hard was deprecated on 2026-11-23: removed from `/` and active `/models` rankings, kept as a frozen archive page at `/hard` and dimmed archived section on `/models/[slug]`.
- Old external links land on active bench sections via the redirects noted at the top of `app/{cuda,mega,multi}/page.tsx`, and on the frozen archive at `app/hard/page.tsx`; keep them working.

## Before `kb deploy`

Skim the homepage chart (`/#mega`, `/#cuda`) and `/models` (dark mode, look at the actual render). Run the redaction scan (`media/AGENTS.md`) before any `public/runs` commit. Site palette tokens live in `app/globals.css`; `media/kbh_theme.py` copies them, so a palette change updates both.

## After every finished run

In this order. The publish gates check steps 2, 4, 5 and 6; nothing checks 1, 3 or 7 for you.

1. Pull the archive to `benchmarks/<bench>/outputs/runs/<run_id>` and regrade in isolation (rules in root `AGENTS.md`).
2. Mega only: write the GPU label to `outputs/runs/<run_id>/gpu` (`RTX PRO 6000 Blackwell`, `H100`, `B200`); `build_mega_leaderboard.py` drops rows without it.
3. Write and `git add` the audit YAML in `results/annotations/`.
4. Cuda RTX only: append the run_id to `benchmarks/cuda/results/published_runs.json` `run_ids` (or to its `excluded` map with a reason); the RTX cuda board publishes only what is listed.
5. New model: add its slug to `LIVE_MODEL_SLUGS` (`app/_lib/models.server.ts`), its display name to `MODEL_NAMES` in `scripts/build_model_index.py`, and every board model id it ships under (bare and provider-prefixed, as they appear as `model` in `public/data/catalog.json`) to `MODEL_NAMES`, `SHORT_NAMES` and `LIVE_MODEL_IDS` in `app/_lib/charts.ts`. A model deliberately off the homepage goes in `RETIRED_MODEL_SLUGS` instead. New lab: `app/AGENTS.md`.
6. `kb publish <bench>` (add `--push` or run `kb push-runs <bench>` for the HF transcripts). If the gate fails, fix what it names and rerun; the commit and deploy do not happen until it passes.
7. `kb deploy`, then open the live homepage and confirm the model is in the roster and the new cell is on the board for that GPU tab. The push is on `master` and Vercel builds it; the commit email must be `elliot@arledge.net` or nothing deploys. Write-ups lead with what the traces show (`media/AGENTS.md`).
