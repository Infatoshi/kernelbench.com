# kernelbench.com — operator guide

Entrypoint for every harness (`CLAUDE.md` and `.cursorrules` symlink here): rules, gates, and pointers only, under 10 KB (Grok truncates there; a test enforces it). Open the file below that your task touches before starting.

Monorepo for the KernelBench website and the evals. Canonical checkout is the Mac at `~/dev/sites/kernelbench.com` (edit, publish, deploy, orchestrate). GPU sessions run on rented workers (Lambda, Brev, Verda); anvil may hold a disposable checkout but is never source of truth. Site: Next.js 16, Tailwind, bun; Vercel builds on push; `kb deploy` from the Mac.

## Where to read next

- `docs/REMOTE.md` — rented GPU workers: `kb lambda` CLI, bootstrap order, ncu on VMs, pulling archives back, Brev teardown, multi-node topology gate.
- `docs/HARNESSES.md` — every harness branch (transport, key, benches, quirks), route notes, runner behaviour, workspace/GPU-lock isolation, broad-sweep launcher. Enforced by `kbtool/tests`.
- `docs/ENV.md` — every `KB_`/`KBH_`/`KBM_`/`KBMINI_` variable, publish-affecting ones flagged. Enforced by `kbtool/tests`.
- `docs/TORCH.md` — torch pins per bench and `patch_torch.sh`.
- `docs/POST.md`, `docs/ARTICLE.md` — short X posts and X Articles: skeleton, names, charts, covers, redaction scan, ship path.
- `benchmarks/hard/README.md` — developer guide for the single-GPU benches: layout, adding a problem, correctness, results, tests, sweep failures.
- `benchmarks/<bench>/SPEC.md` methodology, `DEVLOG.md` history and war stories (newest first), `README.md` the deck.
- `benchmarks/mini/SPEC.md` — mini runbook and pre-debut checklist.
- `benchmarks/<bench>/results/annotations/SCHEMA.md` — audit YAML schema.
- `benchmarks/v3/AGENTS.md` — the archive only; ignore unless working v3.

## Benches

| bench | path | deck | drive it with | site |
| --- | --- | --- | --- | --- |
| hard | `benchmarks/hard/` | 6 per-op kernels, CUDA or Triton; per-GPU decks `problems-rtxpro6000` (default), `-h100`, `-b200` | `kb sweep` / `uv run kbh run` | `/hard` |
| mega | `benchmarks/mega/` | 1 fused megakernel, `02_kimi_linear_decode` | `cd benchmarks/mega && ./scripts/run_hard.sh` | `/mega` |
| cuda | `benchmarks/cuda/` | 4 CUDA-only problems; `src/eval/cuda_language.py` fails Triton, DSLs, pure torch | `kb -b cuda ...` | `/cuda` |
| mini | `benchmarks/mini/` | 4 problems, sub-200B open weights, 30-min cap, 5 repeats, Lambda H100 SXM | `cd benchmarks/mini && ./scripts/sweep_mini.sh` | homepage category; unpublished, keep out of posts |
| multi | `benchmarks/multi/` | 4xH100 NVLink, frontier roster only | `benchmarks/multi/scripts/sweep_wave.sh` | unpublished |

Hard, mega, and cuda share harness, archive, and roofline code and run unlimited wall-clock. Removed problems stay removed: hard `04_kahan_softmax`, mega `01_rl_grid_ppo`. Eligibility is `benchmarks/<bench>/roster.yaml`: open benches do not enumerate models (the leaderboard is the list); multi refuses off-roster before any GPU time because one session holds four H100s; a `KBM_ALLOW_OFF_ROSTER=1` cell is archived but never publishable.

## "Do a sweep of <model>" is an order, not a question

State the assumption in one line ("Assuming: all problems, hard + mega, audited, published") and go. Scope: every problem in the deck for the current GPU plus the mega deck; an audit YAML for every passing cell; contamination check, redaction, `kb publish`, commit, push. Existing valid cells for the same (model, harness, problem, GPU) are not rerun. The human coming back to finished, audited, published results is the success condition. Interrupt only for `STOP: needs $X_API_KEY` (append the key to `~/.env_vars`, rerun), a shared GPU that never frees, or an ambiguous model identity. Post to X only after the live board, this run's kernel file, and this run's HF trace return 200.

```
kb sweep <harness> <model>      # all hard problems, parallel containers, unlimited time
kb publish [hard|cuda|mini|mega] # rebuild leaderboard, viewers, models.json from archives
kb deploy "<msg>"               # publish + commit + push
kb -b <cuda|mini> run|sweep|audit|lint|traces-to-hf ...   # other benches; mega and multi keep their own drivers
kb lambda ... | kb brev ... | kb contamination <bench> | kb push-runs <bench> | kb help
```

`kb` is the `kbtool/` uv package; `bin/kb` shims it. A new provider needs a harness branch (copy `kimi-claude` in `scripts/lib/run_harness.sh`) and a row in `docs/HARNESSES.md`.

## Non-negotiable

- uv only, never bare python or pip. `uv run pytest` before committing. Commit email `elliot@arledge.net` or Vercel silently fails the build.
- Never edit `problems/*/solution.py` (agent output; read it from the run archive). Never change `reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, or `PROMPT.txt` after publish unless deliberately versioning the bench; the runner snapshots them and invalidates a run that mutates them.
- GPU work goes through the harness (`kb`, `uv run kbh run`, or the bench's `run_hard.sh`) in container mode: per-run workspace, isolated caches, per-bench GPU lock. Never hide CUDA from the agent or prohibit check, benchmark, or profile. `benchmark.py` scores `variant=solution` first. Run `./scripts/patch_torch.sh` after every `uv sync`.
- Every artifact stays in this repo, in its subfolder, on every machine: archives in `benchmarks/<bench>/outputs/runs/`, scripts in `scripts/`, locks under `outputs/gpu_lock/`. Nothing in `$HOME` or `/tmp`. Pull worker archives back before teardown; a stranded archive is invisible to publish, contamination, and regrade.
- Archives are thin (`.venv` stripped after scoring). Before any pull: `du -sb`, print full vs tiny, pull the tiny set (`result.json`, `solution.py`, `gpu`, logs, sidecar `.cu`). Over 20 MB say the size and wait; over 1 GB refuse until the user has seen it.
- Secrets never in argv or repo files. Provider credit/rate detection reads CLI/API error events and stderr only, on rows without a solution.
- Teardown by pidfile, never `pkill -f`. Confirm `kb lambda ls` / `brev ls` empty; Brev teardown only via `scripts/brev_teardown.sh`. Idle nodes bill the credits.
- Redact on every publish or push (`uv run python scripts/redaction.py runs public/runs`, then the rg scan in `docs/POST.md` must be empty). Run `kb contamination <bench>` before publishing; agents can read `outputs/runs`, and the builders exclude runs whose transcript touches another archive.
- Post drafts and PNGs are ephemeral; delete after posting. Only `media/*.py` generators are tracked.

## A number is publishable only when all of these hold

1. GPU stamp matches: `result.json` `gpu_name` maps via `src/hardware/identify.py` to the deck's hardware key. Empty `CUDA_VISIBLE_DEVICES=` is illegal (it hides CUDA). H100 SXM and PCIe are different keys; a 3090 or RTX 6000 Ada never grades as RTX PRO 6000.
2. Sequential isolated regrade: `check.py` then `benchmark.py` on the final `solution.py`, one GPU owner, zero compute PIDs, no other CUDA jobs. In-run numbers during concurrent agents are timing-contaminated and stay in the archive for debugging only.
3. Audit YAML exists at `benchmarks/<bench>/results/annotations/<run_id>.yaml` with a `verdict:` (`clean` | `reward_hack` | `contamination` | `rubric_leak` | `interesting` | `bug`). An audit is: read `solution.py` end to end and confirm it computes the real op; read the trace for `check.py` sniffing, tolerance edits, grader tampering, and any read or copy of `outputs/runs*`; for any cache, CUDA-graph, `data_ptr`, or identity pattern, overwrite the same input buffer on a quiet GPU and quote `cos(out1,out2)` low and `cos(ref,sol)` at gate; confirm numeric stress ran. A literal copy of another archive's solution is `contamination` even if the overwrite passes. `kb lint` and isolated regrade are tripwires, not audits. Without the YAML you may not write the cell's peak, speedup, or "the kernel did X" anywhere a human reads it, caveat or not: say `regrade done, audit pending` and stop. While babysitting a live session the in-run number may be printed only with the words `in-run, not a result`. Dispatch the audit without being asked.
4. Served model id parsed from the transcript matches the requested cell. A Fable to Opus swap, z.ai `glm-5.1` to `glm-5.2`, or a dropped `--effort` is not the cell. Claude runs: fast mode off, thinking on, Opus at `--effort max`.
5. Headline winners equal the argmax of the annotation YAMLs. Where the roofline is structurally unreadable (`05_topk_bitonic`, cuda `02_deepseek_nsa`) the headline is milliseconds and geomean speedup vs the deck's frozen eager anchor; the site's best-to-worst span is presentation only.
6. Wave complete means a `result.json` on the Mac for every cell. A live tmux, an SSH 255, or a nohup that inits then exits is not launched.

## Publish

`kb publish <bench>` regenerates `results/leaderboard.json` (mega: `public/data/mega/results.csv`), the redacted `public/runs/*_solution.py.txt` kernels, and `public/data/models.json`; never hand-edit them. `git add` new annotation YAMLs before publishing: models.json only joins annotations that git tracks, and an untracked one ships the cell as `unaudited`, which the homepage column chart drops. Transcripts go to HF (`kernelbench-<bench>-traces`) via `kb push-runs <bench>`; the site links each run to its trace. After a new model: `LIVE_MODEL_SLUGS` in `app/_lib/models.server.ts` (homepage and /models roster), `MODEL_NAMES` and `SHORT_NAMES` in `app/_lib/charts.ts`; a new lab needs `LAB_BRANDS` plus `public/logos/labs/<lab>.svg` in `app/_lib/models.ts`. Skim the homepage chart and `/hard`, then `kb deploy`. Charts use `media/kbh_theme.py` (site palette, bars and axes only); covers via `media/thumb_card.py`; write-ups lead with what the traces show.
