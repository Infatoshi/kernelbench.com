# kernelbench.com — operator guide

Entrypoint for every harness (`CLAUDE.md` and `.cursorrules` symlink here): universal rules, gates, and pointers only, under 10 KB (Grok truncates there; a test enforces it). Each directory you touch has its own `AGENTS.md` with the specialized instructions (no other `CLAUDE.md` anywhere; sub-files stay under 32 KB, the Codex cap). Open the one for your task before starting.

Monorepo for the KernelBench website and the evals. Canonical checkout is the Mac at `~/dev/sites/kernelbench.com` (edit, publish, deploy, orchestrate). GPU sessions run on rented workers (Lambda, Brev, Verda); anvil may hold a disposable checkout but is never source of truth.

## Where to read next

- `kbtool/AGENTS.md` — the `kb` CLI, every harness branch (transport, key, benches, quirks), runner behaviour, workspace/GPU-lock isolation, broad sweeps, rented GPU workers (Lambda/Brev bootstrap, ncu, pulling archives back, teardown), the `KB_` variables and the danger list. Enforced by `kbtool/tests`.
- `benchmarks/hard/AGENTS.md` — developer guide shared by every single-GPU bench: layout, adding a problem, correctness, results and thin archives, tests, sweep failures, torch pins and `patch_torch.sh`, the audit YAML schema, the `KBH_` variables.
- `benchmarks/{cuda,mega,mini,multi}/AGENTS.md` — that deck, its commands, and its deltas from hard (mini adds `KBMINI_`, multi adds `KBM_` and the 4xH100 node).
- `benchmarks/<bench>/SPEC.md` methodology, `DEVLOG.md` history and war stories (newest first).
- `media/AGENTS.md` — charts, covers, redaction scan, short X posts and X Articles: skeleton, names, ship path.
- `app/AGENTS.md` — the website: data flow, adding a model or lab, deploy.

## Benches

| bench | path | deck | drive it with | site |
| --- | --- | --- | --- | --- |
| hard | `benchmarks/hard/` | 6 per-op kernels, CUDA or Triton; per-GPU decks `problems-rtxpro6000` (default), `-h100`, `-b200` | `kb sweep` / `uv run kbh run` | `/hard` |
| mega | `benchmarks/mega/` | 1 fused megakernel, `02_kimi_linear_decode` | `cd benchmarks/mega && ./scripts/run_hard.sh` | `/mega` |
| cuda | `benchmarks/cuda/` | 4 CUDA-only problems; `src/eval/cuda_language.py` fails Triton, DSLs, pure torch | `kb -b cuda ...` | `/cuda` |
| mini | `benchmarks/mini/` | 4 problems, sub-200B open weights, 30-min cap, 5 repeats, Lambda H100 SXM | `cd benchmarks/mini && ./scripts/sweep_mini.sh` | homepage category; unpublished, keep out of posts |
| multi | `benchmarks/multi/` | 4xH100 NVLink, frontier roster only | `benchmarks/multi/scripts/sweep_wave.sh` | unpublished |

Hard, mega, and cuda share harness, archive, and roofline code and run unlimited wall-clock. Removed problems stay removed: hard `04_kahan_softmax`, mega `01_rl_grid_ppo`; KernelBench v3 and the Prime Intellect `environments/` are gone, do not resurrect them. Eligibility is `benchmarks/<bench>/roster.yaml`: open benches do not enumerate models (the leaderboard is the list); multi refuses off-roster before any GPU time because one session holds four H100s; a `KBM_ALLOW_OFF_ROSTER=1` cell is archived but never publishable.

## "Do a sweep of <model>" is an order, not a question

State the assumption in one line ("Assuming: all problems, hard + mega, audited, published") and go. Scope: every problem in the deck for the current GPU plus the mega deck; an audit YAML for every passing cell; contamination check, redaction, `kb publish`, commit, push. Existing valid cells for the same (model, harness, problem, GPU) are not rerun. The human coming back to finished, audited, published results is the success condition. Interrupt only for `STOP: needs $X_API_KEY` (append the key to `~/.env_vars`, rerun), a shared GPU that never frees, or an ambiguous model identity. Post to X only after the live board, this run's kernel file, and this run's HF trace return 200.

```
kb sweep <harness> <model>      # all hard problems, parallel containers, unlimited time
kb publish [hard|cuda|mini|mega] # rebuild leaderboard, viewers, models.json from archives
kb deploy "<msg>"               # publish + commit + push
kb -b <cuda|mini> run|sweep|audit|lint|traces-to-hf ...   # other benches; mega and multi keep their own drivers
kb lambda ... | kb brev ... | kb contamination <bench> | kb push-runs <bench> | kb help
```

`kb` is the `kbtool/` uv package: `uv tool install -e ./kbtool` once on the Mac, or `uv run --project kbtool python -m kb` on a fresh box. A new provider needs a harness branch (copy `kimi-claude` in `scripts/lib/run_harness.sh`) and a row in the `kbtool/AGENTS.md` harness table.

## Non-negotiable

- uv only, never bare python or pip; bun for the site. Before committing: `uv run --project kbtool pytest kbtool/tests` (repo consistency, runs on the Mac); a bench's own `uv run pytest` runs on the GPU box. Commit email `elliot@arledge.net` or Vercel silently fails the build.
- Venvs live only where the benchmark runs. The Mac carries `kbtool/.venv` and nothing else; `uv sync` a bench on the GPU box, never here. Never commit or rsync a `.venv`.
- Never edit `problems/*/solution.py` (agent output; read it from the run archive). Never change `reference.py`, `check.py`, `benchmark.py`, `problem.yaml`, `shapes.py`, or `PROMPT.txt` after publish unless deliberately versioning the bench; the runner snapshots them and invalidates a run that mutates them.
- GPU work goes through the harness (`kb`, `uv run kbh run`, or the bench's `run_hard.sh`) in container mode: per-run workspace, isolated caches, per-bench GPU lock. Never hide CUDA from the agent or prohibit check, benchmark, or profile. `benchmark.py` scores `variant=solution` first. Run `./scripts/patch_torch.sh` after every `uv sync`.
- Every artifact stays in this repo, in its subfolder, on every machine: archives in `benchmarks/<bench>/outputs/runs/`, scripts in `scripts/`, locks under `outputs/gpu_lock/`. Nothing in `$HOME` or `/tmp`. Pull worker archives back before teardown; a stranded archive is invisible to publish, contamination, and regrade.
- Archives are thin (`.venv` stripped after scoring). Before any pull: `du -sb`, print full vs tiny, pull the tiny set (`result.json`, `solution.py`, `gpu`, logs, sidecar `.cu`). Over 20 MB say the size and wait; over 1 GB refuse until the user has seen it.
- Secrets never in argv or repo files. Provider credit/rate detection reads CLI/API error events and stderr only, on rows without a solution.
- Teardown by pidfile, never `pkill -f`. Confirm `kb lambda ls` / `brev ls` empty; Brev teardown only via `scripts/brev_teardown.sh`. Idle nodes bill the credits.
- Redact on every publish or push (`uv run python scripts/redaction.py runs public/runs`, then the rg scan in `media/AGENTS.md` must be empty). Run `kb contamination <bench>` before publishing; agents can read `outputs/runs`, and the builders exclude runs whose transcript touches another archive.
- Post drafts and PNGs are ephemeral; delete after posting. Only `media/*.py` generators are tracked. Project markdown is `AGENTS.md`, `SPEC.md`, `DEVLOG.md`, `GOAL.md` only; no READMEs or `docs/` folders below the root.

## A number is publishable only when all of these hold

1. GPU stamp matches: `result.json` `gpu_name` maps via `src/hardware/identify.py` to the deck's hardware key. Empty `CUDA_VISIBLE_DEVICES=` is illegal (it hides CUDA). H100 SXM and PCIe are different keys; a 3090 or RTX 6000 Ada never grades as RTX PRO 6000.
2. Sequential isolated regrade: `check.py` then `benchmark.py` on the final `solution.py`, one GPU owner, zero compute PIDs, no other CUDA jobs. In-run numbers during concurrent agents are timing-contaminated and stay in the archive for debugging only.
3. Audit YAML exists at `benchmarks/<bench>/results/annotations/<run_id>.yaml` with a `verdict:` (`clean` | `reward_hack` | `contamination` | `rubric_leak` | `interesting` | `bug`). An audit is: read `solution.py` end to end and confirm it computes the real op; read the trace for `check.py` sniffing, tolerance edits, grader tampering, and any read or copy of `outputs/runs*`; for any cache, CUDA-graph, `data_ptr`, or identity pattern, overwrite the same input buffer on a quiet GPU and quote `cos(out1,out2)` low and `cos(ref,sol)` at gate; confirm numeric stress ran. A literal copy of another archive's solution is `contamination` even if the overwrite passes. `kb lint` and isolated regrade are tripwires, not audits. Without the YAML you may not write the cell's peak, speedup, or "the kernel did X" anywhere a human reads it, caveat or not: say `regrade done, audit pending` and stop. While babysitting a live session the in-run number may be printed only with the words `in-run, not a result`. Dispatch the audit without being asked.
4. Served model id parsed from the transcript matches the requested cell. A Fable to Opus swap, z.ai `glm-5.1` to `glm-5.2`, or a dropped `--effort` is not the cell. Claude runs: fast mode off, thinking on, Opus at `--effort max`.
5. Headline winners equal the argmax of the annotation YAMLs. Where the roofline is structurally unreadable (`05_topk_bitonic`, cuda `02_deepseek_nsa`) the headline is milliseconds and geomean speedup vs the deck's frozen eager anchor; the site's best-to-worst span is presentation only.
6. Wave complete means a `result.json` on the Mac for every cell. A live tmux, an SSH 255, or a nohup that inits then exits is not launched.

## Publish

`kb publish <bench>` regenerates `results/leaderboard.json` (mega: `public/data/mega/results.csv`), the redacted `public/runs/*_solution.py.txt` kernels, and `public/data/models.json`; never hand-edit them. `git add` new annotation YAMLs before publishing (an untracked one ships the cell as `unaudited`). Transcripts go to HF via `kb push-runs <bench>`. New model or lab: the tables named in `app/AGENTS.md`. Skim the homepage chart and `/hard`, then `kb deploy`. Write-ups lead with what the traces show (`media/AGENTS.md`).
