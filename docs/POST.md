# KernelBench short post

One file. Future agents write model posts from this. Articles stay in
`docs/ARTICLE.md`. Do not invent a second post skeleton.

Style bar (read, do not copy adjectives):
https://x.com/elliotarledge/status/2090458795172360257
(2026-08-20 DeepSeek V4 Pro Hard. Live short-post.)

Fable Mega (older, same skeleton):
https://x.com/elliotarledge/status/2072814573753975266

Hard vs Hard update (two-model, field is the other model):
https://x.com/elliotarledge/status/2075415715306410147

Live edits from 2090458795172360257:
- "June checkpoint" not "June Pro" in the header.
- No harness ids in "It was tested on".
- Rest of deck is one cell per line.
- Two images are fine when the extra chart is the rest-of-deck.
- Never write `x.data` or `foo.data_ptr`. X autolinks `x.data`. Say "the input pointer" or wrap as `` `data_ptr()` `` without a dotted prefix.
- FILL-IN is Elliot. "no comment" is valid if they have not used the model.

Score unit: `% of roofline` when the ceiling is real (GEMM). Milliseconds or microseconds when the ceiling is a dense-eq artifact (CUDA Native Sparse Attention bills `4*B*H*S*S*D`). ncu only when measured this turn. Never invent occupancy, hit rate, or GB/s. Grader `gbps=` in benchmark.log is allowed. Occupancy is not the Native Sparse Attention story until ncu says it is.

Names in the post: `DeepSeek Native Sparse Attention`, not bare `NSA`. No `S=` / `D=` shape tuples.

The rank chart is the field. Do not reprint it as a `>` list. Do not write "sits between" / "ahead of". Do not say "dense-eq board". Pick the standout cell for this model, not the cell that was interesting in an earlier chat.

Do not repeat the problem name and the score in a header line and then again in "It was tested on". One sentence: model, bench, problem, GPU, score. Peers that are not in the chart may stay. The GPU belongs in that sentence.

## Pipeline — folders are the state

```
media/posts/unaudited/   isolated regrade just landed. Number may live here only.
media/posts/audited/     annotation YAML exists. Draft complete except FILL-IN.
media/posts/posted/      user said post. Tweet id in the folder.
```

Runs land → `unaudited/` → reward-hack YAML → move to `audited/` → Elliot
fills FILL-IN → website live 200s → user says post → `posted/`.

No YAML → the folder is `unaudited/`. A caveat in the text does not move it.
Do not start `audited/` without `benchmarks/<bench>/results/annotations/<run_id>.yaml`.
`verdict: contamination` is not `audited/`. Do not draft that cell's number.
Do not post from `unaudited/`. Do not post without FILL-IN and an explicit "post".
Do not post before the live site has the cell. The website is first.

One post = one model on one bench. Default GPU: RTX PRO 6000. A second SKU
(H100) only if that cell exists and is the point.

## Names

Same table as `docs/ARTICLE.md`. Pull live. Do not guess.

Field numbers: re-read the published board in this turn
(`public/data/mega/results.csv`, `benchmarks/hard/results/leaderboard.json`,
`benchmarks/cuda/results/leaderboard.json`). Same SKU only. Stale memory is
wrong. Unpublished / unaudited cells are not field.

## Skeleton — every post, same order

```
{Header: one line. Model + what it wrote. Punch matches the number.
 Not SOTA if it is not first. No "real". No "isolated regrade".}

It was tested on: {problem} for {GPU}. {Nx} the optimized PyTorch baseline.

> {peer} at {score}
> {peer} at {score}
> {peer} at {score}

{What the kernel did. Short. Isolated Hard problem = a few sentences.
 Mega may take one extra paragraph. Do not invent novelty. If it is a
 known human pattern, say that.}

FILL-IN
(Elliot. Daily use. Do not LLM this.)

https://kernelbench.com/{bench}
```

Tweet 1 is that skeleton plus the image. Keep the board URL there.

Tweet 2 is the rest of the site for this run. Do not leave these only in
the first tweet:

```
https://kernelbench.com/{bench}
https://kernelbench.com/{kernel path}
https://huggingface.co/datasets/Infatoshi/kernelbench-{bench}-traces/blob/main/{run_id}.jsonl
```

Kernel path: Mega is `/data/mega/code/{run_id}.solution.py.txt`. Hard and
CUDA are `/runs/{run_id}_solution.py.txt`. The HF URL is this run's
jsonl, not the dataset root. Catalog `trace_url` is not proof. Every href
must be a live 200 (HF `/blob/` and `/resolve/` both). If convert skipped
the run, fix the parser and upload before posting.

Score wording: Mega/Hard/CUDA public copy says `Nx` / `% of roofline` against
the deck floor (`baseline.py` on Mega = optimized PyTorch). `reference.py`
is the correctness oracle, not the speed number. Do not say "isolated
regrade" in the post. That stay in the annotation.

One image. `kbh_theme`. Current published field, subject green. Inspect the PNG.

Do not add FlashKDA / Marlin / vLLM / other outside kernels. Same factory as
the Fable Mega post.

## Site first

`kb publish` then commit + push (`kb deploy` only when the tree is clean
enough to ship). Prove these three live 200s before any X post:

1. `https://kernelbench.com/{bench}`
2. the kernel file
3. this-run HF trace

A local `public/data` write is not synced.

## FILL-IN

Stop and ask. Paste the draft. Wait. After they type the paragraph, put it
in the FILL-IN slot. Then wait for "post".

## Charts, covers, write-ups (shared with articles)

- Charts use the site palette via `from kbh_theme import C, SERIES, apply` (`media/kbh_theme.py`, tokens copied from `app/globals.css`: bg `#111111`, NVIDIA green `#76b900`, fg `#eeeeee`/`#999999`, warn `#fbbf24`, bad `#fb7185`, grid `#242424`). Green = subject/ceiling; rose hatched = reward hack; amber = warn; grey = fail; faded dotted = real kernel that bugged or timed out. If `globals.css` changes, update the theme module.
- Two chart kinds, nothing else. (1) The rank chart: the field on one bench and one problem, subject green, field grey, numbers on the bars, no title (`save_rank` in `media/posts/unaudited/make_charts.py`). (2) The highlight chart: one run's annotated optimization trajectory, `media/trajectory.py <run_dir>` (the Fable 5 Mega 18.7x chart, the best-performing post to date). It takes a one-line title and subtitle, wall clock on top, estimated cumulative output tokens below, every in-session `benchmark.py` result as a point, the audit's `trajectory:` checkpoints as labels. Nothing else goes on a PNG: no methodology paragraphs, caption essays, or multi-GPU grids. Older `make_*.py` with header essays are not the pattern.
- The highlight cell per model per bench is the argmax clean cell in the annotation YAMLs, or the `interesting` one when the trace is the story. Its post leads with what the trace shows (design share of the session, the moves, the measured-and-reverted regression), then the number. Write the `trajectory:` list into that cell's annotation YAML during the audit (schema in `results/annotations/SCHEMA.md`); the chart is not drawn without it.
- Covers are launch cards, not report cards: 5:2 only via `media/thumb_card.py` (3000x1200). Official lab mark plus one subject token at the same visual size (Grok 5 = black-hole G + `5`, DeepSeek = `0731`). Kimi ships a black tile; DeepSeek / Qwen / Grok do not, so never invent one for a transparent SVG. No lab name, no KernelBench signature, no charts, bars, pass counts, peak_fraction, GPU lists, audit tallies, or taglines. Inspect the PNG before upload.
- Write-ups lead with the unique, interesting, or inconsistent, not adjectives. Read two or three transcripts and solutions for the headline cells first: behavioural shifts, metric artifacts (topk's ~0.02 ceiling is launch-overhead-bound for every model), what the winning kernel did, suspicious cross-model convergence.
- Redaction scan before any HF push, `public/runs` commit, or deploy: `uv run python scripts/redaction.py runs public/runs`, then block if this finds anything:
  `rg -n "# AGENTS\\.md instructions|<proactive-behavior>|~/.codex/AGENTS\\.md|~/.claude/CLAUDE\\.md|GOG_KEYRING_PASSWORD=|[A-Z0-9_]*(API_KEY|TOKEN|SECRET|PASSWORD)=" runs public/runs`
- Post drafts (`X-*.md`/`.txt`) and rendered PNGs are throwaway: drafts are never committed, PNGs are gitignored and regenerate from the tracked `media/*.py`. Ask when the post went live, then delete the drafts and PNGs. Never delete the generators.
- Transcript viewers: the canonical extractor is `scripts/transcript-extraction/` (see its `VENDORED.md`); the per-bench `src/viewer/parsers/*` under-extract. Native `claude` and `codex` encrypt chain-of-thought, so an empty reasoning trace for Opus or GPT is the API, not a viewer bug; the `*-claude` routes to open providers return full thinking.
