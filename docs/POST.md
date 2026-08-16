# KernelBench short post

One file. Future agents write model posts from this. Articles stay in
`docs/ARTICLE.md`. Do not invent a second post skeleton.

Style bar (read, do not copy adjectives):
https://x.com/elliotarledge/status/2072814573753975266

Hard vs Hard update (two-model, same skeleton, field is the other model):
https://x.com/elliotarledge/status/2075415715306410147

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
