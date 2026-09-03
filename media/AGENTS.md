# media/ — charts, covers, short posts, X Articles

One file. Future agents write model posts and articles from this. Do not invent a second skeleton or a second style. Tracked here: chart generators (`kbh_theme.py`, `make_*.py`, `trajectory.py`, `thumb_card.py`, `check_article_links.py`) and the `posts/` pipeline folders. Rendered PNGs and post drafts are throwaway (gitignored, regenerate from the `.py`).

## Charts, covers, write-ups (shared by posts and articles)

- Charts use the site palette via `from kbh_theme import C, SERIES, apply` (`media/kbh_theme.py`, tokens copied from `app/globals.css`: bg `#111111`, NVIDIA green `#76b900`, fg `#eeeeee`/`#999999`, warn `#fbbf24`, bad `#fb7185`, grid `#242424`). Green = subject/ceiling; rose hatched = reward hack; amber = warn; grey = fail; faded dotted = real kernel that bugged or timed out. If `globals.css` changes, update the theme module.
- Two chart kinds, nothing else. (1) The rank chart: the field on one bench and one problem, subject green, field grey, numbers on the bars, no title (`save_rank` in `media/posts/unaudited/make_charts.py`). (2) The highlight chart: one run's annotated optimization trajectory, `media/trajectory.py <run_dir>` (the Fable 5 Mega 18.7x chart, the best-performing post to date). It takes a one-line title and subtitle, wall clock on top, estimated cumulative output tokens below, every in-session `benchmark.py` result as a point, the audit's `trajectory:` checkpoints as labels. Nothing else goes on a PNG: no methodology paragraphs, caption essays, or multi-GPU grids. Older `make_*.py` with header essays are not the pattern.
- The highlight cell per model per bench is the argmax clean cell in the annotation YAMLs, or the `interesting` one when the trace is the story. Its post leads with what the trace shows (design share of the session, the moves, the measured-and-reverted regression), then the number. Write the `trajectory:` list into that cell's annotation YAML during the audit (schema in `benchmarks/hard/AGENTS.md`); the chart is not drawn without it.
- Covers are launch cards, not report cards: 5:2 only via `media/thumb_card.py` (3000x1200). Official lab mark plus one subject token at the same visual size (Grok 5 = black-hole G + `5`, DeepSeek = `0731`). Kimi ships a black tile; DeepSeek / Qwen / Grok do not, so never invent one for a transparent SVG. No lab name, no KernelBench signature, no charts, bars, pass counts, peak_fraction, GPU lists, audit tallies, or taglines. Inspect the PNG before upload.
- Write-ups lead with the unique, interesting, or inconsistent, not adjectives. Read two or three transcripts and solutions for the headline cells first: behavioural shifts, metric artifacts (topk's ~0.02 ceiling is launch-overhead-bound for every model), what the winning kernel did, suspicious cross-model convergence.
- Redaction scan before any HF push, `public/runs` commit, or deploy: `uv run python scripts/redaction.py runs public/runs`, then block if this finds anything:
  `rg -n "# AGENTS\\.md instructions|<proactive-behavior>|~/.codex/AGENTS\\.md|~/.claude/CLAUDE\\.md|GOG_KEYRING_PASSWORD=|[A-Z0-9_]*(API_KEY|TOKEN|SECRET|PASSWORD)=" runs public/runs`
- Post drafts (`X-*.md`/`.txt`) and rendered PNGs are throwaway: drafts are never committed, PNGs are gitignored and regenerate from the tracked `media/*.py`. Ask when the post went live, then delete the drafts and PNGs. Never delete the generators.
- Transcript viewers: the per-bench `src/viewer/parsers/*` under-extract vs a full-harness dump (reference: https://github.com/0xSero/ai-data-extraction). Native `claude` and `codex` encrypt chain-of-thought, so an empty reasoning trace for Opus or GPT is the API, not a viewer bug; the `*-claude` routes to open providers return full thinking.

## Names — never freestyle

Pull from live tables. If a key is missing, print the raw slug and stop to add the pretty name. Do not guess capitalization.

| Thing | Source | Rule |
|---|---|---|
| Display name | `app/_lib/charts.ts` `MODEL_NAMES` | `DeepSeek V4 Flash (0731)`, `Qwen 3.8 Max`, `Claude Opus 5` |
| Short label | `SHORT_NAMES` in the same file | charts only |
| Site slug | `public/data/models.json` / `kb publish` | `deepseek-v4-flash-0731` |
| Harness | `kbtool/AGENTS.md` harness table case label | backtick the runner name: `` `or-fable` ``, `` `qwen-claude` ``, `` `codex` `` |
| Model id | the argv / archive slug | keep the provider path: `deepseek/deepseek-v4-flash-0731` |
| GPU | `app/_lib/models.ts` `HOME_GPU_TABS` | Default prose: `RTX PRO 6000`. Never bare `H100`. A second SKU only if the kernel used a feature that card does not have. |
| Bench | `BENCH_LABELS` | CUDA / Mega / Hard |
| Problem | `PROBLEM_LABELS` | `FP8 GEMM`, `Sonic MoE`, not `01_fp8_gemm` in prose |
| Run id | archive basename | `20260801_205612_or-fable_deepseek_deepseek-v4-flash-0731_01_fp8_gemm` — this is the high-taste ID. Quote it. Do not invent a prettier one. |

Names in prose: `DeepSeek Native Sparse Attention`, not bare `NSA`. No `S=` / `D=` shape tuples. 0731 is a post-train: write `DeepSeek V4 Flash (0731)`, not "V4 Flash" as if it were the base. Harness in the first setup sentence is **how it ran**, not a brand mash: `or-fable` on OpenRouter's Anthropic route, model `qwen/qwen3.8-max`, effort `xhigh`.

Field numbers: re-read the published board in this turn (`public/data/mega/results.csv`, `benchmarks/hard/results/leaderboard.json`, `benchmarks/cuda/results/leaderboard.json`). Same SKU only. Stale memory is wrong. Unpublished / unaudited cells are not field.

## Numbers

- Headline winners = argmax of annotation YAML on that board. If the draft contradicts the YAML, the draft is wrong. If the YAML does not exist, stop; do not draft the number with a caveat. Isolated regrade is not an audit.
- Published timing = sequential isolated re-grade on the stamped SKU (`gpu_name` in `result.json`). In-run flywheel times stay in the archive.
- Score unit: `% of roofline` when the ceiling is real (GEMM). Milliseconds or microseconds when the ceiling is a dense-eq artifact (CUDA Native Sparse Attention bills `4*B*H*S*S*D`). Mega/Hard/CUDA public copy says `Nx` / `% of roofline` against the deck floor (`baseline.py` on Mega = optimized PyTorch). `reference.py` is the correctness oracle, not the speed number.
- ncu only when measured this turn. Never invent occupancy, hit rate, or GB/s. Grader `gbps=` in benchmark.log is allowed. Occupancy is not the Native Sparse Attention story until ncu says it is.
- Wrong-SKU, contamination, reward-hack, rubric-leak: name them. Do not narrate them as board wins.

## Links — live GET, or do not embed

Kernel path: Mega is `/data/mega/code/{run_id}.solution.py.txt`. Hard and CUDA are `/runs/{run_id}_solution.py.txt`. The HF URL is this run's jsonl, not the dataset root: `https://huggingface.co/datasets/Infatoshi/kernelbench-{bench}-traces/blob/main/{run_id}.jsonl`. Every href must be a live 200 (HF `/blob/` and `/resolve/` both). If convert skipped the run, fix the parser and upload before posting.

Hugging Face "unresolved" is a missing file (404), not a private dataset. `Infatoshi/kernelbench-{hard,cuda,mega}-traces` are public. Do not flip them private.

- Never invent a path. Do not prefix `h100/` because the cell is an H100 cell. Canonical RTX files are `/blob/main/{run_id}.jsonl`. H100 files are `/blob/main/h100/{run_id}.jsonl` only when that object exists on HF.
- Catalog `trace_url` is a hint. `build_model_index.py` can bake `h100/` for a run that was never pushed. A catalog URL is not proof the file is live.
- Rejected / unpublished cells have no HF object until someone converts and uploads them. Quote the run id in prose. Embed a href only after `check_article_links.py` returns 200 on that exact URL.
- Wrap every URL in `[label](url)`. Bare `_` in a URL italicizes the markdown.
- The checker also GETs the HF `/resolve/main/` twin and refuses an HTML "unresolved" / "Entry not found" page even if status is 200.

---

# Short posts

Style bar (read, do not copy adjectives): https://x.com/elliotarledge/status/2090458795172360257 (2026-08-20 DeepSeek V4 Pro Hard. Live short-post.) Fable Mega (older, same skeleton): https://x.com/elliotarledge/status/2072814573753975266. Hard vs Hard update (two-model, field is the other model): https://x.com/elliotarledge/status/2075415715306410147.

Live edits from 2090458795172360257:
- "June checkpoint" not "June Pro" in the header.
- No harness ids in "It was tested on".
- Rest of deck is one cell per line.
- Two images are fine when the extra chart is the rest-of-deck.
- Never write `x.data` or `foo.data_ptr`. X autolinks `x.data`. Say "the input pointer" or wrap as `` `data_ptr()` `` without a dotted prefix.
- FILL-IN is Elliot. "no comment" is valid if they have not used the model.

The rank chart is the field. Do not reprint it as a `>` list. Do not write "sits between" / "ahead of". Do not say "dense-eq board". Pick the standout cell for this model, not the cell that was interesting in an earlier chat. Do not repeat the problem name and the score in a header line and then again in "It was tested on". One sentence: model, bench, problem, GPU, score. Peers that are not in the chart may stay. The GPU belongs in that sentence.

## Pipeline — folders are the state

```
media/posts/unaudited/   isolated regrade just landed. Number may live here only.
media/posts/audited/     annotation YAML exists. Draft complete except FILL-IN.
media/posts/posted/      user said post. Tweet id in the folder.
```

Runs land → `unaudited/` → reward-hack YAML → move to `audited/` → Elliot fills FILL-IN → website live 200s → user says post → `posted/`.

No YAML → the folder is `unaudited/`. A caveat in the text does not move it. Do not start `audited/` without `benchmarks/<bench>/results/annotations/<run_id>.yaml`. `verdict: contamination` is not `audited/`. Do not draft that cell's number. Do not post from `unaudited/`. Do not post without FILL-IN and an explicit "post". Do not post before the live site has the cell. The website is first.

One post = one model on one bench. Default GPU: RTX PRO 6000. A second SKU (H100) only if that cell exists and is the point.

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

Tweet 1 is that skeleton plus the image. Keep the board URL there. Tweet 2 is the rest of the site for this run (board URL, kernel path, this-run HF trace); do not leave these only in the first tweet. Do not say "isolated regrade" in the post; that stays in the annotation. One image, `kbh_theme`, current published field, subject green, inspect the PNG. Do not add FlashKDA / Marlin / vLLM / other outside kernels. Same factory as the Fable Mega post.

## Site first

`kb publish` then commit + push (`kb deploy` only when the tree is clean enough to ship). Prove these three live 200s before any X post: `https://kernelbench.com/{bench}`, the kernel file, this-run HF trace. A local `public/data` write is not synced.

## FILL-IN

Stop and ask. Paste the draft. Wait. After they type the paragraph, put it in the FILL-IN slot. Then wait for "post".

---

# X Articles

Style bar (read, do not copy adjectives): https://x.com/elliotarledge/status/2078048144844280315. Locked 2026-08-13. Same skeleton every article.

Worked examples in-repo:
- `media/X-article-v4flash/00_article.txt`
- `media/X-article-qwen38/00_article.txt`
- `media/X-article-opus5/00_article.txt`
- `media/X-article-grok5/00_article.example.txt` — imaginary Grok 5; fake numbers; shape only
- `media/X-article-grok5/preview.html` — phone-width scroll of that example

## Cover

`media/thumb_card.py`. Official lab mark. No fake black tile. No stats. Lockup is **mark + one subject token**, same visual size as the mark.

- Grok 5 → official black-hole G (`xai-logo-1024.png`) + the digit `5`. Do not print `GROK`. The old xAI X-mark is wrong.
- DeepSeek → official whale + `0731`.
- Qwen → official hex pinwheel (`qwen-logo-1024.png` / `public/logos/labs/qwen.svg`, same mark as qwen.ai and HF). The old C-dash chat icon is wrong.
- Do not print the lab name next to the lab mark. No KernelBench signature in the corner.

## GPU and compute — one board, no explosion

Do not report Hard×CUDA×Mega × H100×RTX×B200 × every harness. That is a spreadsheet, not an article.

- Default SKU is **RTX PRO 6000 Blackwell**. Home method: run agents in parallel on the cards that are actually there.
- A second SKU (H100, B200, later Rubin) only appears if the kernel used a feature that card does not have. B200 example: `tcgen05.mma`, tensor memory. Consumer Blackwell does not get those. Otherwise skip B200.
- Infer compute from the run you launched. `gpu_name` in `result.json` plus who spun the box (home / Lambda / Brev). Do not ask the human. Do not write "Thanks to @LambdaAPI". State the box as a fact if it matters (`Lambda gpu_1x_rtx_pro_6000`, `home RTX PRO 6000`). If the provider is not in the launch record, omit it.
- In-body charts: one GPU. Not a 2- or 3-panel SKU grid unless the SKU delta is the point.

## Skeleton — every article, same order

Readers should be able to predict the next heading.

1. **Title.** `{Display name} on KernelBench`
2. **Cover.** See above.
3. **Drop.** Two or three blunt sentences. The model just shipped. What the timeline claimed (rumors, training data, "it is good at X"). Why you ran it: hand-curated problems, then a manual audit of every headline trace and solution so it is not gaming the bench. No adjectives. No speculation. Good and bad, same voice.
4. **How I bench.** Paste the boilerplate below. Do not rewrite it. Do not skip it. New readers need this every time.
5. **Using it.** Authentic daily-use paragraph. Ignore the scores. How it talks, how it ships work, whether it generalizes vs the last model you actually used. **Voice-typed by Elliot. Do not LLM this. If it is not in the prompt, stop and ask.** This is the only required human paragraph.
6. **Board.** Publish-grade counts only, default GPU only. Wrong-SKU / contaminated / reward-hack cells named as excluded. Not a second TLDR essay.
7. **Per-bench body** in this order, skip a bench that did not run: **CUDA → Mega → Hard last**.
   - Do not re-explain why the problem exists. The boilerplate already said what each bench is.
   - Standout cell: what the kernel did, the isolated number, solution + trace links.
   - Quote the agent only if the quote is funny or load-bearing (sandbox break, self-caught cheat, priced a rejected alternative). "this could be a massive head start" is not a quote.
   - Hard is optional. Skip it when CUDA and Mega already showed the same failure (weird fusion, identity cache, language-gate dodge).
8. **What transferred / what did not.** Two short lists. This is the profile.
9. **Honesty list.** Every excluded headline cell, with why. No silent drops.
10. **Close.** One useful sentence. Then `https://kernelbench.com` and the HF traces. Optional board deep-links: `/cuda`, `/mega`, `/hard`. **Never** `/models` or `/models/{slug}`. No thank-you for compute.

Do not put peak tables on the cover. Do not lead with adjectives. Do not invent a second methodology essay around the boilerplate.

## How I bench — paste this, fill the brackets

```
Every KernelBench cell is the same job. The model gets the problem, a live GPU, and a coding-agent harness with the env it needs to compile, profile, and iterate without me in the loop. Unlimited wall-clock. It runs until it decides it is done.

I do not report the in-run flywheel number. After the session I take the finished kernel, re-grade it alone on a quiet GPU, and read the solution and the full reasoning trace by hand. Same-buffer overwrite on anything graph- or cache-shaped. Numeric stress on. If it gamed the checker, that cell is not a board number.

This run: [`harness`] , model [`id`], effort [`tier`]. GPU [`sku from result.json`]. Box [`home | Lambda type | Brev name — inferred, not asked`].

CUDA first: forced CUDA / PTX. Triton and kernel DSLs fail. That is the real writing test — warp specialization, async copies, Blackwell tricks that a Triton path will not carry.

Mega second: fused systems. A decode megakernel, or a Craftax-style RL sim in pure CUDA.

Hard last, and I skip it when CUDA and Mega already showed the same failure. Per-op Triton-or-CUDA (paged attention, TopK, KDA) is no longer the interesting deck.
```

Swap only the bracket slots. Leave the rest.

## What to mine from traces

Read 2–3 headline transcripts and the matching `solution.py` before any sentence. Surface only what a leaderboard cell cannot show: a behavioral shift vs the previous model from that lab; a metric artifact that is true for every model (TopK launch-bound ~0.02); the actual kernel move (one launch, fragment ownership, same-buffer fail); profiling that priced a rejected alternative; cross-run contamination or grader mutation, in the agent's own words. If you cannot point at a run id, do not write the claim. Be blunt about both sides. A clean kernel and a reward-hack in the same wave both get named. Do not speculate about why the lab trained it that way.

## In-body charts

Green on black. `kbh_theme`. Square bars. Short labels (`FP8`, `Sonic`, `Paged`). One GPU. Subject bar `#76b900`, field `#4d5d66`. Numbers on the bars. No title essay. Generate with `media/make_<slug>_article.py`, inspect the PNG, then `![short alt](01_….png)`. The standout cell of each bench section gets the highlight chart instead: `media/trajectory.py <run_dir>` with that run's `trajectory:` checkpoints in its annotation YAML (charts section above).

## Files and ship path

```
media/X-article-<slug>/
  00_article.txt          # frontmatter title: + cover:
  00_thumbnail.png        # from thumb_card / make_*_thumb.py
  01_*.png                # in-body charts from media/make_*.py (kbh_theme)
media/X-article-drafts.json
```

```
uv run --no-project python media/check_article_links.py media/X-article-<slug>/00_article.txt
# refuse on any non-200 or HF "unresolved" page. Do not draft until this exits 0.
x-cli -j article validate media/X-article-<slug>/00_article.txt
# draft only when the user says draft
x-cli -j article draft media/X-article-<slug>/00_article.txt
# publish only when the user says publish / publish both
x-cli -j article publish <id>
```

X Article API is draft + publish only. No list/get/edit/delete. One draft per ~15 min. Do not `--force` a deleted hash. Friend reviews the write-up. You do not publish on taste. After it goes live: delete ephemeral `X-*.md` drafts and generated PNGs. Keep the `.py` generators.

## Do not

- invent a pretty model name
- write `/mini` (Mini is a homepage scroll category)
- post an unpublished Mini deck
- hide CUDA or skip the isolated re-grade to "make the article"
- put report-card stats on the cover
- print `GROK` / the lab name on the cover next to the mark
- put a KernelBench signature on the cover
- link `kernelbench.com/models` or `/models/{slug}`
- thank a compute provider (infer the box; do not ask; do not @)
- re-explain why each problem is in the deck
- LLM the **Using it** paragraph
- mention `@` handles you have not verified
- report a 2- or 3-GPU grid because the cards exist
- paste a Hugging Face href you have not GET'd at 200 (HF "unresolved" is a refuse)
- invent a `h100/` prefix
- draft before `check_article_links.py` exits 0
