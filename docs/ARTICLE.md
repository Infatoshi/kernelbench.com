# KernelBench article workflow

One file. Future agents write model posts from this. Do not invent a second style.

Style bar (read, do not copy adjectives):
https://x.com/elliotarledge/status/2078048144844280315

Worked examples in-repo:
- `media/X-article-v4flash/00_article.txt`
- `media/X-article-qwen38/00_article.txt`
- `media/X-article-opus5/00_article.txt`
- `media/X-article-grok5/00_article.example.txt` — imaginary Grok 5; fake numbers; shape only
- `media/X-article-grok5/preview.html` — phone-width scroll of that example

Locked 2026-08-13. Same skeleton every article.

## Names — never freestyle

Pull from live tables. If a key is missing, print the raw slug and stop to add the pretty name. Do not guess capitalization.

| Thing | Source | Rule |
|---|---|---|
| Display name | `app/_lib/charts.ts` `MODEL_NAMES` | `DeepSeek V4 Flash (0731)`, `Qwen 3.8 Max`, `Claude Opus 5` |
| Short label | `SHORT_NAMES` in the same file | charts only |
| Site slug | `public/data/models.json` / `kb publish` | `deepseek-v4-flash-0731` |
| Harness | `docs/HARNESSES.md` case label | backtick the runner name: `` `or-fable` ``, `` `qwen-claude` ``, `` `codex` `` |
| Model id | the argv / archive slug | keep the provider path: `deepseek/deepseek-v4-flash-0731` |
| GPU | `app/_lib/models.ts` `HOME_GPU_TABS` | Default prose: `RTX PRO 6000`. Never bare `H100`. A second SKU only if the kernel used a feature that card does not have. |
| Bench | `BENCH_LABELS` | CUDA / Mega / Hard |
| Problem | `PROBLEM_LABELS` | `FP8 GEMM`, `Sonic MoE`, not `01_fp8_gemm` in prose |
| Run id | archive basename | `20260801_205612_or-fable_deepseek_deepseek-v4-flash-0731_01_fp8_gemm` — this is the high-taste ID. Quote it. Do not invent a prettier one. |

Harness in the first setup sentence: **how it ran**, not a brand mash. Example: `or-fable` on OpenRouter’s Anthropic route, model `qwen/qwen3.8-max`, effort `xhigh`.

0731 is a post-train. Write `DeepSeek V4 Flash (0731)`, not “V4 Flash” as if it were the base.

## Cover

`media/thumb_card.py`. Official lab mark. No fake black tile. No stats.

Lockup is **mark + one subject token**, same visual size as the mark.

- Grok 5 → official black-hole G (`xai-logo-1024.png`) + the digit `5`. Do not print `GROK`. The old xAI X-mark is wrong.
- DeepSeek → official whale + `0731`.
- Qwen → official hex pinwheel (`qwen-logo-1024.png` / `public/logos/labs/qwen.svg`, same mark as qwen.ai and HF). The old C-dash chat icon is wrong.
- Do not print the lab name next to the lab mark.
- No KernelBench signature in the corner.

Inspect the PNG before upload.

## GPU and compute — one board, no explosion

Do not report Hard×CUDA×Mega × H100×RTX×B200 × every harness. That is a spreadsheet, not an article.

- Default SKU is **RTX PRO 6000 Blackwell**. Home method: run agents in parallel on the cards that are actually there.
- A second SKU (H100, B200, later Rubin) only appears if the kernel used a feature that card does not have. B200 example: `tcgen05.mma`, tensor memory. Consumer Blackwell does not get those. Otherwise skip B200.
- Infer compute from the run you launched. `gpu_name` in `result.json` plus who spun the box (home / Lambda / Brev). Do not ask the human. Do not write “Thanks to @LambdaAPI”. State the box as a fact if it matters (`Lambda gpu_1x_rtx_pro_6000`, `home RTX PRO 6000`). If the provider is not in the launch record, omit it.
- In-body charts: one GPU. Not a 2- or 3-panel SKU grid unless the SKU delta is the point.

## Skeleton — every article, same order

Readers should be able to predict the next heading.

1. **Title.** `{Display name} on KernelBench`
2. **Cover.** See above.
3. **Drop.** Two or three blunt sentences. The model just shipped. What the timeline claimed (rumors, training data, “it is good at X”). Why you ran it: hand-curated problems, then a manual audit of every headline trace and solution so it is not gaming the bench. No adjectives. No speculation. Good and bad, same voice.
4. **How I bench.** Paste the boilerplate below. Do not rewrite it. Do not skip it. New readers need this every time.
5. **Using it.** Authentic daily-use paragraph. Ignore the scores. How it talks, how it ships work, whether it generalizes vs the last model you actually used. **Voice-typed by Elliot. Do not LLM this. If it is not in the prompt, stop and ask.** This is the only required human paragraph.
6. **Board.** Publish-grade counts only, default GPU only. Wrong-SKU / contaminated / reward-hack cells named as excluded. Not a second TLDR essay.
7. **Per-bench body** in this order, skip a bench that did not run: **CUDA → Mega → Hard last**.
   - Do not re-explain why the problem exists. The boilerplate already said what each bench is.
   - Standout cell: what the kernel did, the isolated number, solution + trace links.
   - Quote the agent only if the quote is funny or load-bearing (sandbox break, self-caught cheat, priced a rejected alternative). “this could be a massive head start” is not a quote.
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

Read 2–3 headline transcripts and the matching `solution.py` before any sentence. Surface only what a leaderboard cell cannot show:

- a behavioral shift vs the previous model from that lab
- a metric artifact that is true for every model (TopK launch-bound ~0.02)
- the actual kernel move (one launch, fragment ownership, same-buffer fail)
- profiling that priced a rejected alternative
- cross-run contamination or grader mutation, in the agent’s own words

If you cannot point at a run id, do not write the claim.

Be blunt about both sides. A clean kernel and a reward-hack in the same wave both get named. Do not speculate about why the lab trained it that way.

## Numbers

- Headline winners = argmax of annotation YAML on that board. If the draft contradicts the YAML, the draft is wrong. If the YAML does not exist, stop — do not draft the number with a caveat. Isolated regrade is not an audit.
- Published timing = sequential isolated re-grade on the stamped SKU (`gpu_name` in `result.json`). In-run flywheel times stay in the archive.
- Wrong-SKU, contamination, reward-hack, rubric-leak: name them. Do not narrate them as board wins.

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

X Article API is draft + publish only. No list/get/edit/delete. One draft per ~15 min. Do not `--force` a deleted hash. Friend reviews the write-up. You do not publish on taste.

## Links — live GET, or do not embed

Hugging Face "unresolved" is a missing file (404), not a private dataset. `Infatoshi/kernelbench-{hard,cuda,mega}-traces` are public. Do not flip them private.

- Never invent a path. Do not prefix `h100/` because the cell is an H100 cell. Canonical RTX files are `/blob/main/{run_id}.jsonl`. H100 files are `/blob/main/h100/{run_id}.jsonl` only when that object exists on HF.
- Catalog `trace_url` is a hint. `build_model_index.py` can bake `h100/` for a run that was never pushed. A catalog URL is not proof the file is live.
- Rejected / unpublished cells have no HF object until someone converts and uploads them. Quote the run id in prose. Embed a href only after `check_article_links.py` returns 200 on that exact URL.
- Wrap every URL in `[label](url)`. Bare `_` in a URL italicizes the markdown.
- The checker also GETs the HF `/resolve/main/` twin and refuses an HTML "unresolved" / "Entry not found" page even if status is 200.

## In-body charts

Green on black. `kbh_theme`. Square bars. Short labels (`FP8`, `Sonic`, `Paged`). One GPU. Subject bar `#76b900`, field `#4d5d66`. Numbers on the bars. No title essay. Generate with `media/make_<slug>_article.py`, inspect the PNG, then `![short alt](01_….png)`. The standout cell of each bench section gets the highlight chart instead: `media/trajectory.py <run_dir>` with that run's `trajectory:` checkpoints in its annotation YAML (`docs/POST.md`, charts section).

After it goes live: delete ephemeral `X-*.md` drafts and generated PNGs. Keep the `.py` generators.

## Do not

- invent a pretty model name
- write `/mini` (Mini is a homepage scroll category)
- post an unpublished Mini deck
- hide CUDA or skip the isolated re-grade to “make the article”
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
