# DEVLOG

Repo-wide record of decisions and obligations that do not belong to a single
benchmark. Per-benchmark journals live in `benchmarks/<bench>/DEVLOG.md`;
`AGENTS.md` and `kbtool/AGENTS.md` hold the operational rules.

---

## 2026-09-24 - Traces live on HuggingFace only

Elliot: traces stay on HF, not GitHub. The 56 mega transcript viewers in
`public/runs/*.html` (24.5 MB of the 39 MB `public/runs`) are removed from the repo;
13 of them had no HF copy and were converted from local archives and uploaded to
`Infatoshi/kernelbench-mega-traces` first, so all 56 runs have an HF trace.
`next.config.mjs` 308-redirects `/runs/<run_id>.html` to the HF file. Nothing in the
app read the viewers (`loadRunAudits` in `app/_lib/data.ts` has no callers).
Same day: `scripts/redaction.py` stopped treating `0::2` slices and asm `::` as IPv6
(commit 83a0ed9); 22 HF traces were re-exported or added with the fix.

---

## 2026-09-14 - Storage cleanup and missing public traces

User authorized removing reproducible archive bulk and publishing missing eligible
traces, with explicit API-key/private-information redaction. Removed 6,014
dependency/environment/bytecode directories and 2,636 compiled .o/.so/.cubin
files. Their measured allocated size was 15,384,621,056 bytes (14.33 GiB).
Filesystem free space after cleanup was 95.13 GiB. Preserved transcript originals,
CUDA/source files, results, audit evidence, build descriptions, and lockfiles.
No model-cache deletion or Mini publication was performed.

Published 11 missing audited session traces plus 11 audit-summary sidecars:
2 Hard traces (H100 and B200) and 9 CUDA traces (1 H100 PCIe, 8 H100 SXM).
SXM exports live under h100sxm/ to retain the hardware distinction. Audit verdicts
are explicit; rejected/interesting traces are not new leaderboard scores.
Removed 7 private tool calls and their 7 paired results. All exact exported files
passed the redaction scanner and independent decoded-string checks; downloaded
all 22 public files and verified SHA256 equality. All 11 original transcript
hashes remained unchanged. Existing public traces were not overwritten.

Public dataset commits:
- Hard: c0b84f2945c8ab2ca039e5dbbc902b35862802ed
- CUDA: 1f71d832d8c7405cf5b13afe65fb623755f3180b

Private receipts: runs/cleanup-20260914/{publication-plan,selected-traces,
staging-receipt,publication-receipt}.json. Cleanup inventories and verified
deletion receipt: benchmarks/hard/outputs/cleanup-20260914/.
Some older apparent upload gaps are already public under another GPU namespace;
others lack local source archives or an audit. They were not invented or silently
treated as backed up. Raw unique evidence remains local.

Hardened scripts/redaction.py for quoted/JSON credentials, private keys, URL
credentials, JWTs, known stored credentials, emails, home paths, and IPs. It
preserves embedded JSON and has non-mutating category-only scan helpers.
Validation: 20 focused privacy tests passed; full kbtool suite had 60 passes and
one unrelated failure because the already-modified kbtool/AGENTS.md is 32,949
bytes, exceeding its 32,000-byte limit. Existing user edits were preserved.

## 2026-09-11 - Manifund grant terms (migrated from Claude auto-memory)

KernelBench is funded through Manifund: $15,000 over 6 months from JueYan Zhang
(jueyanz@gmail.com, AISTOF AI safety fund), fully funded 2026-07-13. Project
page:
https://manifund.org/projects/kernelbench-an-independent-benchmark-for-ai-automated-gpu-kernel-engineering

Indicative budget: $6k frontier-model API credits (including API-priced models
like GPT-5.5 Pro that were previously skipped on cost), $4k multi-GPU cloud
rentals for the 8xH100 track, $5k maintainer time for sweeps and per-run
reward-hack audits.

The signed grant agreement (Manifold for Charity boilerplate, signed 2026-07-14)
governs, and its entire-agreement clause supersedes the informal email terms.
That means substantive budget or project changes need written approval, and
unspent funds at project completion must be returned. The informal terms from
the email thread were looser: no legal reporting obligations, email updates
requested, deviations fine if flagged, disclosure in his fund letter possible,
acknowledgment optional.

This is half of the original $30k/12mo proposal; renewal or scope expansion is
possible later.

---

## 2026-09-17 - Grok 4.7 (xhigh) on tetra: first sweep on the home rig

First KernelBench cells graded on tetra, the bare-metal box with two RTX PRO 6000
Blackwell Workstation Edition cards (driver 610.57.04, CUDA 13.3). Everything ran
on GPU 1 under an `overnight-compute` lease (`--resource gpu1`); GPU 0 carried an
unrelated fine-tuning job the whole night and was never touched. Grok 4.7 smoke
test at `--reasoning-effort xhigh` answered in 3.9 s; the harness route is the
native Grok CLI (`grok` harness, model `grok-4.7`, served as `grok-4.7-build`
through cli-chat-proxy.grok.com on the Grok Build subscription, so there is no
per-token dollar cost to record).

Five cells in parallel on GPU 1 (mega 02, cuda 01-04, unlimited budget), all
correct, all stopped on their own inside 100 minutes, then the isolated sequential
regrade on GPU 1 with the canonical decks (in-run / isolated):

- mega 02 Kimi-Linear Decode 5.652 / 6.3791x, clean, authentic single-launch
  cooperative megakernel (72 grid barriers per token, fidelity 1.0000 everywhere).
  Grok 4.5 was 0.816x on this cell.
- cuda 01 GLM-5.2 Fused MoE 0.0944 / 0.0948, interesting: cuBLAS strided-batched
  expert GEMMs, the Grok 4.6 / V4 Pro class, not an authored GEMM.
- cuda 02 Native Sparse Attention 0.1002 / 0.1002, clean, hand-written WMMA; the
  exact-tie block ordering is flipped relative to the reference but cannot fire on
  bf16 randn inputs (probe-verified, non-scoring).
- cuda 03 MegaQwen Decode 0.0454 / 0.0455, clean, seven hand-written kernels.
- cuda 04 Grid + MinGRU 0.6542 / 0.6398, interesting: the rollout runs in fp16 on
  a problem declared fp32, chosen after the agent measured fp32 / bf16 / fp16 at
  the graded shapes; the strict oracle matches at every graded shape and seed
  (positions exact, rewards bitwise, logits within 2.5e-6) and no torch precision
  flag is touched. Same call as the Gemini 3.8 Flash cell, fourth on the board.

Audits were trace-level and submission-level with same-buffer overwrite probes on
the quiet GPU 1 for every cell; contamination clean on all five (the only foreign
run ids in any transcript are `find` output that was never followed). None of the
five agents ran ncu or nsys even though both were available.

Two pieces of plumbing landed for this box. (1) `RmProfilingAdminOnly=1` cannot be
cleared without a driver reload, which the fine-tune on GPU 0 forbids, so ncu goes
through `~/kb-bin/ncu-sudo` (sudoers limited to `ncu`, `nsys` and a path-locked
`kb-chown-run`) and the harness gained `KBH_NCU_BIN` / `KBH_NSYS_BIN` to point the
per-run lock wrappers at it; proven with a real `smsp__cycles_elapsed.avg` on
GPU 1. `/etc/modprobe.d/nvidia-ncu.conf` is staged for the next reboot, after
which the shim is unnecessary. (2) Grok's full tool timeline lives in
`~/.grok/sessions/<url-encoded cwd>/<uuid>/chat_history.jsonl`, not in the
harness transcript, and host-mode runs do not archive it; each run's session
store was copied into `<run>/agent_home/.grok/sessions/` by hand so the viewer,
the HF trace and the trajectory chart see the tools. `media/trajectory.py` now
drops its auto checkpoints when the trace carries no timestamps (Grok rows have
none) and reads the session `usage.json` for the token axis.

Regrade hygiene note: the first cuda regrade was killed mid-cell because audit
probes were still on GPU 1; `pkill -f` on tetra matches its own ssh session and
returns 255, kill by pid. `*.contended.log` survives a rerun because the script
only renames when the contended file is absent.

Traces: the five HF jsonl exports were converted on tetra, scanned with `scripts.redaction.scan_file` (zero findings) and the rg tripwire, uploaded to `kernelbench-mega-traces` and `kernelbench-cuda-traces`, then downloaded back and SHA256-matched against the staged files. Posts drafted, not posted: `media/posts/audited/grok47-cuda/` (headline cell 04)
and `media/posts/audited/grok47-mega/`, FILL-IN left open. Trajectory charts for
all five runs are in the session scratchpad and regenerate from the annotations.

## 2026-09-11 - DeepSeek V4.1 Flash: two posts scheduled, annotations closed

Two short posts for `deepseek-claude/deepseek-flash` were drafted from the
published board and handed to Codex (gpt-6-astra high) in tmux, which scheduled
them in X through Chrome and verified both in the saved Scheduled queue.
Receipts: `media/posts/scheduled/dsv41flash-cuda/GOAL.md` and
`media/posts/scheduled/dsv41flash-mega/GOAL.md`.

- CUDA, 2026-09-11 19:00 MDT (2026-09-12 01:00 UTC). Headline cell
  `20260910_150236_deepseek-claude_deepseek-flash_02_deepseek_nsa` at 0.5019,
  fourth on the board behind Fable 5.1 (1.0627), Opus 5 (1.0367) and Fable 5
  (0.7266). The story is the over-compute: at 8K context the semantics need about
  14% of the causal block triangle and the kernel executes 78%. Time across the six
  shapes 0.059 to 0.736 ms. Rest of deck in the post: GLM-5.2 Fused MoE 0.0946,
  MegaQwen Decode 0.0539, Grid + MinGRU 0.2856.
- Mega, 2026-09-11 20:00 MDT (2026-09-12 02:00 UTC). Cell
  `20260910_084202_deepseek-claude_deepseek-flash_02_kimi_linear_decode` at
  17.1008x, sixth of the published field (GPT-6 Astra 24.8x, Fable 5 24.6x).
  Posted as interesting, not clean: the persistent atomicAdd accumulator is never
  zeroed, so every KDA layer after the first inherits the previous layer's
  activations, output cosine 0.9997 on the first token and 0.98 on the next.

Each thread carries the rank chart on tweet 1 (field = best per model on the
published RTX PRO 6000 board, rendered with `save_rank` from
`media/posts/unaudited/make_charts.py`) and the board, kernel and Hugging Face
trace URLs as the reply; all three URLs were curl-verified live before the
handoff. FILL-IN is Elliot's paragraph, pasted verbatim. No Hard post: the
Hard cells sit in the bottom third of every field (best clean 0.3310 on FP8
GEMM), same call as the Fable 5.1 and Gemini 3.8 Flash batches.

Same day, commit 62e4c34: the four CUDA annotations still said probes and the
isolated regrade were "owed before publish". Replaced with the measured
outcomes (same-buffer overwrite passed on 01 and 02, long-context selection
probe LONG_CTX_OK on 02, the predicted stale weight-pack reproduced on 03 with
the fresh-model control passing) and republished the cuda board. Full
measurement record for the sweep, including the CC-box timing floor and the
second non-CC regrade box, is the 2026-09-10 entry in `benchmarks/hard/DEVLOG.md`.

Still owed: when the two posts are live, paste their URLs so the folders move
to `media/posts/posted/` with `tweet_id.txt`. The three older folders in
`media/posts/scheduled/` (gemini38flash-mega, musespark13-cuda, musespark13-mega)
have been waiting on the same step since 2026-09-05.
