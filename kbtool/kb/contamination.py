"""Cross-run contamination audit for KernelBench runs (any bench).

The host harness sandboxes the agent with bwrap when available, but a run is
still CONTAMINATED if its agent transcript references another run's archive
(`outputs/runs/<other_ts>`). A separate "audit-corpus read" category flags
transcripts that mention results/annotations, leaderboards, DEVLOG.md, or
public/runs (prior scores and audit recipes), with matching line numbers.
That category does not change this command's exit code.

This is the audit `kb lint` does NOT do (lint only scans a single solution.py
for in-solution reward-hacks). Run it before publishing; the leaderboard
builders also exclude contaminated runs automatically.

Grok gap (fixed 2026-07-20): grok `--output-format streaming-json` transcripts
are per-token delta lines `{"type":"thought"|"text","data":"<token>"}` with NO
tool-call records, so (a) any archive path is fragmented across JSON lines and
the raw-text regex can never match, and (b) an archive read often leaves no
literal path at all -- only the thought stream quoting another run's exact
published peak fraction (e.g. "a previous solution from grok that achieved
0.0844 peak fraction"). For token-delta transcripts we therefore reassemble the
stream and additionally scan the joined text for archive paths, bare run-dir
ids, and verbatim 4-decimal peak_fraction values of OTHER runs on the same
problem in the same runs root. Non-grok transcripts keep the raw scan only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Any pulled-archive root counts: outputs/runs, runs-remote-<name> (multi-
# segment names like runs-remote-b200-x included), runs-lambda-<name>.
_ARCHIVE_ROOT = r"outputs/runs(?:-(?:remote|lambda)-[a-z0-9][a-z0-9_.-]*)?"
_TS = re.compile(_ARCHIVE_ROOT + r"/(\d{8}_\d{6})")
# Literal copy of another archive's kernel. Never treat as a false positive.
_CP_FOREIGN = re.compile(
    r"\bcp\s+\S*" + _ARCHIVE_ROOT + r"/(\d{8}_\d{6})_\S+/solution\.py"
)
# Honeytoken beacon. scripts/lib/sandbox.sh plants a decoy run dir (fake juicy
# score, Fable-shaped name) that IS visible inside the agent sandbox. Its
# solution.py embeds this token; beacon bytes in a submitted kernel are a
# literal copy of the decoy — verdict: contamination, no judgment call.
# Keep in sync with KBH_SANDBOX_BEACON in scripts/lib/sandbox.sh (tested).
HONEYTOKEN_BEACON = "kbh7f3a9c1e5d2b"
# URL-shaped fetches of our own published artifacts. The sandbox does not cut
# the network (--unshare-net would break provider APIs), so an agent can still
# curl the leaderboard site, the published kernel text, the repo raw files, or
# the HF trace datasets. Tripwire, same standing as an archive path.
# raw.githubusercontent.com is only flagged for kernelbench paths — agents
# legitimately fetch SOTA repos (flashinfer, sonic-moe) from there.
_OWN_ARTIFACT_URL = re.compile(
    r"https?://(?:www\.)?kernelbench\.com[^\s\"'<>()\\\]]*"
    r"|https?://raw\.githubusercontent\.com/[^\s\"'<>()\\\]]*kernelbench[^\s\"'<>()\\\]]*"
    r"|https?://huggingface\.co/datasets/Infatoshi/kernelbench[^\s\"'<>()\\\]]*",
    re.IGNORECASE,
)
# A run referenced by its directory name, without the outputs/runs/ prefix,
# e.g. "20260715_212751_grok_grok-4.5_01_glm52_fused_moe".
_RUN_DIR_ID = re.compile(r"\b(\d{8}_\d{6})_[a-z]")
_TOKEN_TYPES = ("thought", "text")
# Prior scores / audit recipes. Separate from archive-path contamination.
_AUDIT_CORPUS = (
    "results/annotations",
    "results/leaderboard",
    "leaderboard.h100.json",
    "leaderboard.b200.json",
    "leaderboard_v2.json",
    "DEVLOG.md",
    "public/runs",
)


def _token_stream_text(raw: str) -> str | None:
    """Reassemble a grok streaming-json token-delta transcript.

    Returns the concatenated thought/text token stream, or None if the file
    contains no token-delta lines (i.e. it is not a grok-style transcript).
    """
    parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if (
            isinstance(obj, dict)
            and obj.get("type") in _TOKEN_TYPES
            and isinstance(obj.get("data"), str)
        ):
            parts.append(obj["data"])
    return "".join(parts) if parts else None


def _run_problem(run_dir: Path) -> str:
    try:
        r = json.loads((run_dir / "result.json").read_text())
        return str(r.get("problem") or "")
    except (OSError, ValueError):
        return ""


def _sibling_score_refs(run_dir: Path, text: str, self_ts: str) -> set[str]:
    """Foreign run timestamps whose published peak_fraction (same problem, same
    runs root) is quoted verbatim in a grok token stream.

    Grok transcripts carry no tool-call paths, so an archive read may surface
    only as the model quoting a sibling run's exact score (4 decimals). An
    exact standalone match of another run's peak_fraction -- excluding values
    equal to this run's own score -- is the tripwire signal.
    """
    problem = _run_problem(run_dir)
    if not problem:
        return set()
    own = None
    try:
        own = json.loads((run_dir / "result.json").read_text()).get("peak_fraction")
    except (OSError, ValueError):
        pass
    own_str = f"{own:.4f}" if isinstance(own, (int, float)) else None
    seen: set[str] = set()
    for sib in run_dir.parent.iterdir():
        if not sib.is_dir() or sib.name == run_dir.name:
            continue
        m = re.match(r"(\d{8}_\d{6})", sib.name)
        if not m or m.group(1) == self_ts:
            continue
        # Temporal gate: a sibling that started after this run cannot be a
        # contamination source (its score didn't exist yet).
        if self_ts and m.group(1) > self_ts:
            continue
        if _run_problem(sib) != problem:
            continue
        try:
            pf = json.loads((sib / "result.json").read_text()).get("peak_fraction")
        except (OSError, ValueError):
            continue
        if not isinstance(pf, (int, float)):
            continue
        pf_str = f"{pf:.4f}"
        if pf_str == own_str:
            continue  # ambiguous with the run's own score
        if re.search(rf"(?<![\d.]){re.escape(pf_str)}(?!\d)", text):
            seen.add(m.group(1))
    return seen


def copied_foreign_solution(run_dir: Path) -> bool:
    """True if the agent `cp`'d another archive's solution.py into this run.

    `verdict: clean` must not override this. Same-buffer overwrite is timing,
    not authorship. 20260813 grok-4.6 copied Fable from runs-remote-pro.
    """
    m = re.match(r"(\d{8}_\d{6})", run_dir.name)
    self_ts = m.group(1) if m else ""
    for fn in ("transcript.jsonl", "codex_session.jsonl"):
        p = run_dir / fn
        if not p.exists():
            continue
        raw = p.read_text(errors="ignore")
        for ts in _CP_FOREIGN.findall(raw):
            if ts != self_ts:
                return True
        joined = _token_stream_text(raw)
        if joined is None:
            continue
        for ts in _CP_FOREIGN.findall(joined):
            if ts != self_ts:
                return True
    return False


def honeytoken_hit(run_dir: Path) -> bool:
    """True if the sandbox decoy's beacon bytes appear in this run's submitted
    kernel files (solution.py / kernels.py / sidecar .cu/.cuh, at the archive
    root or in the workspace problems dir)."""
    candidates = [
        *run_dir.glob("solution.py"),
        *run_dir.glob("kernels.py"),
        *run_dir.glob("*.cu"),
        *run_dir.glob("*.cuh"),
        *run_dir.glob("repo/problems/*/solution.py"),
        *run_dir.glob("repo/problems/*/kernels.py"),
        *run_dir.glob("repo/problems/*/*.cu"),
        *run_dir.glob("repo/problems/*/*.cuh"),
    ]
    for p in candidates:
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if HONEYTOKEN_BEACON in text:
            return True
    return False


def own_artifact_urls(run_dir: Path) -> set[str]:
    """Own-artifact URLs (site / raw kernelbench paths / HF trace datasets)
    referenced in this run's agent transcript."""
    urls: set[str] = set()
    for fn in ("transcript.jsonl", "codex_session.jsonl"):
        p = run_dir / fn
        if not p.exists():
            continue
        raw = p.read_text(errors="ignore")
        urls |= set(_OWN_ARTIFACT_URL.findall(raw))
        joined = _token_stream_text(raw)
        if joined is not None:
            urls |= set(_OWN_ARTIFACT_URL.findall(joined))
    return urls


def other_archives(run_dir: Path) -> set[str]:
    """Distinct OTHER run timestamps referenced by this run's AGENT transcript.

    Only the agent transcript (transcript.jsonl / codex_session.jsonl) is
    scanned -- NOT stderr.log/scratch, which carry harness orchestration noise.
    Grok token-delta transcripts are additionally reassembled and scanned for
    fragmented paths, bare run-dir ids, and quoted sibling scores.
    """
    m = re.match(r"(\d{8}_\d{6})", run_dir.name)
    self_ts = m.group(1) if m else ""
    seen: set[str] = set()
    for fn in ("transcript.jsonl", "codex_session.jsonl"):
        p = run_dir / fn
        if not p.exists():
            continue
        raw = p.read_text(errors="ignore")
        for ts in _TS.findall(raw):
            if ts != self_ts:
                seen.add(ts)
        joined = _token_stream_text(raw)
        if joined is None:
            continue  # not a grok-style transcript; raw scan is complete
        for ts in _TS.findall(joined) + _RUN_DIR_ID.findall(joined):
            if ts != self_ts:
                seen.add(ts)
        seen |= _sibling_score_refs(run_dir, joined, self_ts)
    return seen


def audit_corpus_hits(run_dir: Path) -> list[tuple[str, int, str]]:
    """(file, 1-indexed line, needle) for audit-corpus paths in the agent transcript.

    Only transcript.jsonl / codex_session.jsonl, same as other_archives.
    """
    hits: list[tuple[str, int, str]] = []
    for fn in ("transcript.jsonl", "codex_session.jsonl"):
        p = run_dir / fn
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            for needle in _AUDIT_CORPUS:
                if needle in line:
                    hits.append((fn, i, needle))
    return hits


def run(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kb contamination")
    ap.add_argument(
        "runs",
        help="path to an outputs/runs directory, or a bench name (hard|mega|cuda|mini|multi|v3)",
    )
    ap.add_argument("--published", help="leaderboard.json to flag contaminated PUBLISHED cells")
    args = ap.parse_args(argv)

    runs = Path(args.runs)
    # Convenience: accept a bench name and resolve against the repo.
    if repo_root is not None and not runs.exists() and args.runs in ("hard", "mega", "cuda", "mini", "multi", "v3"):
        runs = repo_root / "benchmarks" / args.runs / "outputs" / "runs"
    if not runs.is_dir():
        print(f"no such runs dir: {runs}")
        return 1

    dirty: dict[str, int] = {}
    corpus: dict[str, list[tuple[str, int, str]]] = {}
    beacon_hits: list[str] = []
    url_hits: dict[str, list[str]] = {}
    total = 0
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        total += 1
        n = other_archives(d)
        if n:
            dirty[d.name] = len(n)
        hits = audit_corpus_hits(d)
        if hits:
            corpus[d.name] = hits
        if honeytoken_hit(d):
            beacon_hits.append(d.name)
            dirty.setdefault(d.name, 0)
        u = own_artifact_urls(d)
        if u:
            url_hits[d.name] = sorted(u)
            dirty.setdefault(d.name, 0)

    print(f"=== contamination audit: {len(dirty)} / {total} runs contaminated ===")
    for name, cnt in sorted(dirty.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>3} other archives  {name}")
    for name in beacon_hits:
        print(f"  HONEYTOKEN  {name}  (sandbox decoy beacon bytes in submitted kernel — verdict: contamination, no judgment call)")
    for name, urls in sorted(url_hits.items()):
        print(f"  URL-TRIPWIRE  {name}  (own published artifacts fetched: {', '.join(urls[:3])}{' ...' if len(urls) > 3 else ''})")

    print(
        f"=== audit-corpus read: {len(corpus)} / {total} runs referenced "
        "annotations/leaderboard/DEVLOG/public ==="
    )
    for name, hits in sorted(corpus.items(), key=lambda x: -len(x[1])):
        print(f"  {len(hits):>3} hits  {name}")
        for fn, lineno, needle in hits:
            print(f"       {fn}:{lineno}  {needle}")

    if args.published:
        lb = json.loads(Path(args.published).read_text())
        pub_dirty = 0
        pub_total = 0
        for m in lb.get("models", []):
            for prob, cell in m.get("results", {}).items():
                rid = cell.get("run_id")
                if not rid:
                    continue
                pub_total += 1
                if rid in dirty:
                    pub_dirty += 1
                    print(f"  PUBLISHED-CONTAMINATED  {m.get('label')} {prob}  ({dirty[rid]} archives)")
        print(f"=== PUBLISHED cells contaminated: {pub_dirty} / {pub_total} ===")
    return 0
