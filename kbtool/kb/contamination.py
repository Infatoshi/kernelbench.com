"""Cross-run contamination audit for KernelBench runs (hard or mega).

The harness does NOT sandbox the agent filesystem: an agent has bash + absolute
paths, so it can read the shared `outputs/runs/` archive -- every prior winning
solution -- and reverse-engineer a known answer instead of writing its own
kernel. A run is CONTAMINATED if its agent transcript references another run's
archive (`outputs/runs/<other_ts>`).

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

_TS = re.compile(r"outputs/runs/(\d{8}_\d{6})")
# A run referenced by its directory name, without the outputs/runs/ prefix,
# e.g. "20260715_212751_grok_grok-4.5_01_glm52_fused_moe".
_RUN_DIR_ID = re.compile(r"\b(\d{8}_\d{6})_[a-z]")
_TOKEN_TYPES = ("thought", "text")


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


def run(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kb contamination")
    ap.add_argument(
        "runs",
        help="path to an outputs/runs directory, or a bench name (hard|mega|cuda|v3)",
    )
    ap.add_argument("--published", help="leaderboard.json to flag contaminated PUBLISHED cells")
    args = ap.parse_args(argv)

    runs = Path(args.runs)
    # Convenience: accept a bench name and resolve against the repo.
    if repo_root is not None and not runs.exists() and args.runs in ("hard", "mega", "cuda", "v3"):
        runs = repo_root / "benchmarks" / args.runs / "outputs" / "runs"
    if not runs.is_dir():
        print(f"no such runs dir: {runs}")
        return 1

    dirty: dict[str, int] = {}
    total = 0
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        total += 1
        n = other_archives(d)
        if n:
            dirty[d.name] = len(n)

    print(f"=== contamination audit: {len(dirty)} / {total} runs read other archives ===")
    for name, cnt in sorted(dirty.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>3} other archives  {name}")

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
