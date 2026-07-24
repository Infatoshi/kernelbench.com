"""Viewer CLI.

Examples:
    uv run python -m src.viewer /path/to/run_dir
    uv run python -m src.viewer /path/to/run_dir --transcript transcript.jsonl
    uv run python -m src.viewer /path/to/run_dir --open  # open in browser
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from src.viewer.html import render
from src.viewer.parsers import parse, sniff


def _find_transcript(run_dir: Path, prefer: str | None = None) -> Path:
    """Pick the best transcript file in a run directory."""
    if prefer:
        p = run_dir / prefer
        if p.exists():
            return p
    candidates = [
        "codex_session.jsonl",   # Codex session file (richest)
        "transcript.jsonl",      # claude / kimi / cursor / droid
        "transcript.txt",        # codex stdout fallback
    ]
    for name in candidates:
        p = run_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(f"no transcript found in {run_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KernelBench-Hard transcript viewer")
    parser.add_argument("run_dir", type=Path, help="run directory with transcript + artifacts")
    parser.add_argument("--transcript", help="transcript filename to use (auto-detected by default)")
    parser.add_argument("--out", type=Path, help="output HTML path (default: <run_dir>/index.html)")
    parser.add_argument("--open", action="store_true", help="open the generated HTML in a browser")
    parser.add_argument("--inspect", action="store_true", help="print parsed Session and exit")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 1

    transcript = _find_transcript(run_dir, args.transcript)
    fmt = sniff(transcript)
    print(f"transcript: {transcript}  format: {fmt}")
    session = parse(transcript)

    if args.inspect:
        print(f"harness={session.harness} model={session.model} turns={session.turn_count} "
              f"tools={session.tool_call_count} events={len(session.events)}")
        for i, e in enumerate(session.events[:30]):
            preview = (e.text or e.reasoning or (e.tool_result.content if e.tool_result else "") or "")[:60]
            tcs = ",".join(tc.name for tc in e.tool_calls)
            print(f"  [{i:3d}] {e.role:10s} tools={tcs!s:20s} {preview!r}")
        if len(session.events) > 30:
            print(f"  ... {len(session.events) - 30} more events")
        return 0

    out = render(run_dir, session, out_path=args.out)
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")

    if args.open:
        webbrowser.open(out.as_uri())

    return 0


if __name__ == "__main__":
    sys.exit(main())
