"""Extract per-run token usage from a harness transcript.

Usage:
    uv run python scripts/extract_usage.py <run_dir> <harness>

Emits a JSON object on stdout with normalized fields:
    input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, reasoning_tokens, total_cost_usd

Any field that is not reported by the given harness is null. The point is a
single uniform shape across claude / codex / kimi / opencode so result.json
aggregation is cheap downstream. Coding-plan billing on the CLI does not
expose per-token cost; transcripts still report the raw token counts, which
is what matters for cross-model comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _claude_or_kimi(events: list[dict]) -> dict:
    """Anthropic-shape stream-json: terminal `{"type":"result"}` has cumulative usage."""
    for e in reversed(events):
        if e.get("type") == "result":
            u = e.get("usage", {})
            return {
                "input_tokens": u.get("input_tokens"),
                "output_tokens": u.get("output_tokens"),
                "cache_read_tokens": u.get("cache_read_input_tokens"),
                "cache_creation_tokens": u.get("cache_creation_input_tokens"),
                "reasoning_tokens": None,
                "total_cost_usd": e.get("total_cost_usd"),
            }
    # Fallback: sum per-message usage from assistant events.
    inp = out = cr = cc = 0
    saw = False
    for e in events:
        if e.get("type") != "assistant":
            continue
        msg = e.get("message", {}) or {}
        u = msg.get("usage", {}) or {}
        inp += u.get("input_tokens", 0) or 0
        out += u.get("output_tokens", 0) or 0
        cr += u.get("cache_read_input_tokens", 0) or 0
        cc += u.get("cache_creation_input_tokens", 0) or 0
        saw = saw or bool(u)
    if not saw:
        return _empty()
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cr,
        "cache_creation_tokens": cc,
        "reasoning_tokens": None,
        "total_cost_usd": None,
    }


def _codex(events: list[dict]) -> dict:
    """Codex session.jsonl: token_count events carry running totals."""
    inp = out = cr = reason = 0
    saw = False
    for e in events:
        payload = e.get("payload", {}) or {}
        ptype = payload.get("type")
        info = payload.get("info") if ptype == "token_count" else None
        if not info:
            continue
        last = info.get("last_token_usage") or {}
        inp += last.get("input_tokens", 0) or 0
        out += last.get("output_tokens", 0) or 0
        cr += last.get("cached_input_tokens", 0) or 0
        reason += last.get("reasoning_output_tokens", 0) or 0
        saw = True
    if not saw:
        return _empty()
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cr,
        "cache_creation_tokens": None,
        "reasoning_tokens": reason,
        "total_cost_usd": None,
    }


def _opencode(events: list[dict]) -> dict:
    """OpenCode SST: each step_finish carries part.tokens cumulative-per-step."""
    inp = out = cr = cw = reason = 0
    saw = False
    for e in events:
        if e.get("type") != "step_finish":
            continue
        toks = (e.get("part", {}) or {}).get("tokens", {}) or {}
        inp += toks.get("input", 0) or 0
        out += toks.get("output", 0) or 0
        reason += toks.get("reasoning", 0) or 0
        cache = toks.get("cache", {}) or {}
        cr += cache.get("read", 0) or 0
        cw += cache.get("write", 0) or 0
        saw = True
    if not saw:
        return _empty()
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cr,
        "cache_creation_tokens": cw,
        "reasoning_tokens": reason,
        "total_cost_usd": None,
    }


def _empty() -> dict:
    return {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
        "cache_creation_tokens": None,
        "reasoning_tokens": None,
        "total_cost_usd": None,
    }


def extract(run_dir: Path, harness: str) -> dict:
    if harness == "codex":
        events = _read_jsonl(run_dir / "codex_session.jsonl")
        if not events:
            events = _read_jsonl(run_dir / "transcript.jsonl")
        return _codex(events)
    transcript = _read_jsonl(run_dir / "transcript.jsonl")
    if harness in ("claude", "ccr-claude", "kimi", "cursor"):
        return _claude_or_kimi(transcript)
    if harness == "opencode":
        return _opencode(transcript)
    return _empty()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: extract_usage.py <run_dir> <harness>", file=sys.stderr)
        return 2
    run_dir = Path(argv[1])
    harness = argv[2]
    out = extract(run_dir, harness)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
