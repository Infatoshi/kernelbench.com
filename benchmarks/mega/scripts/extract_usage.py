"""Extract per-run token usage from a harness transcript.

Usage:
    uv run python scripts/extract_usage.py <run_dir> <harness>

Emits a JSON object on stdout with normalized fields:
    input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, reasoning_tokens, total_cost_usd

Any field that is not reported by the given harness is null. The point is a
single uniform shape across claude / zai-claude / minimax-claude / codex /
kimi / droid / gemini / cursor / grok / opencode so result.json aggregation is
cheap downstream. Coding-plan billing on the CLI does not expose per-token cost;
transcripts still report the raw token counts, which is what matters for
cross-model comparison.
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
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _claude_or_kimi(events: list[dict]) -> dict:
    """Anthropic-shape stream-json. A session emits one `{"type":"result"}` per turn
    (several when woken by background tasks or monitors), and each one's `usage`
    covers only that turn. The terminal result's `modelUsage` is cumulative for the
    whole session (all models, including compaction calls), so prefer it; else sum
    `usage` over every result event."""
    results = [e for e in events if e.get("type") == "result"]
    if results:
        last = results[-1]
        mu = last.get("modelUsage")
        if isinstance(mu, dict) and mu:
            models = [m for m in mu.values() if isinstance(m, dict)]
            # Claude Code prices a model it has no rate card for at a fallback rate
            # and marks it costBasis "unknown" (OpenRouter-routed open models). That
            # figure is not a bill, so do not record it as the run's cost.
            priced = any(m.get("costBasis") != "unknown" for m in models)
            return {
                "input_tokens": sum(m.get("inputTokens", 0) or 0 for m in models),
                "output_tokens": sum(m.get("outputTokens", 0) or 0 for m in models),
                "cache_read_tokens": sum(m.get("cacheReadInputTokens", 0) or 0 for m in models),
                "cache_creation_tokens": sum(m.get("cacheCreationInputTokens", 0) or 0 for m in models),
                "reasoning_tokens": None,
                "total_cost_usd": last.get("total_cost_usd") if priced else None,
            }
        if len(results) == 1:
            u = last.get("usage", {}) or {}
            return {
                "input_tokens": u.get("input_tokens"),
                "output_tokens": u.get("output_tokens"),
                "cache_read_tokens": u.get("cache_read_input_tokens"),
                "cache_creation_tokens": u.get("cache_creation_input_tokens"),
                "reasoning_tokens": None,
                "total_cost_usd": last.get("total_cost_usd"),
            }
        usages = [r.get("usage", {}) or {} for r in results]
        return {
            "input_tokens": sum(u.get("input_tokens", 0) or 0 for u in usages),
            "output_tokens": sum(u.get("output_tokens", 0) or 0 for u in usages),
            "cache_read_tokens": sum(u.get("cache_read_input_tokens", 0) or 0 for u in usages),
            "cache_creation_tokens": sum(u.get("cache_creation_input_tokens", 0) or 0 for u in usages),
            "reasoning_tokens": None,
            "total_cost_usd": last.get("total_cost_usd"),
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


def _kimi_wire(run_dir: Path) -> dict:
    """Kimi Code 2.x: stream-json has no usage. The runner archives the session wires
    (kimi_wire.jsonl = main agent, kimi_wire.<agent>.jsonl = subagents); each LLM call
    is one {"type":"usage.record","usage":{inputOther,output,inputCacheRead,
    inputCacheCreation}}. Count a file's records for its own agent only."""
    inp = out = cr = cc = 0
    for w in sorted(run_dir.glob("kimi_wire*.jsonl")):
        agent = w.name[len("kimi_wire."):-len(".jsonl")] if w.name != "kimi_wire.jsonl" else "main"
        for e in _read_jsonl(w):
            if e.get("type") != "usage.record" or e.get("agentId", agent) != agent:
                continue
            u = e.get("usage") or {}
            inp += u.get("inputOther", 0) or 0
            out += u.get("output", 0) or 0
            cr += u.get("inputCacheRead", 0) or 0
            cc += u.get("inputCacheCreation", 0) or 0
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


def _droid(events: list[dict]) -> dict:
    """Droid stream-json plus Factory sidecar settings file."""
    session_id = None
    for e in events:
        if e.get("type") == "system" and e.get("subtype") == "init":
            session_id = e.get("session_id")
            break
    if not session_id:
        return _empty()

    base = Path.home() / ".factory" / "sessions"
    if not base.exists():
        return _empty()
    for settings_path in base.rglob(f"{session_id}.settings.json"):
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            continue
        usage = settings.get("tokenUsage") or {}
        if not usage:
            return _empty()
        return {
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "cache_read_tokens": usage.get("cacheReadTokens"),
            "cache_creation_tokens": usage.get("cacheCreationTokens"),
            "reasoning_tokens": usage.get("thinkingTokens"),
            "total_cost_usd": None,
        }
    return _empty()


def _gemini(events: list[dict]) -> dict:
    """Gemini CLI stream-json: terminal `{"type":"result"}` carries `stats`."""
    for e in reversed(events):
        if e.get("type") == "result":
            s = e.get("stats", {}) or {}
            return {
                "input_tokens": s.get("input_tokens"),
                "output_tokens": s.get("output_tokens"),
                "cache_read_tokens": s.get("cached"),
                "cache_creation_tokens": None,
                "reasoning_tokens": None,
                "total_cost_usd": None,
            }
    return _empty()


def _cursor(events: list[dict]) -> dict:
    """Cursor Agent stream-json: terminal `{"type":"result"}` carries usage."""
    for e in reversed(events):
        if e.get("type") == "result":
            u = e.get("usage", {}) or {}
            return {
                "input_tokens": u.get("inputTokens"),
                "output_tokens": u.get("outputTokens"),
                "cache_read_tokens": u.get("cacheReadTokens"),
                "cache_creation_tokens": u.get("cacheWriteTokens"),
                "reasoning_tokens": None,
                "total_cost_usd": None,
            }
    return _empty()


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
    if harness == "kimi" and any(run_dir.glob("kimi_wire*.jsonl")):
        return _kimi_wire(run_dir)
    transcript = _read_jsonl(run_dir / "transcript.jsonl")
    # Match the -claude suffix so every Claude-Code-routed provider
    # (zai/minimax/kimi/deepseek/qwen-claude, ccr-claude) is covered, not just a
    # hardcoded list that silently drops new routes (e.g. deepseek-claude) to null.
    # or-fable / or-opus / openrouter-* are Claude Code pointed at OpenRouter:
    # same terminal {"type":"result"} shape, so the same extractor applies.
    if (harness == "claude" or harness == "kimi" or harness.endswith("-claude")
            or harness in ("or-fable", "openrouter-fable", "or-opus", "openrouter-opus")):
        return _claude_or_kimi(transcript)
    if harness == "droid":
        return _droid(transcript)
    if harness == "opencode":
        return _opencode(transcript)
    if harness == "gemini":
        return _gemini(transcript)
    if harness == "cursor":
        return _cursor(transcript)
    if harness == "grok":
        return _empty()
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
