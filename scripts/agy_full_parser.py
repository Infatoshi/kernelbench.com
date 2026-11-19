"""Antigravity (`agy`) full transcript parser (`agy_transcript_full.jsonl`).

Lives outside `benchmarks/<bench>/src/` so graded-surface digests stay untouched.
Unlike `agy --stream-json` (`transcript.jsonl`), `transcript_full.jsonl` from
`agent_home/.gemini/antigravity-cli/brain/<id>/.system_generated/logs/` carries
full `thinking` blocks, tool arguments, tool outputs, and per-step token usage.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


def is_agy_full(path: Path) -> bool:
    path = Path(path)
    if not path.is_file() or path.suffix != ".jsonl":
        return False
    if path.name == "agy_transcript_full.jsonl":
        return True
    try:
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                obj = json.loads(raw)
                return (
                    isinstance(obj, dict)
                    and "step_index" in obj
                    and obj.get("type") in {"USER_INPUT", "PLANNER_RESPONSE", "GENERIC", "SYSTEM_MESSAGE", "CHECKPOINT"}
                )
    except (OSError, ValueError):
        return False
    return False


def parse_agy_full(path: Path) -> Any:
    from src.viewer.events import Event, Session, TokenUsage, ToolCall, ToolResult

    path = Path(path)
    events: list[Any] = []
    total = TokenUsage()
    model = None
    sid = path.parent.name
    pending_calls: deque[tuple[str, str]] = deque()

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            typ = obj.get("type")
            si = obj.get("step_index", len(events))
            content = obj.get("content") or ""
            status = obj.get("status") or ""

            if typ == "USER_INPUT":
                if content:
                    events.append(Event(role="user", text=content))
            elif typ == "PLANNER_RESPONSE":
                if obj.get("model"):
                    model = obj["model"]
                thinking = obj.get("thinking") or ""
                calls: list[Any] = []
                for j, tc in enumerate(obj.get("tool_calls") or []):
                    if not isinstance(tc, dict):
                        continue
                    name = tc.get("name") or "tool"
                    args = tc.get("args") or {}
                    if not isinstance(args, dict):
                        args = {"raw": args}
                    cid = f"agy_{si}_{j}"
                    fp = args.get("Path") or args.get("path") or args.get("file_path")
                    calls.append(ToolCall(name=name, args=args, call_id=cid, file_path=fp))
                    pending_calls.append((cid, name))
                u = TokenUsage(
                    input_tokens=int(obj.get("input_tokens") or 0),
                    output_tokens=int(obj.get("output_tokens") or 0),
                    cache_read_tokens=int(obj.get("cache_read_tokens") or 0),
                    cache_write_tokens=int(obj.get("cache_creation_tokens") or 0),
                )
                total = TokenUsage(
                    input_tokens=total.input_tokens + u.input_tokens,
                    output_tokens=total.output_tokens + u.output_tokens,
                    cache_read_tokens=total.cache_read_tokens + u.cache_read_tokens,
                    cache_write_tokens=total.cache_write_tokens + u.cache_write_tokens,
                )
                if thinking or content or calls:
                    events.append(
                        Event(
                            role="assistant",
                            text=content or None,
                            reasoning=thinking or None,
                            tool_calls=calls,
                            usage=u if (u.input_tokens or u.output_tokens or u.cache_read_tokens) else None,
                            model=model,
                        )
                    )
            elif typ == "GENERIC":
                cid, _ = pending_calls.popleft() if pending_calls else (f"agy_res_{si}", "tool")
                events.append(
                    Event(
                        role="tool",
                        tool_result=ToolResult(
                            call_id=cid,
                            content=content,
                            is_error=(status == "ERROR"),
                        ),
                    )
                )
            elif typ == "SYSTEM_MESSAGE":
                if content:
                    events.append(Event(role="user", text=content))
            elif typ == "CHECKPOINT":
                if content:
                    events.append(Event(role="compaction", text=content))

    return Session(
        harness="agy",
        model=model,
        session_id=sid,
        cwd=None,
        events=events,
        total_usage=total,
    )
