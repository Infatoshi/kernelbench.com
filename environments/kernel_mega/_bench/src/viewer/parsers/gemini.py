"""Parse Gemini CLI stream-json output (`gemini -o stream-json`).

Top-level events observed from real runs:
  {type: "init", session_id, model, timestamp}
  {type: "message", role, content, timestamp}
  {type: "tool_use", tool_name, tool_id, parameters, timestamp}
  {type: "tool_result", tool_id, status, output, timestamp}
  {type: "result", status, stats: {total_tokens, input_tokens, output_tokens,
                                   cached, ...}, timestamp}

Container-mode transcripts may be prefixed by the NGC image banner; non-JSON
lines are skipped. Usage comes from the terminal result event's stats block.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.viewer.events import Event, Session, TokenUsage, ToolCall, ToolResult


def _iso_to_dt(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def parse(path: Path) -> Session:
    events: list[Event] = []
    model = None
    session_id = None
    total = TokenUsage()

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue  # NGC banner / non-JSON noise

            t = obj.get("type")
            ts = _iso_to_dt(obj.get("timestamp"))

            if t == "init":
                model = obj.get("model") or model
                session_id = obj.get("session_id") or session_id
                events.append(Event(
                    role="system",
                    text=f"session start  model={model}",
                    subtype="init", timestamp=ts, raw=obj,
                ))
                continue

            if t == "message":
                role = obj.get("role")
                events.append(Event(
                    role=role if role in {"user", "assistant", "system"} else "assistant",
                    text=obj.get("content", "") or obj.get("text", ""),
                    timestamp=ts, session_id=session_id, raw=obj,
                ))
                continue

            if t == "tool_use":
                events.append(Event(
                    role="assistant",
                    tool_calls=[ToolCall(
                        name=obj.get("tool_name", "?"),
                        args=obj.get("parameters") or {},
                        call_id=obj.get("tool_id"),
                    )],
                    timestamp=ts, session_id=session_id, raw=obj,
                ))
                continue

            if t == "tool_result":
                events.append(Event(
                    role="tool",
                    tool_result=ToolResult(
                        content=str(obj.get("output", "")),
                        call_id=obj.get("tool_id"),
                        is_error=obj.get("status") not in (None, "success"),
                    ),
                    timestamp=ts, session_id=session_id, raw=obj,
                ))
                continue

            if t == "result":
                stats = obj.get("stats") or {}
                total = TokenUsage(
                    input_tokens=stats.get("input_tokens", 0),
                    output_tokens=stats.get("output_tokens", 0),
                    cache_read_tokens=stats.get("cached", 0),
                )
                events.append(Event(
                    role="system",
                    text=f"result: {obj.get('status','?')}",
                    subtype="result", timestamp=ts, raw=obj,
                ))
                continue

    return Session(
        harness="gemini",
        model=model,
        session_id=session_id,
        cwd=None,
        events=events,
        total_usage=total,
    )
