"""Parse Cursor Agent stream-json output.

Cleanest format of all: tool calls and results paired in one event.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.viewer.diff_util import FileTracker
from src.viewer.events import Event, Session, TokenUsage, ToolCall, ToolResult


def _ms_to_dt(ms: int | None) -> datetime | None:
    if not ms:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000)
    except Exception:
        return None


def _summarize_tool_call(tc: dict) -> tuple[str, dict, dict | None]:
    """Cursor wraps tool calls as e.g. {readToolCall: {args, result}}.

    Returns (tool_name, args, result_dict_or_none).
    """
    if not isinstance(tc, dict):
        return ("?", {}, None)
    for key, body in tc.items():
        if key.endswith("ToolCall") and isinstance(body, dict):
            name = key[:-len("ToolCall")]
            return (name, body.get("args") or {}, body.get("result"))
    return ("?", {}, None)


def parse(path: Path) -> Session:
    events: list[Event] = []
    model = None
    session_id = None
    cwd = None
    final_text = None
    total = TokenUsage()
    duration_ms = None
    files = FileTracker()

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = obj.get("type")

            if t == "system" and obj.get("subtype") == "init":
                model = obj.get("model") or model
                session_id = obj.get("session_id") or session_id
                cwd = obj.get("cwd") or cwd
                events.append(Event(
                    role="system",
                    text=f"session start  cwd={cwd}  model={model}  permissions={obj.get('permissionMode','?')}",
                    subtype="init", raw=obj,
                ))
                continue

            if t == "user":
                msg = obj.get("message") or {}
                content = msg.get("content") or []
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            parts.append(c.get("text", ""))
                    text = "\n".join(p for p in parts if p)
                else:
                    text = str(content)
                events.append(Event(role="user", text=text, raw=obj))
                continue

            if t == "tool_call":
                subtype = obj.get("subtype")  # "started" | "completed"
                name, args, result = _summarize_tool_call(obj.get("tool_call") or {})
                ts = _ms_to_dt(obj.get("timestamp_ms"))

                if subtype == "started":
                    diff = None
                    file_path = args.get("path") if isinstance(args, dict) else None
                    # Cursor's edit/write tools both carry full new content in
                    # `streamContent`. createToolCall similar.
                    if name in {"edit", "write", "create"} and file_path:
                        new_content = args.get("streamContent") or args.get("content") or ""
                        diff = files.diff_for_write(file_path, new_content)
                    tool_calls = [ToolCall(
                        name=name, args=args, call_id=obj.get("call_id"),
                        diff=diff or None, file_path=file_path,
                    )]
                    events.append(Event(
                        role="assistant",
                        tool_calls=tool_calls,
                        timestamp=ts, raw=obj, model=model,
                    ))
                elif subtype == "completed" and result is not None:
                    if isinstance(result, dict):
                        if "success" in result:
                            r = result["success"]
                            content_str = r.get("content") if isinstance(r, dict) else str(r)
                        elif "error" in result:
                            content_str = json.dumps(result["error"])
                        else:
                            content_str = json.dumps(result)
                    else:
                        content_str = str(result)
                    events.append(Event(
                        role="tool",
                        tool_result=ToolResult(
                            content=content_str if isinstance(content_str, str) else json.dumps(content_str),
                            call_id=obj.get("call_id"),
                            is_error=isinstance(result, dict) and "error" in result,
                        ),
                        timestamp=ts, raw=obj,
                    ))
                continue

            if t == "assistant":
                msg = obj.get("message") or {}
                model = msg.get("model") or model
                content = msg.get("content") or []
                text_parts = []
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            text_parts.append(c.get("text", ""))
                events.append(Event(
                    role="assistant",
                    text="\n".join(p for p in text_parts if p) or None,
                    raw=obj, model=model,
                ))
                if text_parts:
                    final_text = "\n".join(text_parts)
                continue

            if t == "result":
                final_text = obj.get("result") or final_text
                duration_ms = obj.get("duration_ms")
                u = obj.get("usage") or {}
                total = TokenUsage(
                    input_tokens=u.get("inputTokens", 0),
                    output_tokens=u.get("outputTokens", 0),
                    cache_read_tokens=u.get("cacheReadTokens", 0),
                    cache_write_tokens=u.get("cacheWriteTokens", 0),
                )
                continue

    return Session(
        harness="cursor",
        model=model,
        session_id=session_id,
        cwd=cwd,
        events=events,
        final_text=final_text,
        total_usage=total,
        duration_ms=duration_ms,
    )
