"""Parse Kimi CLI stream-json output. OpenAI-chat shape with `think` blocks."""
from __future__ import annotations

import json
from pathlib import Path

from src.viewer.diff_util import FileTracker
from src.viewer.events import Event, Session, TokenUsage, ToolCall, ToolResult


def parse(path: Path) -> Session:
    events: list[Event] = []
    total = TokenUsage()
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

            role = obj.get("role")
            content = obj.get("content") or []

            if role == "user":
                if isinstance(content, str):
                    events.append(Event(role="user", text=content, raw=obj))
                elif isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            parts.append(c.get("text") or c.get("think") or "")
                        elif isinstance(c, str):
                            parts.append(c)
                    events.append(Event(role="user", text="\n".join(parts), raw=obj))
                continue

            if role == "assistant":
                text_parts = []
                reasoning_parts = []
                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        if c.get("type") == "think":
                            reasoning_parts.append(c.get("think", ""))
                        elif c.get("type") in {"text", "output_text"}:
                            text_parts.append(c.get("text", ""))

                tool_calls = []
                for tc in obj.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    args_raw = fn.get("arguments")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                    tname = fn.get("name", "?")
                    targs = args if isinstance(args, dict) else {"raw": args}
                    diff = None
                    file_path = None
                    if isinstance(targs, dict):
                        file_path = targs.get("path") or targs.get("file_path")
                        if tname in {"WriteFile", "CreateFile"} and file_path:
                            diff = files.diff_for_write(file_path, targs.get("content", "") or "")
                        elif tname in {"EditFile", "Edit"} and file_path:
                            diff = files.diff_for_edit(
                                file_path,
                                targs.get("old_string", "") or "",
                                targs.get("new_string", "") or "",
                            )
                    tool_calls.append(ToolCall(
                        name=tname, args=targs, call_id=tc.get("id"),
                        diff=diff or None, file_path=file_path,
                    ))

                events.append(Event(
                    role="assistant",
                    text="\n".join(p for p in text_parts if p) or None,
                    reasoning="\n".join(reasoning_parts) or None,
                    tool_calls=tool_calls,
                    raw=obj,
                ))
                continue

            if role == "tool":
                tr_text = ""
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            parts.append(c.get("text", ""))
                    tr_text = "\n".join(parts)
                elif isinstance(content, str):
                    tr_text = content
                events.append(Event(
                    role="tool",
                    tool_result=ToolResult(
                        content=tr_text,
                        call_id=obj.get("tool_call_id"),
                    ),
                    raw=obj,
                ))
                continue

    return Session(
        harness="kimi",
        model="kimi",
        session_id=None,
        cwd=None,
        events=events,
        total_usage=total,
    )
