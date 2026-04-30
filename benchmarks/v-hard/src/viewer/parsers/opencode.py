"""Parse OpenCode SST `--format json` output.

OpenCode wraps each event in `{type, timestamp, sessionID, part: {...}}`. The
canonical event types we care about:

  step_start              -- new turn/step begins; no payload of interest
  step_finish             -- end of step; carries `part.tokens` usage block
  tool_use                -- tool call; `part.tool` is the name,
                             `part.state.input` is args, `part.state.output`
                             is the result, `part.state.status` in/out
  text                    -- assistant text content
  text_delta              -- streaming chunk (we pass through)
  error                   -- API error (auth, rate limit, etc.)
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


def parse(path: Path) -> Session:
    events: list[Event] = []
    model = None
    session_id = None
    cwd = None
    final_text = None
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

            t = obj.get("type")
            ts = _ms_to_dt(obj.get("timestamp"))
            session_id = obj.get("sessionID") or session_id
            part = obj.get("part") or {}

            if t == "step_start":
                # No content; renderer doesn't need a separate event for these
                # (the next tool_use / text event is more informative).
                continue

            if t == "step_finish":
                tokens = part.get("tokens") or {}
                cache = tokens.get("cache") or {}
                u = TokenUsage(
                    input_tokens=tokens.get("input", 0) or 0,
                    output_tokens=tokens.get("output", 0) or 0,
                    cache_read_tokens=cache.get("read", 0) or 0,
                    cache_write_tokens=cache.get("write", 0) or 0,
                    thinking_tokens=tokens.get("reasoning", 0) or 0,
                )
                total.input_tokens += u.input_tokens
                total.output_tokens += u.output_tokens
                total.cache_read_tokens += u.cache_read_tokens
                total.cache_write_tokens += u.cache_write_tokens
                total.thinking_tokens += u.thinking_tokens
                # Attach to a synthetic assistant event so the renderer shows it
                events.append(Event(
                    role="assistant", text=None, usage=u, timestamp=ts,
                    session_id=session_id, raw=obj,
                ))
                continue

            if t == "tool_use":
                tname = part.get("tool") or "?"
                state = part.get("state") or {}
                args = state.get("input") or {}
                output = state.get("output", "")
                status = state.get("status")
                call_id = part.get("callID")

                # Detect file write / edit tools from opencode's standard kit.
                diff = None
                fpath = None
                if isinstance(args, dict):
                    fpath = args.get("filePath") or args.get("file_path") or args.get("path")
                if tname in {"write", "create"} and fpath:
                    diff = files.diff_for_write(fpath, args.get("content", "") or "")
                elif tname == "edit" and fpath:
                    diff = files.diff_for_edit(
                        fpath,
                        args.get("oldString", args.get("old_string", "")) or "",
                        args.get("newString", args.get("new_string", "")) or "",
                    )

                # Emit tool call as one event, tool result as a follow-up.
                events.append(Event(
                    role="assistant",
                    tool_calls=[ToolCall(
                        name=tname, args=args if isinstance(args, dict) else {"raw": args},
                        call_id=call_id, diff=diff or None, file_path=fpath,
                    )],
                    timestamp=ts, session_id=session_id, raw=obj,
                ))
                if output is not None and output != "":
                    events.append(Event(
                        role="tool",
                        tool_result=ToolResult(
                            content=str(output),
                            call_id=call_id,
                            is_error=(status == "error"),
                        ),
                        timestamp=ts, session_id=session_id, raw=obj,
                    ))
                continue

            if t in {"text", "text_delta"}:
                txt = part.get("text") if isinstance(part, dict) else None
                if not txt:
                    txt = obj.get("text")
                if txt:
                    events.append(Event(
                        role="assistant", text=txt, timestamp=ts,
                        session_id=session_id, raw=obj,
                    ))
                    final_text = txt
                continue

            if t == "error":
                err = obj.get("error") or {}
                msg = err.get("data", {}).get("message") or err.get("name") or "unknown error"
                events.append(Event(
                    role="error", text=str(msg), timestamp=ts,
                    session_id=session_id, raw=obj,
                ))
                continue

    return Session(
        harness="opencode",
        model=model,
        session_id=session_id,
        cwd=cwd,
        events=events,
        final_text=final_text,
        total_usage=total,
    )
