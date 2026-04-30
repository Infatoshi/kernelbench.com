"""Parse Claude Code stream-json transcripts."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.viewer.diff_util import FileTracker
from src.viewer.events import Event, Session, TokenUsage, ToolCall, ToolResult


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
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
    in_subagent = False  # True between task_started and task_notification

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
            if t == "system":
                model = obj.get("model") or model
                session_id = obj.get("session_id") or obj.get("sessionId") or session_id
                cwd = obj.get("cwd") or cwd
                sub = obj.get("subtype")
                if sub == "init":
                    events.append(Event(
                        role="system",
                        text=f"session start  cwd={cwd}  model={model}  permissions={obj.get('permissionMode','?')}",
                        subtype="init", raw=obj,
                    ))
                elif sub == "compact_boundary":
                    summary = obj.get("message", {}).get("summary") or obj.get("summary") or ""
                    events.append(Event(
                        role="compaction",
                        text=str(summary),
                        subtype="compact_boundary",
                        raw=obj,
                    ))
                elif sub == "task_started":
                    in_subagent = True
                    events.append(Event(
                        role="system", subtype="task_started",
                        text="↳ subagent started", is_sidechain=True, raw=obj,
                    ))
                elif sub == "task_notification":
                    events.append(Event(
                        role="system", subtype="task_notification",
                        text="↳ subagent complete", is_sidechain=True, raw=obj,
                    ))
                    in_subagent = False
                elif sub == "task_progress":
                    # Just a heartbeat between subagent turns; skip in default render.
                    pass
                # other system events (file-history-snapshot etc.) — skip
                continue

            if t == "user":
                msg = obj.get("message") or {}
                content = msg.get("content")
                # User content can be a string (initial prompt) or a list of blocks
                # (tool_results sent back to the model on subsequent turns).
                sidechain = bool(obj.get("isSidechain")) or in_subagent
                if isinstance(content, str):
                    events.append(Event(
                        role="user", text=content,
                        timestamp=_parse_ts(obj.get("timestamp")),
                        is_sidechain=sidechain,
                        parent_uuid=obj.get("parentUuid"),
                        raw=obj,
                    ))
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tr_content = block.get("content")
                            if isinstance(tr_content, list):
                                tr_text = "\n".join(
                                    b.get("text", "") for b in tr_content if isinstance(b, dict)
                                )
                            else:
                                tr_text = str(tr_content) if tr_content is not None else ""
                            events.append(Event(
                                role="tool",
                                tool_result=ToolResult(
                                    content=tr_text,
                                    call_id=block.get("tool_use_id"),
                                    is_error=bool(block.get("is_error")),
                                ),
                                timestamp=_parse_ts(obj.get("timestamp")),
                                is_sidechain=sidechain,
                                raw=obj,
                            ))
                        elif isinstance(block, dict) and block.get("type") == "text":
                            events.append(Event(
                                role="user", text=block.get("text", ""),
                                is_sidechain=sidechain, raw=obj,
                            ))
                continue

            if t == "assistant":
                msg = obj.get("message") or {}
                model = msg.get("model") or model
                content = msg.get("content") or []
                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                tool_calls: list[ToolCall] = []
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        bt = block.get("type")
                        if bt == "text":
                            text_parts.append(block.get("text", ""))
                        elif bt in {"thinking", "reasoning"}:
                            reasoning_parts.append(block.get("thinking") or block.get("text") or "")
                        elif bt == "tool_use":
                            name = block.get("name", "?")
                            args = block.get("input") or {}
                            diff = None
                            file_path = args.get("file_path") if isinstance(args, dict) else None
                            if name == "Write" and file_path:
                                diff = files.diff_for_write(file_path, args.get("content", "") or "")
                            elif name == "Edit" and file_path:
                                diff = files.diff_for_edit(
                                    file_path,
                                    args.get("old_string", "") or "",
                                    args.get("new_string", "") or "",
                                )
                            elif name == "MultiEdit" and file_path:
                                diff_chunks = []
                                for ed in args.get("edits") or []:
                                    if isinstance(ed, dict):
                                        d = files.diff_for_edit(
                                            file_path,
                                            ed.get("old_string", "") or "",
                                            ed.get("new_string", "") or "",
                                        )
                                        if d:
                                            diff_chunks.append(d)
                                diff = "\n".join(diff_chunks) if diff_chunks else None
                            tool_calls.append(ToolCall(
                                name=name, args=args if isinstance(args, dict) else {},
                                call_id=block.get("id"),
                                diff=diff or None,
                                file_path=file_path,
                            ))

                usage_obj = msg.get("usage") or {}
                usage = TokenUsage(
                    input_tokens=usage_obj.get("input_tokens", 0),
                    output_tokens=usage_obj.get("output_tokens", 0),
                    cache_read_tokens=usage_obj.get("cache_read_input_tokens", 0),
                    cache_write_tokens=usage_obj.get("cache_creation_input_tokens", 0),
                )
                total.input_tokens += usage.input_tokens
                total.output_tokens += usage.output_tokens
                total.cache_read_tokens += usage.cache_read_tokens
                total.cache_write_tokens += usage.cache_write_tokens

                events.append(Event(
                    role="assistant",
                    text="\n".join(p for p in text_parts if p) or None,
                    reasoning="\n".join(reasoning_parts) or None,
                    tool_calls=tool_calls,
                    usage=usage,
                    model=model,
                    timestamp=_parse_ts(obj.get("timestamp")),
                    is_sidechain=bool(obj.get("isSidechain")) or in_subagent,
                    parent_uuid=obj.get("parentUuid"),
                    raw=obj,
                ))
                if text_parts:
                    final_text = "\n".join(text_parts)
                continue

            if t == "result":
                final_text = obj.get("result") or final_text
                continue

    return Session(
        harness="claude",
        model=model,
        session_id=session_id,
        cwd=cwd,
        events=events,
        final_text=final_text,
        total_usage=total,
    )
