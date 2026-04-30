"""Parse Codex session JSONL (~/.codex/sessions/.../rollout-*.jsonl).

Each line is wrapped: {type, timestamp, payload: {...}}. The canonical content
stream is in `response_item.*` events; `event_msg.agent_*` events are previews
of those for the live UI and are dropped. `event_msg.token_count` carries
running token totals.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.viewer.diff_util import FileTracker, make_diff, parse_codex_apply_patch
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

    if path.suffix == ".txt":
        text = path.read_text()
        events.append(Event(role="assistant", text=text or "(empty)"))
        return Session(harness="codex", model=None, session_id=None, cwd=None,
                       events=events, final_text=text, total_usage=total)

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
            payload = obj.get("payload") or {}
            ts = _parse_ts(obj.get("timestamp"))

            if t == "session_meta":
                model = payload.get("model") or model
                session_id = payload.get("session_id") or session_id
                cwd = payload.get("cwd") or cwd
                events.append(Event(
                    role="system",
                    text=f"session start  model={model}  ctx={payload.get('model_context_window','?')}",
                    timestamp=ts, raw=obj,
                ))
                continue

            if t == "event_msg":
                pt = payload.get("type")
                if pt == "task_started":
                    events.append(Event(
                        role="system", text=f"task_started turn={(payload.get('turn_id') or '')[:8]}",
                        timestamp=ts, raw=obj,
                    ))
                elif pt == "task_complete":
                    events.append(Event(
                        role="system", text="task_complete",
                        timestamp=ts, raw=obj,
                    ))
                elif pt == "user_message":
                    # Skip — covered by response_item.message
                    pass
                elif pt == "token_count":
                    info = payload.get("info") or {}
                    last = info.get("last_token_usage") or {}
                    total.input_tokens += last.get("input_tokens", 0) or 0
                    total.output_tokens += last.get("output_tokens", 0) or 0
                    total.cache_read_tokens += last.get("cached_input_tokens", 0) or 0
                # Drop agent_message / agent_reasoning — duplicates of response_item.*
                continue

            if t == "response_item":
                pt = payload.get("type")
                if pt == "message":
                    role = payload.get("role")
                    content = payload.get("content") or []
                    text_parts: list[str] = []
                    if isinstance(content, list):
                        for c in content:
                            if not isinstance(c, dict):
                                continue
                            txt = c.get("text") or c.get("input_text") or c.get("output_text")
                            if txt:
                                text_parts.append(txt)
                    if not text_parts:
                        continue
                    body = "\n".join(text_parts)
                    norm_role = role if role in {"user", "assistant", "system"} else "system"
                    if role == "developer":
                        # Developer messages are the harness preamble — render as system, low-key
                        norm_role = "system"
                    events.append(Event(
                        role=norm_role,
                        text=body,
                        timestamp=ts, raw=obj,
                        model=model if norm_role == "assistant" else None,
                    ))
                    if norm_role == "assistant":
                        final_text = body
                elif pt == "reasoning":
                    summary = payload.get("summary") or []
                    parts = []
                    if isinstance(summary, list):
                        for s in summary:
                            if isinstance(s, dict):
                                txt = s.get("text") or s.get("input_text") or s.get("output_text")
                                if txt:
                                    parts.append(txt)
                    if parts:
                        events.append(Event(
                            role="assistant",
                            reasoning="\n".join(parts),
                            timestamp=ts, raw=obj, model=model,
                        ))
                elif pt == "function_call":
                    name = payload.get("name") or "?"
                    args_raw = payload.get("arguments")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                    except json.JSONDecodeError:
                        args = {"raw": args_raw}
                    events.append(Event(
                        role="assistant",
                        tool_calls=[ToolCall(
                            name=name,
                            args=args if isinstance(args, dict) else {"raw": args},
                            call_id=payload.get("call_id"),
                        )],
                        timestamp=ts, raw=obj, model=model,
                    ))
                elif pt == "function_call_output":
                    out = payload.get("output") or ""
                    events.append(Event(
                        role="tool",
                        tool_result=ToolResult(
                            content=str(out),
                            call_id=payload.get("call_id"),
                        ),
                        timestamp=ts, raw=obj,
                    ))
                elif pt in {"custom_tool_call", "custom_tool_call_output"}:
                    if pt == "custom_tool_call":
                        cname = payload.get("name") or "custom_tool"
                        cinput = payload.get("input") or ""
                        diff = None
                        file_path = None
                        if cname == "apply_patch" and isinstance(cinput, str):
                            # Parse codex's patch envelope into per-file old/new pairs.
                            entries = parse_codex_apply_patch(cinput)
                            chunks = []
                            for fpath, (old, new) in entries.items():
                                # Use any prior tracked content if Add was applied earlier.
                                tracked = files.get(fpath) or old
                                chunks.append(make_diff(fpath, tracked, new))
                                files.write(fpath, new)
                            diff = "\n".join(c for c in chunks if c) or None
                            file_path = ", ".join(entries.keys()) if entries else None
                        events.append(Event(
                            role="assistant",
                            tool_calls=[ToolCall(
                                name=cname,
                                args={"raw": cinput},
                                call_id=payload.get("call_id"),
                                diff=diff,
                                file_path=file_path,
                            )],
                            timestamp=ts, raw=obj, model=model,
                        ))
                    else:
                        events.append(Event(
                            role="tool",
                            tool_result=ToolResult(
                                content=str(payload.get("output") or ""),
                                call_id=payload.get("call_id"),
                            ),
                            timestamp=ts, raw=obj,
                        ))
                continue

            # turn_context, etc. — ignore

    return Session(
        harness="codex",
        model=model,
        session_id=session_id,
        cwd=cwd,
        events=events,
        final_text=final_text,
        total_usage=total,
    )
