"""Parse Grok Build CLI sessions into the shared Event model.

Preferred source (full tool timeline, correct for HF Agent Trace viewer):
  <run>/agent_home/.grok/sessions/**/chat_history.jsonl

Fallback (token-delta streaming log, no tools — what --output-format
streaming-json writes to the harness transcript.jsonl):
  {type: "thought"|"text"|"end", data: "..."|...}

chat_history message shapes:
  {type: "system"|"user"|"assistant"|"reasoning"|"tool_result", ...}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.viewer.events import Event, Session, TokenUsage, ToolCall, ToolResult


def _model_from_run_dir(path: Path) -> str | None:
    # path may be transcript or chat_history; walk up to run id dir
    for candidate in (path, *path.parents):
        name = candidate.name
        marker = "_grok_"
        if marker in name:
            tail = name.split(marker, 1)[1]
            match = re.match(r"(?P<model>.+)_\d{2}_.+", tail)
            if match:
                return match.group("model") or None
            return tail or None
    return None


def _run_dir_for(path: Path) -> Path | None:
    """Walk parents looking for a KernelBench run archive dir (has result.json)."""
    for p in (path if path.is_dir() else path.parent, *path.parents):
        if (p / "result.json").exists() or (p / "agent_home").is_dir():
            return p
    return None


def _find_chat_history(path: Path) -> Path | None:
    """Locate the richest Grok session chat_history under the run archive."""
    if path.name == "chat_history.jsonl" and path.exists():
        return path
    run = _run_dir_for(path)
    if run is None:
        return None
    sessions = run / "agent_home" / ".grok" / "sessions"
    if not sessions.is_dir():
        return None
    candidates = sorted(
        sessions.rglob("chat_history.jsonl"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _text_from_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif "text" in block:
                    parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        joined = "".join(parts).strip()
        return joined or None
    return str(content)


def _reasoning_from_obj(obj: dict) -> str | None:
    """Prefer visible summary_text over encrypted_content."""
    summary = obj.get("summary")
    if isinstance(summary, list):
        parts = []
        for item in summary:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("summary_text") or ""))
            else:
                parts.append(str(item))
        text = "\n".join(p for p in parts if p).strip()
        if text:
            return text
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    # Never put encrypted_content in the viewer — unreadable base64 noise.
    return None


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": raw}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {"_raw": str(raw)}


def _parse_chat_history(path: Path) -> Session:
    events: list[Event] = []
    session_id = None
    model = _model_from_run_dir(path)
    cwd = None
    pending_reasoning: str | None = None
    final_text: str | None = None

    # session id from parent dir name if uuid-like
    parent = path.parent.name
    if re.match(r"[0-9a-f-]{20,}", parent):
        session_id = parent

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
                text = _text_from_content(obj.get("content"))
                if text:
                    events.append(Event(
                        role="system",
                        text=text,
                        subtype="system",
                        session_id=session_id,
                        raw=obj,
                    ))
                continue

            if t == "user":
                text = _text_from_content(obj.get("content"))
                if text:
                    events.append(Event(
                        role="user",
                        text=text,
                        session_id=session_id,
                        raw=obj,
                    ))
                continue

            if t == "reasoning":
                # Attach to the next assistant message as reasoning.
                r = _reasoning_from_obj(obj)
                if r:
                    pending_reasoning = (
                        f"{pending_reasoning}\n{r}" if pending_reasoning else r
                    )
                continue

            if t == "assistant":
                model = obj.get("model_id") or model
                text = _text_from_content(obj.get("content"))
                tool_calls: list[ToolCall] = []
                for tc in obj.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    name = str(tc.get("name") or "tool")
                    call_id = tc.get("id") or tc.get("call_id")
                    args = _parse_tool_args(tc.get("arguments") or tc.get("args"))
                    tool_calls.append(ToolCall(name=name, args=args, call_id=call_id))
                # Skip empty assistant shells with neither text nor tools
                if not text and not tool_calls and not pending_reasoning:
                    continue
                events.append(Event(
                    role="assistant",
                    text=text,
                    reasoning=pending_reasoning,
                    tool_calls=tool_calls,
                    model=model,
                    session_id=session_id,
                    raw=obj,
                ))
                if text:
                    final_text = text
                pending_reasoning = None
                continue

            if t == "tool_result":
                content = _text_from_content(obj.get("content")) or ""
                call_id = obj.get("tool_call_id") or obj.get("call_id")
                is_error = bool(obj.get("is_error") or obj.get("error"))
                events.append(Event(
                    role="tool",
                    tool_result=ToolResult(
                        content=content,
                        call_id=call_id,
                        is_error=is_error,
                    ),
                    session_id=session_id,
                    raw=obj,
                ))
                continue

    # Prefer problem workspace cwd from summary if present
    summary = path.parent / "summary.json"
    if summary.exists():
        try:
            info = json.loads(summary.read_text())
            cwd = (info.get("info") or {}).get("cwd") or cwd
            session_id = (info.get("info") or {}).get("id") or session_id
        except (json.JSONDecodeError, OSError):
            pass

    return Session(
        harness="grok",
        model=model,
        session_id=session_id,
        cwd=cwd or str(path.parent),
        events=events,
        final_text=final_text,
        total_usage=TokenUsage(),
    )


def _parse_streaming_jsonl(path: Path) -> Session:
    """Legacy streaming-json transcript: token deltas only, no tools."""
    thought_parts: list[str] = []
    text_parts: list[str] = []
    events: list[Event] = []
    session_id = None
    final_raw = None

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw or not raw.startswith("{"):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = obj.get("type")
            if t == "thought":
                thought_parts.append(str(obj.get("data") or ""))
                continue
            if t == "text":
                text_parts.append(str(obj.get("data") or ""))
                continue
            if t == "end":
                session_id = obj.get("sessionId") or session_id
                final_raw = obj
                continue

    # Prepend the problem prompt when available so the viewer has a user turn.
    run = _run_dir_for(path)
    prompt_text = None
    if run is not None:
        for candidate in (
            run / "repo" / "problems",
            run,
        ):
            # e.g. repo/problems/01_fp8_gemm/PROMPT.txt
            hits = list(candidate.rglob("PROMPT.txt")) if candidate.is_dir() else []
            if hits:
                try:
                    prompt_text = hits[0].read_text()
                except OSError:
                    prompt_text = None
                break
    if prompt_text:
        events.append(Event(role="user", text=prompt_text, session_id=session_id))

    reasoning = "".join(thought_parts).strip() or None
    final_text = "".join(text_parts).strip() or None
    if reasoning or final_text:
        events.append(Event(
            role="assistant",
            text=final_text,
            reasoning=reasoning,
            session_id=session_id,
            raw=final_raw,
        ))
    if final_raw:
        events.append(Event(
            role="system",
            text=f"session end  stopReason={final_raw.get('stopReason', '?')}  requestId={final_raw.get('requestId', '?')}",
            subtype="end",
            session_id=session_id,
            raw=final_raw,
        ))

    return Session(
        harness="grok",
        model=_model_from_run_dir(path),
        session_id=session_id,
        cwd=str(path.parent),
        events=events,
        final_text=final_text,
        total_usage=TokenUsage(),
    )


def parse(path: Path) -> Session:
    path = Path(path)
    chat = _find_chat_history(path)
    if chat is not None:
        return _parse_chat_history(chat)
    return _parse_streaming_jsonl(path)
