"""Parse Grok Build CLI streaming-json transcripts.

Observed event shape:
  {type: "thought", data: "..."}
  {type: "text", data: "..."}
  {type: "end", stopReason: "EndTurn", sessionId: "...", requestId: "..."}

The stream is token-delta oriented rather than message oriented, so we join
each channel into one assistant event.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.viewer.events import Event, Session, TokenUsage


def _model_from_run_dir(path: Path) -> str | None:
    name = path.parent.name
    marker = "_grok_"
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    match = re.match(r"(?P<model>.+)_\d{2}_.+", tail)
    if match:
        return match.group("model") or None
    return tail or None


def parse(path: Path) -> Session:
    thought_parts: list[str] = []
    text_parts: list[str] = []
    events: list[Event] = []
    session_id = None
    final_raw = None

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
