"""Parse Droid (Factory) stream-json output. Format observed from real init events.

Top-level events:
  {type: "system", subtype: "init", cwd, session_id, tools[], model, reasoning_effort}
  {type: "message", role, id, text, timestamp, session_id}
  {type: "error", source, message, timestamp, session_id}

Token usage is NOT in the stream — it's in ~/.factory/sessions/<cwd>/<uuid>.settings.json.
If the .settings.json sidecar can be located, we read it.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.viewer.events import Event, Session, TokenUsage


def _ms_to_dt(ms: int | None) -> datetime | None:
    if not ms:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000)
    except Exception:
        return None


def _find_settings(session_id: str | None, cwd: str | None) -> dict | None:
    if not session_id:
        return None
    base = Path.home() / ".factory" / "sessions"
    if not base.exists():
        return None
    # Sessions are organized by flattened cwd. Search broadly.
    for sf in base.rglob(f"{session_id}.settings.json"):
        try:
            return json.loads(sf.read_text())
        except Exception:
            return None
    return None


def parse(path: Path) -> Session:
    events: list[Event] = []
    model = None
    session_id = None
    cwd = None

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
                tools = obj.get("tools") or []
                events.append(Event(
                    role="system",
                    text=f"session start  cwd={cwd}  model={model}  effort={obj.get('reasoning_effort','?')}  tools={len(tools)}",
                    subtype="init", raw=obj,
                ))
                continue

            if t == "message":
                role = obj.get("role")
                ts = _ms_to_dt(obj.get("timestamp"))
                events.append(Event(
                    role=role if role in {"user", "assistant", "system"} else "system",
                    text=obj.get("text", ""),
                    timestamp=ts,
                    raw=obj,
                ))
                continue

            if t == "error":
                events.append(Event(
                    role="error",
                    text=f"[{obj.get('source','?')}] {obj.get('message','')}",
                    timestamp=_ms_to_dt(obj.get("timestamp")),
                    raw=obj,
                ))
                continue

    settings = _find_settings(session_id, cwd)
    total = TokenUsage()
    if settings:
        tu = settings.get("tokenUsage") or {}
        total = TokenUsage(
            input_tokens=tu.get("inputTokens", 0),
            output_tokens=tu.get("outputTokens", 0),
            cache_read_tokens=tu.get("cacheReadTokens", 0),
            cache_write_tokens=tu.get("cacheCreationTokens", 0),
            thinking_tokens=tu.get("thinkingTokens", 0),
        )
        model = settings.get("model") or model

    return Session(
        harness="droid",
        model=model,
        session_id=session_id,
        cwd=cwd,
        events=events,
        total_usage=total,
    )
