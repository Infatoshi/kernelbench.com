"""Parser dispatch — sniff a transcript file and return the right Session."""
from __future__ import annotations

import json
from pathlib import Path

from src.viewer.events import Session


def sniff(path: Path) -> str:
    """Detect the harness format from the first non-empty JSONL line.

    Returns one of: "claude" | "codex" | "kimi" | "cursor" | "droid".
    Falls back to file-extension hints, then raises if undecidable.
    """
    if path.suffix == ".txt":
        return "codex"  # codex `exec` writes plain text to stdout
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Could be codex stdout text mode
                return "codex"

            # type=system, subtype=init: Claude Code, Cursor, and Droid all emit this.
            # Use field-presence heuristics rather than guessing model strings.
            if obj.get("type") == "system" and obj.get("subtype") == "init":
                # Droid: has reasoning_effort + tools but no mcp_servers / apiKeySource
                if "reasoning_effort" in obj and "tools" in obj and "mcp_servers" not in obj:
                    return "droid"
                # Claude Code has very distinctive fields the others don't
                if any(k in obj for k in ("claude_code_version", "mcp_servers", "slash_commands", "agents", "skills")):
                    return "claude"
                # Cursor's init is minimal: cwd, session_id, model, permissionMode, apiKeySource
                if "apiKeySource" in obj or "permissionMode" in obj:
                    return "cursor"

            # Codex session file: wrapped {type, payload, timestamp}
            if "payload" in obj and obj.get("type") in {"session_meta", "event_msg", "response_item", "task_started"}:
                return "codex"

            # Claude Code session file: parentUuid present
            if "parentUuid" in obj or ("type" in obj and "uuid" in obj and "sessionId" in obj):
                return "claude"

            # Kimi: top-level role=tool|assistant|user with content array of {type, text|think}
            if obj.get("role") in {"user", "assistant", "tool"} and "type" not in obj:
                return "kimi"

            # Cursor: type tool_call / message / result
            if obj.get("type") in {"tool_call", "result", "user", "assistant"} and "session_id" in obj:
                # Cursor uses "session_id" with underscore everywhere
                return "cursor"

            # Droid: message events have explicit type and text fields
            if obj.get("type") == "message" and "session_id" in obj:
                return "droid"

            # OpenCode: events wrap a `part` object and use sessionID (camelCase).
            if obj.get("type") in {"step_start", "step_finish", "tool_use", "text", "text_delta", "error"} \
                    and ("sessionID" in obj or "part" in obj):
                return "opencode"

            return "claude"  # fallback

    raise ValueError(f"could not sniff format of {path}")


def parse(path: Path) -> Session:
    fmt = sniff(path)
    if fmt == "claude":
        from src.viewer.parsers import claude
        return claude.parse(path)
    if fmt == "codex":
        from src.viewer.parsers import codex
        return codex.parse(path)
    if fmt == "kimi":
        from src.viewer.parsers import kimi
        return kimi.parse(path)
    if fmt == "cursor":
        from src.viewer.parsers import cursor
        return cursor.parse(path)
    if fmt == "droid":
        from src.viewer.parsers import droid
        return droid.parse(path)
    if fmt == "opencode":
        from src.viewer.parsers import opencode
        return opencode.parse(path)
    raise ValueError(f"unknown format {fmt!r}")
