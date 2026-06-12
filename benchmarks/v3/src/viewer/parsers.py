"""Normalize harness transcripts (droid / claude / codex) into a common event stream."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Event:
    seq: int
    role: str  # user | assistant | thinking | tool_call | tool_result | system
    label: str = ""
    content: str = ""
    is_error: bool = False
    meta: dict = field(default_factory=dict)


def detect_format(path: Path) -> str:
    """Return 'droid' | 'claude' | 'codex' by sniffing the first few non-empty lines."""
    samples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                return "codex"
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                return "codex"
            if len(samples) >= 5:
                break
    if not samples:
        return "codex"
    # Claude init line has apiKeySource/permissionMode; subsequent assistant lines wrap in message.content[]
    for s in samples:
        if "apiKeySource" in s or "permissionMode" in s or "mcp_servers" in s:
            return "claude"
        if s.get("type") == "assistant" and isinstance(s.get("message"), dict):
            return "claude"
    # Droid: flat message/reasoning/tool_call/tool_result with role/text at top level
    for s in samples:
        if s.get("type") in ("reasoning", "tool_call") and ("text" in s or "parameters" in s):
            return "droid"
        if s.get("type") == "message" and "role" in s and "text" in s:
            return "droid"
    return "droid"


def parse_droid(path: Path) -> list[Event]:
    events: list[Event] = []
    seq = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "system":
                seq += 1
                events.append(Event(seq, "system", obj.get("subtype", "system"), f"cwd: {obj.get('cwd','')}  session: {obj.get('session_id','')[:8]}  model: {obj.get('model','')}"))
            elif t == "user":
                seq += 1
                events.append(Event(seq, "user", "user prompt", obj.get("content", "") or obj.get("text", "")))
            elif t == "message" or t == "assistant":
                text = obj.get("text") or obj.get("content") or ""
                if isinstance(text, list):
                    text = "\n".join(p.get("text", "") for p in text if isinstance(p, dict))
                role = obj.get("role", "assistant")
                if text:
                    seq += 1
                    if role == "user":
                        events.append(Event(seq, "user", "user prompt", text))
                    else:
                        events.append(Event(seq, "assistant", "assistant", text))
            elif t in ("thinking", "reasoning"):
                txt = obj.get("text") or obj.get("content") or ""
                if txt:
                    seq += 1
                    events.append(Event(seq, "thinking", "thinking", txt))
            elif t == "completion":
                seq += 1
                events.append(Event(seq, "system", "done",
                    f"usage: {obj.get('usage', '')}  duration: {obj.get('duration_ms','?')}ms"))
            elif t == "tool_call":
                seq += 1
                params = obj.get("parameters", {})
                events.append(Event(seq, "tool_call", obj.get("toolName", "tool"), _summarize_tool_call(obj.get("toolName", ""), params), meta={"call_id": obj.get("id")}))
            elif t == "tool_result":
                seq += 1
                content = obj.get("value") or obj.get("content") or obj.get("output") or ""
                err = obj.get("error")
                is_err = bool(err or obj.get("isError"))
                if err:
                    if isinstance(err, dict):
                        content = f"ERROR: {err.get('message', err)}"
                    else:
                        content = f"ERROR: {err}"
                if isinstance(content, list):
                    content = "\n".join(str(c) for c in content)
                label = obj.get("toolId") or "result"
                events.append(Event(seq, "tool_result", label, str(content), is_error=is_err, meta={"call_id": obj.get("id")}))
            elif t == "result":
                seq += 1
                events.append(Event(seq, "system", "final", obj.get("result", "") or str(obj)))
    return events


def parse_claude(path: Path) -> list[Event]:
    events: list[Event] = []
    seq = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "system":
                seq += 1
                events.append(Event(seq, "system", obj.get("subtype", "system"),
                    f"model: {obj.get('model','')}  cwd: {obj.get('cwd','')}  session: {obj.get('session_id','')[:8]}"))
            elif t == "assistant":
                msg = obj.get("message", {})
                for block in msg.get("content", []):
                    btype = block.get("type")
                    if btype == "text":
                        seq += 1
                        events.append(Event(seq, "assistant", "assistant", block.get("text", "")))
                    elif btype == "thinking":
                        seq += 1
                        events.append(Event(seq, "thinking", "thinking", block.get("thinking", "")))
                    elif btype == "tool_use":
                        seq += 1
                        name = block.get("name", "tool")
                        events.append(Event(seq, "tool_call", name,
                            _summarize_tool_call(name, block.get("input", {})),
                            meta={"call_id": block.get("id")}))
            elif t == "user":
                msg = obj.get("message", {})
                content = msg.get("content", [])
                if isinstance(content, str):
                    seq += 1
                    events.append(Event(seq, "user", "user prompt", content))
                else:
                    for block in content:
                        if block.get("type") == "tool_result":
                            seq += 1
                            raw = block.get("content", "")
                            if isinstance(raw, list):
                                raw = "\n".join(p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in raw)
                            events.append(Event(seq, "tool_result", "result", str(raw),
                                is_error=bool(block.get("is_error")),
                                meta={"call_id": block.get("tool_use_id")}))
                        elif block.get("type") == "text":
                            seq += 1
                            events.append(Event(seq, "user", "user prompt", block.get("text", "")))
            elif t == "result":
                seq += 1
                events.append(Event(seq, "system", "final",
                    f"result: {obj.get('subtype','')} · duration {obj.get('duration_ms',0)}ms · turns {obj.get('num_turns','?')}"))
    return events


# Codex exec format: section markers "user", "codex", "thinking", "exec" followed by blank line;
# or tokenized like "exec <cmd> in <dir>" and " succeeded in Xms:" then captured stdout.
_CODEX_SECTION = re.compile(r"^(user|codex|thinking|tool|exec|turn diff|tokens used)\s*$")


def parse_codex(path: Path) -> list[Event]:
    text = path.read_text()
    lines = text.splitlines()
    events: list[Event] = []
    seq = 0
    i = 0
    # Skip header (everything up to first blank line after "---" or "session id:")
    while i < len(lines) and not _CODEX_SECTION.match(lines[i].strip()):
        i += 1
    current_role = None
    current_label = ""
    current_buf: list[str] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        m = _CODEX_SECTION.match(stripped)
        if m:
            # flush previous
            if current_role is not None and current_buf:
                seq += 1
                events.append(_codex_event(seq, current_role, current_label, "\n".join(current_buf).rstrip()))
                current_buf = []
            tag = m.group(1)
            if tag == "user":
                current_role = "user"
                current_label = "user prompt"
            elif tag == "codex":
                current_role = "assistant"
                current_label = "assistant"
            elif tag == "thinking":
                current_role = "thinking"
                current_label = "thinking"
            elif tag == "exec":
                current_role = "tool_call"
                current_label = "exec"
            elif tag == "tokens used":
                current_role = "system"
                current_label = "tokens"
            elif tag == "turn diff":
                current_role = "tool_call"
                current_label = "diff"
            else:
                current_role = "system"
                current_label = tag
            i += 1
            continue
        if current_role is not None:
            current_buf.append(line)
        i += 1
    if current_role is not None and current_buf:
        seq += 1
        events.append(_codex_event(seq, current_role, current_label, "\n".join(current_buf).rstrip()))
    return events


def _codex_event(seq: int, role: str, label: str, content: str) -> Event:
    is_err = False
    if role == "tool_call" and label == "exec":
        # Try to split exec into call + result
        # Pattern: "<cmd> in <dir>\n succeeded in Xms:\n<stdout>\n" or "... exited N in Xms:\n..."
        m = re.match(r"^(.*?)\n\s*(succeeded|exited \d+) in \d+ms:\n(.*)$", content, re.DOTALL)
        if m:
            cmd, status, output = m.groups()
            is_err = status.startswith("exited")
            content = f"$ {cmd.strip()}\n--- {status} ---\n{output.rstrip()}"
    return Event(seq, role, label, content, is_error=is_err)


def _summarize_tool_call(name: str, params: dict) -> str:
    """Produce a one-line preview for a tool call's inputs."""
    if not isinstance(params, dict):
        return str(params)[:200]
    # Common paths / commands
    path = params.get("file_path") or params.get("path") or params.get("filepath")
    cmd = params.get("command") or params.get("query") or params.get("pattern")
    out = []
    if name:
        if path:
            out.append(f"{path}")
        if cmd:
            snippet = str(cmd).splitlines()[0][:150]
            out.append(f"$ {snippet}")
        for k in ("old_string", "new_string", "content"):
            if k in params:
                v = str(params[k])
                out.append(f"{k}={v[:60]}{'…' if len(v)>60 else ''}")
    if not out:
        out.append(json.dumps(params)[:200])
    return " · ".join(out)


def load(path: Path) -> tuple[str, list[Event]]:
    fmt = detect_format(path)
    if fmt == "droid":
        return fmt, parse_droid(path)
    if fmt == "claude":
        return fmt, parse_claude(path)
    return fmt, parse_codex(path)
