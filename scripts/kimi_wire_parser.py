"""Kimi Code 2.x transcript parser, shared by every bench's trace exporters.

Lives outside `benchmarks/<bench>/src/` on purpose: every byte under src is in
the graded-surface digest (`scripts/lib/graded_surface.py`), so a parser for a
new CLI format added there re-grades every published cell. Legacy kimi-cli
transcripts keep going through the bench's unchanged `src.viewer.parsers`.

Shapes handled here:

1. Kimi Code 2.x `--output-format stream-json` (the runner's transcript.jsonl):
     {"role":"meta","type":"system.version","version":"2.5.3"}
     {"role":"assistant","content":"...","tool_calls":[{"id","function":{"name","arguments"}}]}
     {"role":"tool","tool_call_id":"...","content":"..."}
     {"role":"meta","type":"turn.step.retrying",...}
     {"role":"meta","type":"session.resume_hint","session_id":"session_..."}   (completed turn)
   No reasoning, no usage, no model id.
2. Kimi Code 2.x session wire (agents/<id>/wire.jsonl, archived by the runner as
   kimi_wire.jsonl): llm.request {model, modelAlias, thinkingEffort}, turn.prompt,
   context.append_loop_event {step.begin | content.part {think|text} | tool.call |
   tool.result | step.end {usage}}, usage.record. A transcript.jsonl with a
   non-empty kimi_wire.jsonl beside it is parsed from the wire, which carries
   reasoning, per-step usage and the model id the CLI sent.

`src.viewer.events` / `src.viewer.diff_util` are imported lazily from whichever
bench's `src` is on sys.path when a parse runs.

Usage from an exporter:
    from scripts.kimi_wire_parser import parse_transcript
    session = parse_transcript(path, fallback_parse=src.viewer.parsers.parse)
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

WIRE_NAME = "kimi_wire.jsonl"


def _jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _first(path: Path) -> dict | None:
    for obj in _jsonl(path):
        return obj
    return None


def _is_wire_head(obj: dict | None) -> bool:
    return bool(obj) and obj.get("type") == "metadata" and "protocol_version" in obj


def _sibling_wire(path: Path) -> Path | None:
    wire = path.parent / WIRE_NAME
    if path.name == "transcript.jsonl" and wire.is_file() and wire.stat().st_size > 0:
        return wire
    return None


def is_kimi2(path: Path) -> bool:
    """True for Kimi Code 2.x input: a wire file, a stream-json opening with a
    `{"role":"meta"}` line, or a transcript.jsonl with a non-empty wire beside it."""
    path = Path(path)
    if not path.is_file() or path.suffix != ".jsonl":
        return False
    if _sibling_wire(path) is not None:
        return True
    head = _first(path)
    return _is_wire_head(head) or bool(head) and head.get("role") == "meta"


def parse_transcript(path: Path, fallback_parse: Callable[[Path], Any]):
    """Kimi 2.x input goes to this module; everything else to `fallback_parse`
    (the bench's `src.viewer.parsers.parse`)."""
    path = Path(path)
    return parse(path) if is_kimi2(path) else fallback_parse(path)


def parse(path: Path):
    path = Path(path)
    wire = _sibling_wire(path)
    if wire is not None:
        session = parse_wire(wire)
        for obj in _jsonl(path):
            if obj.get("type") == "session.resume_hint":
                session.session_id = obj.get("session_id")
        return session
    if _is_wire_head(_first(path)):
        return parse_wire(path)
    return parse_stream(path)


def _tool_call(files, name: str, args, call_id):
    from src.viewer.events import ToolCall

    targs = args if isinstance(args, dict) else {"raw": args}
    diff = None
    file_path = targs.get("path") or targs.get("file_path")
    if name in {"WriteFile", "CreateFile", "Write"} and file_path:
        diff = files.diff_for_write(file_path, targs.get("content", "") or "")
    elif name in {"EditFile", "Edit"} and file_path:
        diff = files.diff_for_edit(
            file_path,
            targs.get("old_string", "") or "",
            targs.get("new_string", "") or "",
        )
    return ToolCall(name=name, args=targs, call_id=call_id, diff=diff or None, file_path=file_path)


def _usage(u: dict):
    from src.viewer.events import TokenUsage

    return TokenUsage(
        input_tokens=u.get("inputOther", 0) or 0,
        output_tokens=u.get("output", 0) or 0,
        cache_read_tokens=u.get("inputCacheRead", 0) or 0,
        cache_write_tokens=u.get("inputCacheCreation", 0) or 0,
    )


def parse_wire(path: Path):
    from src.viewer.diff_util import FileTracker
    from src.viewer.events import Event, Session, TokenUsage, ToolResult

    path = Path(path)
    events: list = []
    total = TokenUsage()
    files = FileTracker()
    model = None
    cwd = None
    step: dict | None = None
    # A wire file holds its own agent's records (subagents have their own wires).
    if path.name == "wire.jsonl":
        agent = path.parent.name
    elif path.name.startswith("kimi_wire.") and path.name != WIRE_NAME:
        agent = path.name[len("kimi_wire."):-len(".jsonl")]
    else:
        agent = "main"

    def new_step() -> dict:
        return {"text": [], "reasoning": [], "calls": [], "results": [], "usage": None}

    def flush() -> None:
        nonlocal step
        if step is None:
            return
        if step["text"] or step["reasoning"] or step["calls"]:
            events.append(Event(
                role="assistant",
                text="".join(step["text"]) or None,
                reasoning="".join(step["reasoning"]) or None,
                tool_calls=step["calls"],
                usage=step["usage"],
                model=model,
            ))
        events.extend(step["results"])
        step = None

    for obj in _jsonl(path):
        t = obj.get("type")
        if t == "llm.request":
            model = obj.get("model") or model
            continue
        if t == "usage.record":
            if obj.get("agentId", agent) == agent:
                u = _usage(obj.get("usage") or {})
                total.input_tokens += u.input_tokens
                total.output_tokens += u.output_tokens
                total.cache_read_tokens += u.cache_read_tokens
                total.cache_write_tokens += u.cache_write_tokens
            continue
        if t == "turn.prompt":
            flush()
            text = "\n".join(p.get("text", "") for p in obj.get("input") or []
                             if isinstance(p, dict) and p.get("type") == "text")
            events.append(Event(role="user", text=text, raw=obj))
            continue
        if t in ("full_compaction.complete", "context.apply_compaction"):
            flush()
            events.append(Event(role="compaction", text=t, subtype=t))
            continue
        if t != "context.append_loop_event":
            continue
        ev = obj.get("event") or {}
        et = ev.get("type")
        if et == "step.begin":
            flush()
            step = new_step()
            continue
        if step is None:
            step = new_step()
        if et == "content.part":
            part = ev.get("part") or {}
            if part.get("type") == "think":
                step["reasoning"].append(part.get("think") or "")
            elif part.get("type") == "text":
                step["text"].append(part.get("text") or "")
        elif et == "tool.call":
            cwd = cwd or (ev.get("display") or {}).get("cwd")
            step["calls"].append(_tool_call(files, ev.get("name") or "?", ev.get("args") or {},
                                            ev.get("toolCallId")))
        elif et == "tool.result":
            res = ev.get("result") or {}
            out = res.get("output")
            if not isinstance(out, str):
                out = json.dumps(out, default=str) if out is not None else json.dumps(res, default=str)
            step["results"].append(Event(
                role="tool",
                tool_result=ToolResult(content=out, call_id=ev.get("toolCallId"),
                                       is_error=bool(res.get("isError") or res.get("error"))),
            ))
        elif et == "step.end":
            if isinstance(ev.get("usage"), dict):
                step["usage"] = _usage(ev["usage"])
            flush()
    flush()

    return Session(
        harness="kimi",
        model=model or "kimi",
        session_id=None,
        cwd=cwd,
        events=events,
        total_usage=total,
    )


def parse_stream(path: Path):
    from src.viewer.diff_util import FileTracker
    from src.viewer.events import Event, Session, TokenUsage, ToolResult

    events: list = []
    total = TokenUsage()
    files = FileTracker()
    session_id = None

    for obj in _jsonl(Path(path)):
        role = obj.get("role")
        content = obj.get("content") or []

        if role == "meta":
            if obj.get("type") == "session.resume_hint":
                session_id = obj.get("session_id")
            elif obj.get("type") == "turn.step.retrying":
                events.append(Event(role="system", subtype="retry", raw=obj,
                                    text=f"retry {obj.get('failed_attempt')}: {obj.get('error_message')}"))
            continue
        if obj.get("type") == "error":
            events.append(Event(role="error", text=str(obj.get("message")), raw=obj))
            continue

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
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
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
                tool_calls.append(_tool_call(files, fn.get("name", "?"), args, tc.get("id")))

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
                tr_text = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
            elif isinstance(content, str):
                tr_text = content
            events.append(Event(
                role="tool",
                tool_result=ToolResult(content=tr_text, call_id=obj.get("tool_call_id")),
                raw=obj,
            ))
            continue

    return Session(
        harness="kimi",
        model="kimi",
        session_id=session_id,
        cwd=None,
        events=events,
        total_usage=total,
    )
