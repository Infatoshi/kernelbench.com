"""Parse Muse Code (Meta) headless output (`muse exec --json`).

Each stdout line is a session record: {schema_version, id, stream:{kind:
"session", id}, sequence, recorded_at (us), record_type, payload_type,
payload:{kind, ...}}. Payload types observed from real runs (Muse Code 1.0.2):

  run.model.configured        payload.model_id, provider_id
  turn.input.user             payload.prompt
  task.lifecycle.proposed     payload.event.task_kind: "model.meta.response" |
                              "tool.<name>" (bash, write_file, ...)
  task.lifecycle.side_effect_intent  payload.event.operation (+ details)
  task.lifecycle.output       payload.event.chunk: tool output; bash chunks
                              are JSON with command / exit_code / output
  tool.result                 payload.call_id, text, edit_facts
  run.output.delta            payload.text (assistant text delta)
  run.terminal.<completed|failed|cancelled>  payload.text (final answer)

stdout carries no token usage. The harness runs `muse export` into
muse_export.json next to the transcript; its provider usage records
(usage_family == "provider", quantity{input_tokens, output_tokens,
reasoning_tokens, cached_tokens}) are summed here when the file exists.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.viewer.events import Event, Session, TokenUsage, ToolCall, ToolResult

_INTENT_SKIP = {"kind", "task_id", "idempotency_key", "policy_decision",
                "cancellation_handle", "parent_task_id"}


def _export_usage(path: Path) -> TokenUsage:
    total = TokenUsage()
    exp = path.parent / "muse_export.json"
    if not exp.exists():
        return total
    try:
        doc = json.loads(exp.read_text())
    except (OSError, json.JSONDecodeError):
        return total

    def walk(node):
        if isinstance(node, dict):
            q = node.get("quantity")
            if node.get("usage_family") == "provider" and isinstance(q, dict) and "input_tokens" in q:
                total.input_tokens += int(q.get("input_tokens") or 0)
                total.output_tokens += int(q.get("output_tokens") or 0)
                total.cache_read_tokens += int(q.get("cached_tokens") or 0)
                total.thinking_tokens += int(q.get("reasoning_tokens") or 0)
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    return total


def parse(path: Path) -> Session:
    events: list[Event] = []
    model = None
    session_id = None
    cwd = None
    final_text = None
    first_us = None
    last_us = None
    task_kind: dict[str, str] = {}
    task_args: dict[str, dict] = {}
    text_buf: list[str] = []

    def flush_text():
        if text_buf:
            events.append(Event(role="assistant", text="".join(text_buf),
                                session_id=session_id))
            text_buf.clear()

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            pt = obj.get("payload_type") or ""
            p = obj.get("payload") or {}
            ts = obj.get("recorded_at")
            if isinstance(ts, (int, float)):
                first_us = ts if first_us is None else first_us
                last_us = ts
            stream = obj.get("stream") or {}
            if stream.get("kind") == "session" and stream.get("id"):
                session_id = session_id or stream.get("id")

            if pt == "run.model.configured":
                model = p.get("model_id") or model
                events.append(Event(role="system", text=f"session start  model={model}",
                                    subtype="init", raw=obj))
            elif pt == "turn.input.user":
                events.append(Event(role="user", text=p.get("prompt") or "",
                                    session_id=session_id, raw=obj))
            elif pt == "run.output.delta":
                text_buf.append(p.get("text") or "")
            elif pt.startswith("task.lifecycle."):
                ev = p.get("event") or {}
                tid = ev.get("task_id") or p.get("task_id") or ""
                kind = ev.get("kind")
                if kind == "proposed":
                    tk = ev.get("task_kind") or ""
                    task_kind[tid] = tk
                elif kind == "side_effect_intent":
                    args = {k: v for k, v in ev.items() if k not in _INTENT_SKIP}
                    task_args[tid] = args
                    tk = task_kind.get(tid, "")
                    if tk.startswith("tool."):
                        flush_text()
                        events.append(Event(
                            role="assistant",
                            tool_calls=[ToolCall(name=tk[5:], args=args, call_id=tid)],
                            session_id=session_id, raw=obj,
                        ))
                elif kind == "output":
                    tk = task_kind.get(tid, "")
                    if tk.startswith("tool."):
                        chunk = ev.get("chunk") or ""
                        is_err = False
                        try:
                            cj = json.loads(chunk)
                            if isinstance(cj, dict) and cj.get("exit_code") not in (None, 0):
                                is_err = True
                        except (ValueError, TypeError):
                            pass
                        events.append(Event(
                            role="tool",
                            tool_result=ToolResult(content=chunk, call_id=tid, is_error=is_err),
                            session_id=session_id, raw=obj,
                        ))
            elif pt.startswith("run.terminal."):
                flush_text()
                final_text = p.get("text") or final_text
                events.append(Event(role="system",
                                    text=f"result: {p.get('terminal', pt.rsplit('.', 1)[-1])}",
                                    subtype="result", raw=obj))

    flush_text()
    duration_ms = None
    if first_us is not None and last_us is not None and last_us >= first_us:
        duration_ms = int((last_us - first_us) / 1000)

    return Session(
        harness="muse",
        model=model,
        session_id=session_id,
        cwd=cwd,
        events=events,
        final_text=final_text,
        total_usage=_export_usage(Path(path)),
        duration_ms=duration_ms,
    )
