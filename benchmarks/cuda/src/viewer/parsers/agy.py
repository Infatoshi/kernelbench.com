"""Parse Antigravity CLI stream-json output (`agy --print --output-format stream-json`).

Top-level events observed from real runs (agy 1.1.24):
  {event: "init", conversation_id, init: {model, cwd, tools, permission_mode}}
  {event: "step_update", step_update: {conversation_id, step_index, state,
      step_type: "user_input" | "agent_response" | "tool",
      text_delta?, tool_name?, tool_info: {name, parameters, error?},
      duration_seconds?, usage?: {input_tokens, output_tokens, thinking_tokens,
                                  cache_read_tokens, total_tokens}}}
  {event: "result", result: {conversation_id, status, response,
      duration_seconds, num_turns, usage: {...}}}

A tool step is emitted twice: state ACTIVE (call) then DONE or ERROR
(result). Tool output is not streamed; the DONE row carries only the
parameters, ERROR rows carry tool_info.error. Events have no timestamps.
Usage comes from the terminal result event (output_tokens already includes
thinking_tokens: input + output == total).
"""
from __future__ import annotations

import json
from pathlib import Path

from src.viewer.events import Event, Session, TokenUsage, ToolCall, ToolResult


def parse(path: Path) -> Session:
    events: list[Event] = []
    model = None
    session_id = None
    cwd = None
    total = TokenUsage()
    final_text = None
    duration_ms = None

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            ev = obj.get("event")

            if ev == "init":
                init = obj.get("init") or {}
                model = init.get("model") or model
                session_id = obj.get("conversation_id") or session_id
                cwd = init.get("cwd") or cwd
                events.append(Event(
                    role="system",
                    text=f"session start  model={model}",
                    subtype="init", raw=obj,
                ))
                continue

            if ev == "step_update":
                su = obj.get("step_update") or {}
                st = su.get("step_type")
                state = su.get("state")
                if st == "agent_response":
                    text = su.get("text_delta") or ""
                    if state == "DONE" and text:
                        events.append(Event(
                            role="assistant", text=text,
                            session_id=session_id, raw=obj,
                        ))
                    continue
                if st == "tool":
                    info = su.get("tool_info") or {}
                    call_id = f"{su.get('conversation_id','')}:{su.get('step_index')}"
                    if state == "ACTIVE":
                        events.append(Event(
                            role="assistant",
                            tool_calls=[ToolCall(
                                name=su.get("tool_name") or info.get("name") or "?",
                                args=info.get("parameters") or {},
                                call_id=call_id,
                            )],
                            session_id=session_id, raw=obj,
                        ))
                    elif state in ("DONE", "ERROR"):
                        err = info.get("error")
                        content = json.dumps(err) if err else ""
                        events.append(Event(
                            role="tool",
                            tool_result=ToolResult(
                                content=content, call_id=call_id,
                                is_error=state == "ERROR",
                            ),
                            session_id=session_id, raw=obj,
                        ))
                    continue
                continue

            if ev == "result":
                res = obj.get("result") or {}
                usage = res.get("usage") or {}
                total = TokenUsage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=usage.get("cache_read_tokens", 0),
                    thinking_tokens=usage.get("thinking_tokens", 0),
                )
                final_text = res.get("response") or final_text
                dur = res.get("duration_seconds")
                duration_ms = int(dur * 1000) if isinstance(dur, (int, float)) else None
                events.append(Event(
                    role="system",
                    text=f"result: {res.get('status','?')}",
                    subtype="result", raw=obj,
                ))
                continue

    return Session(
        harness="agy",
        model=model,
        session_id=session_id,
        cwd=cwd,
        events=events,
        final_text=final_text,
        total_usage=total,
        duration_ms=duration_ms,
    )
