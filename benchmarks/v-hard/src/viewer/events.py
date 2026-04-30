"""Common Event model used by all harness parsers.

Every parser emits a list[Event] in chronological order. The HTML generator
renders these without knowing which harness produced them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Role = Literal["user", "assistant", "tool", "system", "compaction", "error"]


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str | None = None
    # If this tool call is a file write/edit, a unified-diff string showing
    # the change. None for non-file-write tools.
    diff: str | None = None
    # Path of the file being written/edited, if applicable.
    file_path: str | None = None


@dataclass
class ToolResult:
    content: str
    call_id: str | None = None
    is_error: bool = False


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass
class Event:
    role: Role
    text: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: ToolResult | None = None
    usage: TokenUsage | None = None
    model: str | None = None
    timestamp: datetime | None = None
    session_id: str | None = None
    is_sidechain: bool = False
    parent_uuid: str | None = None
    subtype: str | None = None
    raw: dict[str, Any] | None = None  # for debugging


@dataclass
class Session:
    harness: str
    model: str | None
    session_id: str | None
    cwd: str | None
    events: list[Event]
    final_text: str | None = None  # what the agent considered its final answer
    total_usage: TokenUsage | None = None
    duration_ms: int | None = None

    @property
    def turn_count(self) -> int:
        return sum(1 for e in self.events if e.role == "assistant")

    @property
    def tool_call_count(self) -> int:
        return sum(len(e.tool_calls) for e in self.events)
