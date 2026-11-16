"""Kimi Code 2.x transcripts parse through scripts/kimi_wire_parser.py, outside
the graded src trees; legacy kimi-cli transcripts still go through the bench's
src.viewer.parsers."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def kwp():
    """Import the parser with hard's src on sys.path; restore afterwards."""
    saved_path = list(sys.path)
    saved = {k: v for k, v in sys.modules.items()
             if k == "src" or k.startswith("src.") or k == "scripts" or k.startswith("scripts.")}
    for k in saved:
        del sys.modules[k]
    sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks" / "hard")]
    import scripts.kimi_wire_parser as mod
    from src.viewer.parsers import parse as src_parse
    yield mod, src_parse
    sys.path[:] = saved_path
    for k in [k for k in sys.modules
              if k == "src" or k.startswith("src.") or k == "scripts" or k.startswith("scripts.")]:
        del sys.modules[k]
    sys.modules.update(saved)


def _write(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


STREAM = [
    {"role": "meta", "type": "system.version", "version": "2.5.3"},
    {"role": "assistant", "content": "looking", "tool_calls": [
        {"id": "c1", "function": {"name": "Shell", "arguments": json.dumps({"command": "nvidia-smi"})}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "GPU 0"},
    {"role": "meta", "type": "turn.step.retrying", "failed_attempt": 1, "error_message": "529"},
    {"role": "assistant", "content": "done"},
    {"role": "meta", "type": "session.resume_hint", "session_id": "session_abc"},
]

WIRE = [
    {"type": "metadata", "protocol_version": "1.5", "created_at": 1},
    {"type": "llm.request", "model": "kimi-k3.1-pro", "modelAlias": "kimi-code/k3.1-pro",
     "thinkingEffort": "max"},
    {"type": "turn.prompt", "input": [{"type": "text", "text": "write a kernel"}]},
    {"type": "context.append_loop_event", "event": {"type": "step.begin"}},
    {"type": "context.append_loop_event", "event": {"type": "content.part",
                                                     "part": {"type": "think", "think": "plan "}}},
    {"type": "context.append_loop_event", "event": {"type": "content.part",
                                                     "part": {"type": "think", "think": "it"}}},
    {"type": "context.append_loop_event", "event": {"type": "content.part",
                                                     "part": {"type": "text", "text": "writing"}}},
    {"type": "context.append_loop_event", "event": {
        "type": "tool.call", "toolCallId": "t1", "name": "WriteFile",
        "args": {"path": "solution.py", "content": "x = 1\n"}, "display": {"cwd": "/workspace"}}},
    {"type": "context.append_loop_event", "event": {
        "type": "tool.result", "toolCallId": "t1", "result": {"output": "ok"}}},
    {"type": "context.append_loop_event", "event": {
        "type": "step.end", "usage": {"inputOther": 10, "output": 5, "inputCacheRead": 100}}},
    {"type": "usage.record", "agentId": "main",
     "usage": {"inputOther": 10, "output": 5, "inputCacheRead": 100, "inputCacheCreation": 2}},
    {"type": "usage.record", "agentId": "sub1", "usage": {"inputOther": 999, "output": 999}},
]


def test_stream_json(kwp, tmp_path):
    mod, src_parse = kwp
    p = _write(tmp_path / "transcript.jsonl", STREAM)
    assert mod.is_kimi2(p)
    s = mod.parse_transcript(p, src_parse)
    assert s.harness == "kimi" and s.session_id == "session_abc"
    roles = [e.role for e in s.events]
    assert roles == ["assistant", "tool", "system", "assistant"]
    call = s.events[0].tool_calls[0]
    assert call.name == "Shell" and call.call_id == "c1" and call.args == {"command": "nvidia-smi"}
    assert s.events[1].tool_result.call_id == "c1" and s.events[1].tool_result.content == "GPU 0"
    assert s.events[3].text == "done"


def test_wire_beside_transcript(kwp, tmp_path):
    mod, src_parse = kwp
    p = _write(tmp_path / "transcript.jsonl", STREAM)
    _write(tmp_path / "kimi_wire.jsonl", WIRE)
    s = mod.parse_transcript(p, src_parse)
    assert s.model == "kimi-k3.1-pro"
    assert s.session_id == "session_abc"  # from the stream's resume hint
    assert s.cwd == "/workspace"
    user, asst, tool = s.events
    assert user.role == "user" and user.text == "write a kernel"
    assert asst.reasoning == "plan it" and asst.text == "writing" and asst.model == "kimi-k3.1-pro"
    assert asst.usage.input_tokens == 10 and asst.usage.output_tokens == 5
    call = asst.tool_calls[0]
    assert call.call_id == "t1" and call.file_path == "solution.py" and call.diff
    assert tool.tool_result.call_id == "t1" and tool.tool_result.content == "ok"
    # Only the main agent's usage records count.
    t = s.total_usage
    assert (t.input_tokens, t.output_tokens, t.cache_read_tokens, t.cache_write_tokens) == (10, 5, 100, 2)


def test_wire_file_directly(kwp, tmp_path):
    mod, src_parse = kwp
    p = _write(tmp_path / "kimi_wire.jsonl", WIRE)
    assert mod.is_kimi2(p)
    s = mod.parse_transcript(p, src_parse)
    assert s.model == "kimi-k3.1-pro" and s.session_id is None


def test_empty_wire_falls_back_to_stream(kwp, tmp_path):
    mod, src_parse = kwp
    p = _write(tmp_path / "transcript.jsonl", STREAM)
    (tmp_path / "kimi_wire.jsonl").write_text("")
    s = mod.parse_transcript(p, src_parse)
    assert s.model == "kimi" and s.session_id == "session_abc"


def test_legacy_kimi_cli_goes_to_src_parser(kwp, tmp_path):
    mod, src_parse = kwp
    rows = [
        {"role": "assistant", "content": [{"type": "think", "think": "hmm"},
                                          {"type": "text", "text": "hi"}],
         "tool_calls": [{"id": "a", "function": {"name": "Shell", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "out"}]},
    ]
    p = _write(tmp_path / "transcript.jsonl", rows)
    assert not mod.is_kimi2(p)
    called = []

    def fallback(path):
        called.append(path)
        return src_parse(path)

    s = mod.parse_transcript(p, fallback)
    assert called == [p]
    assert s.harness == "kimi"
    assert s.events[0].reasoning == "hmm" and s.events[1].tool_result.content == "out"
