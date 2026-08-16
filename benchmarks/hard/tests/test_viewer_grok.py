import json

from src.viewer.parsers import parse, sniff


def test_grok_streaming_json_parser(tmp_path):
    run_dir = tmp_path / "20260528_125852_grok_grok-build_01_fp8_gemm"
    run_dir.mkdir()
    transcript = run_dir / "transcript.jsonl"
    rows = [
        {"type": "thought", "data": "think "},
        {"type": "thought", "data": "twice"},
        {"type": "text", "data": "write "},
        {"type": "text", "data": "solution"},
        {
            "type": "end",
            "stopReason": "EndTurn",
            "sessionId": "sid",
            "requestId": "rid",
        },
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert sniff(transcript) == "grok"
    session = parse(transcript)

    assert session.harness == "grok"
    assert session.model == "grok-build"
    assert session.session_id == "sid"
    assert session.final_text == "write solution"
    assert session.events[0].reasoning == "think twice"


def test_grok_streaming_json_skips_available_commands_preamble(tmp_path):
    run_dir = tmp_path / "20260813_152200_grok_grok-4.6_02_kimi_linear_decode"
    run_dir.mkdir()
    transcript = run_dir / "transcript.jsonl"
    rows = [
        {
            "type": "available_commands",
            "tools": ["run_terminal_command", "read_file", "write"],
            "commands": ["compact"],
        },
        {"type": "thought", "data": "plan"},
        {
            "type": "tool_call",
            "toolCallId": "call-1",
            "toolName": "read_file",
            "rawInput": {"target_file": "solution.py"},
        },
        {
            "type": "tool_call_update",
            "toolCallId": "call-1",
            "status": "completed",
            "rawOutput": "ok",
        },
        {"type": "text", "data": "done"},
        {"type": "end", "stopReason": "end_turn", "sessionId": "sid-46"},
    ]
    transcript.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert sniff(transcript) == "grok"
    session = parse(transcript)

    assert session.harness == "grok"
    assert session.model == "grok-4.6"
    assert session.session_id == "sid-46"
    assert session.final_text == "done"
    roles = [e.role for e in session.events]
    assert "assistant" in roles
    assert "tool" in roles
    tool_event = next(e for e in session.events if e.role == "tool")
    assert tool_event.tool_result is not None
    assert tool_event.tool_result.content == "ok"
    first_asst = next(e for e in session.events if e.role == "assistant")
    assert first_asst.tool_calls and first_asst.tool_calls[0].name == "read_file"
