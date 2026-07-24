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
