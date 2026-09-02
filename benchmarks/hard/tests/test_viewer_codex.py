from pathlib import Path

from src.viewer.parsers import parse, sniff


def test_codex_plain_stderr_parser(tmp_path: Path) -> None:
    path = tmp_path / "stderr.log"
    path.write_text(
        "OpenAI Codex v0.144.6\n"
        "workdir: /workspace/problem\n"
        "model: gpt-test\n"
        "session id: session-1\n"
        "user\n"
        "build the kernel\n"
        "codex\n"
        "I will inspect it.\n"
        "exec\n"
        "/bin/bash -lc 'pwd'\n"
        " succeeded in 1ms:\n"
        "/workspace/problem\n"
        "codex\n"
        "Done.\n"
    )

    assert sniff(path) == "codex"
    session = parse(path)
    assert session.model == "gpt-test"
    assert session.session_id == "session-1"
    assert session.cwd == "/workspace/problem"
    assert [event.role for event in session.events] == [
        "user", "assistant", "assistant", "tool", "assistant"
    ]
    assert session.events[2].tool_calls[0].name == "exec_command"
    assert session.events[3].tool_result is not None
    assert "/workspace/problem" in session.events[3].tool_result.content
    assert session.final_text == "Done."
