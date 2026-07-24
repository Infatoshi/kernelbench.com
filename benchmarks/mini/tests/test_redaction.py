import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.redaction import redact_jsonable, redact_text


def test_redacts_sensitive_assignments_by_name():
    text = "GOG_KEYRING_PASSWORD=gogcli\nZAI_API_KEY=e2aab1c8883a403f83f137a282abbd5e.94LNqWDK1Ox9bQ2U"

    out = redact_text(text)

    assert "gogcli" not in out
    assert "e2aab1c8883a403f83f137a282abbd5e.94LNqWDK1Ox9bQ2U" not in out
    assert "GOG_KEYRING_PASSWORD=REDACTED" in out
    assert "ZAI_API_KEY=REDACTED" in out


def test_redacts_local_agent_instruction_blocks():
    out = redact_text("# AGENTS.md instructions for /tmp/x\n\n<proactive-behavior>do things")

    assert out == "[REDACTED: local agent instructions]"


def test_redacts_nested_jsonable_values():
    obj = {"message": {"content": [{"text": "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz"}]}}

    out = redact_jsonable(obj)

    assert out["message"]["content"][0]["text"] == "HF_TOKEN=REDACTED"
