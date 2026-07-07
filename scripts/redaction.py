"""Secret and local-instruction redaction for published KernelBench artifacts."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SENSITIVE_NAME_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:"
    r"API_KEY|TOKEN|AUTH_TOKEN|OAUTH_TOKEN|SECRET|PASSWORD|"
    r"KEYRING_PASSWORD|PRIVATE_KEY|ACCESS_KEY|REFRESH_TOKEN"
    r"))=([^\n\r\"'\s]+)"
)

TOKEN_PATTERNS = [
    re.compile(p)
    for p in [
        r"sk-ant-oat01-[A-Za-z0-9_\-]+",
        r"sk-ant-api[A-Za-z0-9_\-]+",
        r"sk-proj-[A-Za-z0-9_\-]+",
        r"AIzaSy[A-Za-z0-9_\-]{20,}",
        r"sk-[a-z]{2,}-[A-Za-z0-9_\-]{16,}",
        r"sk-[A-Za-z0-9]{24,}",
        r"ghp_[A-Za-z0-9]{20,}",
        r"gho_[A-Za-z0-9]{20,}",
        r"github_pat_[A-Za-z0-9_]{30,}",
        r"hf_[A-Za-z0-9]{20,}",
        r"Bearer\s+[A-Za-z0-9._\-]{20,}",
    ]
]

LOCAL_INSTRUCTION_MARKERS = (
    "# AGENTS.md instructions",
    "# CLAUDE.md instructions",
    "<proactive-behavior>",
    "This file and ~/.claude/refs/",
    "~/.codex/AGENTS.md",
    "~/.claude/CLAUDE.md",
)

SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:API_KEY|TOKEN|AUTH_TOKEN|OAUTH_TOKEN|SECRET|PASSWORD|"
    r"KEYRING_PASSWORD|PRIVATE_KEY|ACCESS_KEY|REFRESH_TOKEN)$"
)


def _candidate_secret_values() -> list[str]:
    values: list[str] = []
    for name, value in os.environ.items():
        if SENSITIVE_ENV_NAME_RE.search(name) and len(value) >= 6:
            values.append(value)

    env_file = Path(os.path.expanduser("~/.env_vars"))
    if env_file.exists():
        for line in env_file.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            if "=" not in line:
                continue
            _, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if len(value) >= 6:
                values.append(value)

    return sorted(set(values), key=len, reverse=True)


SECRET_VALUES = _candidate_secret_values()


def redact_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    if any(marker in text for marker in LOCAL_INSTRUCTION_MARKERS):
        return "[REDACTED: local agent instructions]"

    for value in SECRET_VALUES:
        text = text.replace(value, "REDACTED")

    text = SENSITIVE_NAME_RE.sub(lambda m: f"{m.group(1)}=REDACTED", text)
    for pattern in TOKEN_PATTERNS:
        text = pattern.sub("REDACTED", text)
    return text


def redact_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_jsonable(item) for key, item in value.items()}
    return value


def redact_jsonl_file(path: Path) -> None:
    out: list[str] = []
    for line in path.read_text(errors="ignore").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            out.append(redact_text(line))
        else:
            out.append(json.dumps(redact_jsonable(obj)))
    path.write_text("\n".join(out) + ("\n" if out else ""))


def redact_file(path: Path) -> None:
    if path.suffix == ".jsonl":
        redact_jsonl_file(path)
    else:
        path.write_text(redact_text(path.read_text(errors="ignore")))


def main(argv: list[str] | None = None) -> int:
    paths = [Path(arg) for arg in (argv if argv is not None else sys.argv[1:])]
    for path in paths:
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    redact_file(child)
        elif path.is_file():
            redact_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
