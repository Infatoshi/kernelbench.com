"""Secret and local-instruction redaction for published KernelBench artifacts."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.published_submission import (  # noqa: E402
    DEFAULT_MAX_PUBLISHED_FILE_BYTES,
    atomic_write_text,
    read_bounded_bytes,
    read_bounded_text,
)

MAX_REDACTION_TOTAL_BYTES = 128 * 1024 * 1024

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
        try:
            env_text = read_bounded_text(
                env_file, max_bytes=1024 * 1024, errors="ignore"
            )
        except OSError:
            env_text = ""
        for line in env_text.splitlines():
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

    # All-digit values are config knobs (timeouts, context windows, ports),
    # never credentials; blind-replacing them corrupts numeric JSON fields
    # (GROK_DEBUG_CONTEXT_WINDOW=300000 turned every `"timeout": 300000` into
    # invalid `"timeout": REDACTED` and broke the HF trace converter).
    return sorted({v for v in values if not v.isdigit()}, key=len, reverse=True)


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


def redact_jsonl_text(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            out.append(redact_text(line))
        else:
            out.append(json.dumps(redact_jsonable(obj)))
    return "\n".join(out) + ("\n" if out else "")


def redacted_file_contents(path: Path) -> tuple[str, int]:
    encoded = read_bounded_bytes(path, max_bytes=DEFAULT_MAX_PUBLISHED_FILE_BYTES)
    raw = encoded.decode("utf-8", errors="ignore")
    if path.suffix == ".jsonl":
        return redact_jsonl_text(raw), len(encoded)
    return redact_text(raw), len(encoded)


def _regular_files(path: Path) -> list[Path]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise OSError(f"cannot inspect redaction path {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise OSError(f"redaction path must not be a symbolic link: {path}")
    if stat.S_ISREG(metadata.st_mode):
        return [path]
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"redaction path is not a regular file or directory: {path}")

    files: list[Path] = []
    for child in sorted(path.rglob("*")):
        child_metadata = child.lstat()
        if stat.S_ISLNK(child_metadata.st_mode):
            raise OSError(f"redaction tree contains a symbolic link: {child}")
        if stat.S_ISDIR(child_metadata.st_mode):
            continue
        if not stat.S_ISREG(child_metadata.st_mode):
            raise OSError(f"redaction tree contains a special file: {child}")
        files.append(child)
    return files


def main(argv: list[str] | None = None) -> int:
    paths = [Path(arg) for arg in (argv if argv is not None else sys.argv[1:])]
    files: list[Path] = []
    for path in paths:
        files.extend(_regular_files(path))

    # Validate and transform every input before changing any output.  A late
    # symlink/FIFO/oversize discovery therefore cannot leave a half-redacted
    # publication tree behind.
    staged: list[tuple[Path, str]] = []
    total = 0
    for path in files:
        contents, source_size = redacted_file_contents(path)
        total += source_size
        if total > MAX_REDACTION_TOTAL_BYTES:
            raise OSError(f"redaction inputs exceed {MAX_REDACTION_TOTAL_BYTES} bytes")
        staged.append((path, contents))
    for path, contents in staged:
        atomic_write_text(path, contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
