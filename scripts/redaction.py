"""Secret and local-instruction redaction for published KernelBench artifacts.

Scanners return categories/counts only. They never include matched values.
This is a publication tripwire, not a proof that arbitrary prose is public.
"""
from __future__ import annotations

from collections import Counter
import ipaddress
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

# Suffix matching deliberately excludes token counts, context windows and key sizes.
SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:api[_-]?key|auth[_-]?token|oauth[_-]?token|secret(?:[_-]?key)?|password|"
    r"passwd|private[_-]?key|access[_-]?key(?:[_-]?id)?|refresh[_-]?token|"
    r"access[_-]?token|id[_-]?token|client[_-]?secret|authorization|cookie|"
    r"credential|credentials|token)$", re.I,
)
_NAME = r"[A-Za-z_][A-Za-z0-9_.-]*"
SENSITIVE_NAME_RE = re.compile(
    rf"(?<![\w.-])(?P<name>{_NAME})[\"']?\s*[:=]\s*"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;}&\]\"']*)",
    re.I,
)
TOKEN_PATTERNS = [re.compile(p, re.I if p.startswith('(?:Bearer') else 0) for p in [
    r"sk-ant-(?:oat01|api)[A-Za-z0-9_\-]+",
    r"sk-proj-[A-Za-z0-9_\-]+",
    r"AIzaSy[A-Za-z0-9_\-]{20,}",
    r"sk-[a-z]{2,}-[A-Za-z0-9_\-]{16,}",
    r"sk-[A-Za-z0-9]{24,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{30,}",
    r"hf_[A-Za-z0-9]{20,}",
    r"LLM_[A-Za-z0-9_\-]{30,}",
    r"xai-[A-Za-z0-9_\-]{20,}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"(?:AKIA|ASIA)[A-Z0-9]{16}",
    r"(?:Bearer|Basic)\s+[A-Za-z0-9+/=._\-]{8,}",
    r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
]]
LOCAL_INSTRUCTION_MARKERS = (
    "# AGENTS.md instructions", "# CLAUDE.md instructions", "<proactive-behavior>",
    "This file and ~/.claude/refs/", "~/.codex/AGENTS.md", "~/.claude/CLAUDE.md",
)
PEM_RE = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*?(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|\Z)", re.S)
URL_AUTH_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/@]+(?::[^\s/@]*)?@", re.I)
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
HOME_RE = re.compile(r"(?<![\w])(?:/(?:Users|home)/[^/\s\"'<>:]+|/root)(?=/|\b)")
IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6_RE = re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F:.]*(?![\w:])")
# A file dump containing these paths can expose personal configuration beyond tokens.
PRIVATE_DUMP_RE = re.compile(
    r"(?:^|[\s\"'])(?:cat|head|tail|less|more|Get-Content)\s+[^\n]*"
    r"(?:\.env(?:_vars|\b)|\.credentials\.json|auth\.json|\.ssh/|\.aws/credentials)", re.I,
)
REDACTED = "[REDACTED]"


def _is_sensitive(name: str) -> bool:
    # camelCase credential keys occur in CLI auth JSON.
    expanded = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return bool(SENSITIVE_ENV_NAME_RE.search(expanded))


def _candidate_secret_values() -> list[str]:
    """Read credential values in memory only; never execute shell configuration."""
    values = [v for k, v in os.environ.items() if _is_sensitive(k)]
    home = Path.home()
    paths = [home / p for p in (
        '.env_vars', '.env', '.kbm_env', '.codex/auth.json', '.claude/.credentials.json',
        '.config/opencode/auth.json', '.local/share/opencode/auth.json',
        '.grok/config.toml', '.aws/credentials', '.cache/huggingface/token',
    )]

    def collect(obj: Any, sensitive: bool = False) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                collect(v, sensitive or _is_sensitive(k) or k.lower() in {'key', 'access', 'refresh'})
        elif isinstance(obj, list):
            for v in obj:
                collect(v, sensitive)
        elif sensitive and isinstance(obj, str):
            values.append(obj)

    for path in paths:
        try:
            raw = path.read_text()
        except (OSError, UnicodeError):
            continue
        if path.name == 'token':
            values.append(raw.strip())
        try:
            collect(json.loads(raw))
        except (ValueError, TypeError):
            for line in raw.splitlines():
                try:
                    parts = shlex.split(line, comments=True)
                except ValueError:
                    continue
                line = ' '.join(parts).removeprefix('export ')
                if '=' in line:
                    name, val = line.split('=', 1)
                    if _is_sensitive(name.strip()):
                        values.append(val.strip())
    # Numeric settings must never replace JSON numbers or benchmark literals.
    usable = {v for v in values if len(v) >= 6 and not v.isdigit() and not v.startswith(('$', '[REDACTED'))}
    return sorted(usable | {quote(v, safe='') for v in usable}, key=len, reverse=True)


SECRET_VALUES = _candidate_secret_values()


def _already_redacted(value: str) -> bool:
    return not value or value.strip("\"'").startswith(('[REDACTED', 'REDACTED'))


def _valid_ip(match: re.Match) -> bool:
    try:
        ipaddress.ip_address(match.group())
        return True
    except ValueError:
        return False


def scan_text(text: str) -> dict[str, int]:
    """Return suspicious category counts without exposing matched data."""
    counts: Counter[str] = Counter()
    if not isinstance(text, str):
        return {}
    # Scan decoded JSON strings: the escape in "\\n@triton.jit" is not an email.
    if text.lstrip().startswith(('{', '[')):
        try:
            obj = json.loads(text)
        except (ValueError, RecursionError):
            pass
        else:
            return scan_jsonable(obj)
    if any(marker in text for marker in LOCAL_INSTRUCTION_MARKERS):
        counts['local_instructions'] += 1
    if PRIVATE_DUMP_RE.search(text):
        counts['private_file_dump'] += 1
    counts['known_secret'] += sum(text.count(v) for v in SECRET_VALUES)
    counts['credential_assignment'] += sum(
        _is_sensitive(m['name']) and not _already_redacted(m['value'])
        for m in SENSITIVE_NAME_RE.finditer(text)
    )
    for pattern in TOKEN_PATTERNS:
        counts['credential_token'] += len(pattern.findall(text))
    for name, pattern in [('private_key', PEM_RE), ('url_credentials', URL_AUTH_RE),
                          ('email', EMAIL_RE), ('home_path', HOME_RE)]:
        counts[name] += len(pattern.findall(text))
    counts['ip_address'] += sum(_valid_ip(m) for p in (IPV4_RE, IPV6_RE) for m in p.finditer(text))
    return {k: v for k, v in counts.items() if v}


def redact_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Decode before filtering: converters parse JSON tool arguments after redaction.
    # Structured JSON embedded in text must remain valid JSON.
    if text.lstrip().startswith(('{', '[')):
        try:
            obj = json.loads(text)
        except (ValueError, RecursionError):
            pass
        else:
            return json.dumps(redact_jsonable(obj), ensure_ascii=False)
    if any(marker in text for marker in LOCAL_INSTRUCTION_MARKERS):
        return "[REDACTED: local agent instructions]"
    if PRIVATE_DUMP_RE.search(text):
        return "[REDACTED: private credential/config file access]"
    text = PEM_RE.sub('[REDACTED: private key]', text)
    text = URL_AUTH_RE.sub('[REDACTED: URL credentials]/', text)
    # Remove the assignment syntax as well, so the mandatory media rg gate is clean.
    text = SENSITIVE_NAME_RE.sub(
        lambda m: '[REDACTED credential assignment]' if _is_sensitive(m['name']) else m.group(), text,
    )
    for value in SECRET_VALUES:
        text = text.replace(value, REDACTED)
    for pattern in TOKEN_PATTERNS:
        text = pattern.sub(REDACTED, text)
    text = EMAIL_RE.sub('[REDACTED: email]', text)
    text = HOME_RE.sub('/[REDACTED-home]', text)
    for pattern in (IPV4_RE, IPV6_RE):
        text = pattern.sub(lambda m: '[REDACTED: IP]' if _valid_ip(m) else m.group(), text)
    return text


def _empty_sensitive(value: Any) -> Any:
    """Keep JSON structure and primitive types, removing credential-bearing leaves."""
    if isinstance(value, dict):
        return {redact_text(k): _empty_sensitive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_empty_sensitive(v) for v in value]
    if isinstance(value, str):
        return REDACTED
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return type(value)(0)
    return None


def redact_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {redact_text(key): _empty_sensitive(item) if _is_sensitive(key) else redact_jsonable(item)
                for key, item in value.items()}
    return value


def scan_jsonable(value: Any) -> dict[str, int]:
    """Non-mutating recursive scan; output contains category names and counts only."""
    counts: Counter[str] = Counter()
    if isinstance(value, str):
        counts.update(scan_text(value))
    elif isinstance(value, list):
        for item in value:
            counts.update(scan_jsonable(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            counts.update(scan_text(key))
            if _is_sensitive(key) and item != _empty_sensitive(item):
                counts['credential_field'] += 1
            counts.update(scan_jsonable(item))
    return dict(counts)


def scan_file(path: Path) -> dict[str, int]:
    """Scan the exact exported file, without modifying it or printing values."""
    counts: Counter[str] = Counter()
    with path.open(errors='strict') as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except ValueError:
                counts.update(scan_text(line))
            else:
                counts.update(scan_jsonable(value))
    return dict(counts)


def redact_jsonl_file(path: Path) -> None:
    out: list[str] = []
    for line in path.read_text(errors='strict').splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            out.append(redact_text(line))
        else:
            out.append(json.dumps(redact_jsonable(obj)))
    path.write_text('\n'.join(out) + ('\n' if out else ''))


def redact_file(path: Path) -> None:
    if path.suffix == '.jsonl':
        redact_jsonl_file(path)
    else:
        path.write_text(redact_text(path.read_text(errors='strict')))


def main(argv: list[str] | None = None) -> int:
    paths = [Path(arg) for arg in (argv if argv is not None else sys.argv[1:])]
    for path in paths:
        if path.is_dir():
            for child in path.rglob('*'):
                if child.is_file():
                    redact_file(child)
        elif path.is_file():
            redact_file(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
