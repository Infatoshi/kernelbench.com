"""Classify KernelBench-Mega run outcomes.

Provider failure detection must not scan arbitrary transcript text. Agents often
read AGENTS.md, run_hard.sh, old result JSON, or problem docs that contain words
like quota, rate_limit, or insufficient_credits. Those are quoted artifacts, not
API failures.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROVIDER_CREDIT_RE = re.compile(
    r"insufficient[_ -]?credits|out[_ -]?of[_ -]?credits|"
    r"\bcredits?\s+(?:balance|remaining)\b|\bpayment required\b|"
    r"\baccount\s+overage\b|add more using"
)
PROVIDER_RATE_RE = re.compile(
    r"rate[_ -]?limit|\b429\b|quota|resource_exhausted|session limit"
)


def classify_run(
    *,
    usage: dict[str, Any],
    correct: bool,
    template_mutated: bool,
    has_solution: bool,
    session_complete: bool,
    harness_exit: int | None,
    check_exit: int | None,
    bench_exit: int | None,
    log_file: str | Path | None,
    stderr_file: str | Path | None,
    minimum_useful_output_tokens: int,
) -> dict[str, Any]:
    output_tokens = usage.get("output_tokens")
    if not isinstance(output_tokens, (int, float)):
        output_tokens = None

    provider_failure_window = not has_solution
    provider_text = provider_signal_text(log_file, stderr_file).lower()
    insufficient_credits = bool(PROVIDER_CREDIT_RE.search(provider_text))
    rate_limited = bool(PROVIDER_RATE_RE.search(provider_text))
    harness_exit = harness_exit or 0

    reason = "pass"
    retryable = False
    if correct and bench_exit in (None, 0):
        reason = "pass"
    elif correct and bench_exit == 124:
        reason = "benchmark_timeout"
        retryable = True
    elif correct and bench_exit not in (None, 0):
        reason = "benchmark_failed"
    elif template_mutated:
        reason = "template_mutated"
    elif provider_failure_window and insufficient_credits:
        reason = "provider_insufficient_credits"
        retryable = False
    elif provider_failure_window and rate_limited:
        reason = "provider_rate_limited"
        retryable = True
    elif harness_exit == 124:
        reason = "timeout"
        retryable = not has_solution
    elif not session_complete:
        reason = "incomplete_session"
        retryable = True
    elif (
        not has_solution
        and output_tokens is not None
        and output_tokens < minimum_useful_output_tokens
    ):
        reason = "provider_early_stop"
        retryable = True
    elif not has_solution:
        reason = "no_solution"
        retryable = False
    elif check_exit == 124:
        reason = "check_timeout"
        retryable = True
    elif check_exit not in (None, 0):
        reason = "check_failed"
    elif bench_exit == 124:
        reason = "benchmark_timeout"
        retryable = True
    elif bench_exit not in (None, 0):
        reason = "benchmark_failed"
    elif harness_exit != 0:
        reason = "harness_error"
        retryable = True
    else:
        reason = "check_failed"

    return {
        "failure_reason": reason,
        "retryable_infra_failure": retryable,
        "minimum_useful_output_tokens": minimum_useful_output_tokens,
    }


def provider_signal_text(
    log_file: str | Path | None,
    stderr_file: str | Path | None,
    *,
    limit: int = 200_000,
) -> str:
    """Return text that can legitimately indicate provider/API failure."""
    parts = [
        _transcript_error_text(log_file, limit=limit),
        _read_tail(stderr_file, limit=limit),
    ]
    return "\n".join(part for part in parts if part)


def _transcript_error_text(path: str | Path | None, *, limit: int) -> str:
    raw = _read_tail(path, limit=limit)
    if not raw:
        return ""

    saw_json = False
    errors: list[str] = []
    for line in raw.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        saw_json = True
        if _is_provider_error_event(obj):
            errors.append(json.dumps(obj, ensure_ascii=False, sort_keys=True))

    if saw_json:
        return "\n".join(errors)
    return raw


def _is_provider_error_event(obj: dict[str, Any]) -> bool:
    if obj.get("type") == "error" or obj.get("error"):
        return True
    if obj.get("type") == "result" and obj.get("subtype") in {"error", "failure"}:
        return True
    if obj.get("is_error") is True and obj.get("subtype") in {"error", "failure"}:
        return True
    return False


def _read_tail(path: str | Path | None, *, limit: int) -> str:
    if not path:
        return ""
    try:
        p = Path(path)
        if not p.exists():
            return ""
        data = p.read_bytes()
        return data[-limit:].decode("utf-8", errors="replace")
    except OSError:
        return ""
