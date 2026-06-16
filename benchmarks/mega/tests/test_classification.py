import json
from pathlib import Path

from src.harness.classification import classify_run, provider_signal_text


def test_provider_detection_ignores_model_read_artifacts(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "tool": "read",
                            "state": {
                                "status": "completed",
                                "output": "quota and rate limits are healthy",
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "text",
                        "part": {
                            "text": "source says insufficient_credits in run_hard.sh"
                        },
                    }
                ),
            ]
        )
    )

    assert provider_signal_text(transcript, None) == ""
    result = classify_run(
        usage={"output_tokens": 25_000},
        correct=False,
        template_mutated=False,
        has_solution=False,
        session_complete=True,
        harness_exit=0,
        check_exit=None,
        bench_exit=None,
        log_file=transcript,
        stderr_file=None,
        minimum_useful_output_tokens=5_000,
    )
    assert result["failure_reason"] == "no_solution"
    assert result["retryable_infra_failure"] is False


def test_provider_detection_uses_explicit_error_events(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "error",
                "error": {
                    "data": {
                        "message": "429 resource_exhausted: rate limit exceeded"
                    }
                },
            }
        )
    )

    result = classify_run(
        usage={"output_tokens": 42},
        correct=False,
        template_mutated=False,
        has_solution=False,
        session_complete=False,
        harness_exit=1,
        check_exit=None,
        bench_exit=None,
        log_file=transcript,
        stderr_file=None,
        minimum_useful_output_tokens=5_000,
    )
    assert result["failure_reason"] == "provider_rate_limited"
    assert result["retryable_infra_failure"] is True


def test_provider_detection_uses_stderr_for_plain_cli_errors(
    tmp_path: Path,
) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text("provider said payment required; add more using console")

    result = classify_run(
        usage={"output_tokens": 0},
        correct=False,
        template_mutated=False,
        has_solution=False,
        session_complete=False,
        harness_exit=1,
        check_exit=None,
        bench_exit=None,
        log_file=None,
        stderr_file=stderr,
        minimum_useful_output_tokens=5_000,
    )
    assert result["failure_reason"] == "provider_insufficient_credits"
    assert result["retryable_infra_failure"] is False
