"""Regression tests for the advisory normal-return stage guard."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SUPERVISOR = REPO / "scripts" / "trusted_stage.py"
sys.path.insert(0, str(REPO))
from scripts import trusted_stage  # noqa: E402


def _run(tmp_path: Path, source: str, *args: str) -> subprocess.CompletedProcess[str]:
    target = tmp_path / "stage.py"
    target.write_text(source)
    return subprocess.run(
        [sys.executable, os.fspath(SUPERVISOR), os.fspath(target), *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def test_normal_return_preserves_stdout_stderr_and_arguments(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
import sys
print("stage stdout:" + ",".join(sys.argv[1:]), flush=True)
print("stage stderr", file=sys.stderr, flush=True)
""",
        "alpha",
        "beta",
    )

    assert result.returncode == 0
    assert result.stdout == "stage stdout:alpha,beta\n"
    assert result.stderr == "stage stderr\n"


def test_candidate_modules_do_not_shadow_trusted_import_roots(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    candidate = tmp_path / "candidate"
    trusted.mkdir()
    candidate.mkdir()
    (trusted / "kbh_transitive_dependency.py").write_text(
        'VALUE = "trusted-dependency"\n'
    )
    (candidate / "kbh_transitive_dependency.py").write_text(
        'print("candidate-shadow-loaded", flush=True)\nVALUE = "candidate-shadow"\n'
    )
    target = candidate / "stage.py"
    target.write_text(
        "import kbh_transitive_dependency\n"
        "print(kbh_transitive_dependency.VALUE, flush=True)\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.fspath(trusted)

    result = subprocess.run(
        [sys.executable, os.fspath(SUPERVISOR), os.fspath(target)],
        cwd=candidate,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout == "trusted-dependency\n"
    assert result.stderr == ""


def test_candidate_only_sidecar_remains_importable(tmp_path: Path) -> None:
    (tmp_path / "candidate_helper.py").write_text('VALUE = "candidate-sidecar"\n')
    result = _run(
        tmp_path,
        "import candidate_helper\nprint(candidate_helper.VALUE, flush=True)\n",
    )

    assert result.returncode == 0
    assert result.stdout == "candidate-sidecar\n"
    assert result.stderr == ""


@pytest.mark.parametrize("code", [0, 1, 17])
def test_system_exit_is_never_normal_completion(tmp_path: Path, code: int) -> None:
    result = _run(
        tmp_path,
        f'print("PASS", flush=True)\nraise SystemExit({code})\n',
    )

    assert result.returncode != 0
    assert result.stdout == "PASS\n"
    assert "normal return is required" in result.stderr


def test_os_exit_zero_cannot_bless_forged_pass(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        'import os\nprint("PASS", flush=True)\nos._exit(0)\n',
    )

    assert result.returncode != 0
    assert result.stdout == "PASS\n"
    assert "without a valid normal-return receipt" in result.stderr


def test_os_exit_zero_cannot_bless_forged_metric(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        'import os\nprint("peak_fraction: 999", flush=True)\nos._exit(0)\n',
    )

    assert result.returncode != 0
    assert result.stdout == "peak_fraction: 999\n"
    assert "without a valid normal-return receipt" in result.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="requires process signals")
def test_signal_termination_fails(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n",
    )

    assert result.returncode != 0
    assert "SIGTERM" in result.stderr


def test_exception_fails_and_preserves_traceback(tmp_path: Path) -> None:
    result = _run(tmp_path, 'print("before", flush=True)\nraise RuntimeError("boom")\n')

    assert result.returncode != 0
    assert result.stdout == "before\n"
    assert "RuntimeError: boom" in result.stderr


def test_timeout_fails(tmp_path: Path) -> None:
    target = tmp_path / "stage.py"
    target.write_text("import time\ntime.sleep(60)\n")
    result = subprocess.run(
        [
            sys.executable,
            os.fspath(SUPERVISOR),
            "--timeout-seconds",
            "0.1",
            os.fspath(target),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "target timed out" in result.stderr


def test_malformed_receipt_fails(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
import inspect
frame = inspect.currentframe()
while frame is not None:
    sender = frame.f_locals.get("sender")
    if sender is not None:
        sender.send_bytes(b"not-a-valid-receipt")
        break
    frame = frame.f_back
""",
    )

    assert result.returncode != 0
    assert "malformed completion receipt" in result.stderr


def test_duplicate_receipt_fails(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        """
import inspect
frame = inspect.currentframe()
while frame is not None:
    sender = frame.f_locals.get("sender")
    receipt = frame.f_locals.get("receipt")
    if sender is not None and receipt is not None:
        sender.send_bytes(receipt)
        break
    frame = frame.f_back
""",
    )

    assert result.returncode != 0
    assert "duplicate completion receipt" in result.stderr


def test_missing_malformed_and_duplicate_receipts_fail_validation() -> None:
    expected = trusted_stage._new_receipt()

    assert not trusted_stage._receipts_match([], expected)
    assert not trusted_stage._receipts_match([b"malformed"], expected)
    assert not trusted_stage._receipts_match([expected, expected], expected)
    assert trusted_stage._receipts_match([expected], expected)


def test_receipts_are_random_per_run() -> None:
    first = trusted_stage._new_receipt()
    second = trusted_stage._new_receipt()

    assert first != second
    assert len(first) == len(trusted_stage._RECEIPT_PREFIX) + 32
    assert len(second) == len(first)
