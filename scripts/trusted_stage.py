#!/usr/bin/env python3
"""Apply an advisory normal-return guard to a canonical Python stage.

The target script runs in a clean ``spawn`` interpreter.  Its stdout and
stderr remain attached to this process, but neither stream is authoritative.
Success requires the worker to return from ``runpy.run_path`` and then send one
per-run receipt to this parent.  This catches ordinary exceptions and
``SystemExit`` even when candidate output contains a forged marker.  It is not
a completion authority: plain candidate Python can inspect worker frames, send
the exposed receipt, and invoke ``os._exit(0)``.  Callers must treat success as
defense-in-depth telemetry and apply an independent publication gate.

Usage::

    python scripts/trusted_stage.py path/to/check.py
    python scripts/trusted_stage.py --timeout-seconds 180 path/to/check.py
"""

from __future__ import annotations

import argparse
import hmac
import importlib
import multiprocessing
import os
import runpy
import secrets
import signal
import stat
import sys
import time
import traceback
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Sequence


_RECEIPT_PREFIX = b"kernelbench-trusted-stage-v1\0"
_RECEIPT_BYTES = 32
_MAX_MESSAGE_BYTES = len(_RECEIPT_PREFIX) + _RECEIPT_BYTES
_FAILURE_EXIT = 70
_POLL_SECONDS = 0.05
_TERMINATE_GRACE_SECONDS = 1.0


def _new_receipt() -> bytes:
    return _RECEIPT_PREFIX + secrets.token_bytes(_RECEIPT_BYTES)


def _receipts_match(messages: Sequence[bytes], expected: bytes) -> bool:
    return len(messages) == 1 and hmac.compare_digest(messages[0], expected)


def _flush_standard_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass


def _remove_import_root(root: Path) -> None:
    """Remove every spelling of ``root`` from the inherited import path."""

    retained: list[str] = []
    for entry in sys.path:
        try:
            resolved = (Path.cwd() if entry == "" else Path(entry)).resolve()
        except (OSError, TypeError, ValueError):
            retained.append(entry)
            continue
        if resolved != root:
            retained.append(entry)
    sys.path[:] = retained


def _worker(
    script: str,
    script_args: tuple[str, ...],
    preload_modules: tuple[str, ...],
    sender: Connection,
    receipt: bytes,
) -> None:
    """Run the target and send a receipt only after an ordinary return."""

    path = Path(script)
    try:
        candidate_root = path.parent.resolve()
        # Never let the candidate directory precede the standard library,
        # environment, or trusted project roots.  This is name-independent:
        # direct and transitive dependencies keep resolving to the environment
        # even when the bundle contains an artifact with the same module name.
        # Safe-path mode protects the spawn bootstrap; this cleanup also removes
        # inherited empty/relative aliases before any explicit preload.
        _remove_import_root(candidate_root)
        for module in preload_modules:
            importlib.import_module(module)
        _remove_import_root(candidate_root)
        sys.path.append(os.fspath(candidate_root))
        os.chdir(candidate_root)
        sys.argv = [os.fspath(path), *script_args]
        runpy.run_path(os.fspath(path), run_name="__main__")
    except SystemExit as exc:
        print(
            "trusted_stage: target raised SystemExit "
            f"({exc.code!r}); normal return is required",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError("target raised SystemExit") from exc
    except BaseException:  # noqa: BLE001 - every abnormal target exit must fail closed
        traceback.print_exc()
        raise

    _flush_standard_streams()
    try:
        sender.send_bytes(receipt)
    except (BrokenPipeError, EOFError, OSError):
        return
    finally:
        sender.close()


def _stop_worker(process: multiprocessing.Process) -> None:
    if process.pid is None:
        return
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join()


def _read_one(connection: Connection) -> tuple[bytes | None, bool]:
    """Return (message, malformed); EOF is neither a message nor malformed."""

    try:
        return connection.recv_bytes(maxlength=_MAX_MESSAGE_BYTES), False
    except EOFError:
        return None, False
    except (OSError, ValueError):
        return None, True


def _supervise(
    script: Path,
    script_args: Sequence[str] = (),
    *,
    timeout_seconds: float | None = None,
    preload_modules: Sequence[str] = (),
) -> int:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    expected = _new_receipt()
    process = context.Process(
        target=_worker,
        args=(
            os.fspath(script),
            tuple(script_args),
            tuple(preload_modules),
            sender,
            expected,
        ),
        name="kernelbench-trusted-stage-worker",
    )

    messages: list[bytes] = []
    malformed = False
    duplicate = False
    timed_out = False
    receiver_at_eof = False
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds

    try:
        process.start()
        sender.close()

        while process.exitcode is None:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            wait_for = _POLL_SECONDS
            if deadline is not None:
                wait_for = max(0.0, min(wait_for, deadline - time.monotonic()))
            watched = (
                (process.sentinel,) if receiver_at_eof else (receiver, process.sentinel)
            )
            ready = wait(watched, timeout=wait_for)
            if not receiver_at_eof and receiver in ready:
                message, bad = _read_one(receiver)
                malformed = malformed or bad
                if message is None and not bad:
                    receiver_at_eof = True
                if message is not None:
                    if not hmac.compare_digest(message, expected):
                        malformed = True
                    elif messages:
                        duplicate = True
                    messages.append(message)
                if malformed or duplicate:
                    break
            process.join(0)

        if timed_out or malformed or duplicate:
            _stop_worker(process)
        else:
            process.join()

        # A receipt can become readable at the same moment the process exits.
        # Drain only already-buffered data; a descendant retaining the write
        # end must not make the trusted parent wait forever.
        while not receiver_at_eof and receiver.poll(0):
            message, bad = _read_one(receiver)
            malformed = malformed or bad
            if message is None:
                receiver_at_eof = not bad
                break
            if not hmac.compare_digest(message, expected):
                malformed = True
            elif messages:
                duplicate = True
            messages.append(message)
            if malformed or duplicate:
                break
    except KeyboardInterrupt:
        _stop_worker(process)
        print("trusted_stage: interrupted", file=sys.stderr)
        return _FAILURE_EXIT
    except (OSError, RuntimeError) as exc:
        _stop_worker(process)
        print(f"trusted_stage: could not run target: {exc}", file=sys.stderr)
        return _FAILURE_EXIT
    finally:
        receiver.close()
        sender.close()

    if timed_out:
        print("trusted_stage: target timed out", file=sys.stderr)
        return _FAILURE_EXIT
    if malformed:
        print("trusted_stage: malformed completion receipt", file=sys.stderr)
        return _FAILURE_EXIT
    if duplicate:
        print("trusted_stage: duplicate completion receipt", file=sys.stderr)
        return _FAILURE_EXIT
    if process.exitcode != 0:
        if process.exitcode is not None and process.exitcode < 0:
            try:
                reason = signal.Signals(-process.exitcode).name
            except ValueError:
                reason = f"signal {-process.exitcode}"
            print(f"trusted_stage: target terminated by {reason}", file=sys.stderr)
        else:
            print(
                f"trusted_stage: target worker exited {process.exitcode}",
                file=sys.stderr,
            )
        return _FAILURE_EXIT
    if not _receipts_match(messages, expected):
        print(
            "trusted_stage: target exited without a valid normal-return receipt",
            file=sys.stderr,
        )
        return _FAILURE_EXIT
    return 0


def _existing_regular_script(value: str) -> Path:
    try:
        path = Path(value).resolve(strict=True)
        metadata = path.stat()
    except OSError as exc:
        raise argparse.ArgumentTypeError(
            f"cannot open target script {value!r}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise argparse.ArgumentTypeError(f"target is not a regular file: {value}")
    return path


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be a number") from exc
    if not timeout > 0:
        raise argparse.ArgumentTypeError("timeout must be positive")
    return timeout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=_positive_timeout)
    parser.add_argument(
        "--preload-module",
        action="append",
        default=[],
        help="trusted dependency to import before exposing the target directory",
    )
    parser.add_argument("script", type=_existing_regular_script)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return _supervise(
        args.script,
        args.script_args,
        timeout_seconds=args.timeout_seconds,
        preload_modules=args.preload_module,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
