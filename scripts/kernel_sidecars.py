"""Resolve and inline the kernel sidecar files a solution.py actually loads.

Some agent solutions are thin host loaders: the real kernel lives in a
co-located file (nsa_kernel.cu, mingru3.cu, kernels.py, ...) that the publish
pipeline historically dropped, so the site's "solution" link showed Python glue
with no kernel. Grok audit 2026-07-28 found 14 published cells like this across
hard/cuda/mega.

Shared by benchmarks/*/scripts/publish_v2.sh, emit_board_solutions.py, and
publish_mega.sh: call augment(solution_text, run_dir) and publish the result
(then redact as usual). Sidecars are appended under loud banners so the
existing single-URL viewer shows the real kernel with zero UI change.

Only files the loader references are shipped -- never vendored trees or build
caches. A referenced-but-missing sidecar emits a WARNING banner instead of
failing silently.
"""

from __future__ import annotations

import glob
import os
import re
import stat
from pathlib import Path

# module names that are never co-located kernel files
_STDLIKE = {
    "torch",
    "triton",
    "numpy",
    "np",
    "os",
    "sys",
    "math",
    "json",
    "re",
    "pathlib",
    "ctypes",
    "subprocess",
    "functools",
    "itertools",
    "typing",
    "dataclasses",
    "collections",
    "contextlib",
    "time",
    "struct",
    "tempfile",
    "shutil",
    "warnings",
    "hashlib",
    "atexit",
    "threading",
    "types",
    "reference",
    "shapes",
    "problem",
    "check",
    "benchmark",
    "solution",
}

_KERNEL_EXT = (".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".ptx")

# max total appended bytes; beyond this something is wrong (vendored tree?)
_CAP = 1_500_000
_OPEN_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_OPEN_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class SidecarError(RuntimeError):
    """A referenced sidecar is absent, unsafe, unstable, or oversized."""


def _referenced_names(text: str) -> list[tuple[str, bool]]:
    """Return [(basename, definite)]. definite=True means the solution
    demonstrably LOADS the file (sources=[...] list); those warn when missing.
    Speculative mentions (bare '.cu' strings, local imports) are appended when
    found and silently skipped when not -- inline-CUDA cells often quote local
    header names (#include "x.h") that exist only inside the source string."""
    definite: list[str] = []
    speculative: list[str] = []

    def _is_filename(s: str) -> bool:
        # load_inline's cpp_sources/cuda_sources carry CODE strings, and the
        # bare word "sources" suffixes them -- so validate every capture as a
        # plausible path before it can reach a glob (a multiline C declaration
        # once became a glob pattern and crashed emit with ENAMETOOLONG).
        return (
            len(s) < 120
            and "\n" not in s
            and "(" not in s
            and " " not in s
            and "." in Path(s).name
        )

    # cpp_extension.load(sources=[...]) -- NOT cpp_sources=/cuda_sources=
    for m in re.finditer(r"(?<![\w])sources\s*=\s*\[([^\]]*)\]", text):
        definite += [
            x for x in re.findall(r"[\"']([^\"']+)[\"']", m.group(1)) if _is_filename(x)
        ]
    # string literals naming kernel files anywhere (Path(__file__)/"x.cu",
    # open("x.cu"), _SRC = "x.cu", env defaults)
    for m in re.finditer(r"[\"']([\w./\-]+\.(?:cu|cuh|cpp|cc|c|h|hpp|ptx))[\"']", text):
        speculative.append(m.group(1))
    # local python module imports: from x import ... / import x
    for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", text, re.M):
        mod = m.group(1)
        if mod not in _STDLIKE and not mod.startswith("_"):
            speculative.append(mod + ".py")
    # Dedupe by normalized relative path, definite entries win.  Keeping the
    # path prevents a nested decoy with the same basename from replacing the
    # sidecar that the graded loader actually opened.
    seen: set[str] = set()
    out: list[tuple[str, bool]] = []
    for n, d in [(x, True) for x in definite] + [(x, False) for x in speculative]:
        path = Path(n)
        if path.is_absolute() or ".." in path.parts:
            continue
        normalized = path.as_posix().removeprefix("./")
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append((normalized, d))
    return out


def _same_file(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _read_relative_sidecar(
    root: Path, relative: str, *, max_bytes: int
) -> tuple[str, int]:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or ".." in parts:
        raise SidecarError(f"unsafe sidecar path: {relative!r}")
    try:
        directory_fd = os.open(root, _OPEN_DIRECTORY_FLAGS)
    except OSError as exc:
        raise SidecarError(f"cannot safely open sidecar root {root}: {exc}") from exc
    descriptor: int | None = None
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, _OPEN_DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(parts[-1], _OPEN_FILE_FLAGS, dir_fd=directory_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SidecarError(f"sidecar is not a regular file: {relative}")
        if before.st_nlink != 1:
            raise SidecarError(f"hard-linked sidecar is not allowed: {relative}")
        if before.st_size > max_bytes:
            raise SidecarError(
                f"sidecar exceeds remaining size cap ({before.st_size} > "
                f"{max_bytes}): {relative}"
            )
        chunks: list[bytes] = []
        copied = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - copied))
            if not chunk:
                break
            copied += len(chunk)
            if copied > max_bytes:
                raise SidecarError(f"sidecar grew past size cap: {relative}")
            chunks.append(chunk)
        if copied != before.st_size or not _same_file(before, os.fstat(descriptor)):
            raise SidecarError(f"sidecar changed while being read: {relative}")
        return b"".join(chunks).decode("utf-8", errors="replace"), copied
    except FileNotFoundError as exc:
        raise SidecarError(f"referenced sidecar was not found: {relative}") from exc
    except OSError as exc:
        raise SidecarError(f"cannot safely read sidecar {relative}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _find_in_archive(run_dir: Path, referenced: str, *, exact: bool) -> list[str]:
    """Return exact bundle path, or constrained legacy compatibility paths."""

    if exact:
        return [referenced]
    hits: list[str] = []
    direct = run_dir.joinpath(*Path(referenced).parts)
    if direct.exists() or direct.is_symlink():
        hits.append(referenced)
    escaped = glob.escape(referenced)
    for pat in (
        f"repo/problems/*/{escaped}",
        f"scratch/{escaped}",
        f"repo/problems/*/scratch/{escaped}",
    ):
        try:
            for hit in sorted(run_dir.glob(pat)):
                hits.append(hit.relative_to(run_dir).as_posix())
        except (OSError, ValueError):
            continue
    output: list[str] = []
    for h in hits:
        # ignore vendored/third-party and build-cache copies
        relative_parts = Path(h).parts
        if any(
            part in {".venv", "cache", "site-packages", "cutlass", "third_party", "tmp"}
            for part in relative_parts
        ):
            continue
        if h not in output:
            output.append(h)
    return output


def augment(
    solution_text: str,
    run_dir: str | Path,
    *,
    exact: bool = False,
    strict: bool = False,
) -> str:
    """Append every referenced sidecar to the solution text under banners."""
    run_dir = Path(run_dir)
    refs = _referenced_names(solution_text)
    if not refs:
        return solution_text
    parts = [solution_text]
    total = 0
    for name, definite in refs:
        candidates = _find_in_archive(run_dir, name, exact=exact)
        body: str | None = None
        body_size = 0
        selected: str | None = None
        missing_errors: list[SidecarError] = []
        for candidate in candidates:
            try:
                body, body_size = _read_relative_sidecar(
                    run_dir,
                    candidate,
                    max_bytes=_CAP - total,
                )
            except SidecarError as exc:
                if "was not found" in str(exc):
                    missing_errors.append(exc)
                    continue
                raise
            selected = candidate
            break
        if body is None or selected is None:
            if strict and definite:
                if missing_errors:
                    raise missing_errors[0]
                raise SidecarError(f"referenced sidecar was not found: {name}")
            if definite:
                parts.append(
                    f"\n\n# {'=' * 66}\n# WARNING: solution loads sidecar "
                    f"'{name}' but it was not found in the run archive.\n# {'=' * 66}\n"
                )
            continue
        total += body_size
        parts.append(
            f"\n\n# {'=' * 66}\n# ===== sidecar: {name} "
            f"({body_size} bytes, loaded by solution.py) =====\n# {'=' * 66}\n\n" + body
        )
    return "".join(parts)
