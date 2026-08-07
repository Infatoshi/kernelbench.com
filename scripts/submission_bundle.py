#!/usr/bin/env python3
"""Create, verify, and extract immutable KernelBench submissions.

A bundle contains a canonical manifest and a path-preserving ``files/`` tree.
The manifest hashes the content, relative path, size, and mode of every file,
and its own ``bundle_sha256`` hashes that complete description.  Bundle and
extraction directories are assembled beside their destination and renamed
only after every file has been copied and checked.

The implementation deliberately treats the input and the bundle as hostile:
it never follows links, rejects non-regular artifacts, validates every
relative path before using it, and applies independent file-count and byte
limits while both writing and reading.

Examples (run from the repository root)::

    uv run --project kbtool python scripts/submission_bundle.py create PROBLEM BUNDLE
    uv run --project kbtool python scripts/submission_bundle.py verify BUNDLE \
        --expected-sha256 SHA256
    uv run --project kbtool python scripts/submission_bundle.py extract BUNDLE DEST \
        --expected-sha256 SHA256

Verification without an expected digest proves internal self-consistency.
Passing the digest recorded at creation binds the bytes to that submission.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import hmac
import importlib.machinery
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Collection, Sequence


SCHEMA = "kernelbench.submission-bundle"
SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256"
MANIFEST_NAME = "manifest.json"
PAYLOAD_DIR = "files"
SOLUTION_PATH = "solution.py"
RUN_BUNDLE_DIR = "submission_bundle"
VERIFIED_REPLAY_STATUS = "verified"
# Historical archives before this rollout have no immutable bundle.  The UTC
# boundary deliberately follows the August 7 legacy jobs that were already
# allocated or in flight; old launchers must be drained before it.  Under the
# default policy, a metadata-free run at or after this boundary is rejected.
# This is an operational admission rule, not proof of an archive's age; that
# requires the archive name and contents to be held on trusted immutable storage.
BUNDLE_CUTOVER_RUN_ID = "20260808_000000"

# These are benchmark definitions, not agent-authored submission artifacts.
DEFAULT_TEMPLATE_NAMES = frozenset(
    {
        "PROMPT.txt",
        "baseline.py",
        "benchmark.py",
        "check.py",
        "problem.yaml",
        "reference.py",
        "shapes.py",
        "sota.py",
    }
)
TRUSTED_MODULE_STEMS = frozenset(
    {SOLUTION_PATH.removesuffix(".py")}
    | {"multiprocessing", "torch", "yaml"}
    | {
        name.removesuffix(".py")
        for name in DEFAULT_TEMPLATE_NAMES
        if name.endswith(".py")
    }
)
PYTHON_EXTENSION_SUFFIXES = frozenset(importlib.machinery.EXTENSION_SUFFIXES)
DEFAULT_EXCLUDED_DIRECTORY_NAMES = frozenset(
    {".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
)

DEFAULT_MAX_FILES = 2_048
DEFAULT_MAX_DIRECTORIES = 2_048
DEFAULT_MAX_FILE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_RESULT_BYTES = 16 * 1024 * 1024
MAX_PATH_BYTES = 4_096
MAX_COMPONENT_BYTES = 255
MAX_PATH_DEPTH = 64

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})(?:_.+)?\Z")
_PORTABLE_EXTENSION_SUFFIX_RE = re.compile(
    r"(?:\.(?:cpython|cp|pypy|abi)[A-Za-z0-9_.-]*\.(?:so|pyd)|\.(?:so|pyd))\Z"
)
_OPEN_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_OPEN_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_OPEN_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_OPEN_SAFE_DIRECTORY_FLAGS = _OPEN_DIRECTORY_FLAGS | _OPEN_NOFOLLOW | _OPEN_NONBLOCK
_OPEN_FILE_FLAGS = os.O_RDONLY | _OPEN_NOFOLLOW | _OPEN_NONBLOCK

_BUNDLE_PROVENANCE_KEYS = frozenset(
    {
        "submission_bundle",
        "submission_bundle_sha256",
        "submission_replay",
        "submission_replay_status",
    }
)


class BundleError(RuntimeError):
    """A submission cannot be represented or fails bundle verification."""


@dataclass(frozen=True)
class BundleLimits:
    """Independent resource limits applied while creating and consuming."""

    max_files: int = DEFAULT_MAX_FILES
    max_directories: int = DEFAULT_MAX_DIRECTORIES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES

    def __post_init__(self) -> None:
        for name, value in (
            ("max_files", self.max_files),
            ("max_directories", self.max_directories),
            ("max_file_bytes", self.max_file_bytes),
            ("max_total_bytes", self.max_total_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _descriptor(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "bundle_sha256"}


def _descriptor_digest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(_descriptor(manifest))).hexdigest()


def _validate_component(component: str) -> None:
    if not component or component in {".", ".."}:
        raise BundleError(f"unsafe empty or dot path component: {component!r}")
    if "/" in component or "\\" in component or "\x00" in component:
        raise BundleError(f"unsafe path component: {component!r}")
    try:
        encoded = component.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise BundleError(f"path component is not valid UTF-8: {component!r}") from exc
    if len(encoded) > MAX_COMPONENT_BYTES:
        raise BundleError(f"path component is too long: {component!r}")


def _validate_relative_path(path: object) -> tuple[str, ...]:
    if not isinstance(path, str) or not path:
        raise BundleError("artifact path must be a non-empty string")
    if path.startswith("/") or "\\" in path or "\x00" in path:
        raise BundleError(f"unsafe artifact path: {path!r}")
    try:
        encoded = path.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise BundleError(f"artifact path is not valid UTF-8: {path!r}") from exc
    if len(encoded) > MAX_PATH_BYTES:
        raise BundleError(f"artifact path is too long: {path!r}")
    if re.match(r"^[A-Za-z]:", path):
        raise BundleError(f"drive-qualified artifact path is unsafe: {path!r}")

    parts = tuple(path.split("/"))
    if len(parts) > MAX_PATH_DEPTH:
        raise BundleError(
            f"artifact path exceeds maximum depth ({MAX_PATH_DEPTH}): {path!r}"
        )
    for component in parts:
        _validate_component(component)
    if PurePosixPath(*parts).as_posix() != path:
        raise BundleError(f"artifact path is not canonical: {path!r}")
    return parts


def _validate_import_safe_artifact(
    parts: tuple[str, ...],
    *,
    top_level_is_directory: bool = False,
) -> None:
    top_level_name = parts[0]
    if top_level_name in TRUSTED_MODULE_STEMS and (
        top_level_is_directory or len(parts) > 1
    ):
        raise BundleError(
            "top-level artifact directory shadows a trusted Python module: "
            f"{'/'.join(parts)}"
        )
    if len(parts) != 1:
        return
    for stem in TRUSTED_MODULE_STEMS:
        if not top_level_name.startswith(stem):
            continue
        suffix = top_level_name[len(stem) :]
        if (
            suffix in PYTHON_EXTENSION_SUFFIXES
            or _PORTABLE_EXTENSION_SUFFIX_RE.fullmatch(suffix)
        ):
            raise BundleError(
                "top-level extension artifact shadows a trusted Python module: "
                f"{top_level_name}"
            )


def _validate_template_names(names: Collection[str]) -> frozenset[str]:
    result: set[str] = set()
    for name in names:
        _validate_component(name)
        result.add(name)
    if SOLUTION_PATH in result:
        raise ValueError(f"{SOLUTION_PATH} cannot be excluded as a template")
    return frozenset(result)


def _validate_excluded_directory_names(names: Collection[str]) -> frozenset[str]:
    result: set[str] = set()
    for name in names:
        _validate_component(name)
        result.add(name)
    return frozenset(result)


def _open_directory(
    path: str | os.PathLike[str],
    *,
    label: str,
    dir_fd: int | None = None,
) -> int:
    """Open a pinned directory without following any component symlink."""

    raw_path = os.fsdecode(os.fspath(path))
    if not raw_path:
        raise BundleError(f"cannot inspect {label} {path}: empty path")

    if os.path.isabs(raw_path) or dir_fd is None:
        absolute = Path(os.path.abspath(raw_path))
        components = absolute.parts[1:]
        try:
            current_fd = os.open(absolute.anchor, _OPEN_SAFE_DIRECTORY_FLAGS)
        except OSError as exc:
            raise BundleError(
                f"cannot open trusted root for {label} {path}: {exc.strerror}"
            ) from exc
    else:
        components = Path(raw_path).parts
        if ".." in components:
            raise BundleError(f"parent traversal is not allowed in {label}: {path}")
        try:
            current_fd = os.dup(dir_fd)
        except OSError as exc:
            raise BundleError(
                f"cannot duplicate trusted parent for {label} {path}: {exc.strerror}"
            ) from exc

    try:
        if not stat.S_ISDIR(os.fstat(current_fd).st_mode):
            raise BundleError(f"trusted parent for {label} is not a directory: {path}")
        for component in components:
            if component in {"", "."}:
                continue
            try:
                observed = os.stat(
                    component,
                    dir_fd=current_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise BundleError(
                    f"cannot inspect {label} {path}: {exc.strerror}"
                ) from exc
            if stat.S_ISLNK(observed.st_mode):
                raise BundleError(
                    f"{label} must be a real directory, not a link: {path}"
                )
            if not stat.S_ISDIR(observed.st_mode):
                raise BundleError(
                    f"{label} must be a real directory, found "
                    f"{_entry_kind(observed.st_mode)}: {path}"
                )

            next_fd: int | None = None
            try:
                next_fd = os.open(
                    component,
                    _OPEN_SAFE_DIRECTORY_FLAGS,
                    dir_fd=current_fd,
                )
                opened = os.fstat(next_fd)
                if not stat.S_ISDIR(opened.st_mode) or not _same_file(observed, opened):
                    raise BundleError(
                        f"{label} changed while it was being opened: {path}"
                    )
            except OSError as exc:
                if next_fd is not None:
                    os.close(next_fd)
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise BundleError(
                        f"{label} must be a real directory, not a link: {path}"
                    ) from exc
                raise BundleError(
                    f"cannot open {label} {path}: {exc.strerror}"
                ) from exc
            except Exception:
                if next_fd is not None:
                    os.close(next_fd)
                raise

            assert next_fd is not None
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _entry_kind(mode: int) -> str:
    if stat.S_ISLNK(mode):
        return "symbolic link"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISREG(mode):
        return "regular file"
    if stat.S_ISFIFO(mode):
        return "FIFO"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character device"
    if stat.S_ISBLK(mode):
        return "block device"
    return "special file"


def _canonical_file_mode(mode: int) -> int:
    return 0o755 if stat.S_IMODE(mode) & 0o111 else 0o644


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


def _copy_open_file(
    source_fd: int,
    before: os.stat_result,
    destination: Path | None,
    *,
    display_path: str,
    limits: BundleLimits,
    total_before: int,
    output_mode: int,
) -> tuple[str, int]:
    if not stat.S_ISREG(before.st_mode):
        raise BundleError(f"artifact is not a regular file: {display_path}")
    if before.st_nlink != 1:
        raise BundleError(f"hard-linked artifacts are not allowed: {display_path}")
    if before.st_size > limits.max_file_bytes:
        raise BundleError(
            f"artifact exceeds per-file limit ({before.st_size} > "
            f"{limits.max_file_bytes} bytes): {display_path}"
        )
    if total_before + before.st_size > limits.max_total_bytes:
        raise BundleError(
            f"submission exceeds total-size limit ({limits.max_total_bytes} bytes)"
        )

    output = None
    digest = hashlib.sha256()
    copied = 0
    try:
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = destination.open("xb")
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > limits.max_file_bytes:
                raise BundleError(
                    f"artifact grew past per-file limit while reading: {display_path}"
                )
            if total_before + copied > limits.max_total_bytes:
                raise BundleError(
                    f"submission grew past total-size limit while reading: {display_path}"
                )
            digest.update(chunk)
            if output is not None:
                output.write(chunk)

        after = os.fstat(source_fd)
        if copied != before.st_size or not _same_file(before, after):
            raise BundleError(
                f"artifact changed while it was being captured: {display_path}"
            )
        if output is not None:
            os.fchmod(output.fileno(), output_mode)
            output.flush()
            os.fsync(output.fileno())
    finally:
        if output is not None:
            output.close()

    return digest.hexdigest(), copied


def _capture_tree_fd(
    root_fd: int,
    destination: Path | None,
    *,
    excluded_root_names: frozenset[str],
    limits: BundleLimits,
    normalize_modes: bool,
    excluded_directory_names: frozenset[str] = frozenset(),
    observed_directories: set[str] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    total_size = 0
    directory_count = 0
    raw_entry_count = 0
    raw_entry_limit = (
        limits.max_files
        + limits.max_directories
        + len(excluded_root_names)
        + len(excluded_directory_names)
    )

    def visit(directory_fd: int, prefix: tuple[str, ...]) -> None:
        nonlocal directory_count, raw_entry_count, total_size
        directory_before = os.fstat(directory_fd)
        try:
            with os.scandir(directory_fd) as iterator:
                directory_entries = []
                for item in iterator:
                    raw_entry_count += 1
                    if raw_entry_count > raw_entry_limit:
                        raise BundleError(
                            f"submission exceeds entry-count limit ({raw_entry_limit} entries)"
                        )
                    directory_entries.append(item)
                directory_entries.sort(key=lambda item: item.name)
        except OSError as exc:
            shown = "/".join(prefix) or "."
            raise BundleError(
                f"cannot scan artifact directory {shown!r}: {exc.strerror}"
            ) from exc

        for item in directory_entries:
            _validate_component(item.name)
            parts = (*prefix, item.name)
            relative_path = "/".join(parts)
            _validate_relative_path(relative_path)
            try:
                observed = item.stat(follow_symlinks=False)
            except OSError as exc:
                raise BundleError(
                    f"cannot inspect artifact {relative_path!r}: {exc.strerror}"
                ) from exc

            kind = _entry_kind(observed.st_mode)
            _validate_import_safe_artifact(
                parts,
                top_level_is_directory=kind == "directory",
            )
            if kind == "symbolic link":
                raise BundleError(f"symbolic links are not allowed: {relative_path}")
            if kind == "regular file" and observed.st_nlink != 1:
                raise BundleError(
                    f"hard-linked artifacts are not allowed: {relative_path}"
                )
            if not prefix and item.name in excluded_root_names:
                if kind != "regular file":
                    raise BundleError(
                        f"template path must be a regular file, found {kind}: {relative_path}"
                    )
                continue

            if kind == "directory":
                if item.name in excluded_directory_names:
                    continue
                directory_count += 1
                if directory_count > limits.max_directories:
                    raise BundleError(
                        "submission exceeds directory-count limit "
                        f"({limits.max_directories} directories)"
                    )
                if observed_directories is not None:
                    observed_directories.add(relative_path)
                try:
                    child_fd = os.open(
                        item.name,
                        _OPEN_SAFE_DIRECTORY_FLAGS,
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise BundleError(
                        f"cannot safely open artifact directory {relative_path!r}: "
                        f"{exc.strerror}"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    if not stat.S_ISDIR(opened.st_mode) or (
                        opened.st_dev,
                        opened.st_ino,
                    ) != (observed.st_dev, observed.st_ino):
                        raise BundleError(
                            f"artifact directory changed while scanning: {relative_path}"
                        )
                    visit(child_fd, parts)
                finally:
                    os.close(child_fd)
                continue

            if kind != "regular file":
                raise BundleError(f"{kind}s are not allowed: {relative_path}")
            if len(entries) >= limits.max_files:
                raise BundleError(
                    f"submission exceeds file-count limit ({limits.max_files} files)"
                )

            try:
                source_fd = os.open(
                    item.name,
                    _OPEN_FILE_FLAGS,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise BundleError(
                    f"cannot safely open artifact {relative_path!r}: {exc.strerror}"
                ) from exc
            try:
                opened = os.fstat(source_fd)
                if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
                    raise BundleError(
                        f"artifact changed while scanning: {relative_path}"
                    )
                target = (
                    destination.joinpath(*parts) if destination is not None else None
                )
                stored_mode = (
                    _canonical_file_mode(opened.st_mode)
                    if normalize_modes
                    else stat.S_IMODE(opened.st_mode)
                )
                digest, size = _copy_open_file(
                    source_fd,
                    opened,
                    target,
                    display_path=relative_path,
                    limits=limits,
                    total_before=total_size,
                    output_mode=stored_mode,
                )
            finally:
                os.close(source_fd)

            total_size += size
            entries.append(
                {
                    "mode": stored_mode,
                    "path": relative_path,
                    "sha256": digest,
                    "size": size,
                }
            )

        if not _same_file(directory_before, os.fstat(directory_fd)):
            shown = "/".join(prefix) or "."
            raise BundleError(f"artifact directory changed while scanning: {shown}")

    visit(root_fd, ())
    return entries


def _capture_tree(
    source: Path,
    destination: Path | None,
    *,
    excluded_root_names: frozenset[str],
    limits: BundleLimits,
    normalize_modes: bool,
    excluded_directory_names: frozenset[str],
) -> list[dict[str, Any]]:
    root_fd = _open_directory(source, label="artifact root")
    try:
        return _capture_tree_fd(
            root_fd,
            destination,
            excluded_root_names=excluded_root_names,
            limits=limits,
            normalize_modes=normalize_modes,
            excluded_directory_names=excluded_directory_names,
        )
    finally:
        os.close(root_fd)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, _OPEN_DIRECTORY_FLAGS)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems support atomic rename but not directory fsync.
        pass
    finally:
        os.close(fd)


def _fsync_directories(root: Path) -> None:
    directories = [Path(path) for path, _dirs, _files in os.walk(root)]
    for directory in reversed(directories):
        _fsync_directory(directory)


def _atomic_publish_directory(source: Path, destination: Path) -> None:
    """Rename a staged directory without ever replacing an existing path."""

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            renameat2.restype = ctypes.c_int
            rc = renameat2(
                -100,  # AT_FDCWD
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,  # RENAME_NOREPLACE
            )
            if rc == 0:
                return
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise BundleError(f"destination already exists: {destination}")
            if error not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
                raise BundleError(
                    f"cannot publish directory at {destination}: {os.strerror(error)}"
                )

    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is not None:
            renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            renamex_np.restype = ctypes.c_int
            rc = renamex_np(
                os.fsencode(source),
                os.fsencode(destination),
                0x00000004,  # RENAME_EXCL
            )
            if rc == 0:
                return
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise BundleError(f"destination already exists: {destination}")
            raise BundleError(
                f"cannot publish directory at {destination}: {os.strerror(error)}"
            )

    raise BundleError(
        "atomic no-replace directory publication is unavailable on this platform/filesystem"
    )


def _prepare_destination(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.exists() or absolute.is_symlink():
        raise BundleError(f"{label} already exists: {absolute}")
    try:
        resolved_parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise BundleError(
            f"parent for {label} must already exist: {absolute.parent}"
        ) from exc
    if resolved_parent != absolute.parent or not absolute.parent.is_dir():
        raise BundleError(
            f"parent for {label} is not a real directory: {absolute.parent}"
        )
    return absolute


def _resolved_existing(path: Path, *, label: str) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"cannot resolve {label} {path}: {exc.strerror}") from exc


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_overlap(first: Path, second: Path, *, labels: tuple[str, str]) -> None:
    first_resolved = _resolved_existing(first, label=labels[0])
    second_parent = _resolved_existing(second.parent, label=f"{labels[1]} parent")
    second_resolved = second_parent / second.name
    if _is_relative_to(second_resolved, first_resolved) or _is_relative_to(
        first_resolved, second_resolved
    ):
        raise BundleError(f"{labels[0]} and {labels[1]} must not overlap")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    encoded = _canonical_json(manifest)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise BundleError(f"manifest exceeds limit ({MAX_MANIFEST_BYTES} bytes)")
    with path.open("xb") as stream:
        stream.write(encoded)
        os.fchmod(stream.fileno(), 0o644)
        stream.flush()
        os.fsync(stream.fileno())


def create_bundle(
    source_dir: str | os.PathLike[str],
    bundle_dir: str | os.PathLike[str],
    *,
    template_names: Collection[str] = DEFAULT_TEMPLATE_NAMES,
    excluded_directory_names: Collection[str] = DEFAULT_EXCLUDED_DIRECTORY_NAMES,
    limits: BundleLimits = BundleLimits(),
) -> dict[str, Any]:
    """Freeze ``source_dir`` into a new, atomically published bundle.

    Canonical root-level benchmark templates and known generated Python cache
    directories are excluded.  ``solution.py`` and every other regular file,
    including nested and dot-named sidecars, are included.  Existing
    destinations are never replaced.
    """

    source = Path(source_dir)
    output = Path(os.path.abspath(os.fspath(bundle_dir)))
    _reject_overlap(source, output, labels=("artifact root", "bundle destination"))
    output = _prepare_destination(output, label="bundle destination")
    excluded = _validate_template_names(template_names)
    excluded_directories = _validate_excluded_directory_names(excluded_directory_names)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=os.fspath(output.parent))
    )
    published = False
    try:
        payload = staging / PAYLOAD_DIR
        payload.mkdir()
        entries = _capture_tree(
            source,
            payload,
            excluded_root_names=excluded,
            limits=limits,
            normalize_modes=True,
            excluded_directory_names=excluded_directories,
        )
        paths = {entry["path"] for entry in entries}
        if SOLUTION_PATH not in paths:
            raise BundleError(f"required artifact is missing: {SOLUTION_PATH}")

        entries.sort(key=lambda entry: entry["path"])
        manifest: dict[str, Any] = {
            "file_count": len(entries),
            "files": entries,
            "hash_algorithm": HASH_ALGORITHM,
            "schema": SCHEMA,
            "total_size": sum(entry["size"] for entry in entries),
            "version": SCHEMA_VERSION,
        }
        manifest["bundle_sha256"] = _descriptor_digest(manifest)
        _write_manifest(staging / MANIFEST_NAME, manifest)
        _fsync_directories(staging)

        _atomic_publish_directory(staging, output)
        published = True
        _fsync_directory(output.parent)
        return manifest
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"manifest contains duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise BundleError(f"manifest contains invalid JSON number: {value}")


def _read_manifest_fd(bundle_fd: int) -> tuple[dict[str, Any], bytes]:
    try:
        manifest_fd = os.open(
            MANIFEST_NAME,
            _OPEN_FILE_FLAGS,
            dir_fd=bundle_fd,
        )
    except OSError as exc:
        raise BundleError(f"cannot open {MANIFEST_NAME}: {exc.strerror}") from exc
    try:
        metadata = os.fstat(manifest_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise BundleError(f"{MANIFEST_NAME} is not a regular file")
        if metadata.st_nlink != 1:
            raise BundleError(f"hard-linked {MANIFEST_NAME} is not allowed")
        if stat.S_IMODE(metadata.st_mode) != 0o644:
            raise BundleError(f"{MANIFEST_NAME} must have mode 0644")
        if metadata.st_size > MAX_MANIFEST_BYTES:
            raise BundleError(f"manifest exceeds limit ({MAX_MANIFEST_BYTES} bytes)")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(
                manifest_fd, min(1024 * 1024, MAX_MANIFEST_BYTES + 1 - size)
            )
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_MANIFEST_BYTES:
                raise BundleError(
                    f"manifest exceeds limit ({MAX_MANIFEST_BYTES} bytes)"
                )
        encoded = b"".join(chunks)
        if not _same_file(metadata, os.fstat(manifest_fd)):
            raise BundleError("manifest changed while it was being read")
    finally:
        os.close(manifest_fd)

    try:
        decoded = encoded.decode("utf-8", errors="strict")
        manifest = json.loads(
            decoded,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except BundleError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise BundleError(f"manifest is not canonical JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BundleError("manifest root must be an object")
    try:
        canonical = _canonical_json(manifest)
    except (RecursionError, TypeError, ValueError) as exc:
        raise BundleError(f"manifest cannot be represented canonically: {exc}") from exc
    if encoded != canonical:
        raise BundleError("manifest encoding is not canonical")
    return manifest, encoded


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise BundleError(f"{label} has invalid fields ({', '.join(details)})")


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    limits: BundleLimits,
    expected_digest: str | None,
) -> list[dict[str, Any]]:
    _require_exact_keys(
        manifest,
        {
            "bundle_sha256",
            "file_count",
            "files",
            "hash_algorithm",
            "schema",
            "total_size",
            "version",
        },
        label="manifest",
    )
    if manifest["schema"] != SCHEMA:
        raise BundleError(f"unsupported manifest schema: {manifest['schema']!r}")
    if type(manifest["version"]) is not int or manifest["version"] != SCHEMA_VERSION:
        raise BundleError(f"unsupported manifest version: {manifest['version']!r}")
    if manifest["hash_algorithm"] != HASH_ALGORITHM:
        raise BundleError(f"unsupported hash algorithm: {manifest['hash_algorithm']!r}")

    digest = manifest["bundle_sha256"]
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise BundleError("bundle_sha256 must be a lowercase SHA-256 digest")
    try:
        calculated_digest = _descriptor_digest(manifest)
    except (RecursionError, TypeError, ValueError) as exc:
        raise BundleError(f"manifest descriptor is invalid: {exc}") from exc
    if not hmac.compare_digest(digest, calculated_digest):
        raise BundleError("manifest digest does not match its contents")
    if expected_digest is not None:
        if (
            not isinstance(expected_digest, str)
            or _SHA256_RE.fullmatch(expected_digest) is None
        ):
            raise BundleError("expected digest must be a lowercase SHA-256 digest")
        if not hmac.compare_digest(digest, expected_digest):
            raise BundleError(
                f"bundle digest mismatch: expected {expected_digest}, found {digest}"
            )

    files = manifest["files"]
    if not isinstance(files, list):
        raise BundleError("manifest files must be an array")
    if len(files) > limits.max_files:
        raise BundleError(
            f"manifest exceeds file-count limit ({limits.max_files} files)"
        )
    if type(manifest["file_count"]) is not int or manifest["file_count"] != len(files):
        raise BundleError("manifest file_count is inconsistent")
    if type(manifest["total_size"]) is not int or manifest["total_size"] < 0:
        raise BundleError("manifest total_size must be a non-negative integer")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_size = 0
    previous_path: str | None = None
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise BundleError(f"manifest file entry {index} must be an object")
        _require_exact_keys(
            entry,
            {"mode", "path", "sha256", "size"},
            label=f"manifest file entry {index}",
        )
        path = entry["path"]
        parts = _validate_relative_path(path)
        _validate_import_safe_artifact(parts)
        if path in seen:
            raise BundleError(f"manifest contains duplicate artifact path: {path}")
        if previous_path is not None and path <= previous_path:
            raise BundleError("manifest artifact paths are not in canonical order")
        seen.add(path)
        previous_path = path

        size = entry["size"]
        if type(size) is not int or size < 0:
            raise BundleError(f"artifact size must be a non-negative integer: {path}")
        if size > limits.max_file_bytes:
            raise BundleError(
                f"artifact exceeds per-file limit ({size} > {limits.max_file_bytes}): {path}"
            )
        total_size += size
        if total_size > limits.max_total_bytes:
            raise BundleError(
                f"manifest exceeds total-size limit ({limits.max_total_bytes} bytes)"
            )

        mode = entry["mode"]
        if type(mode) is not int or mode not in {0o644, 0o755}:
            raise BundleError(f"artifact mode is invalid: {path}")
        sha256 = entry["sha256"]
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise BundleError(f"artifact SHA-256 is invalid: {path}")
        normalized.append({"mode": mode, "path": path, "sha256": sha256, "size": size})

    if SOLUTION_PATH not in seen:
        raise BundleError(
            f"required artifact is missing from manifest: {SOLUTION_PATH}"
        )
    if manifest["total_size"] != total_size:
        raise BundleError("manifest total_size is inconsistent")
    return normalized


def _inspect_bundle_root(bundle_fd: int) -> None:
    try:
        with os.scandir(bundle_fd) as iterator:
            entries = []
            for entry in iterator:
                if len(entries) == 2:
                    raise BundleError("bundle root has extra entries")
                entries.append(entry)
    except OSError as exc:
        raise BundleError(f"cannot inspect bundle root: {exc.strerror}") from exc
    names: set[str] = set()
    for entry in entries:
        _validate_component(entry.name)
        if entry.name in names:
            raise BundleError(f"duplicate bundle entry: {entry.name}")
        names.add(entry.name)
        metadata = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise BundleError(f"symbolic links are not allowed in bundle: {entry.name}")
        if entry.name == MANIFEST_NAME:
            if not stat.S_ISREG(metadata.st_mode):
                raise BundleError(f"{MANIFEST_NAME} is not a regular file")
            if metadata.st_nlink != 1:
                raise BundleError(f"hard-linked {MANIFEST_NAME} is not allowed")
            if stat.S_IMODE(metadata.st_mode) != 0o644:
                raise BundleError(f"{MANIFEST_NAME} must have mode 0644")
        if entry.name == PAYLOAD_DIR and not stat.S_ISDIR(metadata.st_mode):
            raise BundleError(f"{PAYLOAD_DIR} is not a directory")
    expected = {MANIFEST_NAME, PAYLOAD_DIR}
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        raise BundleError(
            f"bundle root is incomplete or has extra entries: missing={missing}, extra={extra}"
        )


def _open_payload_fd(bundle_fd: int) -> int:
    return _open_directory(
        PAYLOAD_DIR,
        label="bundle payload",
        dir_fd=bundle_fd,
    )


def _read_and_validate_bundle_fd(
    bundle_fd: int,
    *,
    limits: BundleLimits,
    expected_digest: str | None,
    copy_to: Path | None = None,
) -> dict[str, Any]:
    bundle_before = os.fstat(bundle_fd)
    _inspect_bundle_root(bundle_fd)
    manifest, _encoded = _read_manifest_fd(bundle_fd)
    expected_entries = _validate_manifest(
        manifest,
        limits=limits,
        expected_digest=expected_digest,
    )
    payload_fd = _open_payload_fd(bundle_fd)
    actual_directories: set[str] = set()
    try:
        actual_entries = _capture_tree_fd(
            payload_fd,
            copy_to,
            excluded_root_names=frozenset(),
            limits=limits,
            normalize_modes=False,
            observed_directories=actual_directories,
        )
    finally:
        os.close(payload_fd)
    if not _same_file(bundle_before, os.fstat(bundle_fd)):
        raise BundleError("bundle root changed while it was being verified")

    actual_entries.sort(key=lambda entry: entry["path"])
    expected_directories = {
        "/".join(parts[:index])
        for entry in expected_entries
        for parts in [_validate_relative_path(entry["path"])]
        for index in range(1, len(parts))
    }
    if actual_directories != expected_directories:
        missing_directories = sorted(expected_directories - actual_directories)
        extra_directories = sorted(actual_directories - expected_directories)
        raise BundleError(
            "bundle directory layout does not match manifest: "
            f"missing={missing_directories}, extra={extra_directories}"
        )
    if actual_entries != expected_entries:
        expected_by_path = {entry["path"]: entry for entry in expected_entries}
        actual_by_path = {entry["path"]: entry for entry in actual_entries}
        missing = sorted(expected_by_path.keys() - actual_by_path.keys())
        extra = sorted(actual_by_path.keys() - expected_by_path.keys())
        changed = sorted(
            path
            for path in expected_by_path.keys() & actual_by_path.keys()
            if expected_by_path[path] != actual_by_path[path]
        )
        raise BundleError(
            "bundle payload does not match manifest: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    return manifest


def _read_and_validate_bundle(
    bundle_dir: Path,
    *,
    limits: BundleLimits,
    expected_digest: str | None,
    copy_to: Path | None = None,
) -> dict[str, Any]:
    bundle_fd = _open_directory(bundle_dir, label="bundle")
    try:
        return _read_and_validate_bundle_fd(
            bundle_fd,
            limits=limits,
            expected_digest=expected_digest,
            copy_to=copy_to,
        )
    finally:
        os.close(bundle_fd)


def verify_bundle(
    bundle_dir: str | os.PathLike[str],
    *,
    limits: BundleLimits = BundleLimits(),
    expected_digest: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify manifest structure and every byte in a bundle payload."""

    expected_digest = _coalesce_expected_digest(expected_digest, expected_sha256)
    return _read_and_validate_bundle(
        Path(bundle_dir),
        limits=limits,
        expected_digest=expected_digest,
    )


def extract_bundle(
    bundle_dir: str | os.PathLike[str],
    destination_dir: str | os.PathLike[str],
    *,
    limits: BundleLimits = BundleLimits(),
    expected_digest: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify and atomically extract a bundle into a new directory."""

    bundle = Path(bundle_dir)
    expected_digest = _coalesce_expected_digest(expected_digest, expected_sha256)
    destination = Path(os.path.abspath(os.fspath(destination_dir)))
    _reject_overlap(bundle, destination, labels=("bundle", "extraction destination"))
    destination = _prepare_destination(destination, label="extraction destination")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=os.fspath(destination.parent),
        )
    )
    published = False
    try:
        manifest = _read_and_validate_bundle(
            bundle,
            limits=limits,
            expected_digest=expected_digest,
            copy_to=staging,
        )
        _fsync_directories(staging)
        _atomic_publish_directory(staging, destination)
        published = True
        _fsync_directory(destination.parent)
        return manifest
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _is_grandfathered_legacy_run(run_id: str) -> bool:
    """Return whether ``run_id`` predates the immutable-bundle cutover."""

    match = _RUN_ID_RE.fullmatch(run_id)
    if match is None:
        return False
    try:
        timestamp = datetime.strptime(match.group("timestamp"), "%Y%m%d_%H%M%S")
        cutover = datetime.strptime(BUNDLE_CUTOVER_RUN_ID, "%Y%m%d_%H%M%S")
    except ValueError:
        return False
    return timestamp < cutover


def _strict_result_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"result.json contains duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_result_json_constant(value: str) -> None:
    raise BundleError(f"result.json contains invalid JSON number: {value}")


def _load_run_result_fd(run_fd: int) -> dict[str, Any]:
    result_fd: int | None = None
    try:
        try:
            observed = os.stat(
                "result.json",
                dir_fd=run_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise BundleError(
                f"cannot inspect archived result.json: {exc.strerror}"
            ) from exc
        if not stat.S_ISREG(observed.st_mode):
            raise BundleError(
                "archived result.json must be a regular file, "
                f"found {_entry_kind(observed.st_mode)}"
            )
        if observed.st_nlink != 1:
            raise BundleError("hard-linked archived result.json is not allowed")
        if observed.st_size > MAX_RESULT_BYTES:
            raise BundleError(
                f"archived result.json exceeds limit ({MAX_RESULT_BYTES} bytes)"
            )

        try:
            result_fd = os.open("result.json", _OPEN_FILE_FLAGS, dir_fd=run_fd)
        except OSError as exc:
            raise BundleError(
                f"cannot safely open archived result.json: {exc.strerror}"
            ) from exc
        before = os.fstat(result_fd)
        if not stat.S_ISREG(before.st_mode) or not _same_file(observed, before):
            raise BundleError("archived result.json changed while being opened")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(
                result_fd,
                min(1024 * 1024, MAX_RESULT_BYTES + 1 - size),
            )
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_RESULT_BYTES:
                raise BundleError(
                    f"archived result.json exceeds limit ({MAX_RESULT_BYTES} bytes)"
                )
        after = os.fstat(result_fd)
        if size != before.st_size or not _same_file(before, after):
            raise BundleError("archived result.json changed while being read")
    finally:
        if result_fd is not None:
            os.close(result_fd)

    try:
        decoded = b"".join(chunks).decode("utf-8", errors="strict")
        loaded = json.loads(
            decoded,
            object_pairs_hook=_strict_result_object,
            parse_constant=_reject_result_json_constant,
        )
    except BundleError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise BundleError(f"cannot decode archived result.json: {exc}") from exc
    if not isinstance(loaded, dict):
        raise BundleError("archived result.json root must be an object")
    return loaded


def load_run_result(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a bounded regular result.json relative to a pinned run directory."""

    root = Path(run_dir)
    run_fd = _open_directory(root, label="run directory")
    run_before = os.fstat(run_fd)
    try:
        result = _load_run_result_fd(run_fd)
        if not _same_file(run_before, os.fstat(run_fd)):
            raise BundleError("run directory changed while result.json was being read")
        return result
    finally:
        os.close(run_fd)


def _validate_result_metrics(
    result: dict[str, Any],
) -> bool:
    correct = result.get("correct")
    peak_fraction = result.get("peak_fraction")
    if type(correct) is not bool:
        raise BundleError("result.json correct must be a boolean")
    if correct is False:
        if peak_fraction is not None:
            raise BundleError("incorrect result must not carry a peak_fraction")
    elif (
        type(peak_fraction) not in (int, float)
        or not 0 <= peak_fraction <= 100
        or not math.isfinite(peak_fraction)
    ):
        raise BundleError(
            "correct result must carry a finite numeric peak_fraction in [0, 100]"
        )
    return correct


def _verify_run_provenance_fd(
    root: Path,
    run_fd: int,
    result: dict[str, Any],
    *,
    allow_legacy: bool,
    enforce_cutover: bool,
    limits: BundleLimits,
) -> dict[str, Any] | None:
    recorded_run_id = result.get("run_id")
    if not isinstance(recorded_run_id, str):
        raise BundleError("result.json run_id must be a string")
    if recorded_run_id != root.name:
        raise BundleError("result.json run_id does not match its archive directory")

    regrade: dict[str, Any] | None = None
    if "regrade" in result:
        candidate_regrade = result["regrade"]
        if not isinstance(candidate_regrade, dict):
            raise BundleError("regrade metadata must be an object when present")
        regrade = candidate_regrade

    bundle_metadata = result.get("submission_bundle")
    replay_metadata = result.get("submission_replay")
    flat_digest = result.get("submission_bundle_sha256")
    flat_status = result.get("submission_replay_status")
    is_bundle_aware = any(key in result for key in _BUNDLE_PROVENANCE_KEYS)
    if not is_bundle_aware:
        if allow_legacy and (
            not enforce_cutover or _is_grandfathered_legacy_run(recorded_run_id)
        ):
            _validate_result_metrics(result)
            return None
        raise BundleError("immutable submission metadata is required for this run")

    has_solution = result.get("has_solution")
    if type(has_solution) is not bool:
        raise BundleError("bundle-aware result.json has_solution must be a boolean")
    correct = _validate_result_metrics(result)

    if bundle_metadata is not None and not isinstance(bundle_metadata, dict):
        raise BundleError("submission_bundle metadata must be an object or null")
    if replay_metadata is not None and not isinstance(replay_metadata, dict):
        raise BundleError("submission_replay metadata must be an object or null")

    nested_digest = (
        bundle_metadata.get("bundle_sha256")
        if isinstance(bundle_metadata, dict)
        else None
    )
    replay_digest = (
        replay_metadata.get("bundle_sha256")
        if isinstance(replay_metadata, dict)
        else None
    )
    digests = [
        value
        for value in (flat_digest, nested_digest, replay_digest)
        if value is not None
    ]
    if digests and any(value != digests[0] for value in digests[1:]):
        raise BundleError("archived bundle digests disagree")
    digest = digests[0] if digests else None

    nested_status = (
        replay_metadata.get("status") if isinstance(replay_metadata, dict) else None
    )
    if (
        flat_status is not None
        and nested_status is not None
        and flat_status != nested_status
    ):
        raise BundleError("archived replay statuses disagree")
    status = flat_status if flat_status is not None else nested_status

    # A run where the agent produced no solution cannot be bundled or graded,
    # but remains useful failure telemetry and is safe to surface as such.
    if status == "not_applicable" and has_solution is False and digest is None:
        return None
    if status != VERIFIED_REPLAY_STATUS:
        raise BundleError(f"submission replay is not verified: {status!r}")
    if has_solution is not True:
        raise BundleError("verified replay must record has_solution=true")
    if digest is None:
        raise BundleError("verified replay is missing its bundle digest")
    if not isinstance(bundle_metadata, dict) or not isinstance(replay_metadata, dict):
        raise BundleError(
            "verified replay requires complete bundle and replay metadata"
        )
    if nested_digest != digest or replay_digest != digest:
        raise BundleError(
            "verified replay metadata is not bound to the recorded digest"
        )
    if (
        bundle_metadata.get("path") != RUN_BUNDLE_DIR
        or bundle_metadata.get("schema") != SCHEMA
        or bundle_metadata.get("version") != SCHEMA_VERSION
    ):
        raise BundleError("verified replay has invalid bundle schema metadata")
    for field in ("fresh_extraction", "fresh_caches"):
        if replay_metadata.get(field) is not True:
            raise BundleError(f"verified replay did not record {field}=true")
    for field in (
        "network_isolated",
        "mount_isolated",
        "root_isolated",
        "pid_isolated",
        "clean_environment",
    ):
        if replay_metadata.get(field) is not True:
            raise BundleError(f"verified replay did not record {field}=true")
    if (
        replay_metadata.get("network_isolation")
        != "unshare-user-mount-pid-net-private-root-v1"
    ):
        raise BundleError("verified replay did not use the full isolation backend")
    replay_stage_count = replay_metadata.get("stage_count")
    if type(replay_stage_count) is not int or replay_stage_count not in {1, 2}:
        raise BundleError("verified replay has an invalid stage_count")
    surface_digest = replay_metadata.get("grader_surface_sha256")
    if (
        not isinstance(surface_digest, str)
        or _SHA256_RE.fullmatch(surface_digest) is None
    ):
        raise BundleError("verified replay is missing grader_surface_sha256")

    stage_count = replay_stage_count
    check_exit_code = result.get("check_exit_code")
    benchmark_exit_code = result.get("benchmark_exit_code")
    if regrade is not None:
        regrade_digest = regrade.get("submission_bundle_sha256")
        # Legacy regrades on grandfathered archives remain readable.  Once a
        # run is bundle-aware, however, the active top-level metrics must be
        # bound to the same bundle and the same isolation guarantees.
        if digest is not None:
            if regrade.get("status") != VERIFIED_REPLAY_STATUS:
                raise BundleError("active regrade is not verified")
            if regrade_digest != digest:
                raise BundleError("active regrade is bound to a different bundle")
            for field in (
                "fresh_extraction",
                "fresh_caches",
                "network_isolated",
                "mount_isolated",
                "root_isolated",
                "pid_isolated",
                "clean_environment",
            ):
                if regrade.get(field) is not True:
                    raise BundleError(f"active regrade did not record {field}=true")
            if (
                regrade.get("network_isolation")
                != "unshare-user-mount-pid-net-private-root-v1"
            ):
                raise BundleError(
                    "active regrade did not use the full isolation backend"
                )
            regrade_surface = regrade.get("grader_surface_sha256")
            if (
                not isinstance(regrade_surface, str)
                or _SHA256_RE.fullmatch(regrade_surface) is None
            ):
                raise BundleError("active regrade is missing grader_surface_sha256")
            regrade_stage_count = regrade.get("stage_count")
            if type(regrade_stage_count) is not int or regrade_stage_count not in {
                1,
                2,
            }:
                raise BundleError("active regrade has an invalid stage_count")
            regrade_check_exit = regrade.get("check_exit_code")
            if type(regrade_check_exit) is not int:
                raise BundleError("active regrade has an invalid check_exit_code")
            regrade_benchmark_exit = regrade.get("benchmark_exit_code")
            if regrade_stage_count == 1:
                if regrade_benchmark_exit is not None:
                    raise BundleError(
                        "one-stage active regrade must not carry a benchmark_exit_code"
                    )
            elif type(regrade_benchmark_exit) is not int:
                raise BundleError("active regrade has an invalid benchmark_exit_code")
            stage_count = regrade_stage_count
            check_exit_code = regrade_check_exit
            benchmark_exit_code = regrade_benchmark_exit

    bundle_fd = _open_directory(
        RUN_BUNDLE_DIR,
        label="archived submission bundle",
        dir_fd=run_fd,
    )
    try:
        manifest = _read_and_validate_bundle_fd(
            bundle_fd,
            limits=limits,
            expected_digest=digest,
        )
    finally:
        os.close(bundle_fd)
    if correct:
        if stage_count != 2:
            raise BundleError("scored replay must complete two isolated stages")
        if type(check_exit_code) is not int or check_exit_code != 0:
            raise BundleError(
                "scored replay is not publishable without check_exit_code=0"
            )
        if type(benchmark_exit_code) is not int or benchmark_exit_code != 0:
            raise BundleError(
                "scored replay is not publishable without benchmark_exit_code=0"
            )
    return manifest


def verify_run_provenance(
    run_dir: str | os.PathLike[str],
    result: dict[str, Any] | None = None,
    *,
    allow_legacy: bool = True,
    enforce_cutover: bool = True,
    limits: BundleLimits = BundleLimits(),
) -> dict[str, Any] | None:
    """Verify the immutable submission attached to an archived run.

    ``None`` classifies an explicitly bundle-less failed attempt, or a result
    admitted by the operational legacy policy.  The timestamp cutoff is not
    proof of when an archive was created: callers enforcing provenance must
    also keep archive names and contents on trusted immutable storage.  A
    caller may deliberately disable the cutoff for a runner that has not
    adopted bundles.

    Once any bundle or replay metadata key is present, partial or null metadata
    never falls back to legacy.  The run directory stays pinned while
    ``result.json`` and ``submission_bundle`` are opened relative to it.  A
    supplied ``result`` is checked against that safely read archive snapshot;
    it never replaces the on-disk read.
    """

    if type(allow_legacy) is not bool:
        raise BundleError("allow_legacy must be a boolean")
    if type(enforce_cutover) is not bool:
        raise BundleError("enforce_cutover must be a boolean")

    root = Path(run_dir)
    run_fd = _open_directory(root, label="run directory")
    run_before = os.fstat(run_fd)
    try:
        archived_result = _load_run_result_fd(run_fd)
        if result is not None and not isinstance(result, dict):
            raise BundleError("archived result.json root must be an object")
        if result is not None and result != archived_result:
            raise BundleError(
                "provided result does not match the pinned archived result.json"
            )
        verified = _verify_run_provenance_fd(
            root,
            run_fd,
            archived_result,
            allow_legacy=allow_legacy,
            enforce_cutover=enforce_cutover,
            limits=limits,
        )
        if not _same_file(run_before, os.fstat(run_fd)):
            raise BundleError("run directory changed while provenance was verified")
        return verified
    finally:
        os.close(run_fd)


def _limits_from_args(args: argparse.Namespace) -> BundleLimits:
    try:
        return BundleLimits(
            max_files=args.max_files,
            max_directories=args.max_directories,
            max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_total_bytes,
        )
    except ValueError as exc:
        raise BundleError(str(exc)) from exc


def _coalesce_expected_digest(
    expected_digest: str | None,
    expected_sha256: str | None,
) -> str | None:
    if (
        expected_digest is not None
        and expected_sha256 is not None
        and expected_digest != expected_sha256
    ):
        raise BundleError("expected_digest and expected_sha256 disagree")
    return expected_digest if expected_digest is not None else expected_sha256


def _add_limit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument("--max-directories", type=int, default=DEFAULT_MAX_DIRECTORIES)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)


def _summary(manifest: dict[str, Any]) -> str:
    return json.dumps(
        {
            "bundle_sha256": manifest["bundle_sha256"],
            "file_count": manifest["file_count"],
            "total_size": manifest["total_size"],
        },
        sort_keys=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="freeze a problem directory")
    create.add_argument("source", type=Path, help="agent-authored problem directory")
    create.add_argument("bundle", type=Path, help="new bundle directory")
    create.add_argument(
        "--template",
        action="append",
        default=[],
        metavar="NAME",
        help="additional root-level benchmark template to exclude",
    )
    _add_limit_arguments(create)

    verify = commands.add_parser("verify", help="verify a complete bundle")
    verify.add_argument("bundle", type=Path)
    verify.add_argument(
        "--expected-digest", "--expected-sha256", dest="expected_digest"
    )
    _add_limit_arguments(verify)

    extract = commands.add_parser("extract", help="verify and extract a bundle")
    extract.add_argument("bundle", type=Path)
    extract.add_argument("destination", type=Path, help="new extraction directory")
    extract.add_argument(
        "--expected-digest", "--expected-sha256", dest="expected_digest"
    )
    _add_limit_arguments(extract)

    gate = commands.add_parser(
        "verify-run",
        help="verify a run's recorded replay status and immutable bundle",
    )
    gate.add_argument("run", type=Path)
    gate.add_argument(
        "--require-bundle",
        action="store_true",
        help="reject historical result.json files without bundle metadata",
    )
    _add_limit_arguments(gate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        limits = _limits_from_args(args)
        if args.command == "create":
            templates = DEFAULT_TEMPLATE_NAMES | frozenset(args.template)
            manifest = create_bundle(
                args.source,
                args.bundle,
                template_names=templates,
                limits=limits,
            )
        elif args.command == "verify":
            manifest = verify_bundle(
                args.bundle,
                limits=limits,
                expected_digest=args.expected_digest,
            )
        elif args.command == "extract":
            manifest = extract_bundle(
                args.bundle,
                args.destination,
                limits=limits,
                expected_digest=args.expected_digest,
            )
        else:
            manifest = verify_run_provenance(
                args.run,
                allow_legacy=not args.require_bundle,
                limits=limits,
            )
            if manifest is None:
                print(
                    json.dumps({"status": "legacy_or_not_applicable"}, sort_keys=True)
                )
                return 0
    except (BundleError, OSError, ValueError) as exc:
        print(f"submission bundle error: {exc}", file=sys.stderr)
        return 2
    print(_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
