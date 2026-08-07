#!/usr/bin/env python3
"""Capture and replay a small, digest-verified submission bundle.

The bundle is manifest.json plus a path-preserving files/ tree. The manifest
is compact, sorted-key JSON with one trailing newline; its exact bytes are the
bundle identity. File modes are normalized to 0644 or 0755, preserving only
whether the source had any execute bit set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FORMAT = 1
MANIFEST = "manifest.json"
PAYLOAD = "files"
SOLUTION = "solution.py"
DERIVED_REPORTS = ("framework.txt", "cuda_language.json")
MAX_FILES = 2_048
MAX_DIRECTORIES = 2_048
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_REPORT_BYTES = 1024 * 1024
MAX_PATH_BYTES = 4_096
MAX_COMPONENT_BYTES = 255
MAX_DEPTH = 64
CHUNK = 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DRIVE = re.compile(r"[A-Za-z]:")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _NOFOLLOW | _NONBLOCK
_FILE_FLAGS = os.O_RDONLY | _NOFOLLOW | _NONBLOCK


class BundleError(RuntimeError):
    """The requested bundle operation is unsafe or invalid."""


@dataclass(frozen=True)
class Blob:
    path: str
    mode: int
    data: bytes

    @property
    def entry(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "path": self.path,
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "size": len(self.data),
        }


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _parts(path: object) -> tuple[str, ...]:
    if type(path) is not str or not path:
        raise BundleError("file path must be a non-empty string")
    if path.startswith("/") or "\\" in path or "\0" in path or _DRIVE.match(path):
        raise BundleError(f"unsafe file path: {path!r}")
    try:
        encoded = path.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise BundleError(f"file path is not valid UTF-8: {path!r}") from exc
    pieces = tuple(path.split("/"))
    if len(encoded) > MAX_PATH_BYTES or len(pieces) > MAX_DEPTH:
        raise BundleError(f"file path exceeds limits: {path!r}")
    for piece in pieces:
        if piece in {"", ".", ".."} or "\0" in piece or "\\" in piece:
            raise BundleError(f"unsafe file path: {path!r}")
        if len(piece.encode("utf-8")) > MAX_COMPONENT_BYTES:
            raise BundleError(f"path component is too long: {path!r}")
    if "__pycache__" in pieces or pieces[-1].endswith((".pyc", ".pyo")):
        raise BundleError(f"Python bytecode artifacts are forbidden: {path}")
    return pieces


def _exclude_names(names: Iterable[str]) -> frozenset[str]:
    result: set[str] = set()
    for name in names:
        if type(name) is not str or not name or "/" in name or "\\" in name:
            raise BundleError(f"--exclude must name one top-level entry: {name!r}")
        _parts(name)
        if name == SOLUTION:
            raise BundleError(f"cannot exclude required file {SOLUTION}")
        result.add(name)
    return frozenset(result)


def _same(before: os.stat_result, after: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(before, field) == getattr(after, field) for field in fields)


def _open_directory(path: Path, label: str) -> int:
    """Open every path component without following symbolic links."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    pieces = absolute.parts
    try:
        fd = os.open(pieces[0], _DIR_FLAGS)
    except OSError as exc:
        raise BundleError(f"cannot open {label} {absolute}: {exc.strerror}") from exc
    try:
        for piece in pieces[1:]:
            next_fd = os.open(piece, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise BundleError(f"{label} is not a directory: {absolute}")
        return fd
    except (OSError, BundleError) as exc:
        os.close(fd)
        if isinstance(exc, BundleError):
            raise
        raise BundleError(
            f"cannot safely open {label} {absolute}: {exc.strerror}"
        ) from exc


def read_regular(path: Path, max_bytes: int) -> bytes:
    """Boundedly read one singly-linked regular file without following links."""
    path = Path(os.path.abspath(os.fspath(path)))
    if max_bytes < 0 or not path.name:
        raise BundleError("invalid bounded-read request")
    parent_fd = _open_directory(path.parent, "file parent")
    file_fd: int | None = None
    try:
        file_fd = os.open(path.name, _FILE_FLAGS, dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise BundleError(f"not one regular file: {path}")
        if before.st_size > max_bytes:
            raise BundleError(f"file exceeds {max_bytes} bytes: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(file_fd, min(CHUNK, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise BundleError(f"file exceeds {max_bytes} bytes: {path}")
        if size != before.st_size or not _same(before, os.fstat(file_fd)):
            raise BundleError(f"file changed while being read: {path}")
        return b"".join(chunks)
    except OSError as exc:
        raise BundleError(f"cannot safely read {path}: {exc.strerror}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _collect(
    root_fd: int,
    *,
    excludes: frozenset[str] = frozenset(),
    exact_modes: bool = False,
) -> tuple[list[Blob], set[str]]:
    blobs: list[Blob] = []
    directories: set[str] = set()
    total = 0
    directory_count = 0
    raw_count = 0

    def visit(directory_fd: int, prefix: tuple[str, ...]) -> None:
        nonlocal total, directory_count, raw_count
        before_dir = os.fstat(directory_fd)
        try:
            with os.scandir(directory_fd) as iterator:
                items = []
                for item in iterator:
                    raw_count += 1
                    if raw_count > MAX_FILES + MAX_DIRECTORIES + len(excludes):
                        raise BundleError("submission exceeds entry-count limit")
                    if not exact_modes and (
                        item.name == "__pycache__"
                        or item.name.endswith((".pyc", ".pyo"))
                    ):
                        continue
                    items.append((item.name, item.stat(follow_symlinks=False)))
        except OSError as exc:
            shown = "/".join(prefix) or "."
            raise BundleError(f"cannot scan {shown}: {exc.strerror}") from exc
        items.sort(key=lambda pair: pair[0])
        for name, observed in items:
            relative = "/".join((*prefix, name))
            parts = _parts(relative)
            if not prefix and name in excludes:
                continue
            if stat.S_ISLNK(observed.st_mode):
                raise BundleError(f"symbolic links are forbidden: {relative}")
            if not stat.S_ISDIR(observed.st_mode) and not stat.S_ISREG(
                observed.st_mode
            ):
                raise BundleError(f"special files are forbidden: {relative}")
            if stat.S_ISREG(observed.st_mode) and observed.st_nlink != 1:
                raise BundleError(f"hard-linked files are forbidden: {relative}")
            if stat.S_ISDIR(observed.st_mode):
                directory_count += 1
                if directory_count > MAX_DIRECTORIES:
                    raise BundleError(
                        f"submission exceeds {MAX_DIRECTORIES} directories"
                    )
                directories.add(relative)
                try:
                    child_fd = os.open(name, _DIR_FLAGS, dir_fd=directory_fd)
                except OSError as exc:
                    raise BundleError(
                        f"cannot safely open directory {relative}: {exc.strerror}"
                    ) from exc
                try:
                    opened = os.fstat(child_fd)
                    if not stat.S_ISDIR(opened.st_mode) or (
                        opened.st_dev,
                        opened.st_ino,
                    ) != (observed.st_dev, observed.st_ino):
                        raise BundleError(
                            f"directory changed while scanning: {relative}"
                        )
                    visit(child_fd, parts)
                finally:
                    os.close(child_fd)
                continue
            if len(blobs) >= MAX_FILES:
                raise BundleError(f"submission exceeds {MAX_FILES} files")
            try:
                file_fd = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
            except OSError as exc:
                raise BundleError(
                    f"cannot safely open file {relative}: {exc.strerror}"
                ) from exc
            try:
                opened = os.fstat(file_fd)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino)
                    != (observed.st_dev, observed.st_ino)
                ):
                    raise BundleError(f"file changed while scanning: {relative}")
                if opened.st_size > MAX_FILE_BYTES:
                    raise BundleError(
                        f"file exceeds {MAX_FILE_BYTES} bytes: {relative}"
                    )
                chunks: list[bytes] = []
                size = 0
                while True:
                    chunk = os.read(file_fd, min(CHUNK, MAX_FILE_BYTES + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
                        raise BundleError("submission exceeds byte limits")
                if size != opened.st_size or not _same(opened, os.fstat(file_fd)):
                    raise BundleError(f"file changed while reading: {relative}")
                actual_mode = stat.S_IMODE(opened.st_mode)
                if exact_modes:
                    if actual_mode not in {0o644, 0o755}:
                        raise BundleError(f"non-canonical file mode for {relative}")
                    mode = actual_mode
                else:
                    mode = 0o755 if opened.st_mode & 0o111 else 0o644
                data = b"".join(chunks)
            finally:
                os.close(file_fd)
            total += len(data)
            blobs.append(Blob(relative, mode, data))
        if not _same(before_dir, os.fstat(directory_fd)):
            shown = "/".join(prefix) or "."
            raise BundleError(f"directory changed while scanning: {shown}")

    visit(root_fd, ())
    blobs.sort(key=lambda blob: blob.path)
    return blobs, directories


def _parse_manifest(raw: bytes, expect: str | None) -> tuple[dict[str, object], str]:
    digest = hashlib.sha256(raw).hexdigest()
    if expect is not None:
        if _SHA256.fullmatch(expect) is None:
            raise BundleError("--expect must be a lowercase SHA-256 digest")
        if digest != expect:
            raise BundleError(
                f"manifest digest mismatch: expected {expect}, found {digest}"
            )
    try:
        manifest = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise BundleError(f"manifest is not valid JSON: {exc}") from exc
    if type(manifest) is not dict or set(manifest) != {"format", "files"}:
        raise BundleError("manifest must contain exactly 'format' and 'files'")
    if type(manifest["format"]) is not int or manifest["format"] != FORMAT:
        raise BundleError(f"unsupported manifest format: {manifest['format']!r}")
    if type(manifest["files"]) is not list:
        raise BundleError("manifest files must be an array")
    if len(manifest["files"]) > MAX_FILES:
        raise BundleError(f"manifest exceeds {MAX_FILES} files")
    previous: str | None = None
    paths: set[str] = set()
    total = 0
    for index, entry in enumerate(manifest["files"]):
        if type(entry) is not dict or set(entry) != {
            "mode",
            "path",
            "sha256",
            "size",
        }:
            raise BundleError(f"invalid manifest file entry {index}")
        parts = _parts(entry["path"])
        path = entry["path"]
        if previous is not None and path <= previous:
            raise BundleError("manifest file paths are not strictly sorted")
        previous = path
        if any("/".join(parts[:cut]) in paths for cut in range(1, len(parts))):
            raise BundleError(f"manifest path collision: {path}")
        paths.add(path)
        if type(entry["mode"]) is not int or entry["mode"] not in {0o644, 0o755}:
            raise BundleError(f"invalid file mode: {path}")
        if type(entry["size"]) is not int or not 0 <= entry["size"] <= MAX_FILE_BYTES:
            raise BundleError(f"invalid file size: {path}")
        if (
            type(entry["sha256"]) is not str
            or _SHA256.fullmatch(entry["sha256"]) is None
        ):
            raise BundleError(f"invalid file SHA-256: {path}")
        total += entry["size"]
        if total > MAX_TOTAL_BYTES:
            raise BundleError(f"manifest exceeds {MAX_TOTAL_BYTES} total bytes")
    if SOLUTION not in paths:
        raise BundleError(f"manifest is missing required file {SOLUTION}")
    try:
        canonical = _canonical(manifest)
    except (TypeError, ValueError, RecursionError) as exc:
        raise BundleError(f"manifest is not canonical JSON: {exc}") from exc
    if raw != canonical:
        raise BundleError("manifest encoding is not canonical")
    return manifest, digest


def _read_manifest(bundle_fd: int) -> bytes:
    try:
        fd = os.open(MANIFEST, _FILE_FLAGS, dir_fd=bundle_fd)
    except OSError as exc:
        raise BundleError(f"cannot open {MANIFEST}: {exc.strerror}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise BundleError(f"{MANIFEST} must be one regular, unlinked file")
        if stat.S_IMODE(before.st_mode) != 0o644:
            raise BundleError(f"{MANIFEST} must have mode 0644")
        if before.st_size > MAX_MANIFEST_BYTES:
            raise BundleError("manifest exceeds size limit")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(fd, min(CHUNK, MAX_MANIFEST_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_MANIFEST_BYTES:
                raise BundleError("manifest exceeds size limit")
        if size != before.st_size or not _same(before, os.fstat(fd)):
            raise BundleError("manifest changed while reading")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _load_bundle(bundle: Path, expect: str | None) -> tuple[str, list[Blob]]:
    bundle_fd = _open_directory(bundle, "bundle")
    try:
        root_before = os.fstat(bundle_fd)
        with os.scandir(bundle_fd) as iterator:
            root = {}
            for item in iterator:
                if len(root) >= 2:
                    raise BundleError(
                        "bundle root must contain exactly manifest.json and files/"
                    )
                root[item.name] = item.stat(follow_symlinks=False)
        if set(root) != {MANIFEST, PAYLOAD}:
            raise BundleError(
                "bundle root must contain exactly manifest.json and files/"
            )
        if stat.S_ISLNK(root[MANIFEST].st_mode) or not stat.S_ISREG(
            root[MANIFEST].st_mode
        ):
            raise BundleError("manifest.json must be a regular file")
        if stat.S_ISLNK(root[PAYLOAD].st_mode) or not stat.S_ISDIR(
            root[PAYLOAD].st_mode
        ):
            raise BundleError("files must be a real directory")
        manifest, digest = _parse_manifest(_read_manifest(bundle_fd), expect)
        try:
            payload_fd = os.open(PAYLOAD, _DIR_FLAGS, dir_fd=bundle_fd)
        except OSError as exc:
            raise BundleError(f"cannot safely open files/: {exc.strerror}") from exc
        try:
            opened = os.fstat(payload_fd)
            if (opened.st_dev, opened.st_ino) != (
                root[PAYLOAD].st_dev,
                root[PAYLOAD].st_ino,
            ):
                raise BundleError("files directory changed while opening")
            blobs, directories = _collect(payload_fd, exact_modes=True)
        finally:
            os.close(payload_fd)
        if not _same(root_before, os.fstat(bundle_fd)):
            raise BundleError("bundle directory changed while verifying")
    finally:
        os.close(bundle_fd)
    specs = {entry["path"]: entry for entry in manifest["files"]}
    actual = {blob.path: blob for blob in blobs}
    if set(specs) != set(actual):
        raise BundleError("bundle payload file set does not match manifest")
    expected_directories = {
        "/".join(parts[:cut])
        for path in specs
        for parts in [_parts(path)]
        for cut in range(1, len(parts))
    }
    if directories != expected_directories:
        raise BundleError("bundle payload directory set does not match manifest")
    for path, spec in specs.items():
        blob = actual[path]
        if (
            blob.mode != spec["mode"]
            or len(blob.data) != spec["size"]
            or hashlib.sha256(blob.data).hexdigest() != spec["sha256"]
        ):
            raise BundleError(f"bundle payload does not match manifest: {path}")
    return digest, blobs


def _write_file(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def _write_blobs(root: Path, blobs: Iterable[Blob]) -> None:
    for blob in blobs:
        _write_file(root.joinpath(*_parts(blob.path)), blob.data, blob.mode)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish a staged path into a harness-owned directory."""
    if os.path.lexists(destination):
        raise BundleError(f"destination already exists: {destination}")
    os.rename(source, destination)


def _new_destination(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if not path.name or os.path.lexists(path):
        raise BundleError(f"{label} already exists: {path}")
    parent_fd = _open_directory(path.parent, f"{label} parent")
    os.close(parent_fd)
    return path


def capture(source: Path, bundle: Path, excludes: Iterable[str] = ()) -> str:
    source_fd = _open_directory(source, "source")
    try:
        blobs, _ = _collect(source_fd, excludes=_exclude_names(excludes))
    finally:
        os.close(source_fd)
    if SOLUTION not in {blob.path for blob in blobs}:
        raise BundleError(f"required file is missing: {SOLUTION}")
    destination = _new_destination(bundle, "bundle")
    manifest = {"format": FORMAT, "files": [blob.entry for blob in blobs]}
    encoded = _canonical(manifest)
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise BundleError("manifest exceeds size limit")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    published = False
    try:
        os.chmod(staging, 0o755)
        payload = staging / PAYLOAD
        payload.mkdir(mode=0o755)
        os.chmod(payload, 0o755)
        _write_blobs(payload, blobs)
        _write_file(staging / MANIFEST, encoded, 0o644)
        _rename_noreplace(staging, destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return hashlib.sha256(encoded).hexdigest()


def verify(bundle: Path, expect: str | None = None) -> str:
    digest, _ = _load_bundle(bundle, expect)
    return digest


def load(bundle: Path, expect: str | None = None) -> tuple[str, dict[str, bytes]]:
    """Verify a bundle and return its exact files keyed by relative path."""
    digest, blobs = _load_bundle(bundle, expect)
    return digest, {blob.path: blob.data for blob in blobs}


def extract(bundle: Path, destination: Path, expect: str | None = None) -> str:
    digest, blobs = _load_bundle(bundle, expect)
    destination = _new_destination(destination, "extraction destination")
    bundle = Path(os.path.abspath(os.fspath(bundle)))
    if (
        destination == bundle
        or destination.is_relative_to(bundle)
        or bundle.is_relative_to(destination)
    ):
        raise BundleError("bundle and extraction destination must not overlap")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    published = False
    try:
        _write_blobs(staging, blobs)
        _rename_noreplace(staging, destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return digest


def project(
    bundle: Path,
    run_dir: Path,
    expect: str | None = None,
    reports_from: Path | None = None,
) -> str:
    digest, blobs = _load_bundle(bundle, expect)
    run_dir = Path(os.path.abspath(os.fspath(run_dir)))
    run_fd = _open_directory(run_dir, "run directory")
    os.close(run_fd)
    solution = next(blob for blob in blobs if blob.path == SOLUTION)
    sidecars = [blob for blob in blobs if blob.path != SOLUTION]
    staging = Path(
        tempfile.mkdtemp(prefix=f".{run_dir.name}.project-", dir=run_dir.parent)
    )
    try:
        _write_file(staging / SOLUTION, solution.data, solution.mode)
        for blob in sidecars:
            target = staging / "scratch"
            _write_file(target.joinpath(*_parts(blob.path)), blob.data, blob.mode)
        if reports_from is not None:
            for name in DERIVED_REPORTS:
                source = reports_from / name
                if not os.path.lexists(source):
                    continue
                data = read_regular(source, MAX_REPORT_BYTES)
                _write_file(staging / "scratch" / name, data, 0o644)
        for name in (SOLUTION, "scratch"):
            target = run_dir / name
            if os.path.lexists(target):
                raise BundleError(f"projection path already exists: {target}")
        for name in (SOLUTION, "scratch"):
            source = staging / name
            if os.path.lexists(source):
                _rename_noreplace(source, run_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture", help="capture a source tree")
    capture_parser.add_argument("source", type=Path)
    capture_parser.add_argument("bundle", type=Path)
    capture_parser.add_argument(
        "--exclude", action="append", default=[], metavar="NAME"
    )
    for name, help_text in (
        ("verify", "verify a bundle"),
        ("extract", "verify and extract a bundle"),
        ("project", "regenerate the legacy run view"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("bundle", type=Path)
        if name != "verify":
            command.add_argument("destination", type=Path)
        command.add_argument("--expect", metavar="DIGEST")
        if name == "project":
            command.add_argument("--reports-from", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            digest = capture(args.source, args.bundle, args.exclude)
        elif args.command == "verify":
            digest = verify(args.bundle, args.expect)
        elif args.command == "extract":
            digest = extract(args.bundle, args.destination, args.expect)
        else:
            digest = project(
                args.bundle, args.destination, args.expect, args.reports_from
            )
    except (BundleError, OSError) as exc:
        print(f"submission_bundle.py: error: {exc}", file=sys.stderr)
        return 1
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
