"""Open the exact submission bytes bound to a publishable run result."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterator, Mapping

from scripts import submission_bundle
from scripts.reward_hack_tripwires import executable_hack_hits


DEFAULT_MAX_PUBLISHED_FILE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_PUBLICATION_JSON_BYTES = 64 * 1024 * 1024
_OPEN_PARENT_FLAGS = (
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


@dataclass(frozen=True)
class SubmissionView:
    solution: Path
    root: Path
    bundled: bool


@dataclass(frozen=True)
class PublicationAnnotation:
    """The audit fields that authorize one bundle-era publication."""

    run_id: str
    verdict: str
    publish_grade: bool | None
    board_eligible: bool | None
    contamination_clean: bool
    text: str

    @property
    def publishable(self) -> bool:
        return (
            self.verdict == "clean"
            and self.publish_grade is True
            and self.board_eligible is not False
        )


def validate_run_id(value: object) -> str:
    """Return one safe archive-directory component from published metadata."""

    if not isinstance(value, str) or not value:
        raise submission_bundle.BundleError("published run_id must be a string")
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or len(value.encode("utf-8", errors="strict")) > 255
    ):
        raise submission_bundle.BundleError(f"unsafe published run_id: {value!r}")
    if submission_bundle._RUN_ID_RE.fullmatch(value) is None:
        raise submission_bundle.BundleError(f"invalid published run_id: {value!r}")
    return value


def load_required_published_run_ids(
    path: str | os.PathLike[str],
) -> frozenset[str]:
    """Load a nonempty, bounded, no-follow curation manifest."""

    if not os.fspath(path):
        raise OSError("a published-run curation manifest is required")
    try:
        manifest = json.loads(
            read_bounded_text(path, max_bytes=DEFAULT_MAX_PUBLICATION_JSON_BYTES),
            object_pairs_hook=_strict_publication_object,
            parse_constant=_reject_publication_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise OSError(f"invalid published-run manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("run_ids"), list):
        raise OSError("published-run manifest must contain a run_ids array")
    raw_ids = manifest["run_ids"]
    if not raw_ids:
        raise OSError("published-run manifest must contain at least one run_id")
    run_ids = [validate_run_id(run_id) for run_id in raw_ids]
    if len(set(run_ids)) != len(run_ids):
        raise OSError("published-run manifest contains duplicate run_ids")
    return frozenset(run_ids)


def _strict_publication_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OSError(f"publication JSON contains duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_publication_json_constant(value: str) -> None:
    raise OSError(f"publication JSON contains invalid number: {value}")


def _annotation_scalar(
    text: str,
    field: str,
    *,
    required: bool,
    unquote: bool = True,
) -> str | None:
    matches = re.findall(rf"^{re.escape(field)}:\s*(.*?)\s*$", text, re.MULTILINE)
    if len(matches) > 1:
        raise OSError(f"publication annotation contains duplicate {field}")
    if not matches:
        if required:
            raise OSError(f"publication annotation is missing {field}")
        return None
    value = re.split(r"\s+#", matches[0], maxsplit=1)[0].strip()
    if unquote and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    if not value:
        raise OSError(f"publication annotation has an empty {field}")
    return value


def _annotation_boolean(text: str, field: str) -> bool | None:
    value = _annotation_scalar(text, field, required=False, unquote=False)
    if value is None:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise OSError(f"publication annotation {field} must be true or false")


def read_publication_annotation(
    path: str | os.PathLike[str],
) -> PublicationAnnotation:
    """Read and validate one exact, independently owned audit annotation."""

    annotation_path = Path(path)
    expected_run_id = validate_run_id(annotation_path.stem)
    text = read_bounded_text(annotation_path, max_bytes=4 * 1024 * 1024)
    if re.search(r"^(?:---|\.\.\.)\s*(?:#.*)?$", text, re.MULTILINE):
        raise OSError("publication annotation must contain one implicit YAML document")
    recorded_run_id = validate_run_id(_annotation_scalar(text, "run_id", required=True))
    if recorded_run_id != expected_run_id:
        raise OSError("publication annotation run_id does not match its filename")
    verdict = _annotation_scalar(text, "verdict", required=True)
    assert verdict is not None
    return PublicationAnnotation(
        run_id=recorded_run_id,
        verdict=verdict,
        publish_grade=_annotation_boolean(text, "publish_grade"),
        board_eligible=_annotation_boolean(text, "board_eligible"),
        contamination_clean=bool(
            re.search(r"^contamination:\s*clean\b", text, re.MULTILINE)
        ),
        text=text,
    )


def require_publishable_annotations(
    selected_run_ids: Collection[str],
    bundle_aware_run_ids: Collection[str],
    annotations: Mapping[str, PublicationAnnotation],
) -> None:
    """Require explicit audit approval for every selected bundle-era solution."""

    bundle_aware = {validate_run_id(run_id) for run_id in bundle_aware_run_ids}
    for run_id in selected_run_ids:
        run_id = validate_run_id(run_id)
        if run_id not in bundle_aware:
            continue
        annotation = annotations.get(run_id)
        if annotation is None:
            raise submission_bundle.BundleError(
                f"selected bundle-aware run is missing its audit annotation: {run_id}"
            )
        if annotation.run_id != run_id or not annotation.publishable:
            raise submission_bundle.BundleError(
                f"selected bundle-aware run lacks explicit publication approval: "
                f"{run_id}"
            )


def reject_bundle_static_hacks(run_id: str, source: str) -> None:
    """Veto high-confidence checker-control patterns in bundle-era source.

    The normal-return receipt is advisory because submission code shares the
    checker interpreter.  This source gate therefore runs again over the exact
    digest-bound bytes selected for publication.  ``FLAG`` findings remain for
    human review; only the deliberately narrow ``HACK`` class is automatic.
    """

    run_id = validate_run_id(run_id)
    blocked = executable_hack_hits(source)
    if not blocked:
        return
    details = ", ".join(f"{label} at line {line}" for _, label, line, _, _ in blocked)
    raise submission_bundle.BundleError(
        f"selected bundle-aware run contains a static HACK tripwire: "
        f"{run_id} ({details})"
    )


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


def _absolute_path(path: str | os.PathLike[str]) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise OSError(f"parent traversal is not allowed in publication path: {path}")
    return Path(os.path.abspath(os.fspath(raw)))


def _open_parent_directory(
    path: str | os.PathLike[str], *, create: bool
) -> tuple[Path, int]:
    absolute = _absolute_path(path)
    if absolute.name in {"", ".", ".."}:
        raise OSError(f"publication path has no safe filename: {path}")
    parent_fd = os.open("/", _OPEN_PARENT_FLAGS)
    try:
        for component in absolute.parent.parts[1:]:
            try:
                next_fd = os.open(component, _OPEN_PARENT_FLAGS, dir_fd=parent_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, 0o755, dir_fd=parent_fd)
                next_fd = os.open(component, _OPEN_PARENT_FLAGS, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
    except BaseException:
        os.close(parent_fd)
        raise
    return absolute, parent_fd


def read_bounded_bytes(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_PUBLISHED_FILE_BYTES,
) -> bytes:
    """Read one stable, single-link regular file without following links."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    absolute, parent_fd = _open_parent_directory(path, create=False)
    descriptor: int | None = None
    try:
        parent_before = os.fstat(parent_fd)
        descriptor = os.open(
            absolute.name,
            _OPEN_FILE_FLAGS,
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"publication input is not a regular file: {absolute}")
        if before.st_nlink != 1:
            raise OSError(f"hard-linked publication input is not allowed: {absolute}")
        if before.st_size > max_bytes:
            raise OSError(f"publication input exceeds {max_bytes} bytes: {absolute}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise OSError(
                    f"publication input grew past {max_bytes} bytes: {absolute}"
                )
        if size != before.st_size or not _same_file(before, os.fstat(descriptor)):
            raise OSError(f"publication input changed while being read: {absolute}")
        if not _same_file(parent_before, os.fstat(parent_fd)):
            raise OSError(
                f"publication input directory changed while being read: {absolute.parent}"
            )
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def read_bounded_text(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_PUBLISHED_FILE_BYTES,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> str:
    return read_bounded_bytes(path, max_bytes=max_bytes).decode(
        encoding,
        errors=errors,
    )


def read_json_file(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_PUBLICATION_JSON_BYTES,
) -> Any:
    try:
        return json.loads(read_bounded_text(path, max_bytes=max_bytes))
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise OSError(f"invalid publication JSON {path}: {exc}") from exc


def selected_solution_run_ids(board: object) -> list[str]:
    """Validate a leaderboard and return every selected solution archive."""

    if not isinstance(board, dict) or not isinstance(board.get("models"), list):
        raise OSError("leaderboard must contain a models array")
    selected: set[str] = set()
    for model in board["models"]:
        if not isinstance(model, dict) or not isinstance(model.get("results"), dict):
            raise OSError("leaderboard model results must be an object")
        for cell in model["results"].values():
            if not isinstance(cell, dict):
                raise OSError("leaderboard result cell must be an object")
            has_solution = cell.get("has_solution")
            if type(has_solution) is not bool:
                raise OSError("leaderboard result has_solution must be a boolean")
            if not has_solution:
                continue
            selected.add(validate_run_id(cell.get("run_id")))
    return sorted(selected)


def prepare_selected_solution_outputs(
    board_path: str | os.PathLike[str],
    runs_root: str | os.PathLike[str],
    publication_root: str | os.PathLike[str],
) -> list[tuple[Path, str]]:
    """Read a board, then verify/render every selected solution."""

    return prepare_selected_solution_outputs_from_board(
        read_json_file(board_path),
        runs_root,
        publication_root,
    )


def prepare_selected_solution_outputs_from_board(
    board: object,
    runs_root: str | os.PathLike[str],
    publication_root: str | os.PathLike[str],
) -> list[tuple[Path, str]]:
    """Verify and render every selected solution before writing any output."""

    # Imports are intentionally delayed: redaction itself uses the safe file
    # primitives in this module, and neither dependency is needed by builders.
    from scripts.kernel_sidecars import augment
    from scripts.redaction import redact_text

    outputs: list[tuple[Path, str]] = []
    for run_id in selected_solution_run_ids(board):
        run = Path(runs_root) / run_id
        with open_verified_submission(run) as view:
            text = read_bounded_text(view.solution, errors="replace")
            if view.bundled:
                reject_bundle_static_hacks(run_id, text)
            text = augment(
                text,
                view.root,
                exact=view.bundled,
                strict=view.bundled,
            )
        outputs.append(
            (Path(publication_root) / f"{run_id}_solution.py.txt", redact_text(text))
        )
    return outputs


def publish_selected_solutions(
    board_path: str | os.PathLike[str],
    runs_root: str | os.PathLike[str],
    publication_root: str | os.PathLike[str],
) -> int:
    outputs = prepare_selected_solution_outputs(
        board_path,
        runs_root,
        publication_root,
    )
    for destination, contents in outputs:
        atomic_write_text(destination, contents)
    return len(outputs)


@contextlib.contextmanager
def trusted_archive_lock() -> Iterator[int]:
    """Exclude host-mode agents and mutating archive consumers."""

    inherited_text = os.environ.get("KBH_TRUST_ARCHIVE_LOCK_FD", "")
    if inherited_text.isdigit():
        inherited = int(inherited_text)
        try:
            opened = os.fstat(inherited)
            root = os.stat("/", follow_symlinks=False)
            if stat.S_ISDIR(opened.st_mode) and (opened.st_dev, opened.st_ino) == (
                root.st_dev,
                root.st_ino,
            ):
                # Re-locking an inherited open-file description is safe and
                # nonblocking.  Never unlock it here: the parent owns the
                # composite publication lifetime.
                fcntl.flock(inherited, fcntl.LOCK_EX | fcntl.LOCK_NB)
                yield inherited
                return
        except (BlockingIOError, OSError, ValueError):
            pass

    descriptor = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    previous = os.environ.get("KBH_TRUST_ARCHIVE_LOCK_FD")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.set_inheritable(descriptor, True)
        os.environ["KBH_TRUST_ARCHIVE_LOCK_FD"] = str(descriptor)
        yield descriptor
    finally:
        if previous is None:
            os.environ.pop("KBH_TRUST_ARCHIVE_LOCK_FD", None)
        else:
            os.environ["KBH_TRUST_ARCHIVE_LOCK_FD"] = previous
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextlib.contextmanager
def open_verified_submission(
    run_dir: str | os.PathLike[str],
) -> Iterator[SubmissionView]:
    """Yield a verified extraction, or an explicit grandfathered legacy view."""

    run = Path(run_dir)
    validate_run_id(run.name)
    result = submission_bundle.load_run_result(run)
    manifest = submission_bundle.verify_run_provenance(run, result)
    if manifest is None:
        if any(key in result for key in submission_bundle._BUNDLE_PROVENANCE_KEYS):
            raise submission_bundle.BundleError(
                "bundle-aware non-solution attempt cannot be published as legacy"
            )
        if result.get("has_solution") is False:
            raise submission_bundle.BundleError(
                "bundle-less attempt does not contain a publishable solution"
            )
        try:
            contents = read_bounded_bytes(run / "solution.py")
        except OSError as exc:
            raise submission_bundle.BundleError(
                f"legacy solution is unavailable or unsafe: {exc}"
            ) from exc
        with tempfile.TemporaryDirectory(prefix="kernelbench-publish-legacy-") as temp:
            solution = Path(temp) / "solution.py"
            with solution.open("xb") as stream:
                stream.write(contents)
                stream.flush()
                os.fchmod(stream.fileno(), 0o644)
                os.fsync(stream.fileno())
            yield SubmissionView(solution=solution, root=run, bundled=False)
        return

    if result.get("agent_container") is not True:
        raise submission_bundle.BundleError(
            "v2 publication requires agent_container=true"
        )
    digest = manifest["bundle_sha256"]
    with tempfile.TemporaryDirectory(prefix="kernelbench-publish-") as temporary:
        extraction = Path(temporary) / "submission"
        submission_bundle.extract_bundle(
            run / submission_bundle.RUN_BUNDLE_DIR,
            extraction,
            expected_digest=digest,
        )
        solution = extraction / "solution.py"
        if not solution.is_file() or solution.is_symlink():
            raise submission_bundle.BundleError(
                "verified bundle does not contain a regular solution.py"
            )
        yield SubmissionView(solution=solution, root=extraction, bundled=True)


def atomic_write_text(
    destination: str | os.PathLike[str],
    contents: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Atomically replace a text output without following its old symlink."""

    path, parent_fd = _open_parent_directory(destination, create=True)
    temporary: str | None = None
    descriptor: int | None = None
    try:
        for _ in range(128):
            candidate = f".{path.name}.tmp-{secrets.token_hex(12)}"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                continue
            temporary = candidate
            break
        if descriptor is None or temporary is None:
            raise OSError("could not allocate publication temporary file")
        with os.fdopen(descriptor, "w", encoding=encoding) as stream:
            descriptor = None
            stream.write(contents)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(
            temporary,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary = None
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
