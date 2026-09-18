"""One digest function for the graded surface.

Every published number is produced by a specific grading surface: the deck's
problem files (``check.py``, ``benchmark.py``, ``problem.yaml``, ``shapes.py``,
``reference.py``, ``sota.py``, ``PROMPT.txt``) plus the shared ``src/`` tree the
check imports. If that surface changes, previously published cells were graded
by something else and their numbers no longer mean what the board says.

Before this module there was no record of which surface scored a cell, which is
how a widened ``check.py`` could ship while the board kept serving numbers
produced under a narrower one. The fix is mechanical: stamp
``graded_surface_sha`` into ``result.json`` at grading time, and make the
publish gate refuse any cell whose stamp is not the current deck's digest.

Three callers share this code and MUST agree byte-for-byte:
  - ``scripts/lib/run_harness.sh`` (in-run grading, host mode)
  - ``benchmarks/*/scripts/regrade_sequential.sh`` (isolated re-grade)
  - ``scripts/check_publish_gates.py`` (gate E, which refuses stale cells)

Keep them pointed at this module. A private reimplementation in any one of them
would silently reintroduce exactly the drift this exists to catch.
"""
from __future__ import annotations

import hashlib
import stat
from pathlib import Path

# The deck files that define how a problem is graded and timed. Any edit to one
# of these changes what a published number means, so all of them are in the
# digest. Mirrors TEMPLATE_FILES in scripts/lib/run_harness.sh.
DECK_FILES = (
    "reference.py",
    "sota.py",
    "shapes.py",
    "problem.yaml",
    "check.py",
    "benchmark.py",
    "PROMPT.txt",
)


def _update_tree(digest: "hashlib._Hash", root: Path, label: str) -> None:
    """Fold one directory tree into *digest*, deterministically.

    Directories, regular files, and symlink targets are all recorded; bytecode
    caches are skipped so a stray ``__pycache__`` cannot change the stamp.
    Anything else (fifo, socket, device) raises: the surface must be plain
    files, or the digest is not meaningful.
    """
    if not root.is_dir():
        raise SystemExit(f"graded surface: {label} is not a directory: {root}")
    digest.update(b"\x01" + label.encode() + b"\0")
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            kind, contents = b"d", b""
        elif stat.S_ISREG(metadata.st_mode):
            kind, contents = b"f", path.read_bytes()
        elif stat.S_ISLNK(metadata.st_mode):
            kind, contents = b"l", str(path.readlink()).encode()
        else:
            raise SystemExit(f"graded surface: unsafe entry: {label}/{relative}")
        digest.update(kind + relative.encode() + b"\0")
        digest.update(hashlib.sha256(contents).digest())


def graded_surface_digest(deck_dir: Path, src_dir: Path) -> str:
    """Return the sha256 of the graded surface for one problem.

    *deck_dir* is a single problem directory (``problems-<gpu>/<problem>``);
    *src_dir* is the bench's ``src/`` tree. Both are hashed, so moving a peak
    constant or a tolerance changes the stamp just as editing ``check.py`` does.
    """
    digest = hashlib.sha256()
    _update_tree(digest, Path(src_dir), "src")
    for name in DECK_FILES:
        path = Path(deck_dir) / name
        if not path.is_file():
            raise SystemExit(f"graded surface: missing {name} in {deck_dir}")
        digest.update(b"f" + name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI so the shell emitters do not reimplement this.

        python graded_surface.py <deck_dir> <src_dir>
    """
    import sys

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: graded_surface.py <deck_problem_dir> <src_dir>", file=sys.stderr)
        return 2
    print(graded_surface_digest(Path(args[0]), Path(args[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())