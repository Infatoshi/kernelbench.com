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

import re
from pathlib import Path

# module names that are never co-located kernel files
_STDLIKE = {
    "torch", "triton", "numpy", "np", "os", "sys", "math", "json", "re",
    "pathlib", "ctypes", "subprocess", "functools", "itertools", "typing",
    "dataclasses", "collections", "contextlib", "time", "struct", "tempfile",
    "shutil", "warnings", "hashlib", "atexit", "threading", "types",
    "reference", "shapes", "problem", "check", "benchmark", "solution",
}

_KERNEL_EXT = (".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".ptx")

# max total appended bytes; beyond this something is wrong (vendored tree?)
_CAP = 1_500_000


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
        return (len(s) < 120 and "\n" not in s and "(" not in s
                and " " not in s and "." in Path(s).name)

    # cpp_extension.load(sources=[...]) -- NOT cpp_sources=/cuda_sources=
    for m in re.finditer(r"(?<![\w])sources\s*=\s*\[([^\]]*)\]", text):
        definite += [x for x in re.findall(r"[\"']([^\"']+)[\"']", m.group(1))
                     if _is_filename(x)]
    # string literals naming kernel files anywhere (Path(__file__)/"x.cu",
    # open("x.cu"), _SRC = "x.cu", env defaults)
    for m in re.finditer(r"[\"']([\w./\-]+\.(?:cu|cuh|cpp|cc|c|h|hpp|ptx))[\"']", text):
        speculative.append(m.group(1))
    # local python module imports: from x import ... / import x
    for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][\w]*)", text, re.M):
        mod = m.group(1)
        if mod not in _STDLIKE and not mod.startswith("_"):
            speculative.append(mod + ".py")
    # dedupe by basename, definite entries win, order preserved
    seen: set[str] = set()
    out: list[tuple[str, bool]] = []
    for n, d in [(x, True) for x in definite] + [(x, False) for x in speculative]:
        b = Path(n).name
        if b and b not in seen:
            seen.add(b)
            out.append((b, d))
    return out


def _find_in_archive(run_dir: Path, basename: str) -> Path | None:
    """Prefer the graded workspace problem dir, then run root, then scratch."""
    hits: list[Path] = []
    for pat in (f"repo/problems/*/{basename}", basename, f"scratch/{basename}",
                f"repo/problems/*/scratch/{basename}"):
        try:
            hits += sorted(run_dir.glob(pat))
        except (OSError, ValueError):
            continue
    for h in hits:
        # ignore vendored/third-party and build-cache copies
        s = str(h)
        if any(x in s for x in ("/.venv/", "/cache/", "/site-packages/",
                                "/cutlass/", "/third_party/", "/tmp/")):
            continue
        if h.is_file():
            return h
    return None


def augment(solution_text: str, run_dir: str | Path) -> str:
    """Append every referenced sidecar to the solution text under banners."""
    run_dir = Path(run_dir)
    refs = _referenced_names(solution_text)
    if not refs:
        return solution_text
    parts = [solution_text]
    total = 0
    for name, definite in refs:
        p = _find_in_archive(run_dir, name)
        if p is None:
            if definite:
                parts.append(
                    f"\n\n# {'='*66}\n# WARNING: solution loads sidecar "
                    f"'{name}' but it was not found in the run archive.\n# {'='*66}\n")
            continue
        try:
            body = p.read_text(errors="replace")
        except OSError:
            continue
        if total + len(body) > _CAP:
            parts.append(
                f"\n\n# {'='*66}\n# sidecar '{name}' omitted: size cap reached\n# {'='*66}\n")
            continue
        total += len(body)
        parts.append(
            f"\n\n# {'='*66}\n# ===== sidecar: {name} "
            f"({len(body)} bytes, loaded by solution.py) =====\n# {'='*66}\n\n"
            + body)
    return "".join(parts)
