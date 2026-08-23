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
failing silently. Embedded ELF fatbins (base64 .so) are stripped; co-located
.cu files are then appended so the public URL is source, not a binary dump.
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

# ELF64 little-endian magic in standard base64. Agents sometimes embed a
# compiled .so as a string blob; the public kernel must be source.
_ELF64_LE_B64 = "f0VMRgIBAQ"
_ELF_BLOB_ASSIGN = re.compile(
    r"([A-Za-z_][\w]*)\s*=\s*\(\s*(?:\"[A-Za-z0-9+/]+=*\"\s*)+\)",
)
_CEILING_CU = {"micro.cu"}


def strip_embedded_elf(text: str) -> tuple[str, bool]:
    """Replace compiled ELF/cubin string blobs with an empty stub.

    Returns (text, stripped). Public /runs files are source viewers; a 1MB
    base64 .so is not a kernel.
    """
    if _ELF64_LE_B64 not in text and "\x7fELF" not in text:
        return text, False

    def repl(m: re.Match[str]) -> str:
        if _ELF64_LE_B64 not in m.group(0) and "\x7fELF" not in m.group(0):
            return m.group(0)
        return (
            f'{m.group(1)} = ""  '
            f"# public copy omits embedded ELF fatbin; CUDA source is below"
        )

    new, n = _ELF_BLOB_ASSIGN.subn(repl, text)
    if n:
        return new, True
    # Fallback: cut from the magic to the next lone-paren closer.
    j = text.find(_ELF64_LE_B64)
    if j < 0:
        return text, False
    eq = text.rfind("=", 0, j)
    ident_start = text.rfind("\n", 0, eq) + 1
    close = text.find("\n)\n", j)
    if eq < 0 or close < 0:
        return text, False
    ident = text[ident_start:eq].strip()
    stub = (
        f'{ident} = ""  '
        f"# public copy omits embedded ELF fatbin; CUDA source is below\n"
    )
    return text[:ident_start] + stub + text[close + 3 :], True


def _extra_cu_after_elf_strip(run_dir: Path) -> list[Path]:
    """When the loader was a fatbin, ship co-located .cu that was not imported."""
    hits: list[Path] = []
    for pat in ("scratch/*.cu", "repo/problems/*/*.cu",
                "repo/problems/*/scratch/*.cu"):
        try:
            hits += sorted(run_dir.glob(pat))
        except (OSError, ValueError):
            continue
    out: list[Path] = []
    seen: set[str] = set()
    for p in hits:
        if not p.is_file() or p.name in _CEILING_CU or p.name in seen:
            continue
        try:
            if p.stat().st_size < 512:
                continue
        except OSError:
            continue
        seen.add(p.name)
        out.append(p)
    return out


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
    solution_text, stripped_elf = strip_embedded_elf(solution_text)
    refs = _referenced_names(solution_text)
    extra = _extra_cu_after_elf_strip(run_dir) if stripped_elf else []
    if not refs and not extra:
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
    already = {n for n, _ in refs}
    for p in extra:
        if p.name in already:
            continue
        try:
            body = p.read_text(errors="replace")
        except OSError:
            continue
        if total + len(body) > _CAP:
            parts.append(
                f"\n\n# {'='*66}\n# sidecar '{p.name}' omitted: size cap reached\n"
                f"# {'='*66}\n")
            continue
        total += len(body)
        parts.append(
            f"\n\n# {'='*66}\n# ===== sidecar: {p.name} "
            f"({len(body)} bytes, CUDA source for stripped fatbin) =====\n"
            f"# {'='*66}\n\n" + body)
    return "".join(parts)
