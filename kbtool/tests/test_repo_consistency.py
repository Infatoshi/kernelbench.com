"""Repo-wide consistency guards.

These tests exist because every failure mode they check has already happened
in this repo (see benchmarks/*/DEVLOG.md):
- forked copies of shared bench code silently diverging (mega's stale roofline
  table shipped 2.5x-wrong peaks for six weeks),
- shell entry points breaking with zero test signal (lambda_worker.sh ssh_base
  dropped its command args and every `kb lambda run` was a no-op),
- docs drifting from code (91 env vars and 15 harnesses were undocumented,
  AGENTS.md carried a flag removed from the harness).

Run via: uv run --project kbtool pytest kbtool/tests/
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCHES = ("hard", "cuda", "mini", "mega")

# Files that are SUPPOSED to be byte-identical across the single-GPU benches.
# If you diverge one on purpose, either sync it back or remove it here with a
# DEVLOG entry recording the deliberate fork.
SHARED_IDENTICAL = [
    "src/eval/correctness.py",
    "src/eval/timing.py",
    "src/eval/roofline.py",
    "src/eval/shapes.py",
    "src/hardware/__init__.py",
    "src/hardware/rtx_pro_6000.py",
    "src/hardware/h100.py",
    "src/hardware/h100_sxm.py",
    "src/hardware/b200.py",
    "src/hardware/m4_max.py",
]


def _read(bench: str, rel: str) -> bytes:
    return (REPO / "benchmarks" / bench / rel).read_bytes()


def test_shared_bench_files_are_identical():
    drifted = []
    for rel in SHARED_IDENTICAL:
        ref_path = REPO / "benchmarks/hard" / rel
        assert ref_path.exists(), f"reference file missing: {ref_path}"
        ref = ref_path.read_bytes()
        for bench in BENCHES[1:]:
            p = REPO / "benchmarks" / bench / rel
            if not p.exists():
                continue  # bench legitimately lacks the component (e.g. mega/kbh)
            if p.read_bytes() != ref:
                drifted.append(f"{bench}/{rel}")
    assert not drifted, (
        "shared bench files drifted from hard's copy (sync them or record a "
        f"deliberate fork in DEVLOG and remove from SHARED_IDENTICAL): {drifted}"
    )


def test_all_shell_scripts_parse():
    scripts = sorted(
        list((REPO / "scripts").glob("*.sh"))
        + [p for b in (*BENCHES, "multi") for p in (REPO / "benchmarks" / b / "scripts").glob("*.sh")]
    )
    assert scripts, "no shell scripts found — path bug in the test"
    bad = []
    for sc in scripts:
        r = subprocess.run(["bash", "-n", str(sc)], capture_output=True, text=True)
        if r.returncode != 0:
            bad.append(f"{sc.relative_to(REPO)}: {r.stderr.strip()[:200]}")
    assert not bad, f"shell syntax errors: {bad}"


def _harness_case_labels() -> set[str]:
    labels: set[str] = set()
    # hard/cuda/mini dispatch from the shared runner; mega and multi keep forks.
    runners = [
        REPO / "scripts/lib/run_harness.sh",
        REPO / "benchmarks/mega/scripts/run_hard.sh",
        REPO / "benchmarks/multi/scripts/run_agent.sh",
    ]
    # PATH-wrapper / plumbing case labels that are not harnesses.
    not_harnesses = {"uv", "python", "python3", "nvidia-smi", "ncu", "nsys", "nvcc"}
    for rn in runners:
        text = rn.read_text()
        for m in re.finditer(r"^\s{4}([a-z0-9_|\- ]+)\)$", text, re.M):
            for label in m.group(1).split("|"):
                label = label.strip()
                if label and label not in not_harnesses and "*" not in label:
                    labels.add(label)
    return labels


def test_harness_doc_covers_all_case_labels():
    doc = (REPO / "docs/HARNESSES.md").read_text()
    missing = sorted(h for h in _harness_case_labels() if f"`{h}`" not in doc)
    assert not missing, f"harness branches with no docs/HARNESSES.md row: {missing}"


def test_env_doc_covers_all_read_vars():
    var_re = re.compile(r"KB(?:H|M|MINI)?_[A-Z][A-Z0-9_]*")
    roots = [REPO / "scripts", REPO / "kbtool/kb"]
    for b in (*BENCHES, "multi"):
        roots += [REPO / "benchmarks" / b / "scripts", REPO / "benchmarks" / b / "src"]
    found: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix not in (".sh", ".py") or not p.is_file():
                continue
            found |= set(var_re.findall(p.read_text(errors="ignore")))
    doc = (REPO / "docs/ENV.md").read_text()
    # ENV.md's footer lists deliberately-excluded scan artifacts.
    missing = sorted(v for v in found if f"`{v}`" not in doc)
    assert not missing, f"env vars read by code but absent from docs/ENV.md: {missing}"


def test_lambda_worker_ssh_forwards_command():
    """ssh_base dropped "$@" once and every remote command silently no-opped."""
    text = (REPO / "scripts/lambda_worker.sh").read_text()
    m = re.search(r"ssh_base\(\) \{.*?\n\}", text, re.S)
    assert m, "ssh_base() not found"
    assert '"$@"' in m.group(0), "ssh_base() no longer forwards its command args"


def test_teardown_scripts_cannot_false_succeed():
    """A failed provider listing must never be reported as a completed teardown."""
    brev = (REPO / "scripts/brev_teardown.sh").read_text()
    assert "cannot confirm state" in brev, "brev_teardown lost its failed-listing guard"
    lam = (REPO / "scripts/lambda_worker.sh").read_text()
    assert "curl -sSf" in lam, "lambda api() no longer fails on HTTP errors"
    assert 'if LISTING="$(api GET /instances)"' in lam, (
        "lambda down poll no longer distinguishes a failed listing from a gone instance"
    )


def test_gpu_lock_bounded_retry_everywhere():
    """The unbounded `flock -x 9` deadlock cost 71 min of an Opus 5 sweep; the
    bounded-retry fix must exist in the shared single-GPU runner."""
    text = (REPO / "scripts/lib/run_harness.sh").read_text()
    assert "until flock -x -w 5 9; do" in text, "bounded flock retry missing from shared runner"


def test_or_proxy_launch_bypasses_gpu_lock_wrapper():
    """The or-provider proxy is a CPU-only daemon; launched via the $RUN_DIR/bin
    python3 wrapper it inherits the flock fd and holds outputs/gpu.lock for its
    whole life, starving every later run (2026-08-01)."""
    text = (REPO / "scripts/lib/run_harness.sh").read_text()
    m = re.search(r'\n\s*OR_PROXY_UPSTREAM=[^\n]*', text)
    assert m, "or-provider proxy launch line not found"
    assert '"$OR_PROXY_PYTHON"' in m.group(0), (
        "proxy launch must use the wrapper-bypassing $OR_PROXY_PYTHON, not bare python3"
    )


def test_bench_wrappers_are_thin_and_use_shared_runner():
    """hard/cuda/mini run_hard.sh are identity-pinning wrappers over
    scripts/lib/run_harness.sh. Logic creeping back into a wrapper is the
    fork-drift failure mode this structure exists to kill (mini's fork shipped
    a KERNELBENCH-CUDA banner and a stale or-fable branch)."""
    for b in ("hard", "cuda", "mini"):
        p = REPO / "benchmarks" / b / "scripts/run_hard.sh"
        text = p.read_text()
        assert "scripts/lib/run_harness.sh" in text, f"{b}: wrapper does not exec the shared runner"
        assert "KB_BENCH_DIR" in text and "KB_BENCH_BANNER" in text, f"{b}: wrapper missing identity pins"
        assert 'case "$HARNESS"' not in text, f"{b}: harness dispatch leaked back into the wrapper"
        assert len(text.splitlines()) < 30, f"{b}: wrapper no longer thin ({len(text.splitlines())} lines)"


def test_lambda_sync_preserves_torch_index_patch():
    """Re-syncing a bootstrapped node once shipped the Mac's cu130 uv.lock over
    the node's cu128-patched one; every later graded env build then died with
    driver-too-old at check time (2026-08-01)."""
    text = (REPO / "scripts/lambda_worker.sh").read_text()
    assert "preserving node torch-index patch" in text, (
        "lambda sync no longer guards the bootstrapped pyproject/uv.lock torch-index patch"
    )


def test_lambda_sync_ships_shared_runner_lib():
    """kb lambda sync copies one bench dir to the node; the wrapper's fallback
    path (bench-local scripts/lib/) only works if sync ships the lib there."""
    text = (REPO / "scripts/lambda_worker.sh").read_text()
    assert 'scripts/lib/' in text, "lambda_worker sync no longer ships scripts/lib to workers"
