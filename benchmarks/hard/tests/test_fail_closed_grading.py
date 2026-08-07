import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

BENCH_ROOT = Path(__file__).resolve().parents[1]
MONOREPO = BENCH_ROOT.parents[1]
ENTRYPOINT = BENCH_ROOT / "src" / "eval" / "trusted_entrypoint.py"


def test_print_pass_then_system_exit_zero_is_not_success(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("import solution\nprint('PASS')\n")
    (tmp_path / "solution.py").write_text("print('PASS')\nraise SystemExit(0)\n")

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stdout.splitlines().count("PASS") == 1
    assert "before normal completion" in completed.stderr


def test_trusted_entrypoint_preserves_nonzero_system_exit(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("raise SystemExit(7)\n")

    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "check.py"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 7


def test_native_runner_uses_the_system_exit_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = _load_native_harness(monkeypatch)
    helper = tmp_path / "src" / "eval" / "trusted_entrypoint.py"
    helper.parent.mkdir(parents=True)
    shutil.copy2(ENTRYPOINT, helper)
    problem = tmp_path / "problems" / "p"
    problem.mkdir(parents=True)
    (problem / "check.py").write_text("import solution\nprint('PASS')\n")
    (problem / "solution.py").write_text("print('PASS')\nraise SystemExit(0)\n")

    returncode, log = native.run_native(str(tmp_path), "p", "check.py", 5)

    assert returncode != 0
    assert log.splitlines().count("PASS") == 1


def test_trusted_entrypoint_is_mirrored_exactly() -> None:
    expected = ENTRYPOINT.read_bytes()
    mirrors = [
        MONOREPO / "benchmarks" / bench / "src" / "eval" / "trusted_entrypoint.py"
        for bench in ("cuda", "mega", "mini")
    ] + [
        MONOREPO
        / "environments"
        / env
        / "_bench"
        / "src"
        / "eval"
        / "trusted_entrypoint.py"
        for env in ("kernel_hard", "kernel_mega")
    ]
    assert all(path.read_bytes() == expected for path in mirrors)


def _load_native_harness(monkeypatch: pytest.MonkeyPatch):
    fake_vf = types.ModuleType("verifiers")
    fake_vf.StatefulToolEnv = object
    fake_vf.Rubric = object
    fake_vf.Environment = object
    fake_vf.Parser = object
    fake_vf.RubricGroup = object
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.Dataset = object
    monkeypatch.setitem(sys.modules, "verifiers", fake_vf)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    path = MONOREPO / "environments" / "kernel_hard" / "kernel_native_harness.py"
    spec = importlib.util.spec_from_file_location("test_kernel_native_harness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("runs", "correct", "score"),
    [
        ([(1, "PASS\n")], False, 0.0),
        ([(0, "PASS\nPASS\n")], False, 0.0),
        ([(0, "PASS\n"), (1, "peak_fraction: 0.75\n")], True, 0.0),
        (
            [(0, "PASS\n"), (0, "peak_fraction: 0.75\npeak_fraction: 0.9\n")],
            True,
            0.0,
        ),
        ([(0, "PASS\n"), (0, "peak_fraction: 0.75\n")], True, 0.75),
    ],
)
def test_native_scoring_requires_exit_codes_and_single_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runs: list[tuple[int, str]],
    correct: bool,
    score: float,
) -> None:
    native = _load_native_harness(monkeypatch)
    problem = tmp_path / "problems" / "p"
    problem.mkdir(parents=True)
    (problem / "solution.py").write_text("# candidate\n")
    results = iter(runs)
    monkeypatch.setattr(native, "run_native", lambda *_args: next(results))

    result = native.score_workspace(str(tmp_path), "p", 1, 1)

    assert result["correct"] is correct
    assert result["peak_fraction"] == score


def test_shell_grading_uses_exact_markers_and_process_status() -> None:
    shared = (MONOREPO / "scripts" / "lib" / "run_harness.sh").read_text()
    assert "grep -axc 'PASS'" in shared
    assert '[ "$CHECK_PASS_COUNT" -eq 1 ]' in shared
    assert '[ "$BENCH_EXIT_CODE" -eq 0 ]' in shared
    assert '[ "$BENCH_METRIC_COUNT" -eq 1 ]' in shared
    assert 'uv run python "$TRUSTED_ENTRYPOINT" check.py' in shared
    assert 'uv run python "$TRUSTED_ENTRYPOINT" benchmark.py' in shared
    shared_check = shared.index('uv run python "$TRUSTED_ENTRYPOINT" check.py')
    shared_benchmark = shared.index('uv run python "$TRUSTED_ENTRYPOINT" benchmark.py')
    assert shared.index('strip_python_bytecode "$PROBLEM_DIR"') < shared_check
    assert shared.index('strip_python_bytecode "$PROBLEM_DIR"', shared_check) < shared_benchmark

    mega = (BENCH_ROOT.parent / "mega" / "scripts" / "run_hard.sh").read_text()
    assert "grep -axc 'PASS'" in mega
    assert '[ "$BENCH_METRIC_COUNT" -eq 1 ]' in mega
    mega_check = mega.index('uv run python "$TRUSTED_ENTRYPOINT" check.py')
    mega_benchmark = mega.index('uv run python "$TRUSTED_ENTRYPOINT" benchmark.py')
    assert mega.index('strip_python_bytecode "$PROBLEM_DIR"') < mega_check
    assert mega.index('strip_python_bytecode "$PROBLEM_DIR"', mega_check) < mega_benchmark
