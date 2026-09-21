"""Unit tests for the launch-time sandbox canary (scripts/lib/sandbox_canary.py).

No bwrap needed: the canary is pure-stdlib path sweeping, so it is exercised
against fake trees here on any OS. The Linux end-to-end path (bwrap argv +
canary inside the sandbox) is scripts/smoke_sandbox_linux.sh.
"""
import importlib.util
from pathlib import Path

_CANARY = Path(__file__).resolve().parents[2] / "scripts/lib/sandbox_canary.py"
_spec = importlib.util.spec_from_file_location("sandbox_canary", _CANARY)
canary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(canary)

RUN_ID = "20260819_101112_claude_claude-opus-5_01_fp8_gemm"
HP_ID = "20260601_030405_or-fable_anthropic_claude-fable-5_01_fp8_gemm"


def _mk_clean_tree(root: Path) -> Path:
    """The tree a correctly built sandbox presents: only this run's dir, the
    honeytoken decoy, and the gpu lock dir are visible."""
    runs = root / "outputs" / "runs"
    own = runs / RUN_ID
    (own / "repo" / "problems" / "01_fp8_gemm").mkdir(parents=True)
    (own / "repo" / "problems" / "01_fp8_gemm" / "check.py").write_text("pass\n")
    hp = runs / HP_ID
    hp.mkdir(parents=True)
    (hp / "solution.py").write_text("decoy\n")
    (hp / "result.json").write_text("{}")
    lock = root / "outputs" / "gpu_lock"
    lock.mkdir(parents=True)
    (lock / "gpu.lock").write_text("")
    return root


def test_clean_sandbox_tree_passes(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    assert canary.scan([str(root)], [], RUN_ID, [HP_ID]) == []


def test_own_run_files_are_ignored(tmp_path):
    """result.json / solution.py inside this run's own dir must not fire."""
    root = _mk_clean_tree(tmp_path / "bench")
    own = root / "outputs" / "runs" / RUN_ID
    (own / "result.json").write_text("{}")
    (own / "repo" / "problems" / "01_fp8_gemm" / "solution.py").write_text("x\n")
    assert canary.scan([str(root)], [], RUN_ID, [HP_ID]) == []


def test_foreign_run_dir_is_detected(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    foreign = root / "outputs" / "runs" / "20260719_121747_or-fable_anthropic_claude-fable-5_01_fp8_gemm"
    foreign.mkdir(parents=True)
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert len(hits) == 1 and "foreign run dir" in hits[0]


def test_stray_solution_py_is_detected(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    pub = root / "public" / "runs"
    pub.mkdir(parents=True)
    (pub / "20260719_121747_solution.py.txt").write_text("kernel\n")
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert any("solution file" in h for h in hits)


def test_stray_result_json_is_detected(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    stray = root / "somewhere"
    stray.mkdir()
    (stray / "result.json").write_text("{}")
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert any("result.json" in h for h in hits)


def test_runs_remote_and_lambda_dirs_are_detected_even_empty(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    (root / "outputs" / "runs-remote-pro").mkdir()
    (root / "outputs" / "runs-lambda-h100-a").mkdir()
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert sum("pulled-archive dir" in h for h in hits) == 2


def test_published_code_dir_is_detected(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    (root / "public" / "data" / "mega" / "code").mkdir(parents=True)
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert any("published-code dir" in h for h in hits)


def test_must_be_hidden_nonempty_fails_empty_passes(tmp_path):
    empty = tmp_path / "kb-remote-archives-tmpfs"
    empty.mkdir()
    assert canary.scan([], [str(empty)], RUN_ID, []) == []

    leaky = tmp_path / "kb-remote-archives"
    (leaky / "20260719_121747_x").mkdir(parents=True)
    hits = canary.scan([], [str(leaky)], RUN_ID, [])
    assert any("must-be-hidden" in h for h in hits)

    gone = tmp_path / "never-existed"
    assert canary.scan([], [str(gone)], RUN_ID, []) == []


def test_main_exit_codes(tmp_path, capsys):
    root = _mk_clean_tree(tmp_path / "bench")
    assert canary.main(["--run-id", RUN_ID, "--allow", HP_ID, "--walk", str(root)]) == 0
    (root / "outputs" / "runs" / "20260719_121747_foreign").mkdir()
    assert canary.main(["--run-id", RUN_ID, "--allow", HP_ID, "--walk", str(root)]) == 1
    out = capsys.readouterr().out
    assert "REFUSING LAUNCH" in out
