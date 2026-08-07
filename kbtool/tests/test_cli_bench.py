"""Bench-selection behavior of the kb CLI (kb -b <bench> ...)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kb import cli


def test_pop_bench_flag_forms():
    argv = ["-b", "cuda", "run", "x"]
    assert cli._pop_bench(argv) == "cuda"
    assert argv == ["run", "x"]
    argv = ["run", "--bench=mini", "x"]
    assert cli._pop_bench(argv) == "mini"
    assert argv == ["run", "x"]
    argv = ["publish"]
    assert cli._pop_bench(argv) == "hard"


def test_pop_bench_env_fallback(monkeypatch):
    monkeypatch.setenv("KB_BENCH", "cuda")
    assert cli._pop_bench(["audit", "rid"]) == "cuda"
    monkeypatch.delenv("KB_BENCH")
    assert cli._pop_bench(["audit", "rid"]) == "hard"


def test_default_problems_root_matches_bench_decks():
    root = Path(__file__).resolve().parents[2]
    for bench, prob_root in cli.DEFAULT_PROBLEMS_ROOT.items():
        assert (root / "benchmarks" / bench / prob_root).is_dir(), (bench, prob_root)


def test_publish_script_map_covers_every_publishable_bench():
    root = Path(__file__).resolve().parents[2]
    for bench, script in (
        ("hard", "publish_v2.sh"),
        ("cuda", "publish_v2.sh"),
        ("mini", "publish_v2.sh"),
        ("mega", "publish_mega.sh"),
    ):
        assert (root / "benchmarks" / bench / "scripts" / script).is_file(), bench


def test_sweep_refuses_benches_with_their_own_drivers():
    root = Path(__file__).resolve().parents[2]
    for bench in ("mega", "multi"):
        with pytest.raises(SystemExit) as e:
            cli.cmd_sweep(root, ["claude", "m"], bench=bench)
        assert "own driver" in str(e.value)


def test_run_dispatches_to_selected_bench(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    calls = {}

    def fake_execvp(prog, argv):
        calls["prog"], calls["argv"], calls["cwd"] = prog, argv, os.getcwd()
        raise SystemExit(0)

    monkeypatch.setenv("KB_ALLOW_LOCAL", "1")
    monkeypatch.delenv("KBH_PROBLEMS_ROOT", raising=False)
    monkeypatch.setattr(os, "execvp", fake_execvp)
    with pytest.raises(SystemExit):
        cli.cmd_run(root, ["claude", "claude-opus-5", "01_dequant_gemv"], bench="mini")
    assert calls["argv"][-1] == "problems-h100/01_dequant_gemv"
    assert calls["cwd"].endswith("benchmarks/mini")


def test_contamination_accepts_all_bench_names(tmp_path):
    from kb import contamination

    for bench in ("hard", "cuda", "mini", "mega", "multi"):
        runs = tmp_path / "benchmarks" / bench / "outputs" / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        rc = contamination.run([bench], repo_root=tmp_path)
        assert rc == 0, bench


def test_audited_run_ids_partition_hardware_namespaces(tmp_path):
    annotations = tmp_path / "results" / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "rtx.yaml").write_text("run_id: run-rtx\ngpu: RTX_PRO_6000\n")
    (annotations / "h100.yaml").write_text(
        "run_id: run-h100\ngpu: NVIDIA H100 PCIe (SM90)\n"
    )
    (annotations / "legacy.yaml").write_text("run_id: run-legacy\n")

    assert cli._audited_run_ids(tmp_path, None) == ["run-legacy", "run-rtx"]
    assert cli._audited_run_ids(tmp_path, "h100") == ["run-h100"]
    assert cli._audited_run_ids(tmp_path, "b200") == []


def test_publication_lock_fd_is_inherited_by_subprocess(monkeypatch):
    monkeypatch.delenv("KBH_TRUST_ARCHIVE_LOCK_FD", raising=False)
    monkeypatch.setenv("KBH_TRUST_ARCHIVE_LOCK_HELD", "1")

    with cli._trusted_archive_lock() as descriptor:
        result = cli._publication_run(
            [
                sys.executable,
                "-c",
                "import os; fd=int(os.environ['KBH_TRUST_ARCHIVE_LOCK_FD']); "
                "a=os.fstat(fd); b=os.stat('/'); "
                "print(a.st_dev == b.st_dev and a.st_ino == b.st_ino)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert descriptor == int(os.environ["KBH_TRUST_ARCHIVE_LOCK_FD"])
        assert result.stdout.strip() == "True"

    assert "KBH_TRUST_ARCHIVE_LOCK_FD" not in os.environ


def test_publish_holds_one_lock_through_all_derived_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    script = tmp_path / "benchmarks/hard/scripts/publish_v2.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n")
    events: list[str] = []

    def assert_locked(event: str) -> None:
        descriptor = cli._ACTIVE_ARCHIVE_LOCK_FD
        assert descriptor is not None
        assert os.environ["KBH_TRUST_ARCHIVE_LOCK_FD"] == str(descriptor)
        os.fstat(descriptor)
        events.append(event)

    def fake_run(*_args, **_kwargs):
        assert_locked("publish")
        return subprocess.CompletedProcess([], 0)

    def fake_detail(_root):
        assert_locked("detail")
        return 0

    def fake_index(_root):
        assert_locked("index")
        return 0

    monkeypatch.setattr(cli, "_publication_run", fake_run)
    monkeypatch.setattr(cli, "_rebuild_run_detail", fake_detail)
    monkeypatch.setattr(cli, "_rebuild_model_index", fake_index)

    assert cli.cmd_publish(tmp_path, [], bench="hard") == 0
    assert events == ["publish", "detail", "index"]
    assert cli._ACTIVE_ARCHIVE_LOCK_FD is None
