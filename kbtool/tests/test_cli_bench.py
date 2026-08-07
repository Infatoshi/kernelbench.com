"""Bench-selection behavior of the kb CLI (kb -b <bench> ...)."""
import os
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
    for bench, script in (("hard", "publish_v2.sh"), ("cuda", "publish_v2.sh"),
                          ("mini", "publish_v2.sh"), ("mega", "publish_mega.sh")):
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
    (annotations / "rtx.yaml").write_text(
        "run_id: run-rtx\n"
        "gpu: RTX_PRO_6000\n"
    )
    (annotations / "h100.yaml").write_text(
        "run_id: run-h100\n"
        "gpu: NVIDIA H100 PCIe (SM90)\n"
    )
    (annotations / "legacy.yaml").write_text("run_id: run-legacy\n")

    assert cli._audited_run_ids(tmp_path, None) == ["run-legacy", "run-rtx"]
    assert cli._audited_run_ids(tmp_path, "h100") == ["run-h100"]
    assert cli._audited_run_ids(tmp_path, "b200") == []
