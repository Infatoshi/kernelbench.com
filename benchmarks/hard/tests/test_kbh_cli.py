from pathlib import Path
from types import SimpleNamespace

from src.kbh import cli

ROOT = Path(__file__).resolve().parents[1]


def test_kbh_run_dry_run_targets_existing_runner(capsys) -> None:
    status = cli.main([
        "run",
        "codex",
        "gpt-5.5",
        "problems-rtxpro6000/01_fp8_gemm",
        "xhigh",
        "--dry-run",
    ])

    assert status == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith(str(ROOT / "scripts" / "run_hard.sh"))
    assert "codex gpt-5.5" in out
    assert str(ROOT / "problems-rtxpro6000" / "01_fp8_gemm") in out
    assert out.endswith("xhigh")


def test_kbh_run_invokes_backend_runner(monkeypatch) -> None:
    calls = []

    def fake_run(command, cwd, check):
        calls.append((command, cwd, check))
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    status = cli.main(["run", "opencode", "model/id", "problems-rtxpro6000/07_topk_softmax"])

    assert status == 17
    assert len(calls) == 1
    command, cwd, check = calls[0]
    assert command == [
        str(ROOT / "scripts" / "run_hard.sh"),
        "opencode",
        "model/id",
        str(ROOT / "problems-rtxpro6000" / "07_topk_softmax"),
    ]
    assert cwd == ROOT
    assert check is False


def test_kbh_run_accepts_claude_code_alias(monkeypatch) -> None:
    calls = []

    def fake_run(command, cwd, check):
        calls.append((command, cwd, check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    status = cli.main(["run", "claude-code", "claude-opus-4-7", "problems-rtxpro6000/01_fp8_gemm"])

    assert status == 0
    command, _, _ = calls[0]
    assert command[:3] == [
        str(ROOT / "scripts" / "run_hard.sh"),
        "claude",
        "claude-opus-4-7",
    ]


def test_kbh_run_preserves_droid_and_opencode_harnesses(capsys) -> None:
    for harness, model in (
        ("droid", "custom:GLM-5.1-[Z.AI-Coding-Plan]-0"),
        ("opencode", "zai/glm-5.1"),
    ):
        status = cli.main([
            "run",
            harness,
            model,
            "problems-rtxpro6000/01_fp8_gemm",
            "--dry-run",
        ])

        assert status == 0
        out = capsys.readouterr().out.strip()
        assert f"run_hard.sh {harness} " in out
        assert model in out


def test_kbh_run_reports_missing_backend(capsys, tmp_path) -> None:
    missing = tmp_path / "missing-runner"

    status = cli.main([
        "run",
        "claude",
        "model",
        "problems-rtxpro6000/01_fp8_gemm",
        "--runner",
        str(missing),
    ])

    assert status == 1
    assert f"runner not found: {missing}" in capsys.readouterr().err
