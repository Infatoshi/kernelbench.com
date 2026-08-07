"""Derived public artifacts fail closed without replacing prior outputs."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_gpu_builders_run_source_veto_before_replacing_board() -> None:
    for bench in ("hard", "cuda", "mini"):
        script = (
            REPO / f"benchmarks/{bench}/scripts/build_v2_leaderboard.py"
        ).read_text()
        annotation_gate = script.rindex("require_publishable_annotations(")
        source_gate = script.rindex("prepare_selected_solution_outputs_from_board(")
        board_write = script.rindex("atomic_write_text(")

        assert annotation_gate < source_gate < board_write, bench


@pytest.mark.parametrize("bench", ["hard", "cuda", "mini"])
@pytest.mark.parametrize("unsafe_kind", ["deleted", "symlink", "empty"])
def test_leaderboard_builder_rejects_unsafe_curation_manifest_before_write(
    tmp_path: Path,
    bench: str,
    unsafe_kind: str,
) -> None:
    manifest = tmp_path / "published_runs.json"
    if unsafe_kind == "symlink":
        target = tmp_path / "real-manifest.json"
        target.write_text(json.dumps({"run_ids": ["20260808_120000_test"]}))
        manifest.symlink_to(target)
    elif unsafe_kind == "empty":
        manifest.write_text(json.dumps({"run_ids": []}))
    runs = tmp_path / "runs"
    runs.mkdir()
    output = REPO / "benchmarks" / bench / "results" / "leaderboard_v2.json"
    previous = output.read_bytes() if output.exists() else None
    env = os.environ.copy()
    env.pop("KBH_TRUST_ARCHIVE_LOCK_FD", None)
    env["KBH_PUBLISHED_MANIFEST"] = str(manifest)
    env["KBH_RUNS_DIR"] = str(runs)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "benchmarks" / bench / "scripts/build_v2_leaderboard.py"),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode != 0
    assert (output.read_bytes() if output.exists() else None) == previous


def test_model_index_invariant_failure_keeps_previous_output(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load("test_build_model_index", REPO / "scripts/build_model_index.py")
    output = tmp_path / "public/data/models.json"
    output.parent.mkdir(parents=True)
    output.write_text("PREVIOUS\n")
    for bench in ("hard", "mega", "cuda"):
        (tmp_path / f"benchmarks/{bench}/results/annotations").mkdir(parents=True)

    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "OUT", output)
    monkeypatch.setattr(module, "load_site_board", lambda *_args: None)
    monkeypatch.setattr(module, "load_mega", lambda *_args: None)
    monkeypatch.setattr(module, "load_legacy_v1", lambda *_args: None)
    monkeypatch.setattr(module, "join_annotations", lambda *_args: ([], 1))
    monkeypatch.setattr(module, "compute_perf", lambda *_args: None)
    monkeypatch.setattr(module, "join_catalog", lambda *_args: None)
    monkeypatch.setattr(module, "finalize", lambda *_args: {"models": []})

    with pytest.raises(RuntimeError, match="annotation count mismatch"):
        module._build()

    assert output.read_text() == "PREVIOUS\n"


def test_per_gpu_emitter_rejects_selected_missing_archive_before_writes(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load(
        "test_emit_board_solutions",
        REPO / "benchmarks/hard/scripts/emit_board_solutions.py",
    )
    bench = tmp_path / "benchmarks/hard"
    board = bench / "results/leaderboard.h100.json"
    board.parent.mkdir(parents=True)
    board.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "results": {
                            "01_problem": {
                                "run_id": "20260807_120000_test_model_01_problem",
                                "has_solution": True,
                            }
                        }
                    }
                ]
            }
        )
    )
    public = tmp_path / "public/runs"
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "PUB", public)
    monkeypatch.setattr(
        module,
        "BOARDS",
        [("hard", "h100", "results/leaderboard.h100.json", ["outputs/runs-h100"])],
    )

    with pytest.raises(Exception, match="run directory|cannot open"):
        module._emit()

    assert not public.exists()
