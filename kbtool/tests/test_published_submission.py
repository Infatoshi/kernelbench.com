"""Publishing must use the submission bytes that were actually graded."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts import published_submission  # noqa: E402
from scripts import redaction  # noqa: E402
from scripts import submission_bundle as bundles  # noqa: E402
from scripts.kernel_sidecars import SidecarError  # noqa: E402


RECEIPT_FORGERY = """
import inspect
import os

frame = inspect.currentframe()
while frame is not None:
    sender = frame.f_locals.get("sender")
    receipt = frame.f_locals.get("receipt")
    if sender is not None and receipt is not None:
        sender.send_bytes(receipt)
        break
    frame = frame.f_back
print("PASS", flush=True)
print("peak_fraction: 99", flush=True)
os._exit(0)
"""


def _result(run: Path, digest: str) -> dict[str, object]:
    return {
        "run_id": run.name,
        "has_solution": True,
        "correct": True,
        "peak_fraction": 0.25,
        "check_exit_code": 0,
        "benchmark_exit_code": 0,
        "agent_container": True,
        "submission_bundle_sha256": digest,
        "submission_replay_status": "verified",
        "submission_bundle": {
            "path": "submission_bundle",
            "schema": bundles.SCHEMA,
            "version": bundles.SCHEMA_VERSION,
            "bundle_sha256": digest,
        },
        "submission_replay": {
            "status": "verified",
            "bundle_sha256": digest,
            "fresh_extraction": True,
            "fresh_caches": True,
            "stage_count": 2,
            "grader_surface_sha256": "1" * 64,
            "network_isolated": True,
            "network_isolation": "unshare-user-mount-pid-net-private-root-v1",
            "mount_isolated": True,
            "root_isolated": True,
            "pid_isolated": True,
            "clean_environment": True,
            "in_process_completion_guard": True,
        },
    }


def _bundled_board_case(
    tmp_path: Path,
    source_text: str,
) -> tuple[Path, Path, dict[str, object]]:
    source = tmp_path / "source"
    run = tmp_path / "runs" / "20260808_120000_test_model_01_problem"
    source.mkdir(parents=True)
    run.mkdir(parents=True)
    (source / "solution.py").write_text(source_text)
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    (run / "result.json").write_text(
        json.dumps(_result(run, manifest["bundle_sha256"]))
    )
    board: dict[str, object] = {
        "models": [
            {
                "results": {
                    "01_problem": {
                        "has_solution": True,
                        "run_id": run.name,
                    }
                }
            }
        ]
    }
    return run, tmp_path / "public", board


@pytest.mark.parametrize("unsafe_kind", ["deleted", "symlink", "empty"])
def test_required_curation_manifest_rejects_unsafe_or_empty_input(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    manifest = tmp_path / "published_runs.json"
    if unsafe_kind == "symlink":
        target = tmp_path / "real-manifest.json"
        target.write_text(json.dumps({"run_ids": ["20260808_120000_test"]}))
        manifest.symlink_to(target)
    elif unsafe_kind == "empty":
        manifest.write_text(json.dumps({"run_ids": []}))

    with pytest.raises((OSError, bundles.BundleError)):
        published_submission.load_required_published_run_ids(manifest)


def test_required_curation_manifest_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "published_runs.json"
    manifest.write_text(
        '{"run_ids":["20260808_120000_first"],"run_ids":["20260808_120000_second"]}'
    )

    with pytest.raises(OSError, match="duplicate key"):
        published_submission.load_required_published_run_ids(manifest)


@pytest.mark.parametrize(
    "unsafe_kind", ["deleted", "symlink", "directory_symlink", "empty"]
)
def test_publication_annotation_rejects_unsafe_or_empty_input(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    run_id = "20260808_120000_test_model_01_problem"
    annotations = tmp_path / "annotations"
    annotation = annotations / f"{run_id}.yaml"
    if unsafe_kind == "directory_symlink":
        real_annotations = tmp_path / "real-annotations"
        real_annotations.mkdir()
        (real_annotations / annotation.name).write_text(
            f"run_id: {run_id}\nverdict: clean\npublish_grade: true\n"
        )
        annotations.symlink_to(real_annotations, target_is_directory=True)
    else:
        annotations.mkdir()
    if unsafe_kind == "symlink":
        target = tmp_path / "real-annotation.yaml"
        target.write_text(f"run_id: {run_id}\nverdict: clean\npublish_grade: true\n")
        annotation.symlink_to(target)
    elif unsafe_kind == "empty":
        annotation.write_text("")

    with pytest.raises((OSError, bundles.BundleError)):
        published_submission.read_publication_annotation(annotation)


def test_bundle_era_selection_requires_explicit_clean_publish_grade(
    tmp_path: Path,
) -> None:
    run_id = "20260808_120000_test_model_01_problem"
    annotation_path = tmp_path / f"{run_id}.yaml"
    selected = [run_id]
    bundled = {run_id}

    with pytest.raises(bundles.BundleError, match="missing its audit annotation"):
        published_submission.require_publishable_annotations(
            selected,
            bundled,
            {},
        )

    # Missing approval remains compatible only for a grandfathered legacy run.
    published_submission.require_publishable_annotations(selected, set(), {})

    for verdict, publish_grade in (
        ("clean", None),
        ("clean", False),
        ("interesting", True),
    ):
        lines = [f"run_id: {run_id}", f"verdict: {verdict}"]
        if publish_grade is not None:
            lines.append(f"publish_grade: {str(publish_grade).lower()}")
        annotation_path.write_text("\n".join(lines) + "\n")
        parsed = published_submission.read_publication_annotation(annotation_path)
        with pytest.raises(bundles.BundleError, match="publication approval"):
            published_submission.require_publishable_annotations(
                selected,
                bundled,
                {run_id: parsed},
            )

    annotation_path.write_text(
        f'run_id: {run_id}\nverdict: clean\npublish_grade: "true"\n'
    )
    with pytest.raises(OSError, match="must be true or false"):
        published_submission.read_publication_annotation(annotation_path)

    annotation_path.write_text(
        f"run_id: {run_id}\n---\nverdict: clean\npublish_grade: true\n"
    )
    with pytest.raises(OSError, match="one implicit YAML document"):
        published_submission.read_publication_annotation(annotation_path)

    annotation_path.write_text(
        f"run_id: {run_id}\nverdict: clean\npublish_grade: true\n"
    )
    parsed = published_submission.read_publication_annotation(annotation_path)
    published_submission.require_publishable_annotations(
        selected,
        bundled,
        {run_id: parsed},
    )


def test_receipt_forgery_is_vetoed_even_after_manual_publication_approval(
    tmp_path: Path,
) -> None:
    run, publication_root, board = _bundled_board_case(tmp_path, RECEIPT_FORGERY)
    annotation_path = tmp_path / f"{run.name}.yaml"
    annotation_path.write_text(
        f"run_id: {run.name}\nverdict: clean\npublish_grade: true\n"
    )
    annotation = published_submission.read_publication_annotation(annotation_path)
    published_submission.require_publishable_annotations(
        [run.name],
        {run.name},
        {run.name: annotation},
    )
    board_path = tmp_path / "leaderboard.json"
    board_path.write_text(json.dumps(board))

    with pytest.raises(bundles.BundleError, match="static HACK tripwire"):
        published_submission.publish_selected_solutions(
            board_path,
            run.parent,
            publication_root,
        )

    assert not publication_root.exists()


@pytest.mark.parametrize(
    "source_text",
    [
        "class Model:\n    pass\n",
        "import torch\nvalue = torch.matmul(left, right)\n",
        "if cached.data_ptr() == value.data_ptr():\n    replay_graph()\n",
        'PHASE = "correctness"\n# Verified manually with check.py.\n',
    ],
)
def test_clean_and_flag_only_sources_are_not_automatically_vetoed(
    tmp_path: Path,
    source_text: str,
) -> None:
    run, publication_root, board = _bundled_board_case(tmp_path, source_text)

    outputs = published_submission.prepare_selected_solution_outputs_from_board(
        board,
        run.parent,
        publication_root,
    )

    assert len(outputs) == 1
    assert outputs[0][0].name == f"{run.name}_solution.py.txt"


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    graded=st.binary(max_size=4_096),
    projection=st.binary(max_size=4_096),
    sidecar=st.binary(max_size=4_096),
)
def test_bundled_publication_ignores_mutable_compatibility_projection(
    tmp_path: Path,
    graded: bytes,
    projection: bytes,
    sidecar: bytes,
) -> None:
    case = tmp_path / "publish-case"
    shutil.rmtree(case, ignore_errors=True)
    source = case / "source"
    run = case / "20260808_120000_test_model_01_problem"
    source.mkdir(parents=True)
    run.mkdir()
    (source / "solution.py").write_bytes(graded)
    (source / "kernel.cu").write_bytes(sidecar)
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    (run / "result.json").write_text(
        json.dumps(_result(run, manifest["bundle_sha256"]))
    )
    (run / "solution.py").write_bytes(projection)

    with published_submission.open_verified_submission(run) as view:
        assert view.bundled
        assert view.solution.read_bytes() == graded
        assert (view.root / "kernel.cu").read_bytes() == sidecar


def test_legacy_publication_rejects_solution_symlink(tmp_path: Path) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("secret")
    (run / "solution.py").symlink_to(outside)
    (run / "result.json").write_text(
        json.dumps({"run_id": run.name, "correct": False, "peak_fraction": None})
    )

    with pytest.raises(bundles.BundleError, match="unavailable or unsafe"):
        with published_submission.open_verified_submission(run):
            pass


def test_bundle_aware_no_solution_attempt_never_downgrades_to_legacy(
    tmp_path: Path,
) -> None:
    run = tmp_path / "20260808_120000_test_model_01_problem"
    run.mkdir()
    (run / "solution.py").write_text("planted")
    (run / "result.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "has_solution": False,
                "correct": False,
                "peak_fraction": None,
                "submission_bundle": None,
                "submission_replay": {"status": "not_applicable"},
                "submission_bundle_sha256": None,
                "submission_replay_status": "not_applicable",
            }
        )
    )

    with pytest.raises(bundles.BundleError, match="cannot be published as legacy"):
        with published_submission.open_verified_submission(run):
            pass


def test_atomic_publisher_replaces_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "public" / "solution.txt"
    destination.parent.mkdir()
    target = tmp_path / "operator-file"
    target.write_text("keep")
    destination.symlink_to(target)

    published_submission.atomic_write_text(destination, "graded")

    assert target.read_text() == "keep"
    assert not destination.is_symlink()
    assert destination.read_text() == "graded"


def test_atomic_publisher_rejects_intermediate_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    public = tmp_path / "public"
    public.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        published_submission.atomic_write_text(public / "runs/solution.txt", "bad")

    assert not (outside / "runs/solution.txt").exists()


def test_bundled_publication_requires_containerized_agent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run = tmp_path / "20260808_120000_test_model_01_problem"
    source.mkdir()
    run.mkdir()
    (source / "solution.py").write_text("graded")
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    result = _result(run, manifest["bundle_sha256"])
    result["agent_container"] = False
    (run / "result.json").write_text(json.dumps(result))

    with pytest.raises(bundles.BundleError, match="agent_container=true"):
        with published_submission.open_verified_submission(run):
            pass


def test_selected_missing_archive_fails_before_any_solution_is_published(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    public = tmp_path / "public"
    valid = runs / "20260807_120000_test_model_01_problem"
    missing = "20260807_120001_test_model_02_problem"
    valid.mkdir(parents=True)
    (valid / "solution.py").write_text("VALID")
    (valid / "result.json").write_text(
        json.dumps(
            {
                "run_id": valid.name,
                "correct": False,
                "peak_fraction": None,
            }
        )
    )
    board = {
        "models": [
            {
                "results": {
                    "01_problem": {"run_id": valid.name, "has_solution": True},
                    "02_problem": {"run_id": missing, "has_solution": True},
                }
            }
        ]
    }
    board_path = tmp_path / "leaderboard.json"
    board_path.write_text(json.dumps(board))

    with pytest.raises(bundles.BundleError):
        published_submission.publish_selected_solutions(board_path, runs, public)

    assert not public.exists()


def test_verified_bundle_publishes_exact_nested_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "source"
    run = tmp_path / "20260808_120000_test_model_01_problem"
    source.mkdir()
    run.mkdir()
    (source / "solution.py").write_text('sources = ["tmp/kernel.cu"]\n')
    (source / "tmp").mkdir()
    (source / "tmp/kernel.cu").write_text("DIGEST_BOUND")
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    (run / "result.json").write_text(
        json.dumps(_result(run, manifest["bundle_sha256"]))
    )
    board = {
        "models": [
            {"results": {"01_problem": {"run_id": run.name, "has_solution": True}}}
        ]
    }
    board_path = tmp_path / "leaderboard.json"
    board_path.write_text(json.dumps(board))

    outputs = published_submission.prepare_selected_solution_outputs(
        board_path, tmp_path, tmp_path / "public"
    )

    assert len(outputs) == 1
    assert "DIGEST_BOUND" in outputs[0][1]


def test_verified_bundle_missing_definite_sidecar_fails_strictly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    run = tmp_path / "20260808_120000_test_model_01_problem"
    source.mkdir()
    run.mkdir()
    (source / "solution.py").write_text('sources = ["missing.cu"]\n')
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    (run / "result.json").write_text(
        json.dumps(_result(run, manifest["bundle_sha256"]))
    )
    board = {
        "models": [
            {"results": {"01_problem": {"run_id": run.name, "has_solution": True}}}
        ]
    }
    board_path = tmp_path / "leaderboard.json"
    board_path.write_text(json.dumps(board))

    with pytest.raises(SidecarError, match="was not found"):
        published_submission.publish_selected_solutions(
            board_path, tmp_path, tmp_path / "public"
        )

    assert not (tmp_path / "public").exists()


def test_redaction_rejects_tree_symlink_without_partial_writes(tmp_path: Path) -> None:
    tree = tmp_path / "public"
    tree.mkdir()
    good = tree / "good.txt"
    good.write_text("sk-" + "a" * 30)
    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP")
    (tree / "planted.txt").symlink_to(victim)

    with pytest.raises(OSError, match="symbolic link"):
        redaction.main([str(tree)])

    assert good.read_text() == "sk-" + "a" * 30
    assert victim.read_text() == "KEEP"


def test_redaction_rejects_hardlink_and_fifo(tmp_path: Path) -> None:
    tree = tmp_path / "public"
    tree.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("KEEP")
    os.link(victim, tree / "hardlink.txt")

    with pytest.raises(OSError, match="hard-linked"):
        redaction.main([str(tree)])
    assert victim.read_text() == "KEEP"

    (tree / "hardlink.txt").unlink()
    fifo = tree / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(OSError, match="special file"):
        redaction.main([str(tree)])
