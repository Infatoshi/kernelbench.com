"""Properties and security regressions for immutable submission bundles."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts import submission_bundle as bundles  # noqa: E402


_COMPONENT = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_-",
    ),
    min_size=1,
    max_size=12,
).filter(lambda value: value not in {".", ".."})
_DIRECTORY_COMPONENT = _COMPONENT.filter(
    lambda value: (
        value not in bundles.DEFAULT_EXCLUDED_DIRECTORY_NAMES
        and value not in bundles.TRUSTED_MODULE_STEMS
    )
)
_SIDECARS = st.dictionaries(
    keys=st.tuples(_DIRECTORY_COMPONENT, _COMPONENT),
    values=st.binary(max_size=1_024),
    max_size=12,
)
_VALID_PEAK_FRACTIONS = st.one_of(
    st.integers(min_value=0, max_value=100),
    st.floats(
        min_value=0,
        max_value=100,
        allow_nan=False,
        allow_infinity=False,
    ),
)
_INVALID_PEAK_FRACTIONS = st.one_of(
    st.none(),
    st.booleans(),
    st.text(max_size=16),
    st.lists(st.integers(), max_size=3),
    st.integers(min_value=-1_000_000, max_value=-1),
    st.integers(min_value=101, max_value=1_000_000),
    st.floats(allow_nan=True, allow_infinity=True).filter(
        lambda value: not math.isfinite(value) or value < 0 or value > 100
    ),
)
_TRUSTED_MODULE_STEMS = st.sampled_from(sorted(bundles.TRUSTED_MODULE_STEMS))
_ABI_EXTENSION_SUFFIXES = st.one_of(
    st.sampled_from(
        sorted(
            bundles.PYTHON_EXTENSION_SUFFIXES
            | {
                ".abi3.so",
                ".pyd",
                ".so",
            }
        )
    ),
    st.integers(min_value=310, max_value=399).map(
        lambda abi: f".cpython-{abi}-x86_64-linux-gnu.so"
    ),
    st.integers(min_value=310, max_value=399).map(
        lambda abi: f".cp{abi}-win_amd64.pyd"
    ),
)


def _write_submission(
    root: Path,
    *,
    solution: bytes = b"def custom_kernel(x):\n    return x\n",
    sidecars: dict[tuple[str, str], bytes] | None = None,
) -> None:
    root.mkdir()
    (root / "solution.py").write_bytes(solution)
    (root / "check.py").write_text("raise AssertionError('template leaked')\n")
    for parts, contents in (sidecars or {}).items():
        path = root.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)


@settings(
    max_examples=35,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(solution=st.binary(max_size=2_048), sidecars=_SIDECARS)
def test_nested_sidecars_round_trip_by_path_and_content(
    tmp_path: Path,
    solution: bytes,
    sidecars: dict[tuple[str, str], bytes],
) -> None:
    shutil.rmtree(tmp_path / "case", ignore_errors=True)
    tmp_path = tmp_path / "case"
    tmp_path.mkdir()
    source = tmp_path / "problem"
    bundle = tmp_path / "bundle"
    extracted = tmp_path / "extracted"
    _write_submission(source, solution=solution, sidecars=sidecars)
    # A template-named file is still an agent sidecar when it is nested.
    nested_template_name = source / "helpers" / "check.py"
    nested_template_name.parent.mkdir(exist_ok=True)
    nested_template_name.write_bytes(b"nested helper")

    created = bundles.create_bundle(source, bundle)
    verified = bundles.verify_bundle(
        bundle,
        expected_digest=created["bundle_sha256"],
    )
    extracted_manifest = bundles.extract_bundle(
        bundle,
        extracted,
        expected_digest=created["bundle_sha256"],
    )

    assert verified == created == extracted_manifest
    assert (extracted / "solution.py").read_bytes() == solution
    assert not (extracted / "check.py").exists()
    assert (extracted / "helpers" / "check.py").read_bytes() == b"nested helper"
    for parts, contents in sidecars.items():
        assert extracted.joinpath(*parts).read_bytes() == contents

    expected_paths = {"solution.py", "helpers/check.py"}
    expected_paths.update("/".join(parts) for parts in sidecars)
    assert {entry["path"] for entry in created["files"]} == expected_paths


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    replacement=st.binary(min_size=1, max_size=2_048).filter(
        lambda value: value != b"def custom_kernel(x):\n    return x\n"
    )
)
def test_payload_tampering_is_always_detected(
    tmp_path: Path, replacement: bytes
) -> None:
    shutil.rmtree(tmp_path / "case", ignore_errors=True)
    tmp_path = tmp_path / "case"
    tmp_path.mkdir()
    source = tmp_path / "problem"
    bundle = tmp_path / "bundle"
    _write_submission(source)
    bundles.create_bundle(source, bundle)

    (bundle / "files" / "solution.py").write_bytes(replacement)
    with pytest.raises(bundles.BundleError, match="does not match manifest"):
        bundles.verify_bundle(bundle)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.py",
        "nested/../../outside.py",
        "/absolute/outside.py",
        "nested//outside.py",
        "nested/./outside.py",
        "nested\\outside.py",
    ],
)
def test_manifest_traversal_is_rejected_before_extraction(
    tmp_path: Path,
    unsafe_path: str,
) -> None:
    source = tmp_path / "problem"
    bundle = tmp_path / "bundle"
    destination = tmp_path / "extracted"
    _write_submission(source)
    bundles.create_bundle(source, bundle)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["path"] = unsafe_path
    manifest["bundle_sha256"] = bundles._descriptor_digest(manifest)
    manifest_path.write_bytes(bundles._canonical_json(manifest))

    with pytest.raises(bundles.BundleError, match="artifact path|path component"):
        bundles.extract_bundle(bundle, destination)
    assert not destination.exists()
    assert not (tmp_path.parent / "outside.py").exists()


def test_missing_payload_artifact_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "problem"
    bundle = tmp_path / "bundle"
    _write_submission(source, sidecars={("kernels", "fused.cu"): b"kernel"})
    bundles.create_bundle(source, bundle)
    (bundle / "files" / "kernels" / "fused.cu").unlink()

    with pytest.raises(bundles.BundleError, match="missing=.*kernels/fused.cu"):
        bundles.verify_bundle(bundle)


def test_missing_solution_fails_without_publishing_partial_bundle(
    tmp_path: Path,
) -> None:
    source = tmp_path / "problem"
    source.mkdir()
    (source / "kernel.cu").write_bytes(b"kernel")
    bundle = tmp_path / "bundle"

    with pytest.raises(bundles.BundleError, match="required artifact is missing"):
        bundles.create_bundle(source, bundle)
    assert not bundle.exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_source_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "problem"
    outside = tmp_path / "outside.cu"
    outside.write_bytes(b"not part of this submission")
    _write_submission(source)
    (source / "kernel.cu").symlink_to(outside)

    with pytest.raises(bundles.BundleError, match="symbolic links are not allowed"):
        bundles.create_bundle(source, tmp_path / "bundle")


def test_source_hard_link_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "problem"
    outside = tmp_path / "outside.cu"
    outside.write_bytes(b"not independently owned by the submission")
    _write_submission(source)
    os.link(outside, source / "kernel.cu")

    with pytest.raises(bundles.BundleError, match="hard-linked artifacts"):
        bundles.create_bundle(source, tmp_path / "bundle")


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(stem=_TRUSTED_MODULE_STEMS)
def test_top_level_package_directory_cannot_shadow_a_trusted_module(
    tmp_path: Path,
    stem: str,
) -> None:
    case = tmp_path / "package-shadow"
    shutil.rmtree(case, ignore_errors=True)
    case.mkdir()
    source = case / "problem"
    _write_submission(source)
    (source / stem).mkdir()

    with pytest.raises(bundles.BundleError, match="shadows a trusted Python module"):
        bundles.create_bundle(source, case / "bundle")


@settings(
    max_examples=35,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(stem=_TRUSTED_MODULE_STEMS, suffix=_ABI_EXTENSION_SUFFIXES)
def test_top_level_extension_cannot_shadow_a_trusted_module(
    tmp_path: Path,
    stem: str,
    suffix: str,
) -> None:
    case = tmp_path / "extension-shadow"
    shutil.rmtree(case, ignore_errors=True)
    case.mkdir()
    source = case / "problem"
    _write_submission(source)
    (source / f"{stem}{suffix}").write_bytes(b"extension")

    with pytest.raises(bundles.BundleError, match="shadows a trusted Python module"):
        bundles.create_bundle(source, case / "bundle")


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(stem=_TRUSTED_MODULE_STEMS, suffix=_ABI_EXTENSION_SUFFIXES)
def test_manifest_cannot_introduce_a_top_level_extension_shadow(
    tmp_path: Path,
    stem: str,
    suffix: str,
) -> None:
    case = tmp_path / "manifest-shadow"
    shutil.rmtree(case, ignore_errors=True)
    case.mkdir()
    source = case / "problem"
    bundle = case / "bundle"
    _write_submission(source)
    (source / "native_sidecar.so").write_bytes(b"extension")
    bundles.create_bundle(source, bundle)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shadow_path = f"{stem}{suffix}"
    entry = next(
        item for item in manifest["files"] if item["path"] == "native_sidecar.so"
    )
    entry["path"] = shadow_path
    manifest["files"].sort(key=lambda item: item["path"])
    manifest["bundle_sha256"] = bundles._descriptor_digest(manifest)
    manifest_path.write_bytes(bundles._canonical_json(manifest))
    (bundle / "files" / "native_sidecar.so").rename(bundle / "files" / shadow_path)

    with pytest.raises(bundles.BundleError, match="shadows a trusted Python module"):
        bundles.verify_bundle(bundle)


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(stem=_TRUSTED_MODULE_STEMS)
def test_manifest_cannot_introduce_a_top_level_package_shadow(
    tmp_path: Path,
    stem: str,
) -> None:
    case = tmp_path / "manifest-package-shadow"
    shutil.rmtree(case, ignore_errors=True)
    case.mkdir()
    source = case / "problem"
    bundle = case / "bundle"
    _write_submission(source)
    (source / "native_sidecar.py").write_bytes(b"helper")
    bundles.create_bundle(source, bundle)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shadow_path = f"{stem}/__init__.py"
    entry = next(
        item for item in manifest["files"] if item["path"] == "native_sidecar.py"
    )
    entry["path"] = shadow_path
    manifest["files"].sort(key=lambda item: item["path"])
    manifest["bundle_sha256"] = bundles._descriptor_digest(manifest)
    manifest_path.write_bytes(bundles._canonical_json(manifest))
    shadow_parent = bundle / "files" / stem
    shadow_parent.mkdir()
    (bundle / "files" / "native_sidecar.py").rename(shadow_parent / "__init__.py")

    with pytest.raises(bundles.BundleError, match="shadows a trusted Python module"):
        bundles.verify_bundle(bundle)


def test_unrelated_top_level_and_nested_module_sidecars_remain_valid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "problem"
    bundle = tmp_path / "bundle"
    extracted = tmp_path / "extracted"
    _write_submission(source)
    (source / "custom_kernel.so").write_bytes(b"extension")
    (source / "helpers" / "solution").mkdir(parents=True)
    (source / "helpers" / "solution" / "__init__.py").write_bytes(b"helper")
    (source / "helpers" / "reference.abi3.so").write_bytes(b"nested extension")

    created = bundles.create_bundle(source, bundle)
    bundles.extract_bundle(bundle, extracted, expected_digest=created["bundle_sha256"])

    assert (extracted / "custom_kernel.so").read_bytes() == b"extension"
    assert (extracted / "helpers" / "solution" / "__init__.py").read_bytes() == (
        b"helper"
    )
    assert (extracted / "helpers" / "reference.abi3.so").read_bytes() == (
        b"nested extension"
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation requires POSIX")
def test_source_special_file_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "problem"
    _write_submission(source)
    os.mkfifo(source / "compile.pipe")

    with pytest.raises(bundles.BundleError, match="FIFO"):
        bundles.create_bundle(source, tmp_path / "bundle")


def test_size_limit_failure_is_atomic(tmp_path: Path) -> None:
    source = tmp_path / "problem"
    _write_submission(source, solution=b"too large")
    bundle = tmp_path / "bundle"

    with pytest.raises(bundles.BundleError, match="per-file limit"):
        bundles.create_bundle(
            source,
            bundle,
            limits=bundles.BundleLimits(
                max_files=2, max_file_bytes=3, max_total_bytes=3
            ),
        )
    assert not bundle.exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_extra_payload_and_manifest_tampering_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "problem"
    bundle = tmp_path / "bundle"
    _write_submission(source)
    created = bundles.create_bundle(source, bundle)

    (bundle / "files" / "unlisted.cu").write_bytes(b"surprise")
    with pytest.raises(bundles.BundleError, match="extra=.*unlisted.cu"):
        bundles.verify_bundle(bundle)
    (bundle / "files" / "unlisted.cu").unlink()

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["total_size"] += 1
    manifest_path.write_bytes(bundles._canonical_json(manifest))
    with pytest.raises(bundles.BundleError, match="manifest digest"):
        bundles.verify_bundle(bundle, expected_digest=created["bundle_sha256"])


def test_extract_never_replaces_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "problem"
    bundle = tmp_path / "bundle"
    destination = tmp_path / "destination"
    _write_submission(source)
    bundles.create_bundle(source, bundle)
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_bytes(b"untouched")

    with pytest.raises(bundles.BundleError, match="already exists"):
        bundles.extract_bundle(bundle, destination)
    assert sentinel.read_bytes() == b"untouched"


def test_expected_sha256_alias_binds_verification(tmp_path: Path) -> None:
    source = tmp_path / "problem"
    bundle = tmp_path / "bundle"
    _write_submission(source)
    manifest = bundles.create_bundle(source, bundle)

    assert (
        bundles.verify_bundle(bundle, expected_sha256=manifest["bundle_sha256"])
        == manifest
    )
    with pytest.raises(bundles.BundleError, match="bundle digest mismatch"):
        bundles.verify_bundle(bundle, expected_sha256="0" * 64)


def _verified_result(run: Path, digest: str) -> dict[str, object]:
    return {
        "run_id": run.name,
        "has_solution": True,
        "correct": True,
        "peak_fraction": 0.25,
        "check_exit_code": 0,
        "benchmark_exit_code": 0,
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


def _verified_regrade(
    digest: str,
    *,
    stage_count: int = 2,
    check_exit_code: int = 0,
    benchmark_exit_code: int | None = 0,
) -> dict[str, object]:
    return {
        "status": "verified",
        "submission_bundle_sha256": digest,
        "grader_surface_sha256": "2" * 64,
        "stage_count": stage_count,
        "check_exit_code": check_exit_code,
        "benchmark_exit_code": benchmark_exit_code,
        "fresh_extraction": True,
        "fresh_caches": True,
        "network_isolated": True,
        "network_isolation": "unshare-user-mount-pid-net-private-root-v1",
        "mount_isolated": True,
        "root_isolated": True,
        "pid_isolated": True,
        "clean_environment": True,
        "in_process_completion_guard": True,
    }


def _fresh_verified_run(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    case = tmp_path / "verified-metrics"
    shutil.rmtree(case, ignore_errors=True)
    run = case / "20260808_120000_test_model_01_problem"
    source = case / "problem"
    run.mkdir(parents=True)
    _write_submission(source)
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    return run, _verified_result(run, manifest["bundle_sha256"]), manifest


def _not_applicable_result(run: Path) -> dict[str, object]:
    return {
        "run_id": run.name,
        "has_solution": False,
        "correct": False,
        "peak_fraction": None,
        "submission_bundle_sha256": None,
        "submission_replay_status": "not_applicable",
        "submission_bundle": None,
        "submission_replay": {
            "status": "not_applicable",
            "bundle_sha256": None,
        },
    }


def _verify_archived_run(
    run: Path,
    result: dict[str, object] | None = None,
    **kwargs: object,
) -> dict[str, object] | None:
    if result is not None:
        (run / "result.json").write_text(json.dumps(result))
    return bundles.verify_run_provenance(run, result, **kwargs)


def test_run_provenance_binds_publishable_score_to_bundle(tmp_path: Path) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    source = tmp_path / "problem"
    run.mkdir()
    _write_submission(source, sidecars={("kernels", "fused.cu"): b"kernel"})
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    result = _verified_result(run, manifest["bundle_sha256"])
    (run / "result.json").write_text(json.dumps(result))

    assert _verify_archived_run(run) == manifest

    result["submission_replay"]["fresh_caches"] = False  # type: ignore[index]
    with pytest.raises(bundles.BundleError, match="fresh_caches"):
        _verify_archived_run(run, result)
    result["submission_replay"]["fresh_caches"] = True  # type: ignore[index]
    (run / bundles.RUN_BUNDLE_DIR / "files" / "solution.py").write_text("tampered")
    with pytest.raises(bundles.BundleError, match="does not match manifest"):
        _verify_archived_run(run, result)


def test_supplied_result_cannot_replace_missing_archived_result(tmp_path: Path) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    result = {
        "run_id": run.name,
        "has_solution": True,
        "correct": False,
        "peak_fraction": None,
    }

    with pytest.raises(
        bundles.BundleError,
        match="inspect archived result.json|safely open archived result.json",
    ):
        bundles.verify_run_provenance(run, result)


def test_supplied_result_must_match_pinned_archive_snapshot(tmp_path: Path) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    archived = {
        "run_id": run.name,
        "has_solution": True,
        "correct": False,
        "peak_fraction": None,
    }
    (run / "result.json").write_text(json.dumps(archived))
    supplied = {**archived, "correct": True, "peak_fraction": 0.5}

    with pytest.raises(bundles.BundleError, match="does not match the pinned"):
        bundles.verify_run_provenance(run, supplied)


@pytest.mark.parametrize("entry_kind", ["symlink", "directory", "hardlink"])
def test_archived_result_rejects_unsafe_filesystem_entries(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    result_path = run / "result.json"
    encoded = json.dumps(
        {
            "run_id": run.name,
            "has_solution": True,
            "correct": False,
            "peak_fraction": None,
        }
    )
    if entry_kind == "directory":
        result_path.mkdir()
    else:
        outside = tmp_path / "outside-result.json"
        outside.write_text(encoded)
        if entry_kind == "symlink":
            result_path.symlink_to(outside)
        else:
            os.link(outside, result_path)

    with pytest.raises(
        bundles.BundleError,
        match="safely open|must be a regular file|hard-linked",
    ):
        bundles.verify_run_provenance(run)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation requires POSIX")
def test_archived_result_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    os.mkfifo(run / "result.json")

    with pytest.raises(bundles.BundleError, match="regular file, found FIFO"):
        bundles.verify_run_provenance(run)


def test_archived_result_size_is_bounded_before_json_decode(tmp_path: Path) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    with (run / "result.json").open("wb") as stream:
        stream.truncate(bundles.MAX_RESULT_BYTES + 1)

    with pytest.raises(bundles.BundleError, match="result.json exceeds limit"):
        bundles.verify_run_provenance(run)


def test_archived_result_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    run_id = json.dumps(run.name)
    (run / "result.json").write_text(
        f'{{"run_id":{run_id},"run_id":{run_id},"correct":false}}'
    )

    with pytest.raises(bundles.BundleError, match="duplicate key.*run_id"):
        bundles.verify_run_provenance(run)


def test_archived_bundle_symlink_is_rejected(tmp_path: Path) -> None:
    run, result, _manifest = _fresh_verified_run(tmp_path)
    bundle = run / bundles.RUN_BUNDLE_DIR
    real_bundle = run / "moved-submission-bundle"
    bundle.rename(real_bundle)
    bundle.symlink_to(real_bundle, target_is_directory=True)

    with pytest.raises(bundles.BundleError, match="real directory"):
        _verify_archived_run(run, result)


def test_bundle_path_rejects_intermediate_symlink(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    source = real_parent / "problem"
    bundle = real_parent / "bundle"
    real_parent.mkdir()
    _write_submission(source)
    bundles.create_bundle(source, bundle)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(bundles.BundleError, match="real directory, not a link"):
        bundles.verify_bundle(linked_parent / bundle.name)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation requires POSIX")
def test_archived_bundle_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    run = tmp_path / "20260808_120000_test_model_01_problem"
    run.mkdir()
    os.mkfifo(run / bundles.RUN_BUNDLE_DIR)
    result = _verified_result(run, "1" * 64)

    with pytest.raises(bundles.BundleError, match="real directory"):
        _verify_archived_run(run, result)


def test_archived_bundle_is_opened_relative_to_pinned_run_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, result, manifest = _fresh_verified_run(tmp_path)
    original_open_directory = bundles._open_directory
    bundle_parent_fds: list[int] = []

    def track_open_directory(
        path: str | os.PathLike[str],
        *,
        label: str,
        dir_fd: int | None = None,
    ) -> int:
        if path == bundles.RUN_BUNDLE_DIR:
            assert dir_fd is not None
            bundle_parent_fds.append(dir_fd)
        return original_open_directory(path, label=label, dir_fd=dir_fd)

    monkeypatch.setattr(bundles, "_open_directory", track_open_directory)
    assert _verify_archived_run(run, result) == manifest
    assert bundle_parent_fds


@pytest.mark.parametrize("correct", [None, 0, 1, "true", [], {}])
def test_bundle_aware_correctness_must_be_an_exact_boolean(
    tmp_path: Path,
    correct: object,
) -> None:
    run, result, _ = _fresh_verified_run(tmp_path)
    result["correct"] = correct

    with pytest.raises(bundles.BundleError, match="correct must be a boolean"):
        _verify_archived_run(run, result)


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(peak_fraction=_VALID_PEAK_FRACTIONS)
def test_every_finite_in_range_numeric_score_is_publishable(
    tmp_path: Path,
    peak_fraction: int | float,
) -> None:
    run, result, manifest = _fresh_verified_run(tmp_path)
    result["peak_fraction"] = peak_fraction

    assert _verify_archived_run(run, result) == manifest


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(peak_fraction=_INVALID_PEAK_FRACTIONS)
def test_every_non_numeric_non_finite_or_out_of_range_passing_score_is_rejected(
    tmp_path: Path,
    peak_fraction: object,
) -> None:
    run, result, _ = _fresh_verified_run(tmp_path)
    result["peak_fraction"] = peak_fraction

    with pytest.raises(
        bundles.BundleError,
        match="finite numeric peak_fraction|invalid JSON number",
    ):
        _verify_archived_run(run, result)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(peak_fraction=_VALID_PEAK_FRACTIONS)
def test_incorrect_result_never_carries_a_score(
    tmp_path: Path,
    peak_fraction: int | float,
) -> None:
    run, result, _ = _fresh_verified_run(tmp_path)
    result["correct"] = False
    result["peak_fraction"] = peak_fraction

    with pytest.raises(bundles.BundleError, match="must not carry a peak_fraction"):
        _verify_archived_run(run, result)


def test_incorrect_result_without_a_score_remains_valid(tmp_path: Path) -> None:
    run, result, manifest = _fresh_verified_run(tmp_path)
    result["correct"] = False
    result["peak_fraction"] = None
    result["submission_replay"]["stage_count"] = 1  # type: ignore[index]
    result["check_exit_code"] = 1
    result["benchmark_exit_code"] = None

    assert _verify_archived_run(run, result) == manifest


@pytest.mark.parametrize("bad_zero", [False, 0.0, "0", None])
@pytest.mark.parametrize("field", ["check_exit_code", "benchmark_exit_code"])
def test_passing_exit_codes_must_be_exact_integer_zero(
    tmp_path: Path,
    field: str,
    bad_zero: object,
) -> None:
    run, result, _ = _fresh_verified_run(tmp_path)
    result[field] = bad_zero

    with pytest.raises(bundles.BundleError, match=rf"{field}=0"):
        _verify_archived_run(run, result)


@pytest.mark.parametrize("has_solution", [None, 0, 1, "true", [], {}])
def test_bundle_aware_has_solution_must_be_an_exact_boolean(
    tmp_path: Path,
    has_solution: object,
) -> None:
    run, result, _ = _fresh_verified_run(tmp_path)
    result["has_solution"] = has_solution

    with pytest.raises(bundles.BundleError, match="has_solution must be a boolean"):
        _verify_archived_run(run, result)


def test_verified_replay_cannot_hide_its_bundled_solution(tmp_path: Path) -> None:
    run, result, _ = _fresh_verified_run(tmp_path)
    result["has_solution"] = False

    with pytest.raises(bundles.BundleError, match="has_solution=true"):
        _verify_archived_run(run, result)


@pytest.mark.parametrize(
    "field",
    [
        "network_isolated",
        "mount_isolated",
        "root_isolated",
        "pid_isolated",
        "clean_environment",
    ],
)
def test_publishable_replay_requires_every_isolation_control(
    tmp_path: Path,
    field: str,
) -> None:
    run = tmp_path / "20260808_120000_test_model_01_problem"
    source = tmp_path / "problem"
    run.mkdir()
    _write_submission(source)
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    result = _verified_result(run, manifest["bundle_sha256"])
    result["submission_replay"][field] = False  # type: ignore[index]

    with pytest.raises(bundles.BundleError, match=field):
        _verify_archived_run(run, result)


def test_in_process_completion_guard_is_advisory_metadata(tmp_path: Path) -> None:
    run, result, manifest = _fresh_verified_run(tmp_path)
    replay = result["submission_replay"]
    assert isinstance(replay, dict)
    replay["in_process_completion_guard"] = False

    # Bundle provenance never promotes the same-interpreter receipt to a
    # completion proof. Publication applies its independent annotation and
    # static-HACK gates after this structural archive verification.
    assert _verify_archived_run(run, result) == manifest

    replay.pop("in_process_completion_guard")
    assert _verify_archived_run(run, result) == manifest

    regrade = _verified_regrade(manifest["bundle_sha256"])
    regrade["in_process_completion_guard"] = False
    result["regrade"] = regrade
    assert _verify_archived_run(run, result) == manifest


def test_publishable_replay_rejects_the_legacy_finite_mask_backend(
    tmp_path: Path,
) -> None:
    run, result, _ = _fresh_verified_run(tmp_path)
    result["submission_replay"]["network_isolation"] = (  # type: ignore[index]
        "unshare-user-mount-pid-net"
    )

    with pytest.raises(bundles.BundleError, match="full isolation backend"):
        _verify_archived_run(run, result)


def test_scored_replay_requires_both_isolated_stages(tmp_path: Path) -> None:
    run = tmp_path / "20260808_120000_test_model_01_problem"
    source = tmp_path / "problem"
    run.mkdir()
    _write_submission(source)
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    result = _verified_result(run, manifest["bundle_sha256"])
    result["submission_replay"]["stage_count"] = 1  # type: ignore[index]

    with pytest.raises(bundles.BundleError, match="two isolated stages"):
        _verify_archived_run(run, result)


def test_legacy_policy_accepts_real_pre_cutover_archive_shape(tmp_path: Path) -> None:
    old_run = tmp_path / "20260807_235959_test_model_01_problem"
    old_run.mkdir()
    old_result = {
        "run_id": old_run.name,
        "has_solution": True,
        "correct": True,
        "peak_fraction": 0.25,
        "regrade": {
            "at": "2026-08-07T23:59:59+00:00",
            "host": "legacy-worker",
            "gpu_index": 0,
            "mode": "sequential_isolated",
        },
    }

    assert _verify_archived_run(old_run, old_result, allow_legacy=True) is None
    with pytest.raises(bundles.BundleError, match="metadata is required"):
        _verify_archived_run(old_run, old_result, allow_legacy=False)


@pytest.mark.parametrize(
    ("correct", "peak_fraction", "message"),
    [
        (1, 0.25, "correct must be a boolean"),
        (False, 0, "must not carry a peak_fraction"),
        (True, float("nan"), "finite numeric peak_fraction|invalid JSON number"),
    ],
)
def test_grandfathered_results_still_require_well_typed_metrics(
    tmp_path: Path,
    correct: object,
    peak_fraction: object,
    message: str,
) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    result = {
        "run_id": run.name,
        "has_solution": True,
        "correct": correct,
        "peak_fraction": peak_fraction,
    }

    with pytest.raises(bundles.BundleError, match=message):
        _verify_archived_run(run, result, allow_legacy=True)


@pytest.mark.parametrize(
    "extra_metadata",
    [
        {},
        {"archived_submission_bundle": None},
    ],
    ids=["deleted", "renamed"],
)
def test_post_cutover_metadata_deletion_or_renaming_is_rejected(
    tmp_path: Path,
    extra_metadata: dict[str, object],
) -> None:
    new_run = tmp_path / "20260808_000000_test_model_01_problem"
    new_run.mkdir()
    result = {"run_id": new_run.name, "has_solution": True, **extra_metadata}

    with pytest.raises(bundles.BundleError, match="metadata is required"):
        _verify_archived_run(new_run, result, allow_legacy=True)


@pytest.mark.parametrize(
    "run_id",
    [
        "20260807_246000_test_model_01_problem",
        "20260807_120000_",
        "legacy_test_model_01_problem",
    ],
)
def test_malformed_run_id_cannot_claim_cutoff_legacy(
    tmp_path: Path,
    run_id: str,
) -> None:
    run = tmp_path / run_id
    run.mkdir()
    result = {
        "run_id": run.name,
        "has_solution": True,
        "correct": False,
        "peak_fraction": None,
    }

    with pytest.raises(bundles.BundleError, match="metadata is required"):
        _verify_archived_run(run, result, allow_legacy=True)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("allow_legacy", 1),
        ("enforce_cutover", 0),
        ("enforce_cutover", None),
    ],
)
def test_legacy_policy_switches_require_exact_booleans(
    tmp_path: Path,
    option: str,
    value: object,
) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    result = {
        "run_id": run.name,
        "has_solution": True,
        "correct": False,
        "peak_fraction": None,
    }

    with pytest.raises(bundles.BundleError, match=rf"{option} must be a boolean"):
        _verify_archived_run(run, result, **{option: value})


def test_disabling_cutoff_is_an_explicit_legacy_archive_policy(
    tmp_path: Path,
) -> None:
    run = tmp_path / "20260809_120000_legacy_runner_01_problem"
    run.mkdir()
    result = {
        "run_id": run.name,
        "has_solution": True,
        "correct": False,
        "peak_fraction": None,
    }

    assert (
        _verify_archived_run(
            run,
            result,
            allow_legacy=True,
            enforce_cutover=False,
        )
        is None
    )


def test_renamed_post_cutover_archive_cannot_claim_legacy(tmp_path: Path) -> None:
    original = tmp_path / "20260808_120000_test_model_01_problem"
    original.mkdir()
    result = {"run_id": original.name, "has_solution": True}
    renamed = tmp_path / "20260807_120000_test_model_01_problem"
    original.rename(renamed)

    with pytest.raises(
        bundles.BundleError, match="does not match its archive directory"
    ):
        _verify_archived_run(renamed, result, allow_legacy=True)


@pytest.mark.parametrize(
    "result",
    [
        {"has_solution": True},
        {"run_id": None, "has_solution": True},
        {"run_id": 7, "has_solution": True},
    ],
    ids=["missing", "null", "non-string"],
)
def test_legacy_return_requires_a_typed_run_id(
    tmp_path: Path,
    result: dict[str, object],
) -> None:
    old_run = tmp_path / "20260807_120000_test_model_01_problem"
    old_run.mkdir()

    with pytest.raises(bundles.BundleError, match="run_id must be a string"):
        _verify_archived_run(old_run, result, allow_legacy=True)


def test_not_applicable_return_still_validates_run_identity(tmp_path: Path) -> None:
    run = tmp_path / "20260808_120000_test_model_01_problem"
    run.mkdir()
    result = _not_applicable_result(run)
    result["run_id"] = "20260808_120000_other_archive"

    with pytest.raises(
        bundles.BundleError, match="does not match its archive directory"
    ):
        _verify_archived_run(run, result)


@pytest.mark.parametrize("mode", ["legacy", "not_applicable"])
def test_run_directory_must_be_real_not_a_symlink(tmp_path: Path, mode: str) -> None:
    target = tmp_path / "target"
    target.mkdir()
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.symlink_to(target, target_is_directory=True)
    result = (
        {"run_id": run.name, "has_solution": True}
        if mode == "legacy"
        else _not_applicable_result(run)
    )

    with pytest.raises(bundles.BundleError, match="real directory, not a link"):
        _verify_archived_run(run, result, allow_legacy=True)


def test_run_path_rejects_intermediate_symlink(tmp_path: Path) -> None:
    real_runs = tmp_path / "real-runs"
    run = real_runs / "20260808_120000_test_model_01_problem"
    run.mkdir(parents=True)
    result = _not_applicable_result(run)
    (run / "result.json").write_text(json.dumps(result))
    linked_runs = tmp_path / "linked-runs"
    linked_runs.symlink_to(real_runs, target_is_directory=True)

    with pytest.raises(bundles.BundleError, match="real directory, not a link"):
        bundles.verify_run_provenance(linked_runs / run.name)


def test_present_null_provenance_keys_do_not_downgrade_to_legacy(
    tmp_path: Path,
) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    result = {
        "run_id": run.name,
        "has_solution": True,
        "correct": False,
        "peak_fraction": None,
        "submission_bundle": None,
        "submission_replay": None,
        "submission_bundle_sha256": None,
        "submission_replay_status": None,
    }

    with pytest.raises(bundles.BundleError, match="submission replay is not verified"):
        _verify_archived_run(
            run,
            result,
            allow_legacy=True,
            enforce_cutover=False,
        )


@pytest.mark.parametrize("regrade", [None, False, [], "legacy"])
def test_present_regrade_metadata_must_be_an_object(
    tmp_path: Path,
    regrade: object,
) -> None:
    run = tmp_path / "20260807_120000_test_model_01_problem"
    run.mkdir()
    result = {"run_id": run.name, "has_solution": True, "regrade": regrade}

    with pytest.raises(bundles.BundleError, match="regrade metadata must be an object"):
        _verify_archived_run(run, result, allow_legacy=True)


def test_valid_bundleless_attempt_remains_not_applicable(tmp_path: Path) -> None:
    run = tmp_path / "20260808_120000_test_model_01_problem"
    run.mkdir()

    assert _verify_archived_run(run, _not_applicable_result(run)) is None


def test_active_regrade_must_bind_to_same_bundle_and_controls(tmp_path: Path) -> None:
    run = tmp_path / "20260807_120001_test_model_01_problem"
    source = tmp_path / "problem"
    run.mkdir()
    _write_submission(source)
    manifest = bundles.create_bundle(source, run / bundles.RUN_BUNDLE_DIR)
    result = _verified_result(run, manifest["bundle_sha256"])
    result["regrade"] = {"status": "verification_failed"}
    with pytest.raises(bundles.BundleError, match="active regrade is not verified"):
        _verify_archived_run(run, result)

    result["regrade"] = _verified_regrade(manifest["bundle_sha256"])
    assert _verify_archived_run(run, result) == manifest


def test_active_regrade_stage_and_exits_override_original_replay(
    tmp_path: Path,
) -> None:
    run, result, manifest = _fresh_verified_run(tmp_path)
    digest = manifest["bundle_sha256"]
    result["submission_replay"]["stage_count"] = 1  # type: ignore[index]
    result["check_exit_code"] = 1
    result["benchmark_exit_code"] = 1
    result["regrade"] = _verified_regrade(digest)

    assert _verify_archived_run(run, result) == manifest


def test_passing_regrade_requires_its_own_two_completed_stages(
    tmp_path: Path,
) -> None:
    run, result, manifest = _fresh_verified_run(tmp_path)
    result["regrade"] = _verified_regrade(
        manifest["bundle_sha256"],
        stage_count=1,
        benchmark_exit_code=None,
    )

    with pytest.raises(bundles.BundleError, match="complete two isolated stages"):
        _verify_archived_run(run, result)


@pytest.mark.parametrize("bad_zero", [False, 0.0, "0", None])
@pytest.mark.parametrize("field", ["check_exit_code", "benchmark_exit_code"])
def test_passing_regrade_exit_codes_are_exact_integer_zero(
    tmp_path: Path,
    field: str,
    bad_zero: object,
) -> None:
    run, result, manifest = _fresh_verified_run(tmp_path)
    regrade = _verified_regrade(manifest["bundle_sha256"])
    regrade[field] = bad_zero
    result["regrade"] = regrade

    message = (
        "invalid check_exit_code"
        if field == "check_exit_code"
        else "invalid benchmark_exit_code"
    )
    with pytest.raises(bundles.BundleError, match=message):
        _verify_archived_run(run, result)


def test_one_stage_regrade_cannot_claim_a_benchmark_exit(tmp_path: Path) -> None:
    run, result, manifest = _fresh_verified_run(tmp_path)
    result["correct"] = False
    result["peak_fraction"] = None
    result["regrade"] = _verified_regrade(
        manifest["bundle_sha256"],
        stage_count=1,
        check_exit_code=1,
        benchmark_exit_code=0,
    )

    with pytest.raises(bundles.BundleError, match="must not carry"):
        _verify_archived_run(run, result)


def test_cli_create_verify_extract_contract(tmp_path: Path) -> None:
    source = tmp_path / "problem"
    bundle = tmp_path / "bundle"
    extracted = tmp_path / "extracted"
    _write_submission(source, sidecars={("helpers", "kernel.py"): b"sidecar"})
    tool = REPO / "scripts" / "submission_bundle.py"

    created = subprocess.run(
        [sys.executable, str(tool), "create", str(source), str(bundle)],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(created.stdout)
    assert set(summary) == {"bundle_sha256", "file_count", "total_size"}
    subprocess.run(
        [
            sys.executable,
            str(tool),
            "extract",
            str(bundle),
            str(extracted),
            "--expected-sha256",
            summary["bundle_sha256"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (extracted / "helpers" / "kernel.py").read_bytes() == b"sidecar"

    failed = subprocess.run(
        [
            sys.executable,
            str(tool),
            "verify",
            str(bundle),
            "--expected-sha256",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 2
    assert failed.stderr.startswith("submission bundle error:")
