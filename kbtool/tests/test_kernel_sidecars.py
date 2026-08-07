"""The publisher resolves the same sidecar path as the graded loader."""

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.kernel_sidecars import SidecarError, augment  # noqa: E402


def test_verified_root_sidecar_wins_over_nested_same_basename(tmp_path: Path) -> None:
    (tmp_path / "kernel.cu").write_text("GRADED_ROOT")
    decoy = tmp_path / "repo/problems/01_problem/kernel.cu"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("DECOY")

    rendered = augment('SOURCES = ["kernel.cu"]\n', tmp_path)

    assert "GRADED_ROOT" in rendered
    assert "DECOY" not in rendered


def test_referenced_relative_sidecar_path_is_preserved(tmp_path: Path) -> None:
    expected = tmp_path / "kernels/fused.cu"
    expected.parent.mkdir()
    expected.write_text("EXPECTED")
    (tmp_path / "fused.cu").write_text("WRONG_BASENAME")

    rendered = augment('SOURCES = ["kernels/fused.cu"]\n', tmp_path)

    assert "EXPECTED" in rendered
    assert "WRONG_BASENAME" not in rendered


def test_temp_extraction_location_does_not_hide_sidecars(tmp_path: Path) -> None:
    (tmp_path / "helper.cu").write_text("VISIBLE")

    rendered = augment('SOURCES = ["helper.cu"]\n', tmp_path)

    assert "VISIBLE" in rendered


def test_glob_metacharacters_are_literal_sidecar_names(tmp_path: Path) -> None:
    (tmp_path / "*.cu").write_text("LITERAL")
    (tmp_path / "!.cu").write_text("DECOY")

    rendered = augment('sources = ["*.cu"]\n', tmp_path)

    assert "LITERAL" in rendered
    assert "DECOY" not in rendered


def test_verified_exact_path_bypasses_legacy_cache_name_filter(tmp_path: Path) -> None:
    sidecar = tmp_path / "tmp/kernel.cu"
    sidecar.parent.mkdir()
    sidecar.write_text("BUNDLE_EXACT")

    rendered = augment(
        'sources = ["tmp/kernel.cu"]\n',
        tmp_path,
        exact=True,
        strict=True,
    )

    assert "BUNDLE_EXACT" in rendered


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_unsafe_exact_sidecar_is_rejected(tmp_path: Path, kind: str) -> None:
    sidecar = tmp_path / "kernel.cu"
    victim = tmp_path / "victim.cu"
    victim.write_text("VICTIM")
    if kind == "symlink":
        sidecar.symlink_to(victim)
    elif kind == "hardlink":
        os.link(victim, sidecar)
    else:
        os.mkfifo(sidecar)

    with pytest.raises(SidecarError):
        augment(
            'sources = ["kernel.cu"]\n',
            tmp_path,
            exact=True,
            strict=True,
        )


def test_oversized_sidecar_is_rejected_before_reading_unbounded(
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "kernel.cu"
    with sidecar.open("wb") as stream:
        stream.truncate(1_500_001)

    with pytest.raises(SidecarError, match="size cap"):
        augment(
            'sources = ["kernel.cu"]\n',
            tmp_path,
            exact=True,
            strict=True,
        )


def test_parent_traversal_reference_is_not_published(tmp_path: Path) -> None:
    victim = tmp_path.parent / "victim.cu"
    victim.write_text("OUTSIDE")

    rendered = augment(
        'sources = ["../victim.cu"]\n', tmp_path, exact=True, strict=True
    )

    assert "OUTSIDE" not in rendered
