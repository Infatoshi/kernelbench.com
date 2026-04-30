"""Tests for hardware target registry."""

from src.hardware import TARGETS, get_target, list_targets


def test_all_targets_registered():
    assert "rtx3090" in TARGETS
    assert "h100" in TARGETS
    assert "b200" in TARGETS
    assert "m4max" in TARGETS


def test_list_targets():
    names = list_targets()
    assert len(names) == 4
    assert "rtx3090" in names


def test_rtx3090_config():
    t = get_target("rtx3090")
    assert t.display_name == "RTX 3090"
    assert t.vram_gb == 24
    assert not t.is_metal
    assert "level1" in t.problem_dirs


def test_m4max_config():
    t = get_target("m4max")
    assert t.display_name == "M4"
    assert t.is_metal
    assert "metal_level1" in t.problem_dirs
    assert "4_FP8_Matmul.py" in t.exclude_problems


def test_problem_discovery():
    from pathlib import Path
    t = get_target("rtx3090")
    problems = t.find_problems(Path("."))
    assert len(problems) > 0
    levels = set(lv for lv, _ in problems)
    assert 1 in levels
