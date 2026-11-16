"""Mega headline cell runs two sessions per model; the board row stays the best
session and names the other audited, scoring sessions (`other_sessions`)."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROB = "02_kimi_linear_decode"
GPU = "RTX PRO 6000 Blackwell"


@pytest.fixture
def builder(monkeypatch):
    """build_mega_leaderboard.py without torch: stub its one bench import."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("src", "src.eval"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    mk = types.ModuleType("src.eval.megakernel")

    def extract_evidence(run_dir):
        raise RuntimeError("stub")

    mk.extract_evidence = extract_evidence
    monkeypatch.setitem(sys.modules, "src.eval.megakernel", mk)
    spec = importlib.util.spec_from_file_location(
        "build_mega_leaderboard_t", REPO / "benchmarks/mega/scripts/build_mega_leaderboard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ann(d: Path, rid: str, **kw) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rid}.yaml").write_text("".join(f"{k}: {v}\n" for k, v in kw.items()))


def _row(rid: str, score: str, model: str = "m-1", correct: str = "true",
         run_group: str = "release-1") -> dict:
    return {"gpu": GPU, "model": model, "problem": PROB, "correct": correct,
            "score": score, "run_id": rid, "run_group": run_group}


def _best(rows):
    # same argmax rule as main()
    best = {}
    for r in rows:
        key = (r["gpu"], r["model"], r["problem"])
        cur = best.get(key)
        if cur is None or (r["score"] and float(r["score"] or 0) > float(cur["score"] or 0)):
            best[key] = r
    return best


def test_only_audited_scoring_siblings_count(builder, tmp_path):
    ann = tmp_path / "annotations"
    rows = [
        _row("a_head", "35.460"),
        _row("a_second", "31.200"),       # clean: printed
        _row("a_third", "20.000"),        # interesting: printed, after a_second
        _row("a_unaudited", "33.000"),    # no YAML: aborted / unaudited retry
        _row("a_bug", "34.000"),          # verdict bug
        _row("a_ungraded", "32.000"),     # clean but publish_grade false
        _row("a_ineligible", "32.500"),   # clean but board_eligible false
        _row("a_wrong", "", correct="false"),
        _row("a_old_release", "30.000", run_group="release-0"),   # clean, earlier release
        _row("a_no_group", "29.000", run_group=""),               # clean, no release group
        _row("b_only", "12.000", model="m-2"),
    ]
    _ann(ann, "a_old_release", verdict="clean")
    _ann(ann, "a_no_group", verdict="clean")
    _ann(ann, "a_head", verdict="clean")
    _ann(ann, "a_second", verdict="clean")
    _ann(ann, "a_third", verdict="interesting")
    _ann(ann, "a_bug", verdict="bug")
    _ann(ann, "a_ungraded", verdict="clean", publish_grade="false")
    _ann(ann, "a_ineligible", verdict="clean", board_eligible="false")
    _ann(ann, "a_wrong", verdict="clean")
    _ann(ann, "b_only", verdict="clean")
    best = _best(rows)
    out = builder.other_sessions(rows, best, ann)
    assert best[(GPU, "m-1", PROB)]["run_id"] == "a_head"   # headline = argmax, unchanged
    assert out == {(GPU, "m-1", PROB): "a_second:31.200;a_third:20.000"}


def test_unaudited_headline_still_lists_audited_sibling(builder, tmp_path):
    # Argmax is unchanged: an unaudited top row stays the headline (the site
    # marks it unaudited); the audited session is still printed beside it.
    ann = tmp_path / "annotations"
    rows = [_row("top_unaudited", "40.000"), _row("audited", "30.000")]
    _ann(ann, "audited", verdict="clean")
    best = _best(rows)
    assert builder.other_sessions(rows, best, ann) == {(GPU, "m-1", PROB): "audited:30.000"}


def _run(runs: Path, rid: str, pf: float, model: str = "m-1") -> None:
    d = runs / rid
    d.mkdir(parents=True)
    (d / "gpu").write_text(GPU + "\n")
    (d / "result.json").write_text(json.dumps({
        "run_id": rid, "has_solution": True, "correct": True, "peak_fraction": pf,
        "model": model, "harness": "claude", "problem": PROB, "run_group": "release-1",
    }))


def test_csv_column_and_model_index_parse(builder, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    _run(runs, "20261201_000000_claude_m-1_02_kimi_linear_decode", 35.46)
    _run(runs, "20261201_000100_claude_m-1_02_kimi_linear_decode", 31.2)
    _run(runs, "20261201_000200_claude_m-2_02_kimi_linear_decode", 12.0, model="m-2")
    scoring = {"20261201_000100_claude_m-1_02_kimi_linear_decode"}
    monkeypatch.setattr(builder, "scoring_session", lambda rid, d: rid in scoring)
    out = tmp_path / "results.csv"
    monkeypatch.setattr(sys, "argv", ["b", "--runs", str(runs), "--out", str(out),
                                      "--runs-html", str(tmp_path / "nohtml")])
    builder.main()

    rows = list(csv.DictReader(out.open()))
    assert list(rows[0])[-2:] == ["run_id", "other_sessions"]
    by_model = {r["model"]: r for r in rows}
    assert by_model["m-1"]["run_id"] == "20261201_000000_claude_m-1_02_kimi_linear_decode"
    assert by_model["m-1"]["score"] == "35.460"
    assert by_model["m-1"]["other_sessions"] == "20261201_000100_claude_m-1_02_kimi_linear_decode:31.200"
    assert by_model["m-2"]["other_sessions"] == ""
    # The site's TS readers split lines on "," without quoting.
    assert all("," not in r["other_sessions"] for r in rows)

    spec = importlib.util.spec_from_file_location(
        "build_model_index_t", REPO / "scripts" / "build_model_index.py"
    )
    bmi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bmi)
    models = bmi.Models()
    bmi.load_mega(models, out)
    cells = {s: e["benches"]["mega"]["cells"][PROB] for s, e in models.by_slug.items()}
    m1 = next(c for c in cells.values() if c["score"] == 35.46)
    assert m1["other_sessions"] == [
        {"run_id": "20261201_000100_claude_m-1_02_kimi_linear_decode", "score": 31.2}
    ]
    assert all("other_sessions" not in c for c in cells.values() if c is not m1)
