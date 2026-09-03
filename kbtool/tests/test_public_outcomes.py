from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_catalog", REPO / "scripts" / "build_catalog.py"
)
assert SPEC and SPEC.loader
build_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_catalog)

MODEL_SPEC = importlib.util.spec_from_file_location(
    "build_model_index", REPO / "scripts" / "build_model_index.py"
)
assert MODEL_SPEC and MODEL_SPEC.loader
build_model_index = importlib.util.module_from_spec(MODEL_SPEC)
MODEL_SPEC.loader.exec_module(build_model_index)


def outcome(
    *,
    correct: bool = False,
    has_solution: bool = True,
    failure_reason: str | None = None,
    verdict: str | None = "clean",
    text: str = "",
) -> str:
    return build_catalog.outcome_from_archive(
        correct=correct,
        has_solution=has_solution,
        failure_reason=failure_reason,
        verdict=verdict,
        text=text,
    )


def test_provider_failures_remain_distinct() -> None:
    assert outcome(has_solution=False, failure_reason="provider_rate_limited") == "rate_limit"
    assert outcome(has_solution=False, failure_reason="provider_insufficient_credits") == "credits"
    assert outcome(has_solution=False, failure_reason="provider_early_stop") == "provider_cut"
    assert outcome(has_solution=False, failure_reason="provider_unavailable") == "provider_unavailable"
    assert outcome(has_solution=False, failure_reason="harness_error") == "harness"


def test_evaluator_failures_remain_distinct() -> None:
    assert outcome(failure_reason="check_timeout") == "check_timeout"
    assert outcome(correct=True, failure_reason="benchmark_timeout") == "benchmark_timeout"
    assert outcome(correct=True, failure_reason="benchmark_failed") == "benchmark_failed"
    assert outcome(has_solution=False, failure_reason="incomplete_session") == "incomplete"
    assert outcome(has_solution=False, failure_reason="no_solution") == "empty"


def test_check_failure_uses_observed_failure_stage() -> None:
    assert outcome(failure_reason="check_failed", text="max_abs mismatch") == "wrong"
    assert outcome(failure_reason="check_failed", text="nvcc compilation failed") == "build"


def test_audit_rejection_preserves_exact_verdict() -> None:
    assert outcome(correct=True, verdict="reward_hack") == "reward_hack"
    assert outcome(correct=True, verdict="contamination") == "contamination"
    assert outcome(correct=True, verdict="rubric_leak") == "rubric_leak"


def test_annotation_outcomes_preserve_audited_failure_metadata() -> None:
    assert (
        build_model_index.annotation_outcome(
            {"failure_reason": "provider_insufficient_credits"}, "clean"
        )
        == "credits"
    )
    assert (
        build_model_index.annotation_outcome(
            {"measurement_status": "wrong_gpu"}, "clean"
        )
        == "hardware"
    )
    assert (
        build_model_index.annotation_outcome(
            {"correct": True, "publish_grade": False, "measurement_status": "ungraded"},
            "clean",
        )
        == "ungraded"
    )
    assert (
        build_model_index.annotation_outcome(
            {"correct": True, "peak_fraction": 19.1}, "clean"
        )
        == "pass"
    )
    assert (
        build_model_index.annotation_outcome(
            {"correct": True, "peak_fraction": 19.1}, "megakernel_not_authentic"
        )
        == "not_megakernel"
    )



def test_annotation_cell_selection_stays_on_active_deck() -> None:
    build_model_index.ACTIVE_PROBLEMS["hard"] = {"01_fp8_gemm"}
    choose = build_model_index.should_replace_annotation_cell

    assert not choose(
        bench="hard",
        problem="retired_problem",
        current=None,
        publishable=False,
        score=None,
        run_id="20260805_failed",
    )
    assert choose(
        bench="hard",
        problem="01_fp8_gemm",
        current={"valid": False, "run_id": "20260804_failed"},
        publishable=True,
        score=0.4,
        run_id="20260805_clean",
    )
    assert not choose(
        bench="hard",
        problem="01_fp8_gemm",
        current={"valid": True, "run_id": "20260804_clean", "score": 0.5},
        publishable=False,
        score=None,
        run_id="20260805_failed",
    )
    assert choose(
        bench="hard",
        problem="01_fp8_gemm",
        current={"valid": False, "run_id": "20260804_failed"},
        publishable=False,
        score=None,
        run_id="20260805_failed",
    )

def test_suspect_is_an_audit_reject_everywhere() -> None:
    """`suspect` (contaminated transcript, no other run's solution opened) is a
    verdict the annotations actually emit. It must never keep its number."""
    assert "suspect" in build_catalog.FLAG_VERDICTS
    assert "suspect" in build_model_index.FLAG_VERDICTS
    assert outcome(correct=True, verdict="suspect") == "suspect"
    assert {x["code"] for x in build_catalog.LEGEND} >= {"suspect"}
    ts = (REPO / "app" / "_lib" / "models.ts").read_text()
    flags = ts.split("export const FLAG_VERDICTS", 1)[1].split("])", 1)[0]
    assert '"suspect"' in flags


def test_only_audited_clean_cells_are_valid() -> None:
    valid = build_model_index.cell_is_valid
    assert valid(correct=True, score=0.4, verdict="clean")
    assert valid(correct=True, score=0.4, verdict="interesting")
    # correctness and a number are not enough on their own
    assert not valid(correct=False, score=0.4, verdict="clean")
    assert not valid(correct=True, score=None, verdict="clean")
    # audit rejects, including suspect
    for bad in ("reward_hack", "contamination", "rubric_leak", "suspect"):
        assert not valid(correct=True, score=0.4, verdict=bad)
    # never reviewed -> visible, but not ranking data
    assert not valid(correct=True, score=0.4, verdict="unaudited")
    assert not valid(correct=True, score=0.4, verdict=None)
    # hardware-ineligible runs stay off the board too
    assert not valid(correct=True, score=0.4, verdict="clean", board_eligible=False)


def test_published_index_has_no_unaudited_or_flagged_valid_cells() -> None:
    """The live artifact, not just the helper: every valid cell in
    public/data/models.json carries a non-flag, non-unaudited verdict."""
    import json

    path = REPO / "public" / "data" / "models.json"
    if not path.exists():
        return
    idx = json.loads(path.read_text())
    bad = []
    for model in idx["models"]:
        for bench, block in (model.get("benches") or {}).items():
            views = [("rtxpro6000", block)] + list((block.get("gpus") or {}).items())
            for gpu, view in views:
                for prob, cell in (view.get("cells") or {}).items():
                    verdict = cell.get("verdict") or "unaudited"
                    if cell.get("valid") and (
                        verdict == "unaudited"
                        or verdict in build_model_index.FLAG_VERDICTS
                    ):
                        bad.append((model["slug"], bench, gpu, prob, verdict))
    assert not bad, f"valid cells without a clean audit verdict: {bad}"
