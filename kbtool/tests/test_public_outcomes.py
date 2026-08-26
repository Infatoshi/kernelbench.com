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