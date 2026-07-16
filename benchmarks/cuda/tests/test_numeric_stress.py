import torch

from src.eval.correctness import check_correctness
from src.eval.numeric_stress import (
    NumericStressCase,
    numeric_stress_cases,
    numeric_stress_context,
    tolerance_for_case,
)


def test_small_numeric_case_catches_zero_output_cheat() -> None:
    ref = torch.full((64,), 1e-3, dtype=torch.float32)
    zero = torch.zeros_like(ref)

    assert check_correctness(ref, zero, override={"float32": 0.1})[0]

    case = NumericStressCase(
        "tiny",
        tolerance={"float32": {"atol": 1e-6, "rtol": 1e-4}},
    )
    ok, msg = check_correctness(ref, zero, override=tolerance_for_case({"float32": 0.1}, case))
    assert not ok
    assert "bad=64/64" in msg


def test_scaled_input_breaks_cached_nominal_answer() -> None:
    case = NumericStressCase("small_input", input_scales={0: 1e-3})
    x = torch.ones(8, dtype=torch.float32)
    cached_nominal = x * 2

    with numeric_stress_context(torch.nn.Identity(), torch.nn.Identity(), [x], case) as inputs:
        stressed_x = inputs[0]
        ref = stressed_x * 2

    ok, _ = check_correctness(ref, cached_nominal, dtype=torch.float32)
    assert not ok


def test_state_scale_applies_to_both_models_and_restores() -> None:
    ref = torch.nn.Linear(2, 2, bias=False)
    sol = torch.nn.Linear(2, 2, bias=False)
    sol.load_state_dict(ref.state_dict(), strict=True)
    original = ref.weight.detach().clone()

    case = NumericStressCase("small_weight", state_scales={"weight": 1e-2})
    with numeric_stress_context(ref, sol, [torch.ones(1, 2)], case):
        assert torch.allclose(ref.weight, original * 1e-2)
        assert torch.allclose(sol.weight, original * 1e-2)

    assert torch.equal(ref.weight, original)
    assert torch.equal(sol.weight, original)


def test_problem_cases_include_nominal_and_targeted_fp8_cases() -> None:
    names = [case.name for case in numeric_stress_cases("01_fp8_gemm")]
    assert names == ["nominal", "small_input", "large_input", "small_weight"]
