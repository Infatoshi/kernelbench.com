"""Unit tests for src/eval helpers. These run without a GPU."""
import pytest
import torch

from src.eval.correctness import check_correctness, tolerance_for_dtype
from src.eval.report import long_rows, render_markdown
from src.eval.roofline import compute_gbps, compute_tflops, peak_fraction
from src.eval.timing import benchmark_baselines_enabled
from src.hardware import get


def test_tolerance_defaults():
    assert tolerance_for_dtype(torch.float32)["atol"] == 1e-4
    assert tolerance_for_dtype(torch.bfloat16)["atol"] == 1e-2
    assert tolerance_for_dtype(torch.int8)["atol"] == 0


def test_tolerance_override_scalar():
    tol = tolerance_for_dtype(torch.float32, override={"torch.float32": 5e-4})
    assert tol == {"atol": 5e-4, "rtol": 5e-4}


def test_correctness_shape_mismatch():
    a = torch.zeros(4, 4)
    b = torch.zeros(4, 5)
    ok, msg = check_correctness(a, b)
    assert not ok
    assert "shape mismatch" in msg


def test_correctness_nan_fails():
    a = torch.zeros(4, 4)
    b = torch.zeros(4, 4)
    b[0, 0] = float("nan")
    ok, _ = check_correctness(a, b)
    assert not ok


def test_correctness_pass_within_tol():
    torch.manual_seed(0)
    a = torch.randn(128, 128, dtype=torch.bfloat16)
    b = a + torch.randn_like(a) * 1e-3  # well within 1e-2 atol
    ok, _ = check_correctness(a, b)
    assert ok


def test_correctness_int_exact():
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int8)
    b = a.clone()
    assert check_correctness(a, b)[0]
    b[0, 0] = 99
    assert not check_correctness(a, b)[0]


def test_tflops_math():
    # 1 TFLOPS for 1 TFLOP in 1 ms -> 1000 TFLOPS
    assert compute_tflops(1e12, 1.0) == pytest.approx(1000.0)
    # 100 GFLOPs in 0.1ms => 1 TFLOPS
    assert compute_tflops(1e11, 0.1) == pytest.approx(1000.0)


def test_gbps_math():
    # 1 GB in 1 ms -> 1000 GB/s
    assert compute_gbps(1e9, 1.0) == pytest.approx(1000.0)


def test_peak_fraction():
    assert peak_fraction(100, 400) == pytest.approx(0.25)
    assert peak_fraction(0, 400) == 0.0
    assert peak_fraction(100, 0) == 0.0


def test_hardware_lookup():
    hw = get("RTX_PRO_6000")
    assert hw.sm == "sm_120a"
    assert hw.peak_bandwidth_gb_s == pytest.approx(1800.0)
    assert hw.peak_tflops_dense["fp8"] == pytest.approx(1000.0)


def test_hardware_lookup_h100():
    hw = get("H100")
    assert hw.sm == "sm_90a"
    assert hw.vram_gb == 80
    assert hw.peak_bandwidth_gb_s == pytest.approx(2039.0)
    # H100 PCIe dense fp8 peak; bf16 is half of fp8.
    assert hw.peak_tflops_dense["fp8"] == pytest.approx(1513.0)
    assert hw.peak_tflops_dense["bf16"] == pytest.approx(756.0)


def test_benchmark_baselines_env_flags(monkeypatch):
    monkeypatch.delenv("KBH_BENCHMARK_BASELINES", raising=False)
    monkeypatch.delenv("KBH_KDA_BENCHMARK_BASELINES", raising=False)
    assert not benchmark_baselines_enabled("KDA")

    monkeypatch.setenv("KBH_KDA_BENCHMARK_BASELINES", "1")
    assert benchmark_baselines_enabled("KDA")

    monkeypatch.delenv("KBH_KDA_BENCHMARK_BASELINES", raising=False)
    monkeypatch.setenv("KBH_BENCHMARK_BASELINES", "1")
    assert benchmark_baselines_enabled("anything")


def test_report_long_rows_keep_problem_as_column():
    rows = long_rows(
        [
            {
                "run_id": "run-a",
                "problem": "01_fp8_gemm",
                "harness": "codex",
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
                "correct": True,
                "peak_fraction": 0.5,
                "usage": {"output_tokens": 123},
            },
            {
                "run_id": "run-b",
                "problem": "01_fp8_gemm",
                "harness": "codex",
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
                "correct": False,
                "peak_fraction": 0.25,
            },
        ]
    )

    aggregate = [r for r in rows if r["scope"] == "aggregate"]
    assert {
        (r["problem"], r["harness"], r["model"], r["metric"], r["value"]) for r in aggregate
    } == {
        ("01_fp8_gemm", "codex", "gpt-5.5", "runs", "2"),
        ("01_fp8_gemm", "codex", "gpt-5.5", "correct", "1"),
        ("01_fp8_gemm", "codex", "gpt-5.5", "gmean_peak_fraction", "0.5"),
    }

    assert {
        (r["run_id"], r["metric"], r["value"]) for r in rows if r["scope"] == "run"
    } >= {
        ("run-a", "correct", "True"),
        ("run-a", "peak_fraction", "0.5"),
        ("run-a", "usage.output_tokens", "123"),
        ("run-b", "correct", "False"),
    }

    markdown = render_markdown(rows)
    assert markdown.startswith("| problem | harness | model | reasoning_effort | run_id | scope | metric | value |")
    assert "| 01_fp8_gemm | codex | gpt-5.5 | xhigh | run-a | run | usage.output_tokens | 123 |" in markdown
