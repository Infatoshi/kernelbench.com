"""Claude Code stream-json usage: a session can end several turns, one result event each.

Each result event's `usage` covers only its own turn; the terminal result's
`modelUsage` is cumulative for the session. Taking the last `usage` alone
undercounted multi-turn cells by up to 170x (2026-11-16 or-fable qwen4-max wave).
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHES = ("hard", "cuda", "mega", "mini")


def _load(bench):
    path = ROOT / "benchmarks" / bench / "scripts" / "extract_usage.py"
    spec = importlib.util.spec_from_file_location(f"extract_usage_{bench}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _result(inp, out, cr, cc, cost, model_usage=None):
    e = {"type": "result", "total_cost_usd": cost,
         "usage": {"input_tokens": inp, "output_tokens": out,
                   "cache_read_input_tokens": cr, "cache_creation_input_tokens": cc}}
    if model_usage is not None:
        e["modelUsage"] = model_usage
    return e


ASSISTANT = {"type": "assistant", "message": {"usage": {"output_tokens": 999}}}


@pytest.mark.parametrize("bench", BENCHES)
def test_model_usage_on_last_result_is_cumulative(bench):
    mu = {
        "main-model": {"inputTokens": 100, "outputTokens": 5000, "cacheReadInputTokens": 7000,
                       "cacheCreationInputTokens": 30, "thinkingTokens": 4000, "costUSD": 1.0},
        "compact-model": {"inputTokens": 20, "outputTokens": 300, "cacheReadInputTokens": 0,
                          "cacheCreationInputTokens": 5, "costUSD": 0.1},
    }
    events = [ASSISTANT, _result(50, 4000, 3000, 10, 0.6), ASSISTANT,
              _result(10, 12, 400, 1, 1.1, model_usage=mu)]
    assert _load(bench)._claude_or_kimi(events) == {
        "input_tokens": 120, "output_tokens": 5300, "cache_read_tokens": 7000,
        "cache_creation_tokens": 35, "reasoning_tokens": None, "total_cost_usd": 1.1,
    }


@pytest.mark.parametrize("bench", BENCHES)
def test_single_result_without_model_usage_unchanged(bench):
    events = [ASSISTANT, _result(50, 4000, 3000, 10, 0.6)]
    assert _load(bench)._claude_or_kimi(events) == {
        "input_tokens": 50, "output_tokens": 4000, "cache_read_tokens": 3000,
        "cache_creation_tokens": 10, "reasoning_tokens": None, "total_cost_usd": 0.6,
    }


@pytest.mark.parametrize("bench", BENCHES)
def test_unpriced_model_records_no_cost(bench):
    """Claude Code's fallback-rate estimate for a model it cannot price is not a bill."""
    mu = {"open/model": {"inputTokens": 100, "outputTokens": 5000, "cacheReadInputTokens": 7000,
                         "cacheCreationInputTokens": 0, "costUSD": 91.9, "costBasis": "unknown"}}
    events = [_result(10, 12, 400, 0, 91.9, model_usage=mu)]
    got = _load(bench)._claude_or_kimi(events)
    assert got["output_tokens"] == 5000
    assert got["total_cost_usd"] is None


@pytest.mark.parametrize("bench", BENCHES)
def test_multiple_results_without_model_usage_are_summed(bench):
    events = [ASSISTANT, _result(50, 4000, 3000, 10, 0.6), ASSISTANT,
              _result(10, 12, 400, 1, 1.1)]
    assert _load(bench)._claude_or_kimi(events) == {
        "input_tokens": 60, "output_tokens": 4012, "cache_read_tokens": 3400,
        "cache_creation_tokens": 11, "reasoning_tokens": None, "total_cost_usd": 1.1,
    }
