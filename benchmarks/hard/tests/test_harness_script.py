from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_HARD = ROOT / "scripts" / "run_hard.sh"
LAUNCH_PARALLEL = ROOT / "scripts" / "launch_parallel_sweep.sh"
CLASSIFICATION = ROOT / "src" / "harness" / "classification.py"


def test_post_run_timeout_starts_inside_gpu_lock() -> None:
    script = RUN_HARD.read_text()
    assert "run_gpu_locked_timeout check.py" in script
    assert "run_gpu_locked_timeout benchmark.py" in script
    assert "timeout 180 uv run python check.py" not in script
    assert "timeout 1800 uv run python benchmark.py" not in script


def test_run_archives_are_allocated_atomically() -> None:
    script = RUN_HARD.read_text()
    assert 'RUN_DIR_BASE="${REPO_ROOT}/outputs/runs/' in script
    assert 'if mkdir "$candidate" 2>/dev/null; then' in script
    assert 'failed to allocate unique run directory' in script


def test_claude_family_runs_from_archive_workspace() -> None:
    script = RUN_HARD.read_text()
    for harness in ("claude)", "ccr-claude)", "zai-claude)", "minimax-claude)"):
        start = script.index(harness)
        end = script.index(";;", start)
        block = script[start:end]
        assert '( cd "$PROBLEM_DIR" &&' in block
        assert 'timeout "$BUDGET_SECONDS"' in block
        assert '--add-dir "$PROBLEM_DIR"' in block


def test_minimax_claude_uses_official_anthropic_endpoint() -> None:
    script = RUN_HARD.read_text()
    start = script.index("minimax-claude)")
    end = script.index(";;", start)
    block = script[start:end]
    assert "MINIMAX_API_KEY" in block
    assert "https://api.minimax.io/anthropic" in block
    assert 'ANTHROPIC_MODEL="$MODEL"' in block
    assert 'ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"' in block
    assert 'env \\' not in block


def test_check_timeouts_are_retryable_not_plain_check_failed() -> None:
    classifier = CLASSIFICATION.read_text()
    assert 'reason = "check_timeout"' in classifier
    assert 'reason = "benchmark_timeout"' in classifier
    assert "elif check_exit == 124:" in classifier


def test_grok_uses_headless_cli_and_end_marker() -> None:
    script = RUN_HARD.read_text()
    start = script.index("grok)")
    end = script.index(";;", start)
    block = script[start:end]
    assert 'timeout "$BUDGET_SECONDS" "${AGENT_CUDA_ENV[@]}" grok' in block
    assert '--cwd "$PROBLEM_DIR"' in block
    assert "--output-format streaming-json" in block
    assert '"type":"end"' in script


def test_parallel_launcher_keeps_run_hard_jobs_waitable() -> None:
    script = LAUNCH_PARALLEL.read_text()
    assert 'LAST_LAUNCH_PID=$!' in script
    assert 'pid="$(launch_one' not in script
    assert 'launch_one "$name" "$harness" "$model" "$effort" "$problem"' in script
    assert 'pid="$LAST_LAUNCH_PID"' in script
    assert 'wait "$pid" || true' in script
