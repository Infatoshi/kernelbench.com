from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_HARD = ROOT / "scripts" / "run_hard.sh"
LAUNCH_PARALLEL = ROOT / "scripts" / "launch_parallel_sweep.sh"
RUN_BASELINES = ROOT / "scripts" / "run_baselines.sh"
CLASSIFICATION = ROOT / "src" / "harness" / "classification.py"
BENCHMARKS = sorted((ROOT / "problems").glob("*/benchmark.py"))
KDA_BENCHMARK = ROOT / "problems" / "02_kda_cutlass" / "benchmark.py"


def test_post_run_timeout_starts_inside_gpu_lock() -> None:
    script = RUN_HARD.read_text()
    assert "run_gpu_locked_timeout check.py" in script
    assert "run_gpu_locked_timeout benchmark.py" in script
    assert "timeout 180 uv run python check.py" not in script
    assert "timeout 1800 uv run python benchmark.py" not in script


def test_agent_phase_probe_commands_do_not_wait_on_gpu_lock() -> None:
    script = RUN_HARD.read_text()
    start = script.index('if [ "${KBH_AGENT_PHASE:-0}" = "1" ]; then')
    end = script.index('owner_file="${KBH_GPU_LOCK}.owner"', start)
    block = script[start:end]
    assert "uv|python|python3|nvidia-smi|nvcc)" in block
    assert "ncu|nsys)" in block
    assert "exit 125" in block
    assert "flock" not in block


def test_kda_has_longer_benchmark_timeout_backstop() -> None:
    script = RUN_HARD.read_text()
    assert 'PROBLEM_NAME" = "02_kda_cutlass' in script
    assert "KBH_BENCHMARK_TIMEOUT_02_KDA_CUTLASS_SECONDS" in script
    assert "benchmark_timeout_seconds" in script


def test_all_benchmarks_score_solution_before_optional_baselines() -> None:
    assert BENCHMARKS
    for path in BENCHMARKS:
        benchmark = path.read_text()
        assert "benchmark_baselines_enabled" in benchmark, path
        assert "time_variant" in benchmark, path
        assert "Solution first" in benchmark, path
        assert benchmark.index('variant="solution"') < benchmark.index("torch.compile"), path
        assert benchmark.index("benchmark_baselines_enabled") < benchmark.index("torch.compile"), path


def test_kda_benchmark_keeps_legacy_baseline_env_alias() -> None:
    benchmark = KDA_BENCHMARK.read_text()
    assert 'benchmark_baselines_enabled("KDA", "02_KDA_CUTLASS")' in benchmark
    assert benchmark.index('variant="solution"') < benchmark.index("if not include_baselines")
    assert benchmark.index("if not include_baselines") < benchmark.index("torch.compile")




def test_baseline_generator_opts_into_reference_diagnostics() -> None:
    script = RUN_BASELINES.read_text()
    assert "KBH_BENCHMARK_BASELINES=1 timeout 300 uv run python benchmark.py" in script


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
