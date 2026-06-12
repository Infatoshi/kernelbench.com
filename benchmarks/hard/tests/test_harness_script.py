from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_HARD = ROOT / "scripts" / "run_hard.sh"
LAUNCH_PARALLEL = ROOT / "scripts" / "launch_parallel_sweep.sh"
SWEEP = ROOT / "scripts" / "sweep.sh"
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


def test_claude_container_mode_uses_clean_namespace() -> None:
    script = RUN_HARD.read_text()
    assert 'KBH_AGENT_CONTAINER="${KBH_AGENT_CONTAINER:-0}"' in script
    assert 'KBH_AGENT_CONTAINER_NETWORK="${KBH_AGENT_CONTAINER_NETWORK:-bridge}"' in script
    assert 'KBH_AGENT_CONTAINER_CODEX_NODE=' in script
    assert 'KBH_AGENT_CONTAINER_OPENCODE_BIN=' in script
    assert 'KBH_AGENT_CONTAINER_DROID_BIN=' in script
    assert 'KBH_AGENT_CONTAINER_CURSOR_DIR=' in script
    assert "agent_container_native_profiling_harness_gpu_lock" in script
    assert 'PROMPT_WORKSPACE_DIR="/workspace/problems/$PROBLEM_NAME"' in script
    assert "is not mounted" in script
    assert 'cp -a "$REPO_ROOT/src" "$WORKSPACE_ROOT/src"' in script
    assert "prepare_claude_container_home()" in script
    assert "prepare_codex_container_home()" in script
    assert "prepare_opencode_container_home()" in script
    assert "prepare_droid_container_home()" in script
    assert "prepare_cursor_container_home()" in script
    assert 'cp -p "$HOME/.claude/.credentials.json"' in script
    assert "printf '{}\\n' > \"$home_dir/.claude.json\"" in script
    assert 'cp -p "$HOME/.codex/auth.json"' in script
    assert 'cp -p "$HOME/.config/opencode/opencode.json"' in script
    assert 'cp -p "$HOME/.config/cursor/auth.json"' in script
    assert '--network "$KBH_AGENT_CONTAINER_NETWORK"' in script
    assert '--cap-add CAP_PERFMON' in script
    assert '-v "$WORKSPACE_ROOT:/workspace:rw"' in script
    assert '-v "$KBH_AGENT_CONTAINER_CUDA_HOME:/usr/local/cuda-host:ro"' in script
    assert '-v "$KBH_AGENT_CONTAINER_CODEX_NODE:/opt/node:ro"' in script
    assert '-v "$KBH_AGENT_CONTAINER_OPENCODE_BIN:/usr/local/bin/opencode:ro"' in script
    assert '-v "$KBH_AGENT_CONTAINER_DROID_BIN:/usr/local/bin/droid:ro"' in script
    assert '-v "$KBH_AGENT_CONTAINER_CURSOR_DIR:/opt/cursor-agent:ro"' in script
    assert '-w "/workspace/problems/$PROBLEM_NAME"' in script
    assert '--kill-after="${KBH_TIMEOUT_KILL_AFTER_SECONDS:-30}s"' in script
    assert "run_docker_locked_timeout()" in script
    assert '--cidfile "$cidfile"' in script
    assert '"$REAL_DOCKER" rm -f "$(cat "$cidfile")"' in script
    assert "run_docker_locked_timeout claude-container" in script
    assert "run_docker_locked_timeout codex-container" in script
    assert "run_docker_locked_timeout opencode-container" in script
    assert "run_docker_locked_timeout droid-container" in script
    assert "run_docker_locked_timeout cursor-container" in script
    assert "host harness memory are not mounted" in script
    assert '"agent_container_network": "$KBH_AGENT_CONTAINER_NETWORK"' in script
    assert '"agent_container": $([ "$KBH_AGENT_CONTAINER" = "1" ]' in script


def test_agent_container_mode_does_not_mount_full_harness_state() -> None:
    script = RUN_HARD.read_text()
    start = script.index("run_claude_container()")
    end = script.index("# Snapshot immutable problem files", start)
    block = script[start:end]
    assert '$HOME/.claude:' not in block
    assert "$HOME/.claude.json" not in block
    assert '$HOME/.codex:' not in block
    assert '$HOME/.config/opencode:' not in block
    assert '$HOME/.local/share/opencode:' not in block
    assert '$HOME/.factory:' not in block
    assert '$HOME/.cursor:' not in block
    assert '$HOME/.config/cursor:' not in block
    assert ".claude/projects" not in block
    assert ".claude/sessions" not in block
    assert "history.jsonl" not in block
    assert "opencode.db" not in block


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


def test_preflight_can_filter_to_one_row() -> None:
    script = (ROOT / "scripts" / "preflight_harnesses.sh").read_text()
    assert "KBH_PREFLIGHT_ONLY" in script
    assert '"$harness" == "$KBH_PREFLIGHT_ONLY"' in script
    assert '"$model" == "$KBH_PREFLIGHT_ONLY"' in script


def test_openrouter_nemotron_is_opt_in_for_sweeps() -> None:
    parallel = LAUNCH_PARALLEL.read_text()
    sweep = SWEEP.read_text()
    preflight = (ROOT / "scripts" / "preflight_harnesses.sh").read_text()
    for script in (parallel, sweep, preflight):
        assert "KBH_USE_OPENROUTER_NEMOTRON" in script
        assert "opencode-nemotron" in script
        assert "nvidia/nemotron-3-ultra-550b-a55b" in script


def test_nvcf_nemotron_is_opt_in_for_sweeps() -> None:
    parallel = LAUNCH_PARALLEL.read_text()
    sweep = SWEEP.read_text()
    for script in (parallel, sweep):
        assert "KBH_USE_NVCF_NEMOTRON" in script
        assert "nvcf-nemotron" in script
        assert "nemotron-3-ultra" in script


def test_openrouter_nemotron_uses_archive_local_opencode_config() -> None:
    script = RUN_HARD.read_text()
    assert "write_openrouter_deepinfra_opencode_config()" in script
    assert "OPENROUTER_API_KEY is required for opencode-nemotron" in script
    assert "openrouter-deepinfra" in script
    assert "DeepInfra" in script
    assert "allow_fallbacks" in script
    start = script.index("opencode-nemotron)")
    end = script.index(";;", start)
    block = script[start:end]
    assert 'env XDG_CONFIG_HOME="$OPENCODE_NEMOTRON_CONFIG_HOME"' in block
    assert 'opencode run --pure --format json -m "$OPENCODE_NEMOTRON_MODEL"' in block
    assert 'run_opencode_container "$OPENCODE_NEMOTRON_MODEL"' in block
    assert "opencode-nemotron|nvcf-nemotron" in script


def test_nvcf_nemotron_uses_local_proxy_and_archive_config() -> None:
    script = RUN_HARD.read_text()
    assert "start_nvcf_proxy()" in script
    assert "write_nvcf_opencode_config()" in script
    assert "scripts/nvcf_openai_proxy.py" in script
    assert "NGC_API_KEY, NVIDIA_API_KEY, or NVCF_API_KEY" in script
    start = script.index("nvcf-nemotron)")
    end = script.index(";;", start)
    block = script[start:end]
    assert "start_nvcf_proxy" in block
    assert 'env XDG_CONFIG_HOME="$NVCF_OPENCODE_CONFIG_HOME"' in block
    assert '-m "nvcf-nemotron/$MODEL"' in block
    assert "opencode run --pure --format json" in block
    assert "droid|kimi|opencode|opencode-nemotron|nvcf-nemotron)" in script


def test_parallel_launcher_keeps_run_hard_jobs_waitable() -> None:
    script = LAUNCH_PARALLEL.read_text()
    assert 'LAST_LAUNCH_PID=$!' in script
    assert 'pid="$(launch_one' not in script
    assert 'launch_one "$name" "$harness" "$model" "$effort" "$problem"' in script
    assert 'pid="$LAST_LAUNCH_PID"' in script
    assert 'wait "$pid" || true' in script


def test_agent_container_uses_workspace_uv_env_and_prewarmed_opencode_home() -> None:
    script = RUN_HARD.read_text()
    # Agents must develop against the same uv.lock env the host scores with;
    # droid is out of the suite, so exactly the six active runners mount uv
    # (claude, codex, opencode, cursor, grok, gemini).
    assert script.count("-v \"$REAL_UV:/usr/local/bin/uv:ro\"") == 6
    assert script.count("-v \"$KBH_AGENT_CONTAINER_UV_CACHE:/uv-cache:rw\"") == 6
    assert script.count("-e UV_CACHE_DIR=/uv-cache") == 6
    assert script.count("-e UV_PYTHON_INSTALL_DIR=/uv-cache/python") == 6
    assert "mkdir -p \"$KBH_AGENT_CONTAINER_UV_CACHE\"" in script
    assert "same uv.lock as the official" in script
    # Pre-warmed opencode home: no per-run sqlite migration, and the template
    # copy must not reach into the host opencode data dir (session leak).
    assert "outputs/opencode_home_template" in script
    start = script.index("prepare_opencode_container_home()")
    end = script.index("prepare_droid_container_home()")
    block = script[start:end]
    assert "cp -a \"$KBH_OPENCODE_HOME_TEMPLATE/.\" \"$home_dir/\"" in block
    assert "$HOME/.local/share/opencode" not in block


def test_opencode_container_has_stall_watchdog_and_retry() -> None:
    script = RUN_HARD.read_text()
    # Generic watchdog in the docker wrapper, opt-in via env.
    assert "KBH_STALL_WATCH_LOG" in script
    assert "stall_watchdog.log" in script
    # Retry loop scoped to the opencode runner (the affected adapter family).
    block = script[script.index("run_opencode_container()"):script.index("run_droid_container()")]
    assert "KBH_OPENCODE_STALL_SECONDS" in block
    assert "KBH_OPENCODE_STALL_RETRIES" in block
    assert "remaining=$(( BUDGET_SECONDS - elapsed ))" in block


def test_agent_container_sessions_parallel_with_per_command_lock() -> None:
    script = RUN_HARD.read_text()
    # Default: sessions do NOT hold the GPU lock; in-container GPU commands
    # serialize per-command through the bind-mounted lock dir.
    assert script.count("-v \"$CONTAINER_LOCK_BIN:/kbh/bin:ro\"") == 6
    assert script.count("-v \"$KBH_GPU_LOCK_DIR:/kbh/lock:rw\"") == 6
    assert script.count("-e KBH_GPU_LOCK=/kbh/lock/gpu.lock") == 6
    assert "KBH_AGENT_CONTAINER_SESSION_LOCK" in script
    assert "agent_container_native_profiling_path_wrapper_gpu_lock" in script
    # The lock lives in a dedicated dir so only the lock is mounted, never
    # the rest of outputs/.
    assert "outputs/gpu_lock" in script
