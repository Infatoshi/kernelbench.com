import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# run_hard.sh is a thin identity wrapper since 2026-07-31; the harness logic
# these tests assert on lives in the shared runner at the monorepo root.
RUN_HARD = ROOT / "scripts" / "run_hard.sh"
if "scripts/lib/run_harness.sh" in RUN_HARD.read_text():
    RUN_HARD = ROOT.parents[1] / "scripts" / "lib" / "run_harness.sh"
LAUNCH_PARALLEL = ROOT / "scripts" / "launch_parallel_sweep.sh"
SWEEP = ROOT / "scripts" / "sweep.sh"
RUN_BASELINES = ROOT / "scripts" / "run_baselines.sh"
REGRADE = ROOT / "scripts" / "regrade_sequential.sh"
CLASSIFICATION = ROOT / "src" / "harness" / "classification.py"
BENCHMARKS = sorted((ROOT / "problems-rtxpro6000").glob("*/benchmark.py"))
KDA_BENCHMARK = ROOT / "problems-rtxpro6000" / "02_kda_cutlass" / "benchmark.py"


def test_post_run_timeout_starts_inside_gpu_lock() -> None:
    script = RUN_HARD.read_text()
    assert "run_gpu_locked_timeout check.py" in script
    assert "run_gpu_locked_timeout benchmark.py" in script
    assert "timeout 180 uv run python check.py" not in script
    assert "timeout 1800 uv run python benchmark.py" not in script


def test_cuda_cannot_be_disabled_for_agent_phase() -> None:
    script = RUN_HARD.read_text()
    parallel = LAUNCH_PARALLEL.read_text()
    retries = (ROOT / "scripts" / "launch_infra_retries.sh").read_text()
    for text in (script, parallel, retries):
        assert "KBH_DISABLE_AGENT_CUDA" not in text
        assert "AGENT_CUDA_ENV" not in text
        assert "KBH_AGENT_PHASE" not in text
        assert "CUDA_VISIBLE_DEVICES=" not in text


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


def test_shared_checker_source_is_copied_and_restored_as_trusted_input() -> None:
    script = RUN_HARD.read_text()
    assert 'cp -a "$REPO_ROOT/src" "$WORKSPACE_ROOT/src"' in script
    assert 'ln -s "$REPO_ROOT/src" "$WORKSPACE_ROOT/src"' not in script
    assert 'TRUSTED_SRC_BACKUP_DIR="$RUN_DIR/trusted_src"' in script
    assert "MUTATED: trusted src/" in script
    assert 'TRUSTED_SRC_DIGEST="$(trusted_src_digest' in script
    assert '"$REAL_PYTHON" - "$1"' in script
    assert "unsafe trusted src root" in script
    assert "BOOTSTRAP_PYTHON" not in script
    assert script.index('REAL_PYTHON="$(command -v') < script.index(
        'TRUSTED_SRC_DIGEST="$(trusted_src_digest'
    )
    assert 'strip_python_bytecode "$WORKSPACE_ROOT/src"' in script
    assert '/bin/cp -a "$trusted_source" "$WORKSPACE_ROOT/src"' in script
    after_harness = script.index('detect_template_mutation "after harness"')
    final_check = script.index('echo "Running check.py..."', after_harness)
    assert script.index("restore_trusted_src", after_harness, final_check) < final_check


def test_canonical_deck_regrade_restores_complete_current_runtime() -> None:
    script = REGRADE.read_text()
    canonical = script.index('if [ -n "${KBH_REGRADE_DECK:-}" ]')
    check = script.index('echo "    check.py..."', canonical)
    block = script[canonical:check]
    assert 'WORKSPACE_ROOT="$RUN_DIR/repo"' in script
    assert '/bin/cp -a "$REPO_ROOT/src" "$WORKSPACE_ROOT/src"' in block
    assert 'cp -p "$REPO_ROOT/pyproject.toml" "$WORKSPACE_ROOT/pyproject.toml"' in block
    assert 'cp -p "$REPO_ROOT/uv.lock" "$WORKSPACE_ROOT/uv.lock"' in block
    assert 'cp -p "$REPO_ROOT/.python-version" "$WORKSPACE_ROOT/.python-version"' in block
    assert "FATAL: canonical grading surface is incomplete" in block
    assert "grading workspace as-is" not in block
    assert 'name = "hypothesis"' in (ROOT / "uv.lock").read_text()

    monorepo = ROOT.parents[1]
    canonical_bytes = REGRADE.read_bytes()
    for bench in ("cuda", "mega"):
        assert (
            monorepo / "benchmarks" / bench / "scripts" / "regrade_sequential.sh"
        ).read_bytes() == canonical_bytes


def test_legacy_regrade_receives_current_property_helper_and_lock(tmp_path) -> None:
    run_dir = tmp_path / "legacy-run"
    workspace = run_dir / "repo"
    problem_dir = workspace / "problems" / "05_topk_bitonic"
    problem_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps({"problem": "05_topk_bitonic", "peak_fraction": 0.1})
    )
    (run_dir / "solution.py").write_text("class Model:\n    pass\n")
    (workspace / "src" / "eval").mkdir(parents=True)
    (workspace / "src" / "eval" / "legacy.py").write_text("OLD = True\n")
    (problem_dir / "__pycache__").mkdir()
    (problem_dir / "__pycache__" / "reference.cpython-311.pyc").write_bytes(b"forged")
    (run_dir / "scratch" / "__pycache__").mkdir(parents=True)
    (run_dir / "scratch" / "__pycache__" / "helper.cpython-311.pyc").write_bytes(b"forged")
    (workspace / "pyproject.toml").write_text("[project]\nname='legacy'\nversion='0'\n")
    (workspace / "uv.lock").write_text("version = 1\n")
    (workspace / ".python-version").write_text("3.11\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    timeout = fake_bin / "timeout"
    timeout.write_text("#!/bin/sh\nexit 42\n")
    timeout.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "KBH_REGRADE_ALLOW_BUSY": "1",
            "KBH_REGRADE_DECK": "problems-rtxpro6000",
            "KBH_KEEP_RUN_VENV": "1",
            "PATH": f"{fake_bin}:{env['PATH']}",
        }
    )

    completed = subprocess.run(
        ["bash", str(REGRADE), str(run_dir)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (workspace / "src" / "eval" / "legacy.py").exists() is False
    assert (workspace / "src" / "eval" / "property_stress.py").read_bytes() == (
        ROOT / "src" / "eval" / "property_stress.py"
    ).read_bytes()
    assert (workspace / "pyproject.toml").read_bytes() == (ROOT / "pyproject.toml").read_bytes()
    assert (workspace / "uv.lock").read_bytes() == (ROOT / "uv.lock").read_bytes()
    assert not tuple((workspace / "src").rglob("*.pyc"))
    assert not tuple(problem_dir.rglob("*.pyc"))
    assert not tuple(problem_dir.rglob("*.pyo"))
    assert not tuple(path for path in problem_dir.rglob("__pycache__") if path.is_dir())
    assert "check FAILED (exit 42)" in completed.stdout


def test_regrade_purges_problem_bytecode_after_candidate_restore() -> None:
    script = REGRADE.read_text()
    scratch_restore = script.index('cp -r "$RUN_DIR/scratch/." "$PROBLEM_DIR/"')
    check_run = script.index('uv run python "$TRUSTED_ENTRYPOINT" check.py', scratch_restore)
    first_purge = script.index('purge_untrusted_bytecode "$PROBLEM_DIR"', scratch_restore)
    benchmark_run = script.index('uv run python "$TRUSTED_ENTRYPOINT" benchmark.py', check_run)
    second_purge = script.index('purge_untrusted_bytecode "$PROBLEM_DIR"', check_run)
    assert scratch_restore < first_purge < check_run < second_purge < benchmark_run

    monorepo = ROOT.parents[1]
    canonical_bytes = REGRADE.read_bytes()
    for bench in ("cuda", "mega"):
        assert (
            monorepo / "benchmarks" / bench / "scripts" / "regrade_sequential.sh"
        ).read_bytes() == canonical_bytes

    mini = (monorepo / "benchmarks" / "mini" / "scripts" / "regrade_sequential.sh").read_text()
    mini_scratch = mini.index('cp -r "$RUN_DIR/scratch/." "$PROBLEM_DIR/"')
    mini_check = mini.index('uv run python "$TRUSTED_ENTRYPOINT" check.py', mini_scratch)
    assert mini_scratch < mini.index(
        'purge_untrusted_bytecode "$PROBLEM_DIR"', mini_scratch
    ) < mini_check


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
    assert 'timeout "$BUDGET_SECONDS" grok' in block
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
    assert "droid|kimi|opencode|opencode-nemotron|nvcf-nemotron|hy3|hy3-claude|tinker|inkling|lfm-opencode|hermes|pi)" in script


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


def test_hy3_tokenhub_uses_measured_context_wall_and_host_stall_watch() -> None:
    script = RUN_HARD.read_text()
    # Live TokenHub hy3 hard-caps input at 196608 (RCA 2026-07-09). Advertising
    # 262144 put OpenCode compaction past the real wall.
    cfg = script[script.index("write_tokenhub_hy3_opencode_config()"):script.index("prepare_claude_container_home()")]
    assert "HY3_TOKENHUB_CONTEXT_LIMIT:-196608" in cfg
    assert "HY3_TOKENHUB_OUTPUT_LIMIT:-32000" in cfg
    assert "262144" not in cfg
    # Host-mode path (KBH_AGENT_CONTAINER=0) must supervise transcript growth.
    assert "run_host_with_stall_watch()" in script
    assert "host_stall_watchdog" in script
    assert 'kill "-${sig}" -- "-${pid}"' in script
    assert 'touch "$LOG_FILE"' in script
    assert ') </dev/null >> "$LOG_FILE" 2>> "$STDERR_FILE"' in script
    hy3 = script[script.index("hy3|hy3-claude)"):script.index("deepseek-claude)")]
    assert "run_host_with_stall_watch" in hy3
    assert "KBH_OPENCODE_STALL_SECONDS:-1500" in hy3


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


def test_codex_agent_and_final_grading_share_immutable_environment_path() -> None:
    script = RUN_HARD.read_text()
    codex = script[script.index("    codex)") : script.index("    kimi)")]
    replay = script[
        script.index("run_replay_stage()") : script.index('if [ "$TEMPLATE_MUTATED"')
    ]

    assert '"$HOST_AGENT_ISOLATOR" "$REAL_TIMEOUT" "$BUDGET_SECONDS"' in codex
    assert 'CODEX_HOME="$HOST_CODEX_HOME" codex exec' in codex
    assert "codex sandbox" in codex
    assert "sandbox_workspace_write.network_access=false" in codex
    assert "/usr/bin/getent hosts example.com" in codex
    assert "/usr/bin/curl -fsS --connect-timeout 1" in codex
    assert "/usr/bin/python3 -c" in codex
    assert "--sandbox workspace-write" in codex
    assert 'approval_policy="never"' in codex
    assert 'web_search="disabled"' in codex
    assert 'mcp_servers={}' in codex
    assert '--add-dir "$RUN_DIR"' in codex
    assert "--dangerously-bypass-approvals-and-sandbox" not in codex
    assert '"$HOST_AGENT_ISOLATOR"' in replay
    assert "/usr/bin/env -i" in replay
    assert "KBH_ISOLATION_NETWORK=off" in replay
    assert '"$TRUSTED_PYTHON" -I -S' in replay


def test_immutable_environment_seals_dependencies_and_trusted_grader_files() -> None:
    script = RUN_HARD.read_text()
    isolate = script[
        script.index("HOST_AGENT_ISOLATOR=") : script.index(
            "# Container-side lock wrappers"
        )
    ]

    assert '/usr/bin/mount --bind "$repo/.venv" "$workspace/.venv"' in isolate
    assert '/usr/bin/mount -o remount,bind,ro "$workspace/.venv"' in isolate
    assert 'printf "%s\\n" "$workspace_trusted"' in isolate
    assert (
        'for path in "$home" "$repo" "$python_runtime" "$trusted_tools" "$trusted_uv"'
        in isolate
    )
    assert (
        'for path in "$cargo_home" "$rustup_home" "$cuda_oxide" "$cutile_rust"'
        in isolate
    )
    assert "UV_NO_SYNC=1 UV_OFFLINE=1 PIP_NO_INDEX=1" in isolate
    assert "CARGO_NET_OFFLINE=true" in isolate
    assert 'WORKSPACE_TRUSTED_PATHS="$WORKSPACE_ROOT/src' in script


def test_immutable_environment_preflights_all_six_cuda_dialects() -> None:
    script = RUN_HARD.read_text()
    helper = (ROOT.parents[1] / "scripts" / "lib" / "dialect_preflight.py").read_text()
    start = script.index("DIALECT_NVCC=")
    end = script.index("TEMPLATE_MUTATED=false", start)
    preflight = script[start:end]

    assert '"$nvcc" -std=c++17 -c' in preflight
    assert '"$cargo" "+$cuda_oxide_toolchain" --version' in preflight
    assert '"$rustc" "+$cutile_rust_toolchain" --version' in preflight
    assert '"+$cuda_oxide_toolchain" check --locked --offline' in preflight
    assert '"+$cutile_rust_toolchain" check --locked --offline' in preflight
    assert "6c5458fe991bbde32c5bee74d87822aef1b5a691" in script
    assert "a3ed99d225befcb19f75ec8d81708eb35818fee2" in script
    assert "0859212ad19f71133a9b940c05323286cbf28a05" in script
    assert '"$python" -I "$python_preflight"' in preflight
    assert "@triton.jit" in helper
    assert "@ct.kernel()" in helper
    assert "cute.compile(_cute_scalar_add" in helper
    assert "torch.testing.assert_close" in helper
    assert (
        "CUDA C++, CUDA Oxide, CuTe DSL, Triton, cuTile Python, cuTile Rust"
        in preflight
    )


def test_worker_bootstrap_pins_and_provisions_all_cuda_dialects() -> None:
    bootstrap = (ROOT.parents[1] / "scripts" / "lib" / "bootstrap_dialects.sh").read_text()
    brev = (ROOT.parents[1] / "scripts" / "brev_worker.sh").read_text()
    lambda_worker = (ROOT.parents[1] / "scripts" / "lambda_worker.sh").read_text()

    for bench in ("hard", "cuda", "mini"):
        bench_root = ROOT.parents[1] / "benchmarks" / bench
        project = (bench_root / "pyproject.toml").read_text()
        lock = (bench_root / "uv.lock").read_text()
        assert '"cuda-tile==1.5.0"' in project
        assert '"nvidia-cutlass-dsl==4.7.0"' in project
        assert 'name = "cuda-tile"\nversion = "1.5.0"' in lock
        assert 'name = "nvidia-cutlass-dsl"\nversion = "4.7.0"' in lock

    for value in (
        "6c5458fe991bbde32c5bee74d87822aef1b5a691",
        "nightly-2026-04-03",
        "a3ed99d225befcb19f75ec8d81708eb35818fee2",
        "1.89.0",
        "0859212ad19f71133a9b940c05323286cbf28a05",
        "CUDA_TOOLKIT_VERSION=\"13.3.1\"",
    ):
        assert value in bootstrap
    assert 'cargo "+$CUDA_OXIDE_TOOLCHAIN" fetch --locked' in bootstrap
    assert 'cargo "+$CUDA_OXIDE_TOOLCHAIN" check --locked' in bootstrap
    assert 'cargo "+$CUTILE_RUST_TOOLCHAIN" fetch --locked' in bootstrap
    assert 'cargo "+$CUTILE_RUST_TOOLCHAIN" check --locked' in bootstrap
    assert 'ln -sfn "$cuda_toolkit_root/bin/nvcc"' in bootstrap
    assert '"cuda-toolkit[all]==$CUDA_TOOLKIT_VERSION"' in bootstrap
    assert "bash scripts/lib/bootstrap_dialects.sh" in brev
    assert "bash scripts/lib/bootstrap_dialects.sh" in lambda_worker


def test_codex_child_command_sandbox_blocks_network() -> None:
    codex = shutil.which("codex")
    if codex is None:
        return
    probe = """
if /usr/bin/getent hosts example.com >/dev/null 2>&1; then exit 31; fi
if /usr/bin/curl -fsS --connect-timeout 1 https://example.com/ >/dev/null 2>&1; then exit 32; fi
if /usr/bin/python3 -c 'import socket; socket.create_connection(("1.1.1.1", 53), 1)' >/dev/null 2>&1; then exit 33; fi
"""
    completed = subprocess.run(
        [
            codex,
            "sandbox",
            "-c",
            'sandbox_mode="workspace-write"',
            "-c",
            "sandbox_workspace_write.network_access=false",
            "--",
            "/bin/sh",
            "-eu",
            "-c",
            probe,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
