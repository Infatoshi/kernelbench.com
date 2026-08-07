"""Focused contracts between immutable bundles and shell grading flows."""

from __future__ import annotations

import json
import shutil
import socket
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st


REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts/lib/submission_replay.sh"
RUNNER = REPO / "scripts/lib/run_harness.sh"


@pytest.fixture
def repo_tmp_path() -> Path:
    path = Path(tempfile.mkdtemp(prefix=".replay-test-", dir=REPO))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def host_runtime_sockets(repo_tmp_path: Path) -> tuple[Path, Path, Path]:
    directory = Path(tempfile.mkdtemp(prefix="kb-replay-socket-", dir="/tmp"))
    operator_home = repo_tmp_path / "operator-home"
    ssh = operator_home / ".ssh"
    ssh.mkdir(parents=True)
    tmp_path = directory / "agent.sock"
    home_path = ssh / "control.sock"
    servers = [socket.socket(socket.AF_UNIX), socket.socket(socket.AF_UNIX)]
    servers[0].bind(str(tmp_path))
    servers[1].bind(str(home_path))
    try:
        yield tmp_path, home_path, operator_home
    finally:
        for server in servers:
            server.close()
        shutil.rmtree(directory, ignore_errors=True)


def _bash(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "$1"; shift; {script}', "bash", str(HELPER), *args],
        capture_output=True,
        text=True,
    )


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    score=st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
    noise=st.lists(
        st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",), blacklist_characters="\r\n"
            ),
            max_size=40,
        ).filter(lambda line: not line.startswith("peak_fraction:")),
        max_size=8,
    ),
)
def test_exactly_one_complete_peak_metric_is_accepted(
    tmp_path: Path, score: float, noise: list[str]
) -> None:
    log = tmp_path / "benchmark.log"
    rendered = format(score, ".12f")
    log.write_text(
        "\n".join([*noise, f"peak_fraction: {rendered}", *reversed(noise)]) + "\n"
    )
    result = _bash('submission_extract_peak_fraction "$1"', str(log))
    assert result.returncode == 0
    assert float(result.stdout) == pytest.approx(float(rendered))


@pytest.mark.parametrize(
    "body",
    [
        "peak_fraction: 0.2\npeak_fraction: 0.3\n",
        "prefix peak_fraction: 0.2\n",
        "peak_fraction: nan\n",
        "peak_fraction: -1\n",
        "peak_fraction: 101\n",
    ],
)
def test_ambiguous_or_partial_peak_metric_is_rejected(
    tmp_path: Path, body: str
) -> None:
    log = tmp_path / "benchmark.log"
    log.write_text(body)
    assert _bash('submission_extract_peak_fraction "$1"', str(log)).returncode != 0


@pytest.mark.parametrize(
    ("body", "accepted"),
    [
        ("details\nPASS\n", True),
        ("details\n", False),
        ("PASS\nPASS\n", False),
        ("forged PASS\n", False),
        ("PASS trailing\n", False),
    ],
)
def test_correctness_requires_one_standalone_pass_marker(
    tmp_path: Path, body: str, accepted: bool
) -> None:
    log = tmp_path / "check.log"
    log.write_text(body)
    result = _bash('submission_check_passed "$1"', str(log))
    assert (result.returncode == 0) is accepted


def test_trusted_surface_digest_detects_src_and_grader_mutation(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    problem = workspace / "problems/01_problem"
    (workspace / "src/eval").mkdir(parents=True)
    problem.mkdir(parents=True)
    (workspace / "pyproject.toml").write_text("[project]\nname='test'\nversion='0'\n")
    (workspace / "uv.lock").write_text("version = 1\n")
    (workspace / "src/eval/gate.py").write_text("TRUSTED = True\n")
    (problem / "check.py").write_text("print('PASS')\n")

    command = 'submission_trusted_surface_digest "$1" 01_problem check.py'
    before = _bash(command, str(workspace))
    assert before.returncode == 0
    (workspace / "src/eval/gate.py").write_text("TRUSTED = False\n")
    after_src = _bash(command, str(workspace))
    assert after_src.returncode == 0
    assert after_src.stdout != before.stdout
    (problem / "check.py").write_text("print('FORGED PASS')\n")
    after_grader = _bash(command, str(workspace))
    assert after_grader.stdout != after_src.stdout


def test_executable_identity_detects_replacement(tmp_path: Path) -> None:
    executable = tmp_path / "uv"
    executable.write_text("first\n")
    executable.chmod(0o755)
    before = _bash('submission_executable_identity "$1"', str(executable))
    assert before.returncode == 0, before.stderr

    replacement = tmp_path / "replacement"
    replacement.write_text("second\n")
    replacement.chmod(0o755)
    replacement.replace(executable)
    after = _bash('submission_executable_identity "$1"', str(executable))
    assert after.returncode == 0, after.stderr
    assert json.loads(before.stdout) != json.loads(after.stdout)


def test_bound_bundle_tool_fails_closed_after_replacement(tmp_path: Path) -> None:
    tool = tmp_path / "submission_bundle.py"
    tool.write_text("first\n")
    result = _bash(
        'SUBMISSION_BUNDLE_TOOL="$1"; '
        'SUBMISSION_BUNDLE_TOOL_IDENTITY="$(submission_executable_identity "$1")"; '
        "printf '%s\\n' changed > \"$1\"; "
        "! submission_bundle_tool",
        str(tool),
    )
    assert result.returncode == 0, result.stderr


def test_atomic_json_writer_replaces_complete_document(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    destination.write_text('{"old": true}\n')
    result = _bash(
        "printf '%s' '{\"new\": [1, 2, 3]}' | submission_atomic_write_json \"$1\"",
        str(destination),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(destination.read_text()) == {"new": [1, 2, 3]}
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert not list(tmp_path.glob(".result.json.tmp-*"))


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    solution=st.binary(max_size=8_192),
    sentinel=st.binary(max_size=8_192),
)
def test_legacy_solution_projection_never_follows_destination_symlink(
    tmp_path: Path,
    solution: bytes,
    sentinel: bytes,
) -> None:
    case = tmp_path / "legacy-projection-case"
    shutil.rmtree(case, ignore_errors=True)
    case.mkdir()
    extracted = case / "extracted"
    run_dir = case / "run"
    extracted.mkdir()
    run_dir.mkdir()
    (extracted / "solution.py").write_bytes(solution)
    target = case / "attacker-target"
    target.write_bytes(sentinel)
    (run_dir / "solution.py").symlink_to(target)

    result = _bash(
        'submission_project_legacy_archive "$1" "$2"',
        str(extracted),
        str(run_dir),
    )

    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == sentinel
    assert not (run_dir / "solution.py").is_symlink()
    assert (run_dir / "solution.py").read_bytes() == solution
    assert not list(run_dir.glob(".solution.py.tmp-*"))


def test_output_preparation_unlinks_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = tmp_path / "operator-state"
    target.write_bytes(b"keep me")
    (run_dir / "check.log").symlink_to(target)

    result = _bash('submission_prepare_output_file "$1"', str(run_dir / "check.log"))

    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == b"keep me"
    assert (run_dir / "check.log").is_file()
    assert not (run_dir / "check.log").is_symlink()
    assert (run_dir / "check.log").read_bytes() == b""


def test_regrade_root_reset_does_not_follow_archive_symlink(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    victim = tmp_path / "victim"
    run_dir.mkdir()
    victim.mkdir()
    sentinel = victim / "keep"
    sentinel.write_bytes(b"untouched")
    (run_dir / "regrade_replays").symlink_to(victim, target_is_directory=True)

    result = _bash(
        'submission_reset_run_subdirectory "$1" regrade_replays',
        str(run_dir),
    )

    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == b"untouched"
    assert (run_dir / "regrade_replays").is_dir()
    assert not (run_dir / "regrade_replays").is_symlink()


def test_clean_replay_environment_hides_credentials_and_operator_home(
    tmp_path: Path,
) -> None:
    replay_home = tmp_path / "home"
    result = _bash(
        "export OPENAI_API_KEY=secret HTTPS_PROXY=http://proxy UNRELATED_STATE=leak; "
        'submission_select_clean_environment "$1"; "${SUBMISSION_CLEAN_ENV_PREFIX[@]}" env',
        str(replay_home),
    )
    assert result.returncode == 0, result.stderr
    environment = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    assert environment["HOME"] == str(replay_home)
    assert environment["UV_OFFLINE"] == "1"
    assert environment["UV_LINK_MODE"] == "copy"
    assert Path(environment["UV_CACHE_DIR"]).is_relative_to(replay_home)
    assert "OPENAI_API_KEY" not in environment
    assert "HTTPS_PROXY" not in environment
    assert "UNRELATED_STATE" not in environment


def test_namespace_replay_exposes_only_stage_writes_and_drops_caps(
    repo_tmp_path: Path,
    host_runtime_sockets: tuple[Path, Path, Path],
) -> None:
    tmp_path = repo_tmp_path
    tmp_socket, home_socket, operator_home = host_runtime_sockets
    bash = shutil.which("bash")
    assert bash is not None

    trust_root = tmp_path / "trusted-root"
    run_root = trust_root / "outputs/run"
    stage = run_root / "replays/check"
    project = stage / "repo"
    problem = project / "problems/01_problem"
    source = project / "src"
    venv = project / ".venv"
    for directory in (problem, source, venv):
        directory.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='sandbox-test'\n")
    (project / "uv.lock").write_text("version = 1\n")
    trusted_source = source / "gate.py"
    trusted_source.write_text("TRUSTED = True\n")
    trusted_venv = venv / "installed.py"
    trusted_venv.write_text("INSTALLED = True\n")
    trusted_template = problem / "check.py"
    trusted_template.write_text("print('PASS')\n")
    outside = run_root / "outside-stage.txt"
    outside.write_text("original\n")
    log = run_root / "parent-opened.log"
    writable = stage / "candidate-output.txt"
    external_executable = tmp_path / "host-tool"
    external_executable.write_text("tool\n")
    external_executable.chmod(0o755)
    outside_isolation_roots = tmp_path / "outside-isolation-roots.txt"
    outside_isolation_roots.write_text("host state\n")
    socket_alias = tmp_path / "socket-hardlink-alias.sock"
    socket_alias.hardlink_to(home_socket)
    host_shm = Path("/dev/shm") / f"kernelbench-host-shm-{stage.stat().st_ino}"
    host_shm.write_text("host state\n")

    candidate = r"""
set -euo pipefail
readonly_path() {
    if { : > "$1"; } 2>/dev/null; then
        echo "unexpected writable path: $1"
        exit 31
    fi
}
absent() {
    if [ -e "$1" ] || [ -L "$1" ]; then
        echo "unexpected host path: $1"
        exit 32
    fi
}
absent "$1"
printf 'candidate write\n' > "$2"
readonly_path "$3"
readonly_path "$4"
readonly_path "$5"
absent "$6"
readonly_path "$9"
absent "${10}"
absent "${11}"
absent "${12}"
absent "${13}"
absent "${15}"
absent /.oldroot
absent /var/lib/kernelbench-unallowlisted
absent /opt/kernelbench-unallowlisted
absent /etc/environment
absent /etc/hostname
absent /etc/machine-id
absent /dev/tty
readonly_path /dev/kernelbench-unallowlisted
exec 7<>/dev/null
printf 'device-rw-probe' >&7
exec 7>&-
[ ! -S /run/docker.sock ]
[ ! -S /var/run/docker.sock ]
[ ! -S "${11}" ]
[ ! -S "${13}" ]
[ ! -S "${15}" ]
[ ! -e "${14}" ]
printf 'private shm\n' > "${14}"
[ "$(cat "${14}")" = "private shm" ]
rm -f "${14}"
[ -z "${OPENAI_API_KEY+x}" ]
[ -z "${UNRELATED_STATE+x}" ]
[ "$UV_CACHE_DIR" = "$7" ]
[ "$UV_LINK_MODE" = copy ]
[ "$HOME" = "$8" ]
case ":$PATH:" in
    *":${3%/src/gate.py}/.venv/bin:"*) exit 34 ;;
esac
[ "$(ulimit -f)" = 131072 ]
read -r self_pid _ < /proc/self/stat
[ "$self_pid" = 1 ]
cap_eff=""
while read -r key value _; do
    if [ "$key" = "CapEff:" ]; then
        cap_eff="$value"
        break
    fi
done < /proc/self/status
[ "$cap_eff" = 0000000000000000 ]
network_devices=0
while IFS=: read -r interface _; do
    case "$interface" in
        *'|'*) continue ;;
    esac
    interface="${interface//[[:space:]]/}"
    [ "$interface" = lo ]
    network_devices=$((network_devices + 1))
done < /proc/net/dev
[ "$network_devices" -eq 1 ]
[ "$(hostname)" = kernelbench-replay ]
if [ "${16}" = 1 ]; then
    [ -d /proc/driver/nvidia ]
else
    [ ! -e /proc/driver/nvidia ]
fi

compile_root="${2%/*}/compiler-probe"
printf 'int main(void) { return 0; }\n' > "$compile_root.c"
cc "$compile_root.c" -o "$compile_root"
"$compile_root"
rm -f "$compile_root" "$compile_root.c"

stage_root="${2%/*}"
root_readonly=0
dev_root_readonly=0
null_device_readonly=0
private_proc_rw=0
while read -r _ _ _ _ mountpoint options _; do
    case ",$options," in
        *,rw,*)
            case "$mountpoint" in
                "$stage_root"|/tmp|/dev/shm) ;;
                /proc)
                    private_proc_rw=1
                    ;;
                *) echo "unexpected rw mount: $mountpoint"; exit 33 ;;
            esac
            ;;
        *,ro,*)
            [ "$mountpoint" != / ] || root_readonly=1
            [ "$mountpoint" != /dev ] || dev_root_readonly=1
            [ "$mountpoint" != /dev/null ] || null_device_readonly=1
            ;;
    esac
done < /proc/self/mountinfo
[ "$root_readonly" -eq 1 ]
[ "$dev_root_readonly" -eq 1 ]
[ "$null_device_readonly" -eq 1 ]
[ "$private_proc_rw" -eq 1 ]
printf 'sandbox-ok\n'
"""
    driver = r"""
export OPENAI_API_KEY=secret UNRELATED_STATE=leak
export HOME="${19}"
submission_resolve_isolation_tools || exit 77
submission_add_isolation_readonly_executable "${16}"
submission_configure_isolation "$1" "$2"
submission_reset_caches "$3/cache"
submission_select_clean_environment "$3/home"
submission_build_isolated_command \
    "$3" "$4" "$5" check.py -- \
    "$6" -c "$7" kernelbench-candidate \
    "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" \
    "${17}" "${18}" "${19}" "${20}" "${21}" "${22}" "${23}"
"${SUBMISSION_ISOLATED_COMMAND[@]}" > "${13}" 2>&1
"""
    try:
        result = _bash(
            driver,
            str(run_root),
            str(trust_root),
            str(stage),
            str(project),
            str(problem),
            bash,
            candidate,
            str(outside),
            str(writable),
            str(trusted_source),
            str(trusted_venv),
            str(trusted_template),
            str(log),
            str(stage / "cache/uv"),
            str(stage / "home"),
            str(external_executable),
            str(outside_isolation_roots),
            str(tmp_socket),
            str(operator_home),
            str(home_socket),
            str(host_shm),
            str(socket_alias),
            "1" if Path("/proc/driver/nvidia").is_dir() else "0",
        )
    finally:
        host_shm_contents = host_shm.read_text()
        host_shm.unlink(missing_ok=True)
    if result.returncode == 77:
        pytest.skip("combined user/mount/PID/network namespaces unavailable")
    assert result.returncode == 0, result.stderr or log.read_text(errors="replace")
    assert outside.read_text() == "original\n"
    assert writable.read_text() == "candidate write\n"
    assert trusted_source.read_text() == "TRUSTED = True\n"
    assert trusted_venv.read_text() == "INSTALLED = True\n"
    assert trusted_template.read_text() == "print('PASS')\n"
    assert external_executable.read_text() == "tool\n"
    assert outside_isolation_roots.read_text() == "host state\n"
    assert host_shm_contents == "host state\n"
    assert log.read_text() == "sandbox-ok\n"


def test_namespace_replay_mounts_the_managed_python_runtime(
    repo_tmp_path: Path,
) -> None:
    run_root = repo_tmp_path / "run"
    stage = run_root / "replay"
    project = stage / "repo"
    problem = project / "problems/01_problem"
    (project / "src").mkdir(parents=True)
    problem.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='runtime-test'\n")
    (project / "uv.lock").write_text("version = 1\n")
    (problem / "check.py").write_text("print('PASS')\n")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is unavailable")
    managed_python = Path(
        subprocess.check_output(
            [uv, "python", "find", "--project", str(REPO / "benchmarks/hard")],
            text=True,
        ).strip()
    ).resolve()
    subprocess.run(
        [uv, "venv", "--python", str(managed_python), str(project / ".venv")],
        check=True,
        capture_output=True,
        text=True,
    )
    trusted_tool = project / ".venv/bin/replay-trusted-tool"
    trusted_tool.write_text("#!/bin/sh\nexit 0\n")
    trusted_tool.chmod(0o755)
    output = stage / "python-runtime.json"
    (problem / "check.py").write_text(
        "import json, os, pathlib, shutil, socket, ssl, sys, threading\n"
        "assert socket.getaddrinfo('localhost', 0)\n"
        "thread_errors = []\n"
        "def rename_native_thread():\n"
        "    try:\n"
        "        comm = pathlib.Path(f'/proc/self/task/{threading.get_native_id()}/comm')\n"
        "        comm.write_text('kb-replay\\n')\n"
        "        assert comm.read_text().strip() == 'kb-replay'\n"
        "    except BaseException as exc:\n"
        "        thread_errors.append(exc)\n"
        "thread = threading.Thread(target=rename_native_thread)\n"
        "thread.start()\n"
        "thread.join()\n"
        "assert not thread_errors, thread_errors\n"
        "tool = pathlib.Path(sys.argv[2])\n"
        "assert os.environ['PATH'].split(':', 1)[0] == str(tool.parent)\n"
        "assert pathlib.Path(shutil.which(tool.name)).samefile(tool)\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({\n"
        "    'base_prefix': sys.base_prefix, 'ssl': ssl.OPENSSL_VERSION,\n"
        "    'path': os.environ['PATH'],\n"
        "}))\n"
    )
    driver = r"""
REAL_PYTHON="$6"
submission_resolve_isolation_tools || exit 77
submission_bind_trusted_stage_tool
submission_add_isolation_readonly_executable "$6"
submission_configure_isolation "$1" "$2"
submission_reset_caches "$3/cache"
submission_select_clean_environment "$3/home"
submission_build_isolated_command "$3" "$4" "$5" check.py -- \
    "$4/.venv/bin/python" -P "$SUBMISSION_TRUSTED_STAGE_TOOL" check.py "$7" "$8"
"${SUBMISSION_ISOLATED_COMMAND[@]}"
"""
    result = _bash(
        driver,
        str(run_root),
        str(repo_tmp_path),
        str(stage),
        str(project),
        str(problem),
        str(managed_python),
        str(output),
        str(trusted_tool),
    )
    if result.returncode == 77:
        pytest.skip("combined user/mount/PID/network namespaces unavailable")
    assert result.returncode == 0, result.stderr
    runtime = json.loads(output.read_text())
    assert Path(runtime["base_prefix"]).resolve() == managed_python.parent.parent
    assert runtime["ssl"]
    assert runtime["path"].split(":", 1)[0] == str(project / ".venv/bin")


def test_namespace_selection_fails_closed_when_full_stack_is_unavailable() -> None:
    result = _bash(
        "SUBMISSION_ISOLATION_TOOLS_RESOLVED=1; "
        "SUBMISSION_ISOLATION_AVAILABLE=0; "
        "if submission_select_network_isolation; then exit 9; fi; "
        'test "$SUBMISSION_NETWORK_MODE" = unavailable',
    )
    assert result.returncode == 0, result.stderr


def test_shared_runner_orders_capture_before_untrusted_grading_and_result_last() -> (
    None
):
    script = RUNNER.read_text()
    capture = script.index("submission_bundle_create")
    check = script.index("run_gpu_locked_timeout check.py", capture)
    benchmark = script.index("run_gpu_locked_timeout benchmark.py", check)
    result = script.rindex('submission_atomic_write_json "$RUN_DIR/result.json"')
    cleanup = script.rindex('rm -rf -- "$RUN_DIR/replays/check"')
    assert capture < check < benchmark < cleanup < result
    assert "submission_extract_peak_fraction" in script
    assert "submission_trusted_surface_digest" in script
    assert 'cp -a "$REPO_ROOT/src" "$WORKSPACE_ROOT/src"' in script
    assert 'ln -s "$REPO_ROOT/src"' not in script
    assert "SUBMISSION_EXPECTED_TRUSTED_SURFACE_DIGEST" in script
    assert '"$SOURCE_PROBLEM_DIR" "$replay_root"' in script
    assert "submission_build_isolated_command" in script
    assert "submission_bind_trusted_stage_tool" in script
    assert script.count('"$SUBMISSION_TRUSTED_STAGE_TOOL"') >= 2
    assert 'python find --project "$REPO_ROOT"' in script
    assert 'submission_executable_matches "$REAL_PYTHON"' in script
    assert '--python "$REAL_PYTHON"' in script
    assert 'cat > "$RUN_DIR/result.json"' not in script
    lock_runner = script[
        script.index("run_gpu_locked_timeout()") : script.index(
            "run_docker_locked_timeout()"
        )
    ]
    assert "$RUN_DIR/bin/gpu-lock-exec" not in lock_runner
    assert 'submission_executable_matches "$REAL_TIMEOUT"' in lock_runner
    prepare = script[
        script.index("prepare_submission_replay_stage()") : script.index(
            'if [ "$HAS_SOLUTION" = "true" ]'
        )
    ]
    sync = prepare.index('"$REAL_UV" sync')
    assert prepare.index("submission_trusted_surface_digest") < sync
    assert prepare.rindex("submission_trusted_surface_digest") > sync


def test_shared_runner_requires_private_container_state() -> None:
    script = RUNNER.read_text()
    assert 'KBH_AGENT_CONTAINER="${KBH_AGENT_CONTAINER:-1}"' in script
    assert 'if [ "$KBH_AGENT_CONTAINER" != "1" ]; then' in script
    assert "KBH_HOST_PID_NAMESPACE_ACTIVE" not in script
    assert 'AGENT_CONTAINER_UV_CACHE="$RUN_DIR/container_uv_cache"' in script
    assert "outputs/container_uv_cache" not in script
    assert "unset KBH_TRUST_PHASE_LOCK_FD" not in script
    assert script.count('-v "$KBH_GPU_LOCK:/kbh/gpu.lock:rw"') == 7
    assert script.count("-e KBH_GPU_LOCK_OWNER_FILE=/home/agent/gpu_lock.owner") == 7
    assert '"$KBH_GPU_LOCK_DIR:/kbh/lock:rw"' not in script
    assert 'getattr(os, "O_NOFOLLOW", 0)' in script
    unsupported = script.index(
        "ccr-claude|kimi|lfm-opencode|lfm-claude|lfm-grok|hermes|pi|nvcf-nemotron"
    )
    allocation = script.index('RUN_DIR_BASE="${REPO_ROOT}/outputs/runs/')
    assert unsupported < allocation


def test_shared_single_gpu_regraders_gate_and_replay_bundles() -> None:
    scripts = [
        REPO / f"benchmarks/{bench}/scripts/regrade_sequential.sh"
        for bench in ("hard", "cuda", "mini")
    ]
    assert len({path.read_bytes() for path in scripts}) == 1
    for path in scripts:
        text = path.read_text()
        assert "submission_bundle_verify" in text, path
        assert "submission_prepare_replay" in text, path
        assert "submission_reset_caches" in text, path
        assert "submission_select_network_isolation" in text, path
        assert "submission_select_clean_environment" in text, path
        assert '"$PROJECT_ROOT/.venv/bin/python"' in text, path
        assert "submission_bind_trusted_stage_tool" in text, path
        assert text.count('"$SUBMISSION_TRUSTED_STAGE_TOOL"') == 2, path
        assert "submission_trusted_surface_digest" in text, path
        assert "submission_atomic_write_json" in text, path
        assert "submission_extract_peak_fraction" in text, path
        assert "submission_check_passed" in text, path
        assert "EXPECTED_SURFACE_DIGEST" in text, path
        assert "archived grader surface differs" in text, path
        assert "post-cutover legacy provenance" in text, path
        assert 'verify-run "$RUN_DIR"' in text, path
        assert 'previous_regrade.get("contended")' in text, path
        assert 'contended = previous_regrade["contended"]' in text, path
        assert "REGRADE_STAGE_COUNT=1" in text, path
        assert "REGRADE_STAGE_COUNT=2" in text, path
        assert '"stage_count": stage_count' in text, path
        assert '"check_exit_code": num(os.environ["CEXIT"])' in text, path
        assert '"benchmark_exit_code": num(os.environ["BEXIT"])' in text, path


def test_shared_regraders_use_full_namespace_but_mega_remains_unchanged() -> None:
    for bench in ("hard", "cuda", "mini"):
        text = (REPO / f"benchmarks/{bench}/scripts/regrade_sequential.sh").read_text()
        assert "submission_resolve_isolation_tools" in text
        assert "submission_configure_isolation" in text
        assert text.count("submission_build_isolated_command") == 2
        assert 'python find --project "$REPO_ROOT"' in text
        assert '--python "$REAL_PYTHON"' in text
        assert '"${SUBMISSION_NETWORK_PREFIX[@]}"' not in text
        assert '"${SUBMISSION_CLEAN_ENV_PREFIX[@]}"' not in text
        for field in (
            "mount_isolated",
            "root_isolated",
            "pid_isolated",
            "clean_environment",
            "in_process_completion_guard",
        ):
            assert f'"{field}"' in text
    mega = (REPO / "benchmarks/mega/scripts/regrade_sequential.sh").read_text()
    assert "submission_build_isolated_command" not in mega


def test_isolation_helper_bounds_output_and_uses_private_uv_cache() -> None:
    text = HELPER.read_text()
    device_block = text[
        text.index('for path in "${device_paths[@]}"; do') : text.index(
            'tmpfs "$new_root/dev/shm"'
        )
    ]
    assert "ulimit -f 131072" in text
    assert 'export UV_CACHE_DIR="$cache_root/uv"' in text
    assert 'export UV_LINK_MODE="copy"' in text
    assert "--user --map-root-user --mount --pid --fork --kill-child=KILL" in text
    assert "--net --ipc --uts" in text
    assert "--bounding-set=-all" in text
    assert '"$pivot_root_bin" . .oldroot' in text
    assert '"$umount_bin" -l /.oldroot' in text
    assert '"$mount_attr_python" -c "$mount_attr_code" / single' in text
    assert text.count('-t proc -o rw,nosuid,nodev,noexec proc') == 2
    assert '-t proc -o ro,nosuid,nodev,noexec proc' not in text
    assert 'remount,bind,ro "$target"' in device_block
    assert '"$mount_attr_python" -c "$mount_attr_code" "$new_root/dev"' in device_block
    assert 'remount,bind,rw' not in device_block
    assert "/var/snap/lxd/common/lxd" not in text
    assert "size=4g,nr_inodes=262144" in text


def test_thin_workers_fail_closed_around_remote_regrades() -> None:
    for name in ("lambda_worker.sh", "brev_worker.sh"):
        text = (REPO / "scripts" / name).read_text()
        assert '"$HERE/scripts/submission_bundle.py"' in text
        assert "$REMOTE_DIR/scripts/submission_bundle.py" in text
        assert '"$HERE/scripts/trusted_stage.py"' in text
        assert "$REMOTE_DIR/scripts/trusted_stage.py" in text
        assert "outputs/imported-regrades/$RID" in text
        assert "./scripts/regrade_sequential.sh" in text
        block = text[text.index("  regrade)") : text.index("  pull)")]
        assert '"$SRC/"' in block
        assert '[ "$RID" = "." ] || [ "$RID" = ".." ]' in block
        assert (
            'LOCAL_PROVENANCE="$(python3 "$HERE/scripts/submission_bundle.py" verify-run "$SRC")"'
            in block
        )
        assert "grep -q '\"submission_replay\"'" not in block
        assert "--require-bundle" not in block
        for root_only in (
            "/.venv",
            "/cache",
            "/tmp",
            "/replays",
            "/regrade_replays",
            "/regrade-reviews",
        ):
            assert f"--exclude '{root_only}'" in block
        assert "--exclude 'cache'" not in block
        assert "--exclude 'tmp'" not in block
        assert "REMOTE_STATUS=0" in block
        assert "|| REMOTE_STATUS=$?" in block
        assert 'if [ "$REMOTE_STATUS" -ne 0 ]' in block
        assert 'REVIEW_ROOT="$SRC/regrade-reviews/' in block
        for artifact in (
            "/result.json",
            "/check*.log",
            "/benchmark*.log",
            "/regrade*.log",
            "/submission_bundle/***",
        ):
            assert f"--include '{artifact}'" in block
        assert 'candidate_regrade == original.get("regrade")' in block
        assert (
            'expected_status = ("bundled", "verified") if bundled else ("legacy", "legacy")'
            in block
        )
        assert 'verify-run "$REVIEW"' in block
        assert 'if [ "$PROVENANCE" != "$LOCAL_PROVENANCE" ]' in block
        assert '"$REVIEW/provenance.json"' in block
        assert "result.regrade.json" not in block
        assert "archive not promoted" in block

        local_gate = block.index('verify-run "$SRC"')
        upload = block.index('"$SRC/"', local_gate)
        remote = block.index("./scripts/regrade_sequential.sh", upload)
        review_pull = block.index("--include '/result.json'", remote)
        status_gate = block.index('if [ "$REMOTE_STATUS" -ne 0 ]', review_pull)
        freshness_gate = block.index(
            'candidate_regrade == original.get("regrade")', status_gate
        )
        returned_gate = block.index('verify-run "$REVIEW"', freshness_gate)
        assert (
            local_gate
            < upload
            < remote
            < review_pull
            < status_gate
            < freshness_gate
            < returned_gate
        )
