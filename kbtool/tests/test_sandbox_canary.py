"""Unit tests for the launch-time sandbox canary (scripts/lib/sandbox_canary.py).

No bwrap needed: the canary is pure-stdlib path sweeping, so it is exercised
against fake trees here on any OS. The Linux end-to-end path (bwrap argv +
canary inside the sandbox) is scripts/smoke_sandbox_linux.sh.
"""
import importlib.util
from pathlib import Path

_CANARY = Path(__file__).resolve().parents[2] / "scripts/lib/sandbox_canary.py"
_spec = importlib.util.spec_from_file_location("sandbox_canary", _CANARY)
canary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(canary)

RUN_ID = "20260819_101112_claude_claude-opus-5_01_fp8_gemm"
HP_ID = "20260601_030405_or-fable_anthropic_claude-fable-5_01_fp8_gemm"


def _mk_clean_tree(root: Path) -> Path:
    """The tree a correctly built sandbox presents: only this run's dir, the
    honeytoken decoy, and the gpu lock dir are visible."""
    runs = root / "outputs" / "runs"
    own = runs / RUN_ID
    (own / "repo" / "problems" / "01_fp8_gemm").mkdir(parents=True)
    (own / "repo" / "problems" / "01_fp8_gemm" / "check.py").write_text("pass\n")
    hp = runs / HP_ID
    hp.mkdir(parents=True)
    (hp / "solution.py").write_text("decoy\n")
    (hp / "result.json").write_text("{}")
    lock = root / "outputs" / "gpu_lock"
    lock.mkdir(parents=True)
    (lock / "gpu.lock").write_text("")
    return root


def test_clean_sandbox_tree_passes(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    assert canary.scan([str(root)], [], RUN_ID, [HP_ID]) == []


def test_own_run_files_are_ignored(tmp_path):
    """result.json / solution.py inside this run's own dir must not fire."""
    root = _mk_clean_tree(tmp_path / "bench")
    own = root / "outputs" / "runs" / RUN_ID
    (own / "result.json").write_text("{}")
    (own / "repo" / "problems" / "01_fp8_gemm" / "solution.py").write_text("x\n")
    assert canary.scan([str(root)], [], RUN_ID, [HP_ID]) == []


def test_foreign_run_dir_is_detected(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    foreign = root / "outputs" / "runs" / "20260719_121747_or-fable_anthropic_claude-fable-5_01_fp8_gemm"
    foreign.mkdir(parents=True)
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert len(hits) == 1 and "foreign run dir" in hits[0]


def test_stray_solution_py_is_detected(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    pub = root / "public" / "runs"
    pub.mkdir(parents=True)
    (pub / "20260719_121747_solution.py.txt").write_text("kernel\n")
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert any("solution file" in h for h in hits)


def test_stray_result_json_is_detected(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    stray = root / "somewhere"
    stray.mkdir()
    (stray / "result.json").write_text("{}")
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert any("result.json" in h for h in hits)


def test_runs_remote_and_lambda_dirs_are_detected_even_empty(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    (root / "outputs" / "runs-remote-pro").mkdir()
    (root / "outputs" / "runs-lambda-h100-a").mkdir()
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert sum("pulled-archive dir" in h for h in hits) == 2


def test_published_code_dir_is_detected(tmp_path):
    root = _mk_clean_tree(tmp_path / "bench")
    (root / "public" / "data" / "mega" / "code").mkdir(parents=True)
    hits = canary.scan([str(root)], [], RUN_ID, [HP_ID])
    assert any("published-code dir" in h for h in hits)


def test_must_be_hidden_nonempty_fails_empty_passes(tmp_path):
    empty = tmp_path / "kb-remote-archives-tmpfs"
    empty.mkdir()
    assert canary.scan([], [str(empty)], RUN_ID, []) == []

    leaky = tmp_path / "kb-remote-archives"
    (leaky / "20260719_121747_x").mkdir(parents=True)
    hits = canary.scan([], [str(leaky)], RUN_ID, [])
    assert any("must-be-hidden" in h for h in hits)

    gone = tmp_path / "never-existed"
    assert canary.scan([], [str(gone)], RUN_ID, []) == []


def test_main_exit_codes(tmp_path, capsys):
    root = _mk_clean_tree(tmp_path / "bench")
    assert canary.main(["--run-id", RUN_ID, "--allow", HP_ID, "--walk", str(root)]) == 0
    (root / "outputs" / "runs" / "20260719_121747_foreign").mkdir()
    assert canary.main(["--run-id", RUN_ID, "--allow", HP_ID, "--walk", str(root)]) == 1
    out = capsys.readouterr().out
    assert "REFUSING LAUNCH" in out


# --- session isolation: PID namespace + GPU (2026-11-16) ----------------------
import re  # noqa: E402

import pytest  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
TOKEN = "kbh-sandbox-sentinel-123-4567"
UUID_A = "GPU-00112233-4455-6677-8899-aabbccddeeff"
UUID_B = "GPU-ffeeddcc-bbaa-9988-7766-554433221100"


def _fake_proc(tmp_path: Path, own_ns: str, procs: dict[int, bytes]) -> str:
    proc = tmp_path / "proc"
    (proc / "self" / "ns").mkdir(parents=True)
    (proc / "self" / "ns" / "pid").symlink_to(own_ns)
    for pid, cmd in procs.items():
        (proc / str(pid)).mkdir()
        (proc / str(pid) / "cmdline").write_bytes(cmd)
    return str(proc)


def _esrch(pid, sig):
    raise ProcessLookupError(pid)


def test_pidns_private_passes(tmp_path):
    proc = _fake_proc(tmp_path, "pid:[2]", {
        1: b"bwrap\0--unshare-pid\0python3\0canary.py\0--sentinel-token\0" + TOKEN.encode() + b"\0",
        2: b"python3\0canary.py\0--sentinel-token\0" + TOKEN.encode() + b"\0"})
    assert canary.check_pidns(99999, TOKEN, "pid:[1]", proc=proc, kill=_esrch) == []


def test_pidns_same_namespace_fails(tmp_path):
    proc = _fake_proc(tmp_path, "pid:[1]", {1: b"init\0"})
    hits = canary.check_pidns(99999, TOKEN, "pid:[1]", proc=proc, kill=_esrch)
    assert any("PID namespace" in h for h in hits)


def test_pidns_visible_sentinel_fails_without_killing_it(tmp_path):
    proc = _fake_proc(tmp_path, "pid:[2]", {99999: TOKEN.encode() + b"\x00300\x00"})
    killed = []
    hits = canary.check_pidns(99999, TOKEN, "pid:[1]", proc=proc,
                              kill=lambda p, s: killed.append(p))
    assert any("sentinel visible" in h for h in hits) and killed == []


def test_pidns_signallable_sentinel_fails(tmp_path):
    proc = _fake_proc(tmp_path, "pid:[2]", {1: b"bwrap\0"})
    hits = canary.check_pidns(99999, TOKEN, "pid:[1]", proc=proc, kill=lambda p, s: None)
    assert any("signallable" in h for h in hits)


def test_pidns_requires_token(tmp_path):
    proc = _fake_proc(tmp_path, "pid:[2]", {})
    assert canary.check_pidns(99999, "", "pid:[1]", proc=proc, kill=_esrch)


def test_uuid_format():
    raw = bytes.fromhex("00112233445566778899aabbccddeeff")
    assert canary._uuid_str(raw) == UUID_A


def _cuda_ok(envs):
    return {k: ([UUID_A] if k in ("unset", "0", "all-indices") else []) for k in envs}


def test_gpu_only_assigned_passes():
    hits = canary.check_gpu(UUID_A, [UUID_B], 4, nvml=lambda: [(UUID_A, 2)], cuda=_cuda_ok)
    assert hits == []


def test_gpu_probe_envs_cover_overrides():
    envs = canary._cuda_probe_envs([UUID_B], 4)
    assert envs["unset"] is None and envs["0"] == "0"
    assert envs["all-indices"] == "0,1,2,3" and envs[UUID_B] == UUID_B


def test_gpu_second_card_in_nvml_fails():
    hits = canary.check_gpu(UUID_A, [UUID_B], 4,
                            nvml=lambda: [(UUID_A, 2), (UUID_B, 1)], cuda=_cuda_ok)
    assert any("NVML sees" in h for h in hits)


def test_gpu_override_reaching_other_card_fails():
    def cuda(envs):
        r = _cuda_ok(envs)
        r[UUID_B] = [UUID_B]
        return r
    hits = canary.check_gpu(UUID_A, [UUID_B], 4, nvml=lambda: [(UUID_A, 2)], cuda=cuda)
    assert any("reaches another GPU" in h for h in hits)


def test_gpu_assigned_card_unreachable_fails():
    def cuda(envs):
        r = _cuda_ok(envs)
        r["unset"] = []
        return r
    hits = canary.check_gpu(UUID_A, [], 1, nvml=lambda: [(UUID_A, 0)], cuda=cuda)
    assert any("does not reach the assigned GPU" in h for h in hits)


def test_gpu_nvml_failure_or_empty_fails():
    def boom():
        raise OSError("libnvidia-ml.so.1 missing")
    assert canary.check_gpu(UUID_A, [], 1, nvml=boom, cuda=_cuda_ok)
    assert canary.check_gpu(UUID_A, [], 1, nvml=lambda: [], cuda=_cuda_ok)


def test_sandbox_sh_builds_isolated_argv():
    """The shared builder must create the PID namespace, a fresh /proc and /dev with
    only the assigned GPU node, pin CUDA_VISIBLE_DEVICES=0 inside, and run the
    sentinel + GPU canary on every launch."""
    text = (_ROOT / "scripts/lib/sandbox.sh").read_text()
    argv = re.search(r"^\s*KBH_SBX=\(bwrap [^)]*\)", text, re.M).group(0)
    for flag in ("--die-with-parent", "--unshare-pid", "--dev-bind / /", "--proc /proc", "--dev /dev"):
        assert flag in argv, flag
    assert argv.index("--dev-bind / /") < argv.index("--proc /proc")
    assert argv.index("--dev-bind / /") < argv.index("--dev /dev")
    assert "--setenv CUDA_VISIBLE_DEVICES 0" in text
    assert '--dev-bind "$node" "$node"' in text
    assert "--gpu-info" in text and "--sentinel-pid" in text and "--gpu-uuid" in text
    assert '"pid_namespace": true' in text and '"gpu_uuid"' in text
    assert "KBH_GPU" in text and "exit 4" in text


def _lock_wrapper(path: Path) -> str:
    text = path.read_text()
    m = re.search(r"^cat > \"\$LOCK_WRAPPER_DIR/gpu-lock-exec\" <<'EOF'\n(.*?)^EOF$", text, re.M | re.S)
    assert m, path
    return m.group(1)


def test_gpu_lock_wrapper_copies_identical_and_namespace_safe():
    shared = _lock_wrapper(_ROOT / "scripts/lib/run_harness.sh")
    mega = _lock_wrapper(_ROOT / "benchmarks/mega/scripts/run_hard.sh")
    assert shared == mega, "gpu-lock-exec in run_harness.sh and mega run_hard.sh diverged"
    # A bare `kill -0 <owner>` is meaningless across PID namespaces: owners are
    # trusted only through the pidns + start-time tag.
    assert "kill -0" not in shared
    assert "readlink /proc/self/ns/pid" in shared
    assert '"$(pid_tag $$)" > "$owner_file"' in shared
    # The namespace id itself contains a colon ("pid:[4026...]"): strip only the
    # trailing ":<starttime>", or the ancestor walk never runs.
    assert '"${held_tag%:*}" = "$my_pidns"' in shared and "held_tag%%:" not in shared


@pytest.mark.parametrize("script", ["scripts/lib/run_harness.sh", "benchmarks/mega/scripts/run_hard.sh"])
def test_runners_still_source_shared_sandbox(script):
    text = (_ROOT / script).read_text()
    assert "kbh_sandbox_init" in text and "sandbox.sh" in text
