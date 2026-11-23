"""Public trace exports drop host process listings and private config reads.

The agent sandbox shares the host process table, so `ps`/`top`/nvidia-smi
output carries sibling sessions' argv and unrelated projects' paths.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("trace_privacy", ROOT / "scripts/trace_privacy.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)


@pytest.mark.parametrize("cmd", [
    "ps aux",
    "ps -eo pid,comm,args --sort=-pid | grep -E 'python|ncu|nsys'",
    "ps -eo pid,ppid,stat,etime,args --forest | awk '/gpu-lock/'",
    "ps -o pid,ppid,stat,etime,args -p 2252363,2255831 --forest",
    "ps --ppid 123 -o pid,args; ls -la",
    "ls -la && ps -ef",
    "false || ps -ef",
    "echo start\nps -ef | head",
    "echo $(ps -o args= -p 1)",
    "echo `ps -o args= -p 1`",
    'echo "lock: $(ps -p 7 -o args=)"',
    "timeout 5 ps aux",
    "timeout -k 3 10s ps aux",
    "sudo ps aux",
    "sudo -u root ps aux",
    "env FOO=1 ps aux",
    "FOO=1 ps aux",
    "/bin/ps aux",
    "cat x | xargs ps -o args= -p",
    "watch -n 1 nvidia-smi",
    "bash -lc 'ps aux | grep python'",
    "top -b -n 1",
    "htop",
    "pstree -ap",
    "pgrep -a python",
    "pgrep -fl gpu-lock",
    "pgrep --list-full python",
    "cat /proc/1234/cmdline",
    "tr '\\0' ' ' < /proc/*/cmdline",
    "for p in $(pgrep python); do cat /proc/$p/cmdline; done",
    "nvidia-smi",
    "nvidia-smi -i 0",
    "nvidia-smi -q",
    "nvidia-smi -q -d PIDS",
    "nvidia-smi pmon -c 1",
    "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv",
    "nvidia-smi --query-compute-apps=pid,name --format=csv",
    "uv run python check.py; nvidia-smi",
    "if true; then ps -ef; fi",
    "bash <<'EOF'\nps aux\nEOF",
    "python - <<'PY'\nprint(1)\nPY\nps aux",
])
def test_process_listing_positive(cmd):
    assert tp.lists_host_processes(cmd), cmd
    assert tp.private_tool_call({"command": cmd}), cmd


@pytest.mark.parametrize("cmd", [
    "uv run python -c 'import psutil; print(psutil.cpu_count())'",
    "lpstat -p cups",
    "python bench.py --ps 4",
    "python ps.py",
    "cat ps.py",
    "./ps.py",
    "python -c 'from x import top_k; top_k(3)'",
    "grep -n top_k solution.py",
    'grep -E "python|top" log.txt',
    'echo "top 5 kernels"',
    "git log --oneline | head -3",
    "ls -la /proc/driver/nvidia",
    "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv",
    "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader",
    "nvidia-smi -L",
    "nvidia-smi topo -m",
    "nvidia-smi -q -d MEMORY",
    "pgrep python",
    "pgrep -f gpu-lock",
    "pidof python3",
    "kill -0 1234",
    "uv run python check.py && uv run python benchmark.py",
    "python - <<'PY'\ntop = 5\nps = [1]\nprint(top, ps)\nPY",
    "cat > notes.txt <<EOF\ntop kernels\nEOF\nls",
])
def test_process_listing_negative(cmd):
    assert not tp.lists_host_processes(cmd), cmd
    assert not tp.private_tool_call({"command": cmd}), cmd


def test_existing_rules_kept():
    for args in (
        {"command": "cat ~/.env_vars"},
        {"command": "env | sort"},
        {"command": "printenv"},
        {"file_path": "/home/u/.claude/settings.json"},
        {"path": "/home/u/.kimi-code/config.toml"},
        {"command": "cat /proc/1/environ"},
        {"file_path": "/proc/123/cmdline"},
        # operator skill files and the host's GPU lease listing
        {"target_file": "/home/u/.grok/skills/anvil-shared-gpu/SKILL.md"},
        {"command": "sed -n '1,240p' /home/u/.agents/skills/kernel/SKILL.md"},
        {"command": "/usr/bin/zsh -lc \"overnight-compute --help\""},
        {"command": "overnight-compute status | head"},
    ):
        assert tp.private_tool_call(args), args
    assert not tp.private_tool_call({"file_path": "solution.py"})


def test_command_list_and_raw_only_for_shell_tools():
    class Call:
        def __init__(self, name, args):
            self.name, self.args = name, args

    assert tp.private_tool_call(Call("exec_command", {"command": ["bash", "-lc", "ps aux"]}))
    assert tp.private_tool_call(Call("shell", {"raw": "ps -ef"}))
    # Codex code mode: JS calling tools.exec_command({cmd: "..."}).
    js = 'text(await tools.exec_command({cmd:"ps -eo pid,args | rg \'nvcc\' | head; tail log",max_output_tokens:2000}));\n'
    assert tp.private_tool_call(Call("exec", {"raw": js}))
    js_ok = 'text(await tools.exec_command({cmd:"nvidia-smi --query-gpu=index --format=csv; ls -la"}));\n'
    assert not tp.private_tool_call(Call("exec", {"raw": js_ok}))
    # apply_patch input is text, never a command line.
    assert not tp.private_tool_call(Call("apply_patch", {"raw": "*** Begin Patch\n top = x\n"}))


def _write_run(run_dir: Path) -> None:
    lines = [
        {"type": "system", "subtype": "init", "session_id": "s1", "model": "m", "cwd": "/workspace"},
        {"type": "assistant", "message": {"model": "m", "content": [
            {"type": "tool_use", "id": "t_ok", "name": "Bash", "input": {"command": "ls -la"}},
            {"type": "tool_use", "id": "t_ps", "name": "Bash",
             "input": {"command": "ps -eo pid,args --forest | awk '/gpu-lock/'"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t_ok", "content": "solution.py check.py"},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t_ps",
             "content": "99 bwrap --die-with-parent /data/sibling/private_project/run.py"},
        ]}},
        {"type": "assistant", "message": {"model": "m", "content": [{"type": "text", "text": "done"}]}},
    ]
    run_dir.mkdir(parents=True)
    (run_dir / "transcript.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


# mega has no exporter of its own: kb falls back to cuda's.
@pytest.mark.parametrize("bench", ["hard", "cuda", "mini"])
def test_exporter_drops_paired_ps_call_keeps_neighbour(bench, tmp_path):
    run = tmp_path / "runs" / "20261116_000000_claude_m_01_x"
    _write_run(run)
    out = tmp_path / "out"
    script = ROOT / "benchmarks" / bench / "scripts" / "traces_to_hf.py"
    proc = subprocess.run([sys.executable, str(script), str(out), str(run)],
                          capture_output=True, text=True, cwd=ROOT / "benchmarks" / bench)
    assert proc.returncode == 0, proc.stderr
    assert "removed 1 paired private tool calls/results" in proc.stdout
    text = (out / f"{run.name}.jsonl").read_text()
    assert "bwrap" not in text and "private_project" not in text and "t_ps" not in text
    assert '"t_ok"' in text and "solution.py check.py" in text and "ls -la" in text


def test_mega_falls_back_to_a_filtering_exporter():
    assert not (ROOT / "benchmarks/mega/scripts/traces_to_hf.py").exists()
    cli = (ROOT / "kbtool/kb/cli.py").read_text()
    assert '_bench_dir(root, "cuda") / "scripts" / "traces_to_hf.py"' in cli
    for bench in ("hard", "cuda", "mini"):
        src = (ROOT / "benchmarks" / bench / "scripts" / "traces_to_hf.py").read_text()
        assert "from scripts.trace_privacy import private_call_ids" in src, bench
