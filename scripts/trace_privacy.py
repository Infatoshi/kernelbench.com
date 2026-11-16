"""Which agent tool calls stay out of public trace exports.

Shared by every `benchmarks/<bench>/scripts/traces_to_hf.py` (mega borrows
hard's). A private call is dropped together with its paired tool result.

Two families:
  * private config/environment reads (`~/.claude`, `~/.env_vars`, `env |`, ...);
  * host process listings. The sandbox shares the host's process table, so
    `ps`, `top`, `pstree`, `pgrep -a`, `/proc/<pid>/cmdline` and nvidia-smi's
    process table print sibling benchmark sessions' argv (run ids, the bwrap
    hide list) and unrelated projects' command lines and data paths.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Iterable
from typing import Any

PRIVATE_PATH = re.compile(
    r"(?:^|[/~])\.(?:claude|codex|ssh|kimi-code|agents|grok)(?:/|$)|\.env_vars\b|/proc/\S+/environ\b"
)
PRIVATE_SHELL = re.compile(
    r"(?:^|[;&|])\s*(?:env\s*\||printenv\b|(?:cat|ls|find)\s+~(?:/|\s|$))"
)
# /proc/<pid>/cmdline, /proc/*/cmdline, /proc/$p/cmdline, or a loop that builds it.
PROC_CMDLINE = re.compile(r"/proc/\S*cmdline\b|/proc\b[^\n]*\bcmdline\b")

# Commands that print other processes' names/argv in any form.
_PROCESS_LISTERS = {"ps", "top", "htop", "btop", "atop", "pstree", "procs", "glances",
                    # the host's GPU lease tool: every form prints other projects' job names
                    "overnight-compute"}
# Prefixes that run the next word as a command. Value = options taking an argument.
_WRAPPERS: dict[str, set[str]] = {
    "sudo": {"-u", "-g", "-C", "-D", "-p", "-r", "-t", "-U"},
    "env": {"-u", "-C", "-S", "--unset", "--chdir"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p"},
    "nohup": set(), "time": set(), "command": set(), "exec": set(), "builtin": set(),
    "stdbuf": {"-i", "-o", "-e"},
    "xargs": {"-I", "-n", "-P", "-L", "-d", "-E", "-s", "-a"},
    "watch": {"-n", "--interval"},
    "setsid": set(), "taskset": set(), "chrt": set(),
    "then": set(), "do": set(), "else": set(), "if": set(), "elif": set(),
    "while": set(), "until": set(), "!": set(),
}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_DURATION = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")


def _segments(cmd: str) -> list[str]:
    """Split a shell line into simple commands, quote-aware.

    Separators: ; & | && || newline ( ) { } and command substitution
    ($( and backticks, also inside double quotes). Single-quoted text is opaque.
    """
    segs: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    i, n = 0, len(cmd)

    def flush() -> None:
        s = "".join(cur).strip()
        if s:
            segs.append(s)
        cur.clear()

    while i < n:
        c = cmd[i]
        if quote == "'":
            cur.append(c)
            if c == "'":
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            if cmd[i + 1] == "\n":  # line continuation
                i += 2
                continue
            cur.append(cmd[i:i + 2])
            i += 2
            continue
        if quote == '"':
            if c == '"':
                quote = None
                cur.append(c)
                i += 1
            elif cmd.startswith("$(", i) or c == "`":
                flush()
                i += 2 if c == "$" else 1
            else:
                cur.append(c)
                i += 1
            continue
        if c in "'\"":
            quote = c
            cur.append(c)
            i += 1
            continue
        if cmd.startswith("$(", i):
            flush()
            i += 2
            continue
        if c in ";&|\n()`{}":
            flush()
            i += 1
            continue
        cur.append(c)
        i += 1
    flush()
    return segs


def _words(seg: str) -> list[str]:
    try:
        return shlex.split(seg, comments=True)
    except ValueError:  # unbalanced quote after splitting at a substitution
        return seg.replace('"', " ").replace("'", " ").split()


def _strip_wrappers(words: list[str]) -> list[str]:
    i = 0
    while i < len(words):
        w = words[i]
        if _ASSIGN.match(w):
            i += 1
            continue
        base = os.path.basename(w)
        if base not in _WRAPPERS:
            break
        takes_arg = _WRAPPERS[base]
        i += 1
        while i < len(words) and words[i].startswith("-") and words[i] != "-":
            opt = words[i]
            i += 1
            if opt in takes_arg and "=" not in opt and i < len(words):
                i += 1
            if opt == "--":
                break
        if base == "env":
            while i < len(words) and _ASSIGN.match(words[i]):
                i += 1
        if base == "timeout" and i < len(words) and _DURATION.match(words[i]):
            i += 1
        if base == "taskset" and i < len(words):  # mask or cpu-list
            i += 1
        if base == "chrt" and i < len(words) and words[i].isdigit():
            i += 1
    return words[i:]


def _pgrep_lists(args: list[str]) -> bool:
    for a in args:
        if a in ("--list-name", "--list-full"):
            return True
        if re.match(r"^-[A-Za-z]+$", a) and ("l" in a or "a" in a):
            return True
    return False


_NVSMI_SAFE_SUB = {"topo", "nvlink", "dmon", "clocks", "mig", "-L", "--list-gpus",
                   "-h", "--help", "--version", "conf-compute", "drain"}


def _nvidia_smi_lists(args: list[str]) -> bool:
    """True when nvidia-smi prints process names / command lines."""
    if "pmon" in args:
        return True
    joined = " ".join(args)
    apps = re.search(r"--query-(?:compute|accounted)-apps(?:=|\s+)(\S+)", joined)
    if apps:
        fields = {f.strip() for f in apps.group(1).split(",")}
        return bool(fields & {"name", "process_name"})
    if re.search(r"--query-[a-z-]+", joined):  # --query-gpu, --query-retired-pages, ...
        return False
    if any(a in _NVSMI_SAFE_SUB for a in args):
        return False
    if any(a in ("-q", "--query") for a in args):
        disp = re.search(r"(?:-d|--display)(?:=|\s+)(\S+)", joined)
        return not disp or "PIDS" in disp.group(1).upper()
    # Default table (with or without -i/-l/--loop) ends in the Processes table.
    return True


def _segment_lists_processes(words: list[str], depth: int) -> bool:
    words = _strip_wrappers(words)
    if not words:
        return False
    prog = os.path.basename(words[0])
    args = words[1:]
    if prog in _PROCESS_LISTERS:
        return True
    if prog == "pgrep":
        return _pgrep_lists(args)
    if prog == "nvidia-smi":
        return _nvidia_smi_lists(args)
    # `bash -lc '...'`, `timeout 5 sh -c "..."`, `watch 'ps aux'` (one quoted arg).
    if depth < 4:
        if prog in _SHELLS:
            for j, a in enumerate(args):
                if re.match(r"^-[A-Za-z]*c[A-Za-z]*$", a) and j + 1 < len(args):
                    return lists_host_processes(args[j + 1], depth + 1)
        if len(words) == 1 and " " in words[0]:
            return lists_host_processes(words[0], depth + 1)
    return False


_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[^\n]*\n")
_SHELL_WORD = re.compile(r"(?:^|[\s/;&|(])(?:ba|z|da|k)?sh(?:\s|$)")


def _drop_heredoc_bodies(command: str) -> str:
    """Heredoc bodies fed to python/cat are data, not commands (`top = 5`).
    Bodies fed to a shell (`bash <<EOF`) stay and are scanned."""
    out, pos = [], 0
    for m in _HEREDOC.finditer(command):
        if m.start() < pos:
            continue
        line_start = command.rfind("\n", 0, m.start()) + 1
        end = re.compile(rf"^\s*{re.escape(m.group(2))}\s*$", re.M).search(command, m.end())
        stop = end.end() if end else len(command)
        out.append(command[pos:m.end()])
        if _SHELL_WORD.search(command[line_start:m.start()]):
            out.append(command[m.end():stop])
        pos = stop
    out.append(command[pos:])
    return "".join(out)


def lists_host_processes(command: str, _depth: int = 0) -> bool:
    """True when a shell line lists host processes anywhere in it."""
    if not command:
        return False
    if PROC_CMDLINE.search(command):
        return True
    command = _drop_heredoc_bodies(command)
    return any(_segment_lists_processes(_words(seg), _depth) for seg in _segments(command))


# `raw` holds unparsed args (Codex apply_patch input, odd JSON); read it only
# for shell-shaped tools so patch text is never parsed as a command line.
_COMMAND_KEYS = ("command", "cmd", "CommandLine", "commandLine")
_SHELL_TOOL = re.compile(r"shell|bash|exec|terminal|command|run", re.I)
_PATH_KEYS = ("file_path", "path", "filePath", "absolute_path", "AbsolutePath", "target_file", "TargetFile")


def _as_text(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
        return shlex.join(v)
    return ""


# Codex code mode sends JS: `tools.exec_command({cmd:"ps -eo ...", ...})`.
_JS_CMD = re.compile(r"""\b(?:cmd|command)["']?\s*:\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""")


def _js_commands(raw: str) -> list[str]:
    out = []
    for m in _JS_CMD.finditer(raw):
        lit = m.group(1)
        try:
            out.append(json.loads(lit) if lit[0] == '"' else lit[1:-1])
        except ValueError:
            out.append(lit[1:-1])
    return out


def _call_text(call: Any) -> tuple[str, list[str]]:
    args = getattr(call, "args", None)
    if args is None and isinstance(call, dict):
        args = call
    args = args if isinstance(args, dict) else {}
    target = next((_as_text(args.get(k)) for k in _PATH_KEYS if _as_text(args.get(k))), "")
    command = next((_as_text(args.get(k)) for k in _COMMAND_KEYS if _as_text(args.get(k))), "")
    if command:
        return target, [command]
    if _SHELL_TOOL.search(str(getattr(call, "name", "") or "")):
        raw = _as_text(args.get("raw"))
        js = _js_commands(raw)
        return target, js or [raw]
    return target, []


def private_tool_call(call: Any) -> bool:
    """A ToolCall (or args dict) whose call and result must not be published."""
    target, commands = _call_text(call)
    if PRIVATE_PATH.search(target) or PROC_CMDLINE.search(target):
        return True
    return any(
        PRIVATE_PATH.search(c) or PRIVATE_SHELL.search(c) or lists_host_processes(c)
        for c in commands
    )


def private_call_ids(events: Iterable[Any], rid: str) -> set[str]:
    """IDs of private calls; raises when one cannot be paired with its result."""
    events = list(events)
    ids: set[str] = set()
    for event in events:
        for call in getattr(event, "tool_calls", None) or []:
            if private_tool_call(call):
                if not call.call_id:
                    raise ValueError(f"private tool call without pairable ID in {rid}")
                ids.add(call.call_id)
    paired = {e.tool_result.call_id for e in events if getattr(e, "tool_result", None) is not None}
    if ids - paired:
        raise ValueError(f"private tool call missing its paired result in {rid}")
    return ids
