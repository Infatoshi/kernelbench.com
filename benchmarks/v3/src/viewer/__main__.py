"""Terminal transcript viewer.

Usage:
  uv run python -m src.viewer <transcript-file>
  uv run python -m src.viewer <problem-dir>                 # looks for transcript.jsonl or transcript.log
  uv run python -m src.viewer <run-dir>                     # lists problems (status, speedup, duration)
  uv run python -m src.viewer <run-dir> <problem-name>      # shortcut

Flags:
  --full            don't truncate long tool outputs
  --max-lines N     cap long outputs to N lines (default 40)
  --no-thinking     hide thinking blocks
  --no-tools        hide tool calls/results
  --since SEQ       only show events at seq >= SEQ
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.markup import escape as _escape

from .parsers import Event, load

CONSOLE = Console(highlight=False)


ROLE_STYLE = {
    "user":        ("bright_black",  "bold",       "👤 USER"),
    "assistant":   ("cyan",          "bold cyan",  "🤖 ASSISTANT"),
    "thinking":    ("magenta",       "italic dim", "💭 THINKING"),
    "tool_call":   ("yellow",        "bold yellow","🔧 TOOL CALL"),
    "tool_result": ("green",         "",           "📄 RESULT"),
    "system":      ("bright_black",  "dim",        "⚙ SYSTEM"),
}
ERROR_STYLE = ("red",  "bold red", "❌ ERROR RESULT")


def render_event(evt: Event, max_lines: int | None, hide_thinking: bool, hide_tools: bool) -> None:
    if hide_thinking and evt.role == "thinking":
        return
    if hide_tools and evt.role in ("tool_call", "tool_result"):
        return
    border, title_style, icon = ERROR_STYLE if evt.is_error else ROLE_STYLE.get(evt.role, ROLE_STYLE["system"])
    icon_md = f"[{title_style}]{icon}[/]" if title_style else icon
    header = f"{icon_md} [dim]#{evt.seq}[/]"
    if evt.label and evt.label not in (icon.split(" ", 1)[-1].lower(), "assistant", "user prompt"):
        header += f" [bold]{_escape(evt.label)}[/]"

    body = evt.content or ""
    lines = body.splitlines() or [body]
    truncated = 0
    if max_lines and len(lines) > max_lines:
        truncated = len(lines) - max_lines
        lines = lines[:max_lines]
        lines.append(f"… (truncated {truncated} more lines — pass --full)")

    rendered_body = _render_body(evt, "\n".join(lines))
    CONSOLE.print(Panel(rendered_body, title=header, border_style=border, title_align="left", padding=(0, 1)))


def _render_body(evt: Event, body: str):
    if not body.strip():
        return Text("(empty)", style="dim")
    # Syntax-highlight code-looking content
    if evt.role == "tool_call" and evt.label in ("Write", "Edit") and "\n" in body:
        return Syntax(body, "python", theme="ansi_dark", word_wrap=True, line_numbers=False)
    if evt.role == "tool_result" and _looks_like_code(body):
        return Syntax(body, _guess_lang(body), theme="ansi_dark", word_wrap=True, line_numbers=False)
    if evt.role == "tool_call" and evt.label in ("Execute", "Bash", "exec"):
        return Syntax(body, "bash", theme="ansi_dark", word_wrap=True, line_numbers=False)
    return Text(body)


def _looks_like_code(s: str) -> bool:
    head = s.lstrip()[:200]
    return head.startswith(("import ", "from ", "def ", "class ", "#include", "__global__", "@triton", "template<"))


def _guess_lang(s: str) -> str:
    head = s.lstrip()[:200]
    if head.startswith(("import ", "from ", "def ", "class ", "@triton")):
        return "python"
    if head.startswith(("#include", "__global__", "template<", "__device__")):
        return "cpp"
    return "text"


def render_transcript(path: Path, args) -> None:
    fmt, events = load(path)
    CONSOLE.print(Rule(f"[bold]{path}[/] [dim]({fmt}, {len(events)} events)[/]"))
    for evt in events:
        if args.since and evt.seq < args.since:
            continue
        render_event(evt, None if args.full else args.max_lines, args.no_thinking, args.no_tools)


def list_run(run_dir: Path) -> None:
    table = Table(title=str(run_dir), expand=False)
    table.add_column("Problem", overflow="fold")
    table.add_column("Status")
    table.add_column("Speedup", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Framework")
    table.add_column("Events", justify="right")

    problems = sorted([p for p in run_dir.iterdir() if p.is_dir()])
    for pd in problems:
        result_f = pd / "result.json"
        result = {}
        if result_f.exists():
            try:
                result = json.loads(result_f.read_text())
            except Exception:
                pass
        if result.get("correct"):
            status = "[green]PASS[/]"
        elif result.get("has_solution"):
            status = "[yellow]FAIL[/]"
        else:
            status = "[red]NOSOL[/]"
        speedup = result.get("speedup")
        speedup_s = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "—"
        elapsed = result.get("elapsed_seconds")
        elapsed_s = f"{elapsed}s" if elapsed else "—"
        fw_f = pd / "solution.py"
        framework = "—"
        if fw_f.exists():
            # Reuse detection regex inline; this is read once per problem so cheap
            from ..eval.framework_detect import detect_framework
            framework = detect_framework(fw_f.read_text())
        # Count events by loading if we have a transcript
        tr = pd / "transcript.jsonl"
        if not tr.exists():
            tr = pd / "transcript.log"
        n_events = "—"
        if tr.exists():
            try:
                _, evs = load(tr)
                n_events = str(len(evs))
            except Exception:
                pass
        table.add_row(pd.name, status, speedup_s, elapsed_s, framework, n_events)
    CONSOLE.print(table)


def resolve_target(target: Path, args) -> Path | None:
    """Pick the transcript file inside a path. Returns None if we should list_run instead."""
    if target.is_file():
        return target
    if target.is_dir():
        # Problem dir?
        for name in ("transcript.jsonl", "transcript.log"):
            f = target / name
            if f.exists():
                return f
        # Run dir? Look for subdirs with transcripts
        subs = [p for p in target.iterdir() if p.is_dir()]
        if args.problem:
            match = target / args.problem
            if match.is_dir():
                return resolve_target(match, args)
        return None
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("problem", nargs="?", default=None, help="problem name (if path is a run dir)")
    p.add_argument("--full", action="store_true")
    p.add_argument("--max-lines", type=int, default=40)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--since", type=int, default=0)
    args = p.parse_args()

    if not args.path.exists():
        print(f"path not found: {args.path}", file=sys.stderr)
        return 2

    tr = resolve_target(args.path, args)
    if tr is None:
        list_run(args.path)
        return 0
    render_transcript(tr, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
