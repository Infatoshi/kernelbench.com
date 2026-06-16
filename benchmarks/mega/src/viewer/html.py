"""Self-contained static HTML generator.

Produces a single .html with embedded CSS+JS and event content. Loads Prism.js
from CDN for syntax highlighting (no offline-mode for v1).
"""
from __future__ import annotations

import html as html_mod
import json
from pathlib import Path

from src.viewer.events import Event, Session

CSS = """
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
       background: #0f1115; color: #d8dee9; line-height: 1.5; }
header { background: #1b1f27; border-bottom: 1px solid #2c313a; padding: 12px 24px;
         position: sticky; top: 0; z-index: 50; display: flex; gap: 24px; align-items: baseline; flex-wrap: wrap; }
header .title { font-weight: 600; font-size: 15px; }
header .meta { font-size: 13px; color: #8c95a8; }
header .meta b { color: #d8dee9; font-weight: 500; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
.event { margin-bottom: 16px; border-left: 3px solid #2c313a; padding: 10px 14px; background: #161922;
         border-radius: 4px; }
.event .role { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
               margin-bottom: 6px; color: #6c7689; }
.event .body { font-size: 14px; white-space: pre-wrap; word-break: break-word; }
.event .meta { font-size: 11px; color: #6c7689; margin-top: 6px; }
.event[data-role="user"] { border-left-color: #4dc4ff; }
.event[data-role="user"] .role { color: #4dc4ff; }
.event[data-role="assistant"] { border-left-color: #4ade80; }
.event[data-role="assistant"] .role { color: #4ade80; }
.event[data-role="tool"] { border-left-color: #fbbf24; }
.event[data-role="tool"] .role { color: #fbbf24; }
.event[data-role="system"] { border-left-color: #6c7689; opacity: 0.85; }
.event[data-role="system"] .role { color: #8c95a8; }
.event[data-role="error"] { border-left-color: #ef4444; background: #2b1517; }
.event[data-role="error"] .role { color: #ef4444; }
.event[data-role="compaction"] { border-left-color: #c084fc; background: #221a2e; }
.event[data-role="compaction"] .role { color: #c084fc; }
/* Sidechain styling outside a subagent dropdown (e.g. true Claude Code
   isSidechain events not framed by task_started/task_notification). */
.event[data-sidechain="1"]:not(.subagent-inner .event) {
    margin-left: 32px; background: #131722; border-left-style: dashed; position: relative; }
.event[data-sidechain="1"]:not(.subagent-inner .event)::before {
    content: "subagent"; position: absolute; margin-left: -42px;
    margin-top: 2px; font-size: 10px; color: #6c89bf; letter-spacing: 0.05em; }
.event[data-role="system"][data-subtype="task_started"],
.event[data-role="system"][data-subtype="task_notification"] {
    background: #131c2a; border-left-color: #6c89bf; }
.event[data-role="system"][data-subtype="task_started"] .role,
.event[data-role="system"][data-subtype="task_notification"] .role { color: #6c89bf; }
.subagent-block { margin-bottom: 16px; background: #131c2a; border: 1px solid #2a3548;
    border-left: 3px solid #6c89bf; border-radius: 6px; padding: 0; overflow: hidden; }
.subagent-block > summary { padding: 14px 18px; cursor: pointer; font-size: 14px;
    color: #a3aab8; font-weight: 500; user-select: none; list-style: none;
    display: flex; align-items: center; gap: 12px; transition: background 0.15s; }
.subagent-block > summary::-webkit-details-marker,
.subagent-block > summary::marker { display: none; }
.subagent-block > summary:hover { background: #182236; color: #d8dee9; }
.subagent-block > summary::before {
    content: "";
    display: inline-block;
    width: 0; height: 0;
    border-left: 9px solid #6c89bf;
    border-top: 6px solid transparent;
    border-bottom: 6px solid transparent;
    transition: transform 0.18s ease-out;
    flex-shrink: 0;
}
.subagent-block[open] > summary::before { transform: rotate(90deg); }
.subagent-block[open] > summary { border-bottom: 1px solid #2a3548; color: #d8dee9;
    background: #182236; }
.subagent-inner { padding: 14px 16px 6px; }
.subagent-inner .event { background: #1a2233; border-left-style: dashed; }
.collapsible { background: #1f242e; border: 1px solid #2c313a; border-radius: 4px;
               padding: 6px 10px; margin-top: 8px; cursor: pointer; user-select: none; }
.collapsible summary { font-size: 12px; color: #a3aab8; outline: none; }
.collapsible summary::-webkit-details-marker { color: #6c7689; }
.collapsible[open] summary { color: #d8dee9; margin-bottom: 8px; }
.collapsible .inner { font-size: 13px; white-space: pre-wrap; color: #b8bdc8; }
.tool-call { margin-top: 8px; }
.tool-call .name { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-size: 12px;
                   color: #fbbf24; font-weight: 600; }
.tool-call .args { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-size: 12px;
                   color: #8c95a8; margin-left: 8px; }
.tool-call .filepath { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
                       font-size: 11px; color: #8c95a8; margin-left: 8px; font-weight: 400; }
.diff-block { border-color: #2a3548; }
.diff-block summary { color: #6c89bf; }
.diff-block .inner pre { background: #0e1218; }
.token .deleted, .token.deleted, .language-diff .deleted { color: #fb7185; }
.token .inserted, .token.inserted, .language-diff .inserted { color: #4ade80; }
.usage { display: inline-block; font-family: ui-monospace, monospace; font-size: 11px;
         color: #6c7689; margin-right: 12px; }
.usage b { color: #a3aab8; font-weight: 500; }
pre { margin: 0; padding: 8px; background: #1a1d24; border-radius: 4px; overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-size: 12.5px;
      max-height: 480px; }
code { font-family: inherit; }
.token-bar { height: 4px; background: #2c313a; margin: 2px 0; border-radius: 2px; overflow: hidden; }
.token-bar .fill { height: 100%; background: linear-gradient(90deg, #4ade80, #fbbf24); }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px;
                margin: 16px 0; }
.summary-card { background: #161922; border: 1px solid #2c313a; border-radius: 6px; padding: 12px; }
.summary-card .k { font-size: 11px; text-transform: uppercase; color: #6c7689; letter-spacing: 0.05em; }
.summary-card .v { font-size: 18px; color: #d8dee9; margin-top: 4px; font-family: ui-monospace, monospace; }
.tab-bar { display: flex; gap: 4px; border-bottom: 1px solid #2c313a; margin: 16px 0 0 0; }
.tab { padding: 8px 16px; background: #161922; border: 1px solid #2c313a; border-bottom: none;
       cursor: pointer; font-size: 13px; color: #8c95a8; border-radius: 4px 4px 0 0; }
.tab.active { background: #1b1f27; color: #d8dee9; }
.tab-pane { display: none; padding: 16px; background: #1b1f27; border: 1px solid #2c313a; border-top: none; }
.tab-pane.active { display: block; }
.incomplete-banner { background: #2b1517; border: 1px solid #ef4444; border-left: 4px solid #ef4444;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 16px; font-size: 14px; color: #fca5a5; }
.incomplete-banner b { color: #fecaca; font-weight: 600; }
"""

PRISM_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css">
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
"""

JS = """
document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => {
        const pane = t.dataset.pane;
        document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(x => x.classList.remove('active'));
        t.classList.add('active');
        document.getElementById(pane).classList.add('active');
    });
});
"""


def _esc(s: str | None) -> str:
    return html_mod.escape(s or "", quote=False)


def _truncate(s: str, n: int = 1500) -> tuple[str, bool]:
    if len(s) <= n:
        return s, False
    return s[:n] + f"\n\n... ({len(s) - n} more chars)", True


def _render_code(code: str, lang: str = "python") -> str:
    return f'<pre><code class="language-{lang}">{_esc(code)}</code></pre>'


def _maybe_pretty_json(s: str) -> str:
    """If s is a JSON object/array, return a flattened version where every
    string value with embedded newlines is materialized with real line breaks
    (one field per line, indented). Falls back to the raw string otherwise.
    """
    s = s.lstrip()
    if not s or s[0] not in "{[":
        return s
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s
    return _flatten_for_display(obj)


def _flatten_for_display(obj, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, str) and "\n" in v:
                lines.append(f'{pad}{k}:')
                for ln in v.splitlines():
                    lines.append(f'{pad}  {ln}')
            elif isinstance(v, (dict, list)):
                lines.append(f'{pad}{k}:')
                lines.append(_flatten_for_display(v, indent + 1))
            else:
                lines.append(f'{pad}{k}: {v!r}' if isinstance(v, str) else f'{pad}{k}: {v}')
        return "\n".join(lines)
    if isinstance(obj, list):
        return "\n".join(_flatten_for_display(x, indent) for x in obj)
    return f'{pad}{obj}'


# Argument keys that often hold whole files / large multiline blobs.
# When a diff is already shown, strip these from the JSON view entirely;
# otherwise hoist them out into their own code blocks so newlines render
# as real line breaks instead of literal "\n".
_HEAVY_ARG_KEYS = {
    "content", "streamContent", "new_string", "old_string",
    "file_text", "patch", "input",
}


def _guess_lang(key: str, file_path: str | None, value: str) -> str:
    p = (file_path or "").lower()
    if p.endswith((".py",)):
        return "python"
    if p.endswith((".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc")):
        return "cpp"
    if p.endswith((".js", ".ts", ".tsx", ".jsx")):
        return "javascript"
    if p.endswith((".sh", ".bash")):
        return "bash"
    if p.endswith((".md",)):
        return "markdown"
    if p.endswith((".json",)):
        return "json"
    if p.endswith((".yaml", ".yml")):
        return "yaml"
    if "diff" in key.lower() or "patch" in key.lower():
        return "diff"
    return "text"


def _render_args(args: dict, has_diff: bool) -> str:
    if not isinstance(args, dict) or not args:
        return ""

    extras: list[tuple[str, str, str]] = []  # (key, value, lang)
    cleaned: dict = {}
    file_path = args.get("file_path") or args.get("path") or args.get("filepath")

    for k, v in args.items():
        if isinstance(v, str) and "\n" in v:
            if has_diff and k in _HEAVY_ARG_KEYS:
                cleaned[k] = f"<{len(v)} chars — see diff>"
            else:
                extras.append((k, v, _guess_lang(k, file_path, v)))
                cleaned[k] = f"<{len(v)} chars — see below>"
        else:
            cleaned[k] = v

    out = []
    args_pretty = json.dumps(cleaned, indent=2, default=str)
    args_snippet, _ = _truncate(args_pretty, 600)
    out.append(
        f'<details class="collapsible"><summary>args</summary>'
        f'<div class="inner">{_render_code(args_snippet, "json")}</div></details>'
    )
    for key, value, lang in extras:
        snippet, truncated = _truncate(value, 4000)
        label = f"{key} ({len(value)} chars"
        label += " — TRUNCATED" if truncated else ""
        label += ")"
        out.append(
            f'<details class="collapsible" open><summary>{_esc(label)}</summary>'
            f'<div class="inner">{_render_code(snippet, lang)}</div></details>'
        )
    return "".join(out)


def _render_event(e: Event, idx: int) -> str:
    role = e.role
    parts: list[str] = []
    sub_attr = f' data-subtype="{e.subtype}"' if e.subtype else ""
    side_attr = ' data-sidechain="1"' if e.is_sidechain else ""
    parts.append(f'<div class="event" data-role="{role}"{sub_attr}{side_attr} id="e{idx}">')
    parts.append(f'<div class="role">{role}{f" — {e.subtype}" if e.subtype else ""}</div>')

    if e.text:
        body = _esc(e.text)
        parts.append(f'<div class="body">{body}</div>')

    if e.reasoning:
        parts.append(
            f'<details class="collapsible"><summary>reasoning ({len(e.reasoning)} chars)</summary>'
            f'<div class="inner">{_esc(e.reasoning)}</div></details>'
        )

    for tc in e.tool_calls:
        parts.append('<div class="tool-call">')
        name_label = _esc(tc.name)
        if tc.file_path:
            name_label += f' <span class="filepath">{_esc(tc.file_path)}</span>'
        parts.append(f'<span class="name">{name_label}</span>')
        if tc.diff:
            diff_snippet, diff_truncated = _truncate(tc.diff, 6000)
            label = f"diff ({tc.diff.count(chr(10))} lines"
            label += " — TRUNCATED" if diff_truncated else ""
            label += ")"
            parts.append(
                f'<details class="collapsible diff-block" open>'
                f'<summary>{_esc(label)}</summary>'
                f'<div class="inner">{_render_code(diff_snippet, "diff")}</div></details>'
            )
        parts.append(_render_args(tc.args, has_diff=bool(tc.diff)))
        parts.append('</div>')

    if e.tool_result:
        content = e.tool_result.content or ""
        # If the tool output is itself a JSON object (common in Codex
        # apply_patch / function_call_output), pretty-print it so embedded
        # newlines aren't shown as literal \n.
        content = _maybe_pretty_json(content)
        snippet, truncated = _truncate(content, 1500)
        kind = "stderr" if e.tool_result.is_error else "stdout"
        parts.append(
            f'<details class="collapsible"{" open" if len(content) < 600 else ""}>'
            f'<summary>{kind} ({len(content)} chars)'
            f'{" — TRUNCATED" if truncated else ""}</summary>'
            f'<div class="inner">{_render_code(snippet, "text")}</div></details>'
        )

    if e.usage:
        u = e.usage
        parts.append(
            f'<div class="meta">'
            f'<span class="usage">in <b>{u.input_tokens}</b></span>'
            f'<span class="usage">out <b>{u.output_tokens}</b></span>'
            f'<span class="usage">cache_r <b>{u.cache_read_tokens}</b></span>'
            f'<span class="usage">cache_w <b>{u.cache_write_tokens}</b></span>'
            f'</div>'
        )

    parts.append('</div>')
    return "\n".join(parts)


def _render_timeline(events: list[Event]) -> str:
    """Render the event list, wrapping subagent ranges in a collapsed dropdown.

    A subagent range starts at a system event with subtype=task_started and
    ends at task_notification. The boundary events themselves become part of
    the dropdown summary, not rendered as separate events.
    """
    out: list[str] = []
    i = 0
    n = len(events)
    while i < n:
        e = events[i]
        if e.role == "system" and e.subtype == "task_started":
            # Find the matching task_notification.
            j = i + 1
            while j < n and not (
                events[j].role == "system" and events[j].subtype == "task_notification"
            ):
                j += 1
            inner = events[i + 1 : j]  # exclude both boundary events
            # Pull subagent metadata from the preceding Agent tool call (if any)
            subagent_type = None
            agent_prompt = None
            for k in range(i - 1, max(-1, i - 4), -1):
                for tc in events[k].tool_calls:
                    if tc.name in {"Agent", "Task"}:
                        subagent_type = (tc.args or {}).get("subagent_type")
                        agent_prompt = (tc.args or {}).get("description") or (tc.args or {}).get("prompt", "")
                        if isinstance(agent_prompt, str):
                            agent_prompt = agent_prompt[:80]
                        break
                if subagent_type:
                    break

            tool_count = sum(len(ev.tool_calls) for ev in inner)
            label_parts = ["subagent"]
            if subagent_type:
                label_parts.append(_esc(subagent_type))
            label_parts.append(f"{len(inner)} events · {tool_count} tools")
            if agent_prompt:
                label_parts.append(_esc(agent_prompt))
            summary = " · ".join(label_parts)

            inner_html = "\n".join(_render_event(ev, i + 1 + k) for k, ev in enumerate(inner))
            out.append(
                f'<details class="subagent-block">'
                f'<summary><span>{summary}</span></summary>'
                f'<div class="subagent-inner">{inner_html}</div>'
                f'</details>'
            )
            i = j + 1  # skip past task_notification
            continue

        out.append(_render_event(e, i))
        i += 1
    return "\n".join(out)


def _read_artifact(run_dir: Path, name: str) -> str | None:
    p = run_dir / name
    if p.exists() and p.is_file():
        try:
            return p.read_text()
        except Exception:
            return None
    return None


def render(run_dir: Path, session: Session, out_path: Path | None = None) -> Path:
    """Generate index.html in run_dir and return its path."""
    run_dir = Path(run_dir)
    out = out_path or (run_dir / "index.html")

    # Extract optional artifacts
    solution = _read_artifact(run_dir, "solution.py")
    check_log = _read_artifact(run_dir, "check.log")
    bench_log = _read_artifact(run_dir, "benchmark.log")
    result_json = _read_artifact(run_dir, "result.json")
    final_text = session.final_text

    u = session.total_usage or None

    summary = [
        ("harness", session.harness),
        ("model", session.model or "?"),
        ("turns", str(session.turn_count)),
        ("tools called", str(session.tool_call_count)),
        ("events", str(len(session.events))),
    ]
    if u:
        summary.append(("input toks", f"{u.input_tokens:,}"))
        summary.append(("output toks", f"{u.output_tokens:,}"))
        summary.append(("cache hit", f"{u.cache_read_tokens:,}"))
    if session.duration_ms:
        summary.append(("duration", f"{session.duration_ms/1000:.1f}s"))

    summary_html = '<div class="summary-grid">' + "".join(
        f'<div class="summary-card"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div></div>'
        for k, v in summary
    ) + '</div>'

    # Render an INCOMPLETE banner if result.json says session_complete=false
    incomplete_banner = ""
    if result_json:
        try:
            r = json.loads(result_json)
            if r.get("session_complete") is False:
                exit_code = r.get("harness_exit_code", "?")
                reason = ("hit wall-clock budget (SIGTERM)" if exit_code == 124
                          else f"harness exited with code {exit_code}")
                incomplete_banner = (
                    '<div class="incomplete-banner">'
                    '<b>INCOMPLETE SESSION.</b> '
                    f'{_esc(reason)}. The transcript below is usable but may be '
                    'missing the agent\'s final tool calls or summary. '
                    'Don\'t score this run as a clean failure or success.'
                    '</div>'
                )
        except json.JSONDecodeError:
            pass

    events_html = _render_timeline(session.events)

    tabs: list[tuple[str, str, str]] = []  # (id, label, html)
    if solution:
        tabs.append(("tab-solution", "solution.py", _render_code(solution, "python")))
    if final_text:
        tabs.append(("tab-final", "final answer", f'<div style="white-space:pre-wrap">{_esc(final_text)}</div>'))
    if bench_log:
        tabs.append(("tab-bench", "benchmark.log", _render_code(bench_log, "text")))
    if check_log:
        tabs.append(("tab-check", "check.log", _render_code(check_log, "text")))
    if result_json:
        tabs.append(("tab-result", "result.json", _render_code(result_json, "json")))

    if tabs:
        tab_bar = '<div class="tab-bar">' + "".join(
            f'<div class="tab{" active" if i == 0 else ""}" data-pane="{tid}">{_esc(label)}</div>'
            for i, (tid, label, _) in enumerate(tabs)
        ) + '</div>'
        tab_panes = "\n".join(
            f'<div class="tab-pane{" active" if i == 0 else ""}" id="{tid}">{body}</div>'
            for i, (tid, _, body) in enumerate(tabs)
        )
    else:
        tab_bar = ""
        tab_panes = ""

    title = f"{session.harness} / {session.model or '?'}"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{_esc(title)} — KernelBench-Hard</title>
  {PRISM_HEAD}
  <style>{CSS}</style>
</head>
<body>
  <header>
    <span class="title">{_esc(title)}</span>
    <span class="meta">session: <b>{_esc(session.session_id or '?')}</b></span>
    <span class="meta">cwd: <b>{_esc(session.cwd or '?')}</b></span>
  </header>
  <div class="container">
    {incomplete_banner}
    {summary_html}
    {tab_bar}
    {tab_panes}
    <h2 style="margin-top:32px;font-size:16px;color:#a3aab8">timeline ({len(session.events)} events)</h2>
    {events_html}
  </div>
  <script>{JS}</script>
</body>
</html>
"""
    out.write_text(html_doc)
    return out
