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
:root {
  --bg: #111111; --bg-depth: #000000; --surface: #1a1a1a; --surface-muted: #222222;
  --surface-hover: #242424; --fg: #eeeeee; --fg-bright: #ffffff; --fg-dim: #666666;
  --fg-muted: #999999; --accent: #76b900; --accent-dim: #004831; --warn: #fbbf24;
  --bad: #fb7185; --border: #333333;
  --mono: "Roboto Mono", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: Arial, Helvetica, ui-sans-serif, system-ui, -apple-system, sans-serif;
       background: var(--bg); color: var(--fg); line-height: 1.6; font-size: 15px;
       -webkit-font-smoothing: antialiased; }
a { color: var(--accent); }
header { background: var(--bg-depth); border-bottom: 1px solid var(--border); padding: 18px 24px;
         position: sticky; top: 0; z-index: 50; }
header .crumb { font-size: 12px; color: var(--fg-muted); margin-bottom: 8px; }
header .crumb a { color: var(--accent); text-decoration: none; }
header .crumb a:hover { text-decoration: underline; }
header .title { font-weight: 600; font-size: 20px; color: var(--fg-bright); display: flex;
                align-items: center; gap: 12px; flex-wrap: wrap; }
header .title .sep { color: var(--fg-dim); font-weight: 400; }
header .title .harness { color: var(--accent); }
header .title .model { color: var(--fg-bright); font-family: var(--mono); font-size: 17px; }
header .title .problem { color: var(--fg); }
.pill { font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px;
        letter-spacing: 0.02em; }
.pill-pass { color: var(--accent); border: 1px solid var(--accent); }
.pill-fail { color: var(--bad); border: 1px solid var(--bad); }
.container { max-width: 1080px; margin: 0 auto; padding: 28px 24px 80px; }
.event { margin-bottom: 14px; border-left: 3px solid var(--border); padding: 12px 16px; background: var(--surface);
         border-radius: 6px; }
.event .role { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
               margin-bottom: 6px; color: var(--fg-dim); }
.event .body { font-size: 15px; white-space: pre-wrap; word-break: break-word; color: var(--fg); }
.event .meta { font-size: 11px; color: var(--fg-dim); margin-top: 8px; }
.event[data-role="user"] { border-left-color: #eeeeee; }
.event[data-role="user"] .role { color: #eeeeee; }
.event[data-role="assistant"] { border-left-color: #76b900; }
.event[data-role="assistant"] .role { color: #76b900; }
.event[data-role="tool"] { border-left-color: #fbbf24; }
.event[data-role="tool"] .role { color: #fbbf24; }
.event[data-role="system"] { border-left-color: #666666; opacity: 0.85; }
.event[data-role="system"] .role { color: #999999; }
.event[data-role="error"] { border-left-color: #fb7185; background: #211417; }
.event[data-role="error"] .role { color: #fb7185; }
.event[data-role="compaction"] { border-left-color: #999999; background: #1d1d1d; }
.event[data-role="compaction"] .role { color: #999999; }
/* Sidechain styling outside a subagent dropdown (e.g. true Claude Code
   isSidechain events not framed by task_started/task_notification). */
.event[data-sidechain="1"]:not(.subagent-inner .event) {
    margin-left: 32px; background: #151a10; border-left-style: dashed; position: relative; }
.event[data-sidechain="1"]:not(.subagent-inner .event)::before {
    content: "subagent"; position: absolute; margin-left: -42px;
    margin-top: 2px; font-size: 10px; color: #a3c266; letter-spacing: 0.05em; }
.event[data-role="system"][data-subtype="task_started"],
.event[data-role="system"][data-subtype="task_notification"] {
    background: #151a10; border-left-color: #4d7a00; }
.event[data-role="system"][data-subtype="task_started"] .role,
.event[data-role="system"][data-subtype="task_notification"] .role { color: #a3c266; }
.subagent-block { margin-bottom: 16px; background: #151a10; border: 1px solid #2c3a18;
    border-left: 3px solid #4d7a00; border-radius: 6px; padding: 0; overflow: hidden; }
.subagent-block > summary { padding: 14px 18px; cursor: pointer; font-size: 14px;
    color: #999999; font-weight: 500; user-select: none; list-style: none;
    display: flex; align-items: center; gap: 12px; transition: background 0.15s; }
.subagent-block > summary::-webkit-details-marker,
.subagent-block > summary::marker { display: none; }
.subagent-block > summary:hover { background: #1c2414; color: #eeeeee; }
.subagent-block > summary::before {
    content: "";
    display: inline-block;
    width: 0; height: 0;
    border-left: 9px solid #76b900;
    border-top: 6px solid transparent;
    border-bottom: 6px solid transparent;
    transition: transform 0.18s ease-out;
    flex-shrink: 0;
}
.subagent-block[open] > summary::before { transform: rotate(90deg); }
.subagent-block[open] > summary { border-bottom: 1px solid #2c3a18; color: #eeeeee;
    background: #1c2414; }
.subagent-inner { padding: 14px 16px 6px; }
.subagent-inner .event { background: #1c2414; border-left-style: dashed; }
.collapsible { background: #222222; border: 1px solid #333333; border-radius: 4px;
               padding: 6px 10px; margin-top: 8px; cursor: pointer; user-select: none; }
.collapsible summary { font-size: 12px; color: #999999; outline: none; }
.collapsible summary::-webkit-details-marker { color: #666666; }
.collapsible[open] summary { color: #eeeeee; margin-bottom: 8px; }
.collapsible .inner { font-size: 13px; white-space: pre-wrap; color: #cccccc; }
.tool-call { margin-top: 8px; }
.tool-call .name { font-family: "Roboto Mono", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
                   color: #fbbf24; font-weight: 600; }
.tool-call .args { font-family: "Roboto Mono", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
                   color: #999999; margin-left: 8px; }
.tool-call .filepath { font-family: "Roboto Mono", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
                       font-size: 11px; color: #999999; margin-left: 8px; font-weight: 400; }
.diff-block { border-color: #2c3a18; }
.diff-block summary { color: #a3c266; }
.diff-block .inner pre { background: #000000; }
.token .deleted, .token.deleted, .language-diff .deleted { color: #fb7185; }
.token .inserted, .token.inserted, .language-diff .inserted { color: #76b900; }
.usage { display: inline-block; font-family: "Roboto Mono", ui-monospace, monospace; font-size: 11px;
         color: #666666; margin-right: 12px; }
.usage b { color: #999999; font-weight: 500; }
pre { margin: 0; padding: 12px; background: #000000; border-radius: 6px; overflow-x: auto;
      font-family: var(--mono); font-size: 13.5px; line-height: 1.55; max-height: 520px; }
code { font-family: inherit; }
/* AA-style stat tiles */
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px;
             margin: 0 0 28px 0; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
             padding: 16px 18px; }
.stat-card .k { font-size: 11px; text-transform: uppercase; color: var(--fg-muted); letter-spacing: 0.06em;
                font-weight: 600; }
.stat-card .v { font-size: 26px; color: var(--fg-bright); margin-top: 6px; font-family: var(--mono);
                font-weight: 500; line-height: 1.1; }
.stat-card .v.accent { color: var(--accent); }
.stat-card .v.bad { color: var(--bad); }
.stat-card .sub { font-size: 11px; color: var(--fg-dim); margin-top: 4px; }
/* Prominent solution link (no inline code window) */
.solution-link { display: inline-flex; align-items: center; gap: 10px; margin: 0 0 28px 0;
                 padding: 14px 20px; background: var(--surface); border: 1px solid var(--accent);
                 border-radius: 10px; color: var(--accent); text-decoration: none; font-size: 15px;
                 font-weight: 600; transition: background 0.15s; }
.solution-link:hover { background: var(--surface-hover); }
.solution-link .arrow { font-size: 18px; }
.solution-link .lines { color: var(--fg-muted); font-weight: 400; font-family: var(--mono); font-size: 13px; }
/* Section headings */
.section-head { font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--fg-muted);
                font-weight: 600; margin: 36px 0 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
/* Metadata footer */
.meta-block { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
              margin-bottom: 10px; overflow: hidden; }
.meta-block > summary { padding: 12px 16px; cursor: pointer; font-size: 14px; color: var(--fg);
                        list-style: none; user-select: none; display: flex; gap: 10px; align-items: baseline; }
.meta-block > summary::-webkit-details-marker { display: none; }
.meta-block > summary:hover { background: var(--surface-hover); }
.meta-block > summary .desc { color: var(--fg-dim); font-size: 12px; font-family: var(--mono); }
.meta-block .meta-body { padding: 0 16px 14px; }
.meta-block .meta-body a { text-decoration: underline; text-underline-offset: 3px; }
.incomplete-banner { background: #211417; border: 1px solid #fb7185; border-left: 4px solid #fb7185;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 16px; font-size: 14px; color: #fb7185; }
.incomplete-banner b { color: #ffd1da; font-weight: 600; }
/* Prism token theme — matches the kernelbench.com NVIDIA-green palette. */
code[class*="language-"], pre[class*="language-"] { color: #eeeeee; background: none; text-shadow: none; }
.token.comment, .token.prolog, .token.doctype, .token.cdata { color: #666666; }
.token.punctuation { color: #999999; }
.token.operator, .token.entity, .token.url { color: #999999; }
.token.boolean, .token.number, .token.constant, .token.symbol { color: #fbbf24; }
.token.keyword, .token.atrule, .token.important { color: #76b900; }
.token.selector, .token.attr-name, .token.string, .token.char, .token.attr-value { color: #b8d97a; }
.token.function, .token.class-name, .token.builtin { color: #ffffff; }
.token.property, .token.tag, .token.regex, .token.variable { color: #a3c266; }
.token.deleted { color: #fb7185; }
.token.inserted { color: #76b900; }
"""

PRISM_HEAD = """
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
"""

JS = """
// Open the metadata <details> targeted by a URL hash (e.g. #meta-result) and
// scroll to it. Legacy #tab-solution links jump to the solution link instead.
(function () {
  const h = location.hash.slice(1);
  if (!h) return;
  if (h === 'tab-solution') {
    document.getElementById('solution-link')?.scrollIntoView({ behavior: 'smooth' });
    return;
  }
  const el = document.getElementById(h);
  if (el) { if (el.tagName === 'DETAILS') el.open = true; el.scrollIntoView({ behavior: 'smooth' }); }
})();
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


_HARNESS_LABELS = {
    "claude": "Claude Code",
    "zai-claude": "Claude Code",
    "minimax-claude": "Claude Code",
    "kimi-claude": "Claude Code",
    "deepseek-claude": "Claude Code",
    "qwen-claude": "Claude Code",
    "codex": "Codex",
    "cursor": "Cursor",
    "gemini": "Gemini CLI",
    "grok": "Grok",
    "opencode": "OpenCode",
    "droid": "Droid",
}

_PROBLEM_LABELS = {
    "01_fp8_gemm": "FP8 GEMM",
    "02_kda_cutlass": "KimiDeltaAttention CUTLASS",
    "03_paged_attention": "Paged Attention",
    "05_topk_bitonic": "TopK Bitonic",
    "06_sonic_moe_swiglu": "Sonic MoE SwiGLU",
    "07_w4a16_gemm": "W4A16 GEMM",
}


def _harness_label(h: str | None) -> str:
    return _HARNESS_LABELS.get(h or "", h or "?")


def _problem_label(p: str | None) -> str:
    return _PROBLEM_LABELS.get(p or "", (p or "").replace("_", " ") or "?")


def _fmt_dur(seconds) -> str:
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return "-"
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60}s"


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "-"


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

    r: dict = {}
    if result_json:
        try:
            r = json.loads(result_json)
        except json.JSONDecodeError:
            r = {}

    harness = _harness_label(r.get("harness") or session.harness)
    model = r.get("model") or session.model or "?"
    problem = _problem_label(r.get("problem"))
    correct = bool(r.get("correct"))
    pf = r.get("peak_fraction")
    effort = r.get("reasoning_effort") or ""

    # ---- Header status pill ----
    if correct:
        status_pill = '<span class="pill pill-pass">PASS</span>'
    else:
        status_pill = '<span class="pill pill-fail">FAIL</span>'

    # ---- AA-style stat tiles ----
    u = session.total_usage
    tiles: list[tuple[str, str, str, str]] = []  # (key, value, css_class, sub)
    if pf is not None:
        tiles.append(("peak fraction", f"{pf*100:.1f}%", "accent" if correct else "", "of hardware roofline"))
    tiles.append(("correct", "yes" if correct else "no", "accent" if correct else "bad",
                  _esc(str(r.get("failure_reason") or ""))))
    tiles.append(("turns", str(session.turn_count), "", f"{session.tool_call_count} tool calls"))
    out_tok = (u.output_tokens if u else None)
    if r.get("usage"):
        out_tok = r["usage"].get("output_tokens", out_tok)
    tiles.append(("output tokens", _fmt_int(out_tok), "", "agent generation"))
    if u and u.cache_read_tokens:
        tiles.append(("cache hits", _fmt_int(u.cache_read_tokens), "", "input tokens"))
    if r.get("elapsed_seconds") is not None:
        tiles.append(("agent time", _fmt_dur(r.get("elapsed_seconds")), "",
                      f"check {_fmt_dur(r.get('check_elapsed_seconds'))} · bench {_fmt_dur(r.get('benchmark_elapsed_seconds'))}"))
    elif session.duration_ms:
        tiles.append(("duration", f"{session.duration_ms/1000:.0f}s", "", ""))

    def _tile(k: str, v: str, cls: str, sub: str) -> str:
        sub_html = f'<div class="sub">{sub}</div>' if sub else ""
        return (f'<div class="stat-card"><div class="k">{_esc(k)}</div>'
                f'<div class="v {cls}">{_esc(v)}</div>{sub_html}</div>')

    tiles_html = '<div class="stat-grid">' + "".join(
        _tile(k, v, cls, sub) for k, v, cls, sub in tiles
    ) + '</div>'

    # ---- Incomplete banner ----
    incomplete_banner = ""
    if r.get("session_complete") is False:
        exit_code = r.get("harness_exit_code", "?")
        reason = ("hit the wall-clock budget (still iterating when time ran out)" if exit_code == 124
                  else f"harness exited with code {exit_code}")
        incomplete_banner = (
            '<div class="incomplete-banner">'
            '<b>Session ran out the clock.</b> '
            f'The agent {_esc(reason)}; the timeline below is complete up to that point. '
            'The submitted kernel was still graded normally.'
            '</div>'
        )

    # ---- Prominent solution link (raw file, no inline window) ----
    solution_html = ""
    if solution:
        raw_name = f"{out.stem}_solution.py.txt"
        (out.parent / raw_name).write_text(solution)
        n_lines = solution.count("\n") + 1
        solution_html = (
            f'<a class="solution-link" id="solution-link" '
            f'href="{html_mod.escape(raw_name, quote=True)}" target="_blank" rel="noopener">'
            f'<span>View submitted kernel</span>'
            f'<span class="lines">solution.py · {n_lines} lines</span>'
            f'<span class="arrow">&rarr;</span></a>'
        )

    events_html = _render_timeline(session.events)

    # ---- Metadata footer (collapsed) ----
    meta_blocks: list[str] = []
    if result_json:
        desc_bits = []
        if r.get("failure_reason"):
            desc_bits.append(str(r["failure_reason"]))
        if pf is not None:
            desc_bits.append(f"peak {pf:.4f}")
        desc_bits.append(f"agent {_fmt_dur(r.get('elapsed_seconds'))}")
        if out_tok is not None:
            desc_bits.append(f"{_fmt_int(out_tok)} out tok")
        desc = " · ".join(desc_bits)
        meta_blocks.append(
            f'<details class="meta-block" id="meta-result"><summary>result.json '
            f'<span class="desc">[{_esc(desc)}]</span></summary>'
            f'<div class="meta-body">{_render_code(result_json, "json")}</div></details>'
        )
    if check_log:
        verdict = "passed" if r.get("check_exit_code") == 0 else f"exit {r.get('check_exit_code', '?')}"
        meta_blocks.append(
            f'<details class="meta-block" id="meta-check"><summary>check.log '
            f'<span class="desc">[correctness {_esc(str(verdict))}]</span></summary>'
            f'<div class="meta-body">{_render_code(check_log, "text")}</div></details>'
        )
    if bench_log:
        meta_blocks.append(
            f'<details class="meta-block" id="meta-bench"><summary>benchmark.log '
            f'<span class="desc">[roofline timing]</span></summary>'
            f'<div class="meta-body">{_render_code(bench_log, "text")}</div></details>'
        )
    if final_text:
        meta_blocks.append(
            f'<details class="meta-block" id="meta-final"><summary>agent final answer</summary>'
            f'<div class="meta-body" style="white-space:pre-wrap;font-size:14px">{_esc(final_text)}</div></details>'
        )
    meta_blocks.append(
        f'<details class="meta-block" id="meta-session"><summary>session info '
        f'<span class="desc">[{_esc(session.session_id or "?")}]</span></summary>'
        f'<div class="meta-body" style="font-size:13px;color:var(--fg-muted)">'
        f'run_id: {_esc(r.get("run_id") or out.stem)}<br>'
        f'harness: {_esc(r.get("harness") or session.harness)}<br>'
        f'model: {_esc(model)}<br>'
        f'reasoning effort: {_esc(effort or "default")}<br>'
        f'cwd: {_esc(session.cwd or "?")}<br>'
        f'events: {len(session.events)}</div></details>'
    )
    meta_html = (
        '<div class="section-head">run metadata &amp; logs</div>' + "".join(meta_blocks)
    )

    eff_str = f" [{effort}]" if effort else ""
    title = f"{harness} / {model} — {problem}"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(title)} — KernelBench-Hard</title>
  {PRISM_HEAD}
  <style>{CSS}</style>
</head>
<body>
  <header>
    <div class="crumb"><a href="/hard">&larr; KernelBench-Hard</a></div>
    <div class="title">
      <span class="harness">{_esc(harness)}</span>
      <span class="sep">/</span>
      <span class="model">{_esc(model)}{_esc(eff_str)}</span>
      <span class="sep">·</span>
      <span class="problem">{_esc(problem)}</span>
      {status_pill}
    </div>
  </header>
  <div class="container">
    {incomplete_banner}
    {tiles_html}
    {solution_html}
    <div class="section-head">agent timeline &middot; {len(session.events)} events</div>
    {events_html}
    {meta_html}
  </div>
  <script>{JS}</script>
</body>
</html>
"""
    out.write_text(html_doc)
    return out
