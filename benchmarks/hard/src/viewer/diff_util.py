"""Unified-diff helpers for rendering file edits in the viewer.

We track per-file prior content as we scan a transcript so diffs reflect
the agent's view of the file (what it last wrote), not the filesystem.
"""
from __future__ import annotations

import difflib
import re


def make_diff(path: str, old: str, new: str, context: int = 3) -> str:
    """Produce a unified diff string. Empty old means new file."""
    old_lines = old.splitlines(keepends=True) if old else []
    new_lines = new.splitlines(keepends=True) if new else []
    if old_lines == new_lines:
        return ""
    label_old = "/dev/null" if not old_lines else f"a/{path}"
    label_new = f"b/{path}"
    diff_lines = difflib.unified_diff(
        old_lines, new_lines, fromfile=label_old, tofile=label_new, n=context
    )
    return "".join(diff_lines)


def parse_codex_apply_patch(patch_text: str) -> dict[str, tuple[str, str]]:
    """Parse codex's apply_patch envelope into {path: (old, new)} pairs.

    The format looks like:
      *** Begin Patch
      *** Add File: foo.py
      +line1
      +line2
      *** Update File: bar.py
      @@
      -old
      +new
      *** Delete File: baz.py
      *** End Patch

    For Add: old="", new="<concatenated + lines>".
    For Update: we approximate by treating the lines as a diff hunk and
    rebuilding old/new from -/+/space prefixes (best-effort).
    For Delete: new="" (we don't know old from the patch alone).
    """
    out: dict[str, tuple[str, str]] = {}
    if not patch_text:
        return out
    lines = patch_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m_add = re.match(r"\*\*\* Add File:\s*(.+)$", line)
        m_upd = re.match(r"\*\*\* Update File:\s*(.+)$", line)
        m_del = re.match(r"\*\*\* Delete File:\s*(.+)$", line)
        if m_add:
            path = m_add.group(1).strip()
            i += 1
            new_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("*** "):
                if lines[i].startswith("+"):
                    new_lines.append(lines[i][1:])
                i += 1
            out[path] = ("", "\n".join(new_lines) + ("\n" if new_lines else ""))
            continue
        if m_upd:
            path = m_upd.group(1).strip()
            i += 1
            old_buf: list[str] = []
            new_buf: list[str] = []
            while i < len(lines) and not lines[i].startswith("*** "):
                ln = lines[i]
                if ln.startswith("@@"):
                    pass
                elif ln.startswith("+"):
                    new_buf.append(ln[1:])
                elif ln.startswith("-"):
                    old_buf.append(ln[1:])
                else:
                    if ln.startswith(" "):
                        ln = ln[1:]
                    old_buf.append(ln)
                    new_buf.append(ln)
                i += 1
            out[path] = ("\n".join(old_buf) + "\n", "\n".join(new_buf) + "\n")
            continue
        if m_del:
            path = m_del.group(1).strip()
            out[path] = ("<unknown prior content>\n", "")
            i += 1
            continue
        i += 1
    return out


class FileTracker:
    """Tracks the agent's last-written content per file across a transcript.

    Use to compute realistic diffs for repeated edits to the same file.
    """

    def __init__(self) -> None:
        self._state: dict[str, str] = {}

    def get(self, path: str) -> str:
        return self._state.get(path, "")

    def write(self, path: str, content: str) -> None:
        self._state[path] = content

    def diff_for_write(self, path: str, new_content: str) -> str:
        old = self.get(path)
        d = make_diff(path, old, new_content)
        self.write(path, new_content)
        return d

    def diff_for_edit(self, path: str, old_string: str, new_string: str) -> str:
        # Edit replaces a substring of the file. If we have prior content,
        # apply the substitution; otherwise show just the local change.
        prior = self.get(path)
        if prior and old_string in prior:
            new_full = prior.replace(old_string, new_string, 1)
            d = make_diff(path, prior, new_full)
            self.write(path, new_full)
            return d
        # No prior content known — show just the substring change as a tiny diff.
        return make_diff(path, old_string, new_string)
