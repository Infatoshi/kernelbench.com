"""Tool schemas (Anthropic/OpenAI/Gemini) and tool dispatch logic."""

import re
from typing import Any, Dict, List

BLOCKED_COMMANDS = re.compile(
    r"(?:^|\s|;|&&|\|\|)"
    r"(?:pkill|killall|kill\s+-9|kill\s+-KILL|kill\s+-SIGKILL"
    r"|rm\s+-rf\s+/(?:$|\s|;)|rm\s+-rf\s+/(?:workspace|home|usr|etc|var|opt|bin|sbin|lib|tmp)\b"
    r"|>\s*_benchmark\.py|>\s*/workspace/_benchmark"
    r"|cat\s*>\s*_benchmark"
    r"|chmod|chown"
    r"|nvidia-smi\s+--(?:reset|drain|persist)"
    r"|CUDA_VISIBLE_DEVICES=\s*$"
    r"|unset\s+CUDA"
    r")",
    re.IGNORECASE,
)

BLOCKED_WRITE_PATHS = {"_benchmark.py", "reference.py", "ENVIRONMENT.md", "BACKEND_API.md", "TASK_CONTEXT.md"}


def _dispatch_tool(tool_name: str, tool_input: dict, sandbox) -> str:
    """Execute a tool call and return the output string."""
    if tool_name == "read_file":
        path = tool_input.get("path", "")
        content = sandbox.read_file(path)
        if content is None:
            return f"Error: file not found: {path}"
        offset = tool_input.get("offset")
        limit = tool_input.get("limit")
        if offset or limit:
            lines = content.splitlines(keepends=True)
            start = max((offset or 1) - 1, 0)
            end = start + (limit or len(lines))
            return "".join(lines[start:end])
        return content

    if tool_name == "write_file":
        path = tool_input.get("path", "")
        basename = path.rsplit("/", 1)[-1] if "/" in path else path
        if basename in BLOCKED_WRITE_PATHS:
            return f"Error: cannot overwrite protected file: {path}"
        file_content = tool_input.get("content", "")
        success = sandbox.write_file(path, file_content)
        return f"Successfully wrote to {path}" if success else f"Error writing file: {path}"

    if tool_name == "edit_file":
        path = tool_input.get("path", "")
        basename = path.rsplit("/", 1)[-1] if "/" in path else path
        if basename in BLOCKED_WRITE_PATHS:
            return f"Error: cannot edit protected file: {path}"
        old_str = tool_input.get("old_str", "")
        new_str = tool_input.get("new_str", "")
        content = sandbox.read_file(path)
        if content is None:
            return f"Error: file not found: {path}"
        if old_str not in content:
            preview = content[:500] + "..." if len(content) > 500 else content
            return f"Error: old_str not found in {path}. File content:\n{preview}"
        count = content.count(old_str)
        if count > 1:
            return f"Error: old_str matches {count} locations in {path} -- must be unique"
        new_content = content.replace(old_str, new_str, 1)
        sandbox.write_file(path, new_content)
        return f"Applied edit to {path}"

    if tool_name == "bash":
        cmd = tool_input.get("command", "")
        if BLOCKED_COMMANDS.search(cmd):
            return "Error: command blocked by sandbox security policy. Do not kill processes, modify benchmark files, or alter GPU settings."
        timeout = tool_input.get("timeout")
        cmd_result = sandbox.run_command(cmd, timeout=timeout) if timeout else sandbox.run_command(cmd)
        return f"stdout:\n{cmd_result['stdout']}\nstderr:\n{cmd_result['stderr']}\nreturn_code: {cmd_result['returncode']}"

    if tool_name == "submit":
        return f"Submitted: {tool_input.get('solution_path', 'solution.py')}"

    return f"Error: unknown tool '{tool_name}'"


def _tool_schema(name: str, desc: str, props: dict, required: list) -> dict:
    return {"name": name, "description": desc, "input_schema": {"type": "object", "properties": props, "required": required}}


def _openai_tool(name: str, desc: str, props: dict, required: list) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": required}}}


_READ_DESC = "Read a file from the workspace. Returns the file contents as text."
_READ_PROPS: Dict[str, Any] = {
    "path": {"type": "string", "description": "File path (relative to /workspace/ or absolute)"},
    "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed). Optional."},
    "limit": {"type": "integer", "description": "Number of lines to read. Optional."},
}
_WRITE_DESC = "Create or overwrite a file with the given content."
_WRITE_PROPS: Dict[str, Any] = {
    "path": {"type": "string", "description": "File path (relative to /workspace/ or absolute)"},
    "content": {"type": "string", "description": "The full file content to write"},
}
_EDIT_DESC = "Edit a file by replacing a unique string with new content. The old_str must appear exactly once in the file."
_EDIT_PROPS: Dict[str, Any] = {
    "path": {"type": "string", "description": "File path to edit"},
    "old_str": {"type": "string", "description": "The exact string to find and replace (must be unique in the file)"},
    "new_str": {"type": "string", "description": "The replacement string"},
}
_BASH_DESC = "Execute a shell command in the sandbox. Use for compilation, running code, testing, and system commands."
_BASH_PROPS: Dict[str, Any] = {
    "command": {"type": "string", "description": "The shell command to execute"},
    "timeout": {"type": "integer", "description": "Timeout in seconds. Optional."},
}
_SUBMIT_DESC = "Submit your solution for benchmarking. Call this when your solution compiles and passes correctness checks."
_SUBMIT_PROPS: Dict[str, Any] = {
    "solution_path": {"type": "string", "description": "Path to solution file (default: solution.py)"},
}

TOOLS_ANTHROPIC: List[Dict[str, Any]] = [
    _tool_schema("read_file", _READ_DESC, _READ_PROPS, ["path"]),
    _tool_schema("write_file", _WRITE_DESC, _WRITE_PROPS, ["path", "content"]),
    _tool_schema("edit_file", _EDIT_DESC, _EDIT_PROPS, ["path", "old_str", "new_str"]),
    _tool_schema("bash", _BASH_DESC, _BASH_PROPS, ["command"]),
    _tool_schema("submit", _SUBMIT_DESC, _SUBMIT_PROPS, ["solution_path"]),
]

TOOLS_OPENAI: List[Dict[str, Any]] = [
    _openai_tool("read_file", _READ_DESC, _READ_PROPS, ["path"]),
    _openai_tool("write_file", _WRITE_DESC, _WRITE_PROPS, ["path", "content"]),
    _openai_tool("edit_file", _EDIT_DESC, _EDIT_PROPS, ["path", "old_str", "new_str"]),
    _openai_tool("bash", _BASH_DESC, _BASH_PROPS, ["command"]),
    _openai_tool("submit", _SUBMIT_DESC, _SUBMIT_PROPS, ["solution_path"]),
]


def build_gemini_tools():
    from google.generativeai.types import FunctionDeclaration, Tool

    decls = [
        FunctionDeclaration(name="read_file", description=_READ_DESC, parameters={"type": "object", "properties": _READ_PROPS, "required": ["path"]}),
        FunctionDeclaration(name="write_file", description=_WRITE_DESC, parameters={"type": "object", "properties": _WRITE_PROPS, "required": ["path", "content"]}),
        FunctionDeclaration(name="edit_file", description=_EDIT_DESC, parameters={"type": "object", "properties": _EDIT_PROPS, "required": ["path", "old_str", "new_str"]}),
        FunctionDeclaration(name="bash", description=_BASH_DESC, parameters={"type": "object", "properties": _BASH_PROPS, "required": ["command"]}),
        FunctionDeclaration(name="submit", description=_SUBMIT_DESC, parameters={"type": "object", "properties": _SUBMIT_PROPS, "required": ["solution_path"]}),
    ]
    return Tool(function_declarations=decls)
