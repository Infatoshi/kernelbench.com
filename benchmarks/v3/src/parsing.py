"""XML tool call parsing and Python code extraction from model responses."""

import re
from typing import Any, Dict, List, Optional


def unescape_html(text: str) -> str:
    import html
    return html.unescape(text)


def _parse_xml_tool(match: str, tool_calls: list) -> bool:
    rf_match = re.search(r'<read_file[^>]*>\s*<path>(.*?)</path>', match, re.DOTALL)
    if rf_match:
        inp: Dict[str, Any] = {"path": unescape_html(rf_match.group(1).strip())}
        offset_m = re.search(r'<offset>(.*?)</offset>', match, re.DOTALL)
        limit_m = re.search(r'<limit>(.*?)</limit>', match, re.DOTALL)
        if offset_m:
            inp["offset"] = int(offset_m.group(1).strip())
        if limit_m:
            inp["limit"] = int(limit_m.group(1).strip())
        tool_calls.append({"id": f"xml_read_file_{len(tool_calls)}", "name": "read_file", "input": inp})
        return True

    wf_match = re.search(r'<write_file[^>]*>\s*<path>(.*?)</path>\s*<content>(.*?)</content>\s*</write_file', match, re.DOTALL)
    if wf_match:
        tool_calls.append({
            "id": f"xml_write_file_{len(tool_calls)}", "name": "write_file",
            "input": {"path": unescape_html(wf_match.group(1).strip()), "content": unescape_html(wf_match.group(2))},
        })
        return True

    ef_match = re.search(r'<edit_file[^>]*>\s*<path>(.*?)</path>\s*<old_str>(.*?)</old_str>\s*<new_str>(.*?)</new_str>\s*</edit_file', match, re.DOTALL)
    if ef_match:
        tool_calls.append({
            "id": f"xml_edit_file_{len(tool_calls)}", "name": "edit_file",
            "input": {
                "path": unescape_html(ef_match.group(1).strip()),
                "old_str": unescape_html(ef_match.group(2)),
                "new_str": unescape_html(ef_match.group(3)),
            },
        })
        return True

    bash_match = re.search(r'<bash[^>]*>\s*<command>(.*?)</command>\s*</bash[^>]*>', match, re.DOTALL)
    if bash_match:
        tool_calls.append({
            "id": f"xml_bash_{len(tool_calls)}", "name": "bash",
            "input": {"command": unescape_html(bash_match.group(1).strip())},
        })
        return True

    submit_match = re.search(r'<submit[^>]*>\s*<solution_path>(.*?)</solution_path>\s*</submit[^>]*>', match, re.DOTALL)
    if submit_match:
        tool_calls.append({
            "id": f"xml_submit_{len(tool_calls)}", "name": "submit",
            "input": {"solution_path": unescape_html(submit_match.group(1).strip())},
        })
        return True

    return False


def parse_xml_tool_calls(content: str) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for match in re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
        _parse_xml_tool(match, tool_calls)
    if not tool_calls:
        _parse_xml_tool(content, tool_calls)
    return tool_calls


def extract_python_code(text: str) -> Optional[str]:
    python_blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', text, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip()

    generic_blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    for block in reversed(generic_blocks):
        if any(marker in block for marker in ["import ", "def ", "class ", "torch.", "cuda_source"]):
            return block.strip()

    solution_match = re.search(r'# solution\.py\s*\n(.*?)(?=\n#\s*\w+\.py|\Z)', text, re.DOTALL)
    if solution_match:
        return solution_match.group(1).strip()

    return None
