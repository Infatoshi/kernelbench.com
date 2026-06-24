from __future__ import annotations

import datetime as _dt
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api import (
    _estimate_cost,
    _extract_token_usage,
    _format_assistant_message,
    _format_tool_results,
    _get_model_response,
    _parse_response,
)
from src.eval.context import (
    augment_system_prompt,
    build_initial_user_message,
    extract_reference_metadata,
    inject_workspace_context,
    prepare_workspace_context,
    seed_workspace_context,
)
from src.eval.fingerprint import get_fingerprint
from src.eval.results import EvalResult, apply_benchmark_metrics, attach_solution_metadata
from src.hardware import HardwareTarget
from src.models import ModelConfig, get_provider_client
from src.parsing import extract_python_code
from src.prompts import get_reasoning_prompt, get_system_prompt
from src.tools import BLOCKED_COMMANDS, _dispatch_tool, build_gemini_tools

MAX_PROBLEM_TIME_SECONDS = {1: 300, 2: 600, 3: 900, 4: 1200}


def _get_turn_artifact_dir() -> Optional[Path]:
    raw = os.environ.get("KB_TURN_ARTIFACT_DIR")
    if not raw:
        return None
    artifact_dir = Path(raw)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _write_turn_artifact(turn: int, suffix: str, content: str) -> None:
    artifact_dir = _get_turn_artifact_dir()
    if artifact_dir is None:
        return
    try:
        path = artifact_dir / f"turn_{turn}_{suffix}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass


def _format_command_result(cmd: str, cmd_result: dict) -> str:
    return (
        f"command: {cmd}\n"
        f"return_code: {cmd_result.get('returncode')}\n"
        f"stdout:\n{cmd_result.get('stdout', '')}\n"
        f"stderr:\n{cmd_result.get('stderr', '')}\n"
    )


def _transcript_path() -> Optional[Path]:
    artifact_dir = _get_turn_artifact_dir()
    if artifact_dir is None:
        return None
    return artifact_dir / "transcript.jsonl"


def _log_event(event_type: str, payload: dict) -> None:
    path = _transcript_path()
    if path is None:
        return
    entry = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "type": event_type,
        "payload": payload,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


def _auto_submit_if_compilable(
    sandbox, submitted: bool, solution_path: Optional[str]
) -> tuple[bool, Optional[str]]:
    if submitted:
        return submitted, solution_path

    if not sandbox.file_exists("solution.py"):
        return submitted, solution_path

    compile_checks = [
        ('python -c "from solution import Model; m = Model(); print(\'OK\')"', "Model import check OK"),
        ('python -c "import solution; print(\'OK\')"', "module import check OK"),
    ]
    compile_logs: List[str] = []

    for compile_cmd, success_label in compile_checks:
        compile_result = sandbox.run_command(compile_cmd, timeout=120)
        compile_logs.append(_format_command_result(compile_cmd, compile_result))
        if compile_result["returncode"] == 0 and "OK" in compile_result["stdout"]:
            _write_turn_artifact(999, "compile.log", "\n\n".join(compile_logs))
            print(f"  AUTO-SUBMITTED: solution.py ({success_label})", flush=True)
            return True, "solution.py"

    _write_turn_artifact(999, "compile.log", "\n\n".join(compile_logs))
    return submitted, solution_path


def _begin_problem_alarm(level: int, max_seconds_by_level: Optional[Dict[int, int]] = None) -> Optional[Any]:
    timeout_map = max_seconds_by_level or MAX_PROBLEM_TIME_SECONDS
    timeout_seconds = timeout_map.get(level)
    if timeout_seconds is None or timeout_seconds <= 0:
        return None
    if not hasattr(signal, "SIGALRM"):
        return None

    def _timeout_handler(_signum, _frame):
        raise TimeoutError("Problem time limit exceeded")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    return previous_handler


def _clear_problem_alarm(previous_handler: Optional[Any]) -> None:
    if previous_handler is None:
        return
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous_handler)


def _run_gemini_agent(
    model_config: ModelConfig,
    sandbox,
    system_prompt: str,
    initial_user_message: str,
    max_turns: int,
    max_time: Optional[int] = None,
) -> tuple:
    import google.generativeai as genai

    tools = build_gemini_tools()

    model = genai.GenerativeModel(
        model_config.model_id,
        system_instruction=system_prompt,
        tools=[tools],
    )
    chat = model.start_chat()

    submitted = False
    solution_path = None
    turns_used = 0
    total_input_tokens = 0
    total_output_tokens = 0

    response = chat.send_message(initial_user_message)

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        total_input_tokens += getattr(response.usage_metadata, "prompt_token_count", 0)
        total_output_tokens += getattr(response.usage_metadata, "candidates_token_count", 0)

    agent_deadline = time.time() + (max_time or max_turns * 60)
    turn = 0
    while True:
        remaining = agent_deadline - time.time()
        if remaining <= 0:
            print(f"\n[Wall clock timeout after {turn} turns]", flush=True)
            break
        turn += 1
        turns_used = turn
        print(f"\n[Turn {turn} | {remaining / 60:.1f}min remaining]", flush=True)

        function_calls = []
        text_parts = []

        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                function_calls.append(part.function_call)
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if text_parts:
            text = " ".join(text_parts)
            text = text[:200] + "..." if len(text) > 200 else text
            print(f"Assistant: {text}", flush=True)
        _write_turn_artifact(turn, "response.txt", "\n".join(text_parts))

        if not function_calls:
            print("No tool calls - agent finished", flush=True)
            break

        function_responses = []
        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)
            print(f"  Tool: {tool_name}", flush=True)

            if tool_name == "bash":
                cmd = tool_args.get("command", "")
                print(f"    $ {cmd[:80]}..." if len(cmd) > 80 else f"    $ {cmd}", flush=True)
                if BLOCKED_COMMANDS.search(cmd):
                    output = "Error: command blocked by sandbox security policy. Do not kill processes, modify benchmark files, or alter GPU settings."
                    print("    -> BLOCKED", flush=True)
                else:
                    cmd_result = sandbox.run_command(cmd)
                    output = f"stdout:\n{cmd_result['stdout']}\nstderr:\n{cmd_result['stderr']}\nreturn_code: {cmd_result['returncode']}"
                    _write_turn_artifact(turn, "compile.log", _format_command_result(cmd, cmd_result))
                    if cmd_result["stdout"]:
                        out = cmd_result["stdout"][:150] + "..." if len(cmd_result["stdout"]) > 150 else cmd_result["stdout"]
                        print(f"    -> {out}", flush=True)
                function_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name, response={"result": output}
                        )
                    )
                )

            elif tool_name == "submit":
                solution_path = tool_args.get("solution_path", "solution.py")
                submitted = True
                print(f"  SUBMITTED: {solution_path}", flush=True)
                function_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name, response={"result": f"Submitted: {solution_path}"}
                        )
                    )
                )

            else:
                output = _dispatch_tool(tool_name, tool_args, sandbox)
                print(f"    -> {output[:150]}..." if len(output) > 150 else f"    -> {output}", flush=True)
                function_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name, response={"result": output}
                        )
                    )
                )

        if submitted:
            break

        response = chat.send_message(function_responses)

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            total_input_tokens += getattr(response.usage_metadata, "prompt_token_count", 0)
            total_output_tokens += getattr(response.usage_metadata, "candidates_token_count", 0)

    submitted, solution_path = _auto_submit_if_compilable(sandbox, submitted, solution_path)
    return submitted, solution_path, turns_used, total_input_tokens, total_output_tokens


def _run_reasoning_agent(
    model_config: ModelConfig,
    sandbox,
    system_prompt: str,
    initial_user_message: str,
    max_turns: int,
    max_time: Optional[int] = None,
) -> tuple:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    submitted = False
    solution_path = "solution.py"
    turns_used = 0
    total_input_tokens = 0
    total_output_tokens = 0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_message},
    ]

    agent_deadline = time.time() + (max_time or max_turns * 60)
    turn = 0
    while True:
        remaining = agent_deadline - time.time()
        if remaining <= 0:
            print(f"\n[Wall clock timeout after {turn} turns]", flush=True)
            break
        turn += 1
        turns_used = turn
        print(f"\n[Turn {turn} | {remaining / 60:.1f}min remaining]", flush=True)

        try:
            response = client.chat.completions.create(
                model=model_config.model_id,
                messages=messages,
                max_tokens=16384,
                extra_body={"reasoning": {"enabled": True}},
            )
        except Exception as e:
            print(f"API error: {e}", flush=True)
            break

        if hasattr(response, "usage") and response.usage:
            input_toks = getattr(response.usage, "prompt_tokens", 0)
            output_toks = getattr(response.usage, "completion_tokens", 0)
            total_input_tokens += input_toks
            total_output_tokens += output_toks

        content = response.choices[0].message.content or ""
        text_preview = content[:300] + "..." if len(content) > 300 else content
        print(f"Assistant: {text_preview}", flush=True)
        _write_turn_artifact(turn, "response.txt", content)

        messages.append({"role": "assistant", "content": content})

        code = extract_python_code(content)
        if not code:
            print("  No Python code found in response", flush=True)
            messages.append({
                "role": "user",
                "content": "I couldn't find a Python code block in your response. Please provide the complete solution.py in a ```python code block.",
            })
            continue

        print(f"  Extracted {len(code)} chars of Python code", flush=True)

        sandbox.write_file("solution.py", code)
        _write_turn_artifact(turn, "solution.py", code)

        print("  Testing compilation...", flush=True)
        compile_result = sandbox.run_command(
            'python -c "from solution import Model; m = Model(); print(\'OK\')"',
            timeout=120,
        )
        _write_turn_artifact(
            turn + 1,
            "compile.log",
            _format_command_result(
                'python -c "from solution import Model; m = Model(); print(\'OK\')"',
                compile_result,
            ),
        )

        if compile_result["returncode"] == 0 and "OK" in compile_result["stdout"]:
            print("  Compilation: OK", flush=True)
            submitted = True
            break
        else:
            error_msg = compile_result["stderr"] or compile_result["stdout"] or "Unknown error"
            if len(error_msg) > 2000:
                error_msg = error_msg[:2000] + "\n... (truncated)"
            print(f"  Compilation FAILED: {error_msg[:200]}...", flush=True)

            messages.append({
                "role": "user",
                "content": f"Your code failed to compile/import with this error:\n\n```\n{error_msg}\n```\n\nPlease fix the error and provide the corrected solution.py in a ```python code block.",
            })

    submitted, solution_path = _auto_submit_if_compilable(sandbox, submitted, solution_path)
    return submitted, solution_path, turns_used, total_input_tokens, total_output_tokens


def _finalize_agent_result(
    result: EvalResult,
    submitted: bool,
    solution_path: Optional[str],
    input_tokens: int,
    output_tokens: int,
    turns_used: int,
    model_config: ModelConfig,
    sandbox,
    hardware_target: HardwareTarget,
    level: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    judge_model_key: Optional[str] = None,
    problem_code: str = "",
) -> None:
    result.turns = turns_used
    result.submitted = submitted
    result.input_tokens = input_tokens
    result.output_tokens = output_tokens
    result.total_tokens = input_tokens + output_tokens
    result.cache_creation_tokens = cache_creation_tokens
    result.cache_read_tokens = cache_read_tokens
    result.estimated_cost_usd = _estimate_cost(
        model_config.model_id,
        model_config.provider,
        input_tokens,
        output_tokens,
        cache_creation_tokens,
        cache_read_tokens,
    )

    cache_info = ""
    if cache_creation_tokens or cache_read_tokens:
        cache_info = f" | Cache Create: {cache_creation_tokens:,} | Cache Read: {cache_read_tokens:,}"
    print(
        f"\n[Token Usage] Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {input_tokens + output_tokens:,}{cache_info}",
        flush=True,
    )
    if result.estimated_cost_usd:
        print(f"[Est. Cost] ${result.estimated_cost_usd:.4f}", flush=True)

    result.hardware_fingerprint = get_fingerprint(hardware_target, sandbox)

    if submitted and solution_path:
        print("\n" + "=" * 60, flush=True)
        print("RUNNING BENCHMARK", flush=True)
        print("=" * 60, flush=True)

        attach_solution_metadata(result, solution_path, sandbox)

        from src.eval.benchmark import run_benchmark

        benchmark_result = run_benchmark(
            sandbox,
            solution_path,
            hardware=hardware_target.gpu_sku,
            level=level,
            is_metal=hardware_target.is_metal,
        )
        if benchmark_result is not None:
            apply_benchmark_metrics(result, benchmark_result)

        if judge_model_key and result.correct and result.speedup and result.speedup > 1.0:
            print("\n" + "-" * 60, flush=True)
            print(f"JUDGE REVIEW ({judge_model_key})", flush=True)
            print("-" * 60, flush=True)

            solution_code = sandbox.read_file(solution_path.replace("/workspace/", "")) or ""
            from src.eval.judge import judge_solution

            verdict = judge_solution(
                judge_model_key=judge_model_key,
                problem_code=problem_code,
                solution_code=solution_code,
                problem_name=result.problem,
                benchmark_metrics=benchmark_result or {},
            )

            result.judge_model = judge_model_key
            result.judge_legitimate = verdict.get("legitimate", True)
            result.judge_reason = verdict.get("reason", "")

            if not result.judge_legitimate:
                print(f"JUDGE VERDICT: FAIL — {result.judge_reason}", flush=True)
                result.correct = False
                result.error = f"judge_rejected: {result.judge_reason}"
            else:
                print(f"JUDGE VERDICT: PASS — {result.judge_reason}", flush=True)
    else:
        result.error = "No solution submitted"


def run_eval(
    hardware_target: HardwareTarget,
    model_config: ModelConfig,
    problem_code: str,
    problem_name: str,
    level: int,
    max_turns: int = 20,
    max_time: Optional[int] = None,
    judge_model_key: Optional[str] = None,
) -> EvalResult:
    result = EvalResult(
        model=model_config.name,
        gpu=hardware_target.gpu_sku,
        problem=problem_name,
        level=level,
        reasoning_effort=model_config.reasoning_effort,
    )

    start_time = time.time()

    gpu_name = hardware_target.display_name
    vram = hardware_target.vram_gb

    system_prompt = get_system_prompt(
        hardware_name=hardware_target.name, gpu_name=gpu_name, vram_gb=vram,
        is_metal=hardware_target.is_metal, use_xml_tools=model_config.use_xml_tools,
    )
    system_prompt = augment_system_prompt(system_prompt, is_metal=hardware_target.is_metal)

    reference_metadata = extract_reference_metadata(problem_code)

    sandbox = hardware_target.create_sandbox(problem_code)

    alarm_handler = _begin_problem_alarm(level)
    try:
        sandbox.start()
        print(f"Sandbox started: {sandbox.get_gpu_info()}", flush=True)

        _log_event("session_start", {
            "model": model_config.name, "model_id": model_config.model_id,
            "provider": model_config.provider, "hardware": hardware_target.name,
            "gpu": hardware_target.gpu_sku, "vram_gb": hardware_target.vram_gb,
            "problem": problem_name, "level": level,
            "max_time": max_time, "max_turns": max_turns,
            "judge_model": judge_model_key,
        })

        context_bundle = prepare_workspace_context(
            hardware_name=hardware_target.name,
            gpu_name=gpu_name,
            vram_gb=vram,
            level=level,
            problem_name=problem_name,
            metadata=reference_metadata,
            sandbox=sandbox,
            is_metal=hardware_target.is_metal,
        )
        seed_workspace_context(sandbox=sandbox, context_bundle=context_bundle)
        system_prompt = inject_workspace_context(system_prompt, context_bundle)

        initial_user_message = build_initial_user_message(
            hardware_name=hardware_target.name,
            problem_name=problem_name,
            level=level,
            gpu_name=gpu_name,
            max_turns=max_turns,
            reference_code=problem_code,
            metadata=reference_metadata,
        )

        if model_config.provider == "gemini":
            submitted, solution_path, turns_used, input_tokens, output_tokens = _run_gemini_agent(
                model_config, sandbox, system_prompt, initial_user_message, max_turns, max_time=max_time
            )
            _finalize_agent_result(
                result, submitted, solution_path, input_tokens, output_tokens, turns_used,
                model_config, sandbox, hardware_target, level,
                judge_model_key=judge_model_key, problem_code=problem_code,
            )
            result.elapsed_seconds = time.time() - start_time
            return result

        if model_config.reasoning_mode:
            reasoning_prompt = get_reasoning_prompt(
                hardware_name=hardware_target.name, gpu_name=gpu_name, vram_gb=vram,
                is_metal=hardware_target.is_metal,
            )
            reasoning_prompt = augment_system_prompt(reasoning_prompt, is_metal=hardware_target.is_metal)
            reasoning_prompt = inject_workspace_context(reasoning_prompt, context_bundle)

            submitted, solution_path, turns_used, input_tokens, output_tokens = _run_reasoning_agent(
                model_config, sandbox, reasoning_prompt, initial_user_message, max_turns, max_time=max_time
            )
            _finalize_agent_result(
                result, submitted, solution_path, input_tokens, output_tokens, turns_used,
                model_config, sandbox, hardware_target, level,
                judge_model_key=judge_model_key, problem_code=problem_code,
            )
            result.elapsed_seconds = time.time() - start_time
            return result

        client = get_provider_client(model_config.provider)
        messages: list = []

        if model_config.provider == "anthropic":
            messages = [{"role": "user", "content": initial_user_message}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_user_message},
            ]

        submitted = False
        solution_path = None
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_creation_tokens = 0
        total_cache_read_tokens = 0

        _log_event("system_prompt", {"content": system_prompt})
        _log_event("user_message", {"content": initial_user_message})

        agent_deadline = time.time() + (max_time or max_turns * 60)
        turn = 0
        while True:
            remaining = agent_deadline - time.time()
            if remaining <= 0:
                print(f"\n[Wall clock timeout after {turn} turns]", flush=True)
                break

            turn += 1
            result.turns = turn
            print(f"\n[Turn {turn} | {remaining / 60:.1f}min remaining]", flush=True)

            response = None
            for _retry in range(3):
                try:
                    response = _get_model_response(client, model_config, system_prompt, messages)
                    break
                except Exception as e:
                    print(f"API error (attempt {_retry + 1}/3): {e}", flush=True)
                    if _retry == 2:
                        result.error = f"API error: {e}"
                    else:
                        time.sleep(5 * (_retry + 1))
            if response is None:
                break

            input_toks, output_toks, cache_create, cache_read = _extract_token_usage(response, model_config)
            total_input_tokens += input_toks
            total_output_tokens += output_toks
            total_cache_creation_tokens += cache_create
            total_cache_read_tokens += cache_read

            assistant_content, tool_calls = _parse_response(response, model_config)

            reasoning_content = None
            if hasattr(response, "choices") and response.choices:
                reasoning_content = getattr(response.choices[0].message, "reasoning_content", None)

            _log_event("assistant_message", {
                "turn": turn, "content": assistant_content, "tool_calls": tool_calls,
                "reasoning_content": reasoning_content,
                "input_tokens": input_toks, "output_tokens": output_toks,
                "cache_creation_tokens": cache_create, "cache_read_tokens": cache_read,
            })

            if isinstance(assistant_content, str) and assistant_content:
                text = assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content
                print(f"Assistant: {text}", flush=True)
            turn_payload = {
                "assistant_content": assistant_content,
                "tool_calls": tool_calls,
            }
            _write_turn_artifact(turn, "response.txt", json.dumps(turn_payload, indent=2, default=str))

            messages.append(_format_assistant_message(assistant_content, tool_calls, model_config))

            if not tool_calls:
                print("No tool calls - agent finished", flush=True)
                _log_event("agent_finished", {"turn": turn, "reason": "no_tool_calls"})
                break

            tool_results = []
            turn_tool_logs: List[str] = []
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_input = tc["input"]
                tool_id = tc.get("id", f"tool_{turn}")

                print(f"  Tool: {tool_name}", flush=True)

                if tool_name == "bash":
                    cmd = tool_input.get("command", "")
                    print(f"    $ {cmd[:80]}..." if len(cmd) > 80 else f"    $ {cmd}", flush=True)
                    if BLOCKED_COMMANDS.search(cmd):
                        output = "Error: command blocked by sandbox security policy. Do not kill processes, modify benchmark files, or alter GPU settings."
                        print("    -> BLOCKED", flush=True)
                    else:
                        cmd_result = sandbox.run_command(cmd)
                        output = f"stdout:\n{cmd_result['stdout']}\nstderr:\n{cmd_result['stderr']}\nreturn_code: {cmd_result['returncode']}"
                        turn_tool_logs.append(_format_command_result(cmd, cmd_result))
                        if cmd_result["stdout"]:
                            out = (
                                cmd_result["stdout"][:150] + "..."
                                if len(cmd_result["stdout"]) > 150
                                else cmd_result["stdout"]
                            )
                            print(f"    -> {out}", flush=True)
                    tool_results.append({"id": tool_id, "name": tool_name, "content": output})

                elif tool_name == "submit":
                    solution_path = tool_input.get("solution_path", "solution.py")
                    submitted = True
                    result.submitted = True
                    print(f"  SUBMITTED: {solution_path}", flush=True)
                    tool_results.append({"id": tool_id, "name": tool_name, "content": f"Submitted: {solution_path}"})

                else:
                    output = _dispatch_tool(tool_name, tool_input, sandbox)
                    print(f"    -> {output[:150]}..." if len(output) > 150 else f"    -> {output}", flush=True)
                    tool_results.append({"id": tool_id, "name": tool_name, "content": output})

            _log_event("tool_results", {
                "turn": turn,
                "results": [{"id": tr["id"], "name": tr["name"], "content": tr["content"]} for tr in tool_results],
            })

            _write_turn_artifact(turn, "compile.log", "\n\n".join(turn_tool_logs))
            if sandbox.file_exists("solution.py"):
                sol_snapshot = sandbox.read_file("solution.py") or ""
                _write_turn_artifact(turn, "solution.py", sol_snapshot)

            messages.extend(_format_tool_results(tool_results, model_config))

            if submitted:
                _log_event("agent_finished", {"turn": turn, "reason": "submitted", "solution_path": solution_path})
                break

        submitted, solution_path = _auto_submit_if_compilable(sandbox, submitted, solution_path)
        result.submitted = submitted

        _finalize_agent_result(
            result,
            submitted,
            solution_path,
            total_input_tokens,
            total_output_tokens,
            result.turns,
            model_config,
            sandbox,
            hardware_target,
            level,
            total_cache_creation_tokens,
            total_cache_read_tokens,
            judge_model_key=judge_model_key,
            problem_code=problem_code,
        )

        from dataclasses import asdict
        _log_event("session_end", asdict(result))

    except TimeoutError:
        result.error = "timeout_exceeded"
        _log_event("session_end", {"error": "timeout_exceeded"})
    except Exception as e:
        import traceback

        traceback.print_exc()
        result.error = str(e)
        _log_event("session_end", {"error": str(e)})

    finally:
        _clear_problem_alarm(alarm_handler)
        sandbox.stop()

    result.elapsed_seconds = time.time() - start_time
    return result
