"""API communication, response parsing, token tracking, and cost estimation."""

import json
from typing import Optional, Tuple

from src.models import ModelConfig, get_openrouter_pricing
from src.parsing import parse_xml_tool_calls
from src.tools import TOOLS_ANTHROPIC, TOOLS_OPENAI


def _extract_token_usage(response, model_config: ModelConfig) -> tuple:
    """Returns: (input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens)"""
    input_tokens = output_tokens = cache_creation_tokens = cache_read_tokens = 0

    if model_config.provider == "anthropic":
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            cache_creation_tokens = getattr(response.usage, "cache_creation_input_tokens", 0)
            cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", 0)
    else:
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)
            details = getattr(response.usage, "prompt_tokens_details", None)
            if details:
                cache_read_tokens = getattr(details, "cached_tokens", 0)

    return input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens


def _estimate_cost(
    model_id: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> Optional[float]:
    pricing = get_openrouter_pricing(model_id)
    if pricing is None:
        return None

    input_price, output_price = pricing
    base_input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000
    cache_creation_cost = cache_creation_tokens * (input_price * 1.25) / 1_000_000
    cache_read_cost = cache_read_tokens * (input_price * 0.10) / 1_000_000

    return round(base_input_cost + output_cost + cache_creation_cost + cache_read_cost, 6)


def _get_model_response(client, model_config: ModelConfig, system_prompt: str, messages: list):
    max_tokens = model_config.max_output_tokens or 8192

    if model_config.provider == "anthropic":
        system_with_cache = [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ]
        kwargs = {
            "model": model_config.model_id,
            "max_tokens": max_tokens,
            "system": system_with_cache,
            "messages": messages,
        }
        if not model_config.use_xml_tools:
            kwargs["tools"] = TOOLS_ANTHROPIC
        return client.messages.create(**kwargs)

    elif model_config.provider == "openai":
        kwargs = {
            "model": model_config.model_id,
            "max_completion_tokens": max_tokens,
            "messages": messages,
        }
        if model_config.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": model_config.reasoning_effort}
        if not model_config.use_xml_tools:
            kwargs["tools"] = TOOLS_OPENAI
        return client.chat.completions.create(**kwargs)

    else:
        if model_config.provider == "openrouter":
            cached_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    cached_messages.append({
                        "role": "system",
                        "content": [
                            {"type": "text", "text": msg["content"], "cache_control": {"type": "ephemeral"}}
                        ],
                    })
                else:
                    cached_messages.append(msg)
            kwargs = {"model": model_config.model_id, "max_tokens": max_tokens, "messages": cached_messages}
            if model_config.provider_order:
                kwargs["extra_body"] = {
                    "provider": {"order": model_config.provider_order, "allow_fallbacks": True},
                }
        else:
            kwargs = {"model": model_config.model_id, "max_tokens": max_tokens, "messages": messages}
        if not model_config.use_xml_tools:
            kwargs["tools"] = TOOLS_OPENAI
        return client.chat.completions.create(**kwargs)


def _parse_response(response, model_config: ModelConfig) -> Tuple:
    if model_config.provider == "anthropic":
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
        if model_config.use_xml_tools and not tool_calls:
            tool_calls = parse_xml_tool_calls(content)
        return content, tool_calls

    else:
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })
        if model_config.use_xml_tools and not tool_calls and content:
            tool_calls = parse_xml_tool_calls(content)
        return content, tool_calls


def _format_assistant_message(content, tool_calls, model_config: ModelConfig) -> dict:
    if model_config.use_xml_tools:
        return {"role": "assistant", "content": content or ""}

    if model_config.provider == "anthropic":
        blocks = []
        if content:
            blocks.append({"type": "text", "text": content})
        for tc in tool_calls:
            blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]})
        return {"role": "assistant", "content": blocks}

    msg: dict = {"role": "assistant"}
    if content:
        msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = [
            {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])}}
            for tc in tool_calls
        ]
    return msg


def _format_tool_results(tool_results: list, model_config: ModelConfig) -> list:
    if model_config.use_xml_tools:
        result_text = ""
        for tr in tool_results:
            result_text += f'<tool_result name="{tr["name"]}">\n{tr["content"]}\n</tool_result>\n\n'
        return [{"role": "user", "content": result_text.strip()}]

    if model_config.provider == "anthropic":
        return [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tr["id"], "content": tr["content"]}
                for tr in tool_results
            ],
        }]

    return [
        {"role": "tool", "tool_call_id": tr["id"], "content": tr["content"]}
        for tr in tool_results
    ]
