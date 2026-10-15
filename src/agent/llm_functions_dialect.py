"""Adapt chat-completions payloads to the legacy function-call dialect.

The provider expects ``functions`` and ``function_call`` fields in the body.
Assistant messages use ``function_call`` instead of ``tool_calls``. Tool results
use the ``function`` message role instead of ``tool``. Responses use
``message.function_call`` instead of ``message.tool_calls``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any


def _arguments_object(arguments: Any) -> dict[str, Any]:
    """Return the arguments as a JSON object.

    Parse the JSON string. Wrap the raw string when it is not valid JSON.
    """
    if isinstance(arguments, dict):
        return copy.deepcopy(arguments)
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            return {"_raw": arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"_raw": arguments}
    return {"_raw": str(arguments)}


def _tool_index(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map each tool-call id to its function dict from the assistant messages."""
    index: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls", []):
            call_id = call.get("id")
            if call_id:
                index[call_id] = call.get("function", {})
    return index


def _result_entry(tool: dict[str, Any], tool_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the dialect result message for one tool message."""
    call_id = tool.get("tool_call_id", "")
    function = tool_index.get(call_id)
    if function is None:
        raise ValueError(f"unknown tool_call_id {call_id!r}")
    entry: dict[str, Any] = {"role": "function"}
    if "content" in tool:
        entry["content"] = tool["content"]
    entry["name"] = function.get("name", "")
    return entry


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the message list to the dialect list."""
    tool_index = _tool_index(messages)
    result: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id", "")
            if call_id in consumed:
                continue
            result.append(_result_entry(message, tool_index))
            consumed.add(call_id)
            continue
        if role != "assistant":
            result.append(copy.deepcopy(message))
            continue
        calls = message.get("tool_calls", [])
        if not calls:
            result.append(copy.deepcopy(message))
            continue
        base: dict[str, Any] = {}
        for key, value in message.items():
            if key not in ("tool_calls", "function_call"):
                base[key] = copy.deepcopy(value)
        for call in calls:
            entry = copy.deepcopy(base)
            function = call.get("function", {})
            name = function.get("name", "")
            arguments = _arguments_object(function.get("arguments"))
            entry["function_call"] = {"name": name, "arguments": arguments}
            result.append(entry)
            call_id = call.get("id")
            if not call_id:
                continue
            for tool in messages:
                if tool.get("role") == "tool" and tool.get("tool_call_id") == call_id:
                    result.append(_result_entry(tool, tool_index))
                    consumed.add(call_id)
                    break
    return result


def to_functions_request(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a chat-completions request body to the dialect body.

    Return a copy. Keep all fields that are not converted.
    """
    converted = copy.deepcopy(body)
    if "tools" in converted:
        # The dialect takes the bare function objects, without the {"type", "function"} wrapper.
        converted["functions"] = [
            tool.get("function", tool) if isinstance(tool, dict) else tool
            for tool in converted.pop("tools")
        ]
    converted["messages"] = _convert_messages(converted["messages"])
    if "tool_choice" in converted:
        choice = converted.pop("tool_choice")
        if choice == "auto":
            converted["function_call"] = "auto"
        elif choice == "none":
            converted["function_call"] = "none"
        elif isinstance(choice, dict):
            fn = choice.get("function", {})
            converted["function_call"] = {"name": fn.get("name", "")}
    return converted


def _tool_calls_from_function_call(
    function_call: dict[str, Any], index: int, *, streamed: bool = False
) -> list[dict[str, Any]]:
    """Build the chat-completions tool_calls list from one function_call."""
    name = function_call.get("name", "")
    arguments = function_call.get("arguments", {})
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    digest = hashlib.sha256(f"{name}{raw}{index}".encode()).hexdigest()
    call_id = "call_" + digest[:12]
    call: dict[str, Any] = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }
    if streamed:
        # A streamed tool call carries its position, as in the chat-completions stream.
        call["index"] = index
    return [call]


def _convert_choice(choice: dict[str, Any]) -> dict[str, Any]:
    """Convert one response choice to the chat-completions format.

    Convert the function_call in either ``message`` or ``delta``.
    """
    converted = copy.deepcopy(choice)
    container = converted.get("message")
    streamed = container is None
    if container is None:
        container = converted.get("delta")
    if isinstance(container, dict):
        function_call = container.get("function_call")
        if isinstance(function_call, dict):
            container["tool_calls"] = _tool_calls_from_function_call(
                function_call, 0, streamed=streamed
            )
            container.pop("function_call", None)
    if converted.get("finish_reason") == "function_call":
        converted["finish_reason"] = "tool_calls"
    return converted


def from_functions_response(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a dialect response to the chat-completions format.

    Return a copy. Keep the message unchanged when there is no function_call.
    """
    converted = copy.deepcopy(data)
    if "choices" in converted:
        converted["choices"] = [_convert_choice(choice) for choice in converted["choices"]]
    return converted


def from_functions_stream_delta(chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert one dialect stream chunk to the chat-completions format."""
    converted = copy.deepcopy(chunk)
    if "choices" in converted:
        converted["choices"] = [_convert_choice(choice) for choice in converted["choices"]]
    return converted
