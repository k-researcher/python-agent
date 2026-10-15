"""Adapt chat-completions payloads to the legacy function-call dialect.

The provider expects ``functions`` and ``function_call`` fields in the body.
Assistant messages use ``function_call`` instead of ``tool_calls``. Tool results
use the ``function`` message role instead of ``tool``. Responses use
``message.function_call`` instead of ``message.tool_calls``.
"""

from __future__ import annotations

import copy
import json
import uuid
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


class DialectError(ValueError):
    """The history cannot be expressed in the functions dialect without losing data."""


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
    """Convert the message list to the dialect list.

    Each call takes the first unused result with its ID that follows it. A result is
    never looked up in the whole history: an equal ID earlier or later belongs to
    another call.
    """
    tool_index = _tool_index(messages)
    result: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for position, message in enumerate(messages):
        role = message.get("role")
        if role == "tool":
            if position in consumed:
                continue
            result.append(_result_entry(message, tool_index))
            consumed.add(position)
            continue
        if role != "assistant":
            result.append(copy.deepcopy(message))
            continue
        calls = message.get("tool_calls", [])
        if not calls:
            result.append(copy.deepcopy(message))
            continue
        if len(calls) > 1:
            raise DialectError(
                "the functions dialect takes one tool call per assistant message; "
                f"this message has {len(calls)}"
            )
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
            state_id = call.get("functions_state_id")
            if isinstance(state_id, str) and state_id:
                # The provider links the function result to its state through this ID.
                entry["functions_state_id"] = state_id
            result.append(entry)
            call_id = call.get("id")
            if not call_id:
                continue
            for later in range(position + 1, len(messages)):
                tool = messages[later]
                if (
                    later not in consumed
                    and tool.get("role") == "tool"
                    and tool.get("tool_call_id") == call_id
                ):
                    result.append(_result_entry(tool, tool_index))
                    consumed.add(later)
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
    if "arguments" in function_call:
        arguments = function_call["arguments"]
        raw = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    else:
        # A stream sends the arguments in string fragments: a fragment without them adds
        # nothing. A full reply without arguments means a call with no arguments.
        raw = "" if streamed else "{}"
    # Each reply gets its own ID: two calls with the same name and arguments are different
    # calls, and each must keep its own result.
    call_id = "call_" + uuid.uuid4().hex[:24]
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
            state_id = container.get("functions_state_id")
            if isinstance(state_id, str) and state_id:
                # Keep the ID with the call: the call is what the history stores.
                container["tool_calls"][0]["functions_state_id"] = state_id
            container.pop("function_call", None)
        elif streamed and isinstance(container.get("functions_state_id"), str):
            # A late chunk can carry only the state ID of the call that started earlier.
            container["tool_calls"] = [
                {"index": 0, "functions_state_id": container["functions_state_id"]}
            ]
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
