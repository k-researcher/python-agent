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


def _is_null(schema: Any) -> bool:
    return isinstance(schema, dict) and schema.get("type") == "null"


_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _matches(value: Any, kind: Any) -> bool:
    """Return True when a default value has the JSON type of its property."""
    if value is None:
        return False
    expected = _TYPES.get(kind) if isinstance(kind, str) else None
    if expected is None:
        return True
    if isinstance(value, bool) and kind != "boolean":
        return False
    return isinstance(value, expected)


def _function_schema(schema: Any) -> Any:
    """Return a copy of a JSON schema that the functions dialect accepts.

    The provider needs one ``type`` for each property and refuses unions.
    A nullable union, for example ``anyOf: [integer, null]``, becomes its
    non-null branch. The parameter stays optional through ``required``.
    A union of several types keeps its first branch. A default value that does
    not have the property type is removed: the provider refuses it.
    """
    if isinstance(schema, list):
        return [_function_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: value for key, value in schema.items() if key not in ("anyOf", "oneOf")}
    branches = [
        branch
        for key in ("anyOf", "oneOf")
        if isinstance(schema.get(key), list)
        for branch in schema[key]
        if isinstance(branch, dict) and not _is_null(branch)
    ]
    if branches and "type" not in result:
        # The keys of the property, for example its description, have priority.
        for key, value in _function_schema(branches[0]).items():
            result.setdefault(key, value)
    types = result.get("type")
    if isinstance(types, list):
        concrete = [name for name in types if name != "null"]
        result["type"] = concrete[0] if concrete else "string"
    if "default" in result and not _matches(result["default"], result.get("type")):
        del result["default"]
    for key in ("properties", "$defs", "definitions"):
        if isinstance(result.get(key), dict):
            result[key] = {name: _function_schema(item) for name, item in result[key].items()}
    for key in ("items", "additionalProperties", "allOf"):
        if isinstance(result.get(key), (dict, list)):
            result[key] = _function_schema(result[key])
    return result


def _function(tool: Any) -> Any:
    """Return the bare function object with a schema that the dialect accepts."""
    function = tool.get("function", tool) if isinstance(tool, dict) else tool
    if isinstance(function, dict) and isinstance(function.get("parameters"), dict):
        function = {**function, "parameters": _function_schema(function["parameters"])}
    return function


def to_functions_request(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a chat-completions request body to the dialect body.

    Return a copy. Keep all fields that are not converted.
    """
    converted = copy.deepcopy(body)
    if "tools" in converted:
        # The dialect takes the bare function objects, without the {"type", "function"} wrapper.
        converted["functions"] = [_function(tool) for tool in converted.pop("tools")]
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
        # Missing fragments add nothing. Complete arguments are validated before adaptation.
        raw = ""
    # Each reply gets its own ID: two calls with the same name and arguments are different
    # calls, and each must keep its own result.
    call: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }
    if streamed:
        # A streamed tool call carries its position, as in the chat-completions stream.
        call["index"] = index
    else:
        call["id"] = "call_" + uuid.uuid4().hex[:24]
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
