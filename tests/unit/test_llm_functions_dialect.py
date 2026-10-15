"""Unit tests for the legacy function-call dialect adapter."""

from __future__ import annotations

import copy
import json

import pytest

from agent.llm_functions_dialect import (
    from_functions_response,
    from_functions_stream_delta,
    to_functions_request,
)


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"The {name} tool.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }


def _call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_simple_request_is_unchanged() -> None:
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 10,
    }
    result = to_functions_request(body)
    assert result == body
    assert result is not body


def test_tools_and_tool_choice_auto() -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_tool("search")],
        "tool_choice": "auto",
    }
    result = to_functions_request(body)
    assert "tools" not in result
    assert "tool_choice" not in result
    assert result["functions"] == [tool["function"] for tool in body["tools"]]
    assert result["function_call"] == "auto"


def test_tool_choice_none() -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_tool("search")],
        "tool_choice": "none",
    }
    result = to_functions_request(body)
    assert result["function_call"] == "none"


def test_tool_choice_specific_function() -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [_tool("search")],
        "tool_choice": {"type": "function", "function": {"name": "search"}},
    }
    result = to_functions_request(body)
    assert result["function_call"] == {"name": "search"}


def test_single_tool_call_with_result() -> None:
    messages = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [_call("c1", "search", '{"q": "x"}')]},
        _result("c1", "answer"),
    ]
    body = {"model": "m", "messages": messages, "tools": [_tool("search")]}
    result = to_functions_request(body)
    converted = result["messages"]
    assert converted[0] == messages[0]
    assistant = converted[1]
    assert assistant["role"] == "assistant"
    assert assistant["function_call"] == {"name": "search", "arguments": {"q": "x"}}
    assert "tool_calls" not in assistant
    function = converted[2]
    assert function["role"] == "function"
    assert function["name"] == "search"
    assert function["content"] == "answer"


def test_two_tool_calls_split_with_results() -> None:
    messages = [
        {"role": "user", "content": "do both"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _call("c1", "search", '{"q": "a"}'),
                _call("c2", "calc", '{"op": "sum"}'),
            ],
        },
        _result("c1", "first"),
        _result("c2", "second"),
    ]
    body = {"model": "m", "messages": messages}
    result = to_functions_request(body)
    converted = result["messages"]
    assert converted[0] == messages[0]
    assert converted[1]["role"] == "assistant"
    assert converted[1]["function_call"] == {
        "name": "search",
        "arguments": {"q": "a"},
    }
    assert converted[2] == {"role": "function", "name": "search", "content": "first"}
    assert converted[3]["role"] == "assistant"
    assert converted[3]["function_call"] == {
        "name": "calc",
        "arguments": {"op": "sum"},
    }
    assert converted[4] == {"role": "function", "name": "calc", "content": "second"}
    assert len(converted) == 5


def test_unknown_tool_call_id_raises() -> None:
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [_call("c1", "search", "{}")]},
        _result("nope", "answer"),
    ]
    body = {"model": "m", "messages": messages}
    with pytest.raises(ValueError, match="unknown tool_call_id"):
        to_functions_request(body)


def test_invalid_arguments_wrap_raw() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("c1", "search", "not json")],
        },
        _result("c1", "answer"),
    ]
    body = {"model": "m", "messages": messages}
    result = to_functions_request(body)
    assistant = result["messages"][1]
    assert assistant["function_call"]["arguments"] == {"_raw": "not json"}


def test_non_dict_arguments_wrap_raw() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("c1", "search", "42")],
        },
        _result("c1", "answer"),
    ]
    body = {"model": "m", "messages": messages}
    result = to_functions_request(body)
    assistant = result["messages"][1]
    assert assistant["function_call"]["arguments"] == {"_raw": "42"}


def test_response_with_function_call() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "search",
                        "arguments": {"q": "x"},
                    },
                    "functions_state_id": "state-1",
                },
                "finish_reason": "function_call",
            }
        ]
    }
    result = from_functions_response(data)
    choice = result["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert message["functions_state_id"] == "state-1"
    assert "function_call" not in message
    calls = message["tool_calls"]
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "search"
    parsed = json.loads(call["function"]["arguments"])
    assert parsed == {"q": "x"}
    assert call["id"].startswith("call_")
    assert len(call["id"]) == len("call_") + 12


def test_deterministic_call_id() -> None:
    function_call = {"name": "search", "arguments": {"q": "x"}}
    choice = {"message": {"function_call": function_call}, "finish_reason": "function_call"}
    data = {"choices": [choice]}
    first = from_functions_response(data)
    second = from_functions_response(data)
    assert first == second


def test_response_without_function_call_is_copy() -> None:
    data = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    result = from_functions_response(data)
    assert result == data
    assert result is not data
    assert "tool_calls" not in result["choices"][0]["message"]


def test_stream_delta_with_function_call() -> None:
    chunk = {
        "choices": [
            {
                "delta": {"function_call": {"name": "search", "arguments": {"q": "x"}}},
                "finish_reason": "function_call",
            }
        ]
    }
    result = from_functions_stream_delta(chunk)
    delta = result["choices"][0]["delta"]
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert "function_call" not in delta
    calls = delta["tool_calls"]
    assert calls[0]["function"]["name"] == "search"


def test_stream_delta_without_function_call_is_copy() -> None:
    chunk = {"choices": [{"delta": {"content": "a"}, "finish_reason": None}]}
    result = from_functions_stream_delta(chunk)
    assert result == chunk
    assert result is not chunk


def test_original_request_is_not_mutated() -> None:
    messages = [
        {"role": "user", "content": "do both"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _call("c1", "search", '{"q": "a"}'),
                _call("c2", "calc", '{"op": "sum"}'),
            ],
        },
        _result("c1", "first"),
        _result("c2", "second"),
    ]
    body = {
        "model": "m",
        "messages": messages,
        "tools": [_tool("search"), _tool("calc")],
        "tool_choice": "auto",
    }
    snapshot = copy.deepcopy(body)
    to_functions_request(body)
    assert body == snapshot


def test_original_response_is_not_mutated() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "search", "arguments": {"q": "x"}},
                },
                "finish_reason": "function_call",
            }
        ]
    }
    snapshot = copy.deepcopy(data)
    from_functions_response(data)
    assert data == snapshot


def test_tools_become_bare_function_objects() -> None:
    from agent.llm_functions_dialect import to_functions_request

    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "read_file", "description": "Read", "parameters": {}},
            }
        ],
    }
    assert to_functions_request(body)["functions"] == [
        {"name": "read_file", "description": "Read", "parameters": {}}
    ]


def test_streamed_tool_call_has_an_index() -> None:
    from agent.llm_functions_dialect import from_functions_response, from_functions_stream_delta

    call = {"name": "read_file", "arguments": {"path": "a.txt"}}
    chunk = {"choices": [{"delta": {"function_call": call}, "finish_reason": "function_call"}]}
    streamed = from_functions_stream_delta(chunk)["choices"][0]
    assert streamed["delta"]["tool_calls"][0]["index"] == 0
    assert streamed["finish_reason"] == "tool_calls"
    full = {"choices": [{"message": {"role": "assistant", "function_call": call}}]}
    assert "index" not in from_functions_response(full)["choices"][0]["message"]["tool_calls"][0]
