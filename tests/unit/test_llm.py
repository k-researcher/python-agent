from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from agent.llm import LLMClient, LLMError, LLMTransientError
from agent.llm_errors import LLMErrorKind
from agent.model_config import ReasoningSpec, ResolvedModel


def test_parse_chat_completion() -> None:
    response = LLMClient._parse_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path":"a"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0]["id"] == "call-1"
    assert response.prompt_tokens == 10


def _client(handler: Any) -> LLMClient:
    model = ResolvedModel(
        id="m",
        provider_id="p",
        kind="openai",
        base_url="https://llm.example/v1",
        api_key="sk-test-secret-value",
        configured=True,
        model="m",
        context_window=16_000,
        max_tokens=1_000,
        timeout_seconds=10,
        max_retries=0,
        reasoning=ReasoningSpec(),
        pricing=None,
    )
    client = LLMClient(model)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def _sse(*chunks: dict[str, Any]) -> bytes:
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)


async def _ignore(kind: str, text: str) -> None:
    return None


def test_overflow_is_typed_and_redacted() -> None:
    body = {"error": {"message": "maximum context length; key sk-test-secret-value", "code": None}}
    client = _client(lambda request: httpx.Response(400, json=body))
    with pytest.raises(LLMError) as caught:
        asyncio.run(client.chat([{"role": "user", "content": "x"}], []))
    assert caught.value.kind == LLMErrorKind.context_overflow
    assert not isinstance(caught.value, LLMTransientError)
    assert "sk-test-secret-value" not in str(caught.value)
    assert "sk-test-secret-value" not in caught.value.info.message


def test_generic_400_is_bad_request() -> None:
    body = {"error": {"message": "invalid tool schema"}}
    client = _client(lambda request: httpx.Response(400, json=body))
    with pytest.raises(LLMError) as caught:
        asyncio.run(client.chat([{"role": "user", "content": "x"}], []))
    assert caught.value.kind == LLMErrorKind.bad_request


def test_server_error_is_transient() -> None:
    client = _client(lambda request: httpx.Response(503, text="busy"))
    with pytest.raises(LLMTransientError) as caught:
        asyncio.run(client.chat([{"role": "user", "content": "x"}], []))
    assert caught.value.kind == LLMErrorKind.server
    assert caught.value.started is False


def test_stream_error_after_tool_delta_is_started() -> None:
    tool = {"index": 0, "id": "c1", "function": {"name": "read_file", "arguments": "{"}}
    content = _sse(
        {"choices": [{"delta": {"tool_calls": [tool]}}]},
        {"error": {"message": "boom"}},
    )
    client = _client(lambda request: httpx.Response(200, content=content))
    with pytest.raises(LLMError) as caught:
        asyncio.run(client.chat([{"role": "user", "content": "x"}], [], on_delta=_ignore))
    assert caught.value.started is True


def test_stream_without_usage_reports_unknown_tokens() -> None:
    content = _sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})
    client = _client(lambda request: httpx.Response(200, content=content + b"data: [DONE]\n\n"))
    response = asyncio.run(client.chat([{"role": "user", "content": "x"}], [], on_delta=_ignore))
    assert response.content == "hi"
    assert response.prompt_tokens is None
    assert response.completion_tokens is None
