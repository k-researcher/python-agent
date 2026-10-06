from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from agent.context_summary import SUMMARY_MAX_TOKENS
from agent.llm import LLMClient, LLMError, LLMResponse, LLMTransientError
from agent.model_config import ReasoningSpec, ResolvedModel
from agent.summary_runner import SummaryRunner, make_summary_runner, summary_effort

_HISTORY = [{"role": "user", "content": "earlier work"}]


class StubClient(LLMClient):
    """Minimal chat double that records call arguments and returns a fixed result."""

    def __init__(self, model: ResolvedModel, response: LLMResponse | Exception) -> None:
        super().__init__(model)
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1
        await super().aclose()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        on_delta: Any = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "reasoning_effort": reasoning_effort,
                "max_tokens": max_tokens,
                "on_delta": on_delta,
            }
        )
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _model(
    *,
    supported: bool = True,
    allowed_efforts: list[str] | None = None,
    max_tokens: int = 16_000,
    max_retries: int = 0,
) -> ResolvedModel:
    return ResolvedModel(
        id="m",
        provider_id="p",
        kind="openai",
        base_url="https://llm.example/v1",
        api_key="sk-test",
        configured=True,
        model="m",
        context_window=32_000,
        max_tokens=max_tokens,
        timeout_seconds=10,
        max_retries=max_retries,
        reasoning=ReasoningSpec(supported=supported, allowed_efforts=allowed_efforts or []),
        pricing=None,
    )


def _response(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        prompt_tokens=10,
        completion_tokens=5,
    )


def _stub_runner(
    model: ResolvedModel, response: LLMResponse | Exception
) -> tuple[StubClient, SummaryRunner]:
    stub = StubClient(model, response)
    runner = make_summary_runner(model, client_factory=lambda _model: stub)
    return stub, runner


async def test_normal_response_returns_text() -> None:
    _, runner = _stub_runner(
        _model(allowed_efforts=["low", "medium"]),
        _response(content="summary text", finish_reason="stop"),
    )
    assert await runner(_HISTORY) == "summary text"


async def test_tool_calls_return_none_even_with_content() -> None:
    _, runner = _stub_runner(
        _model(allowed_efforts=["low"]),
        _response(
            content="summary text",
            tool_calls=[{"id": "call_1"}],
            finish_reason="tool_calls",
        ),
    )
    assert await runner(_HISTORY) is None


async def test_length_finish_returns_none() -> None:
    _, runner = _stub_runner(
        _model(allowed_efforts=["low"]),
        _response(content="partial", finish_reason="length"),
    )
    assert await runner(_HISTORY) is None


async def test_empty_content_returns_none() -> None:
    _, runner = _stub_runner(
        _model(allowed_efforts=["low"]),
        _response(content="", finish_reason="stop"),
    )
    assert await runner(_HISTORY) is None


async def test_none_content_with_empty_tool_calls_returns_none() -> None:
    _, runner = _stub_runner(
        _model(allowed_efforts=["low"]),
        _response(content=None, finish_reason="stop"),
    )
    assert await runner(_HISTORY) is None


@pytest.mark.parametrize(("supported", "efforts"), [(False, []), (True, [])])
async def test_model_without_reasoning_still_summarizes(
    supported: bool, efforts: list[str]
) -> None:
    stub, runner = _stub_runner(
        _model(supported=supported, allowed_efforts=efforts),
        _response(content="summary text"),
    )
    assert await runner(_HISTORY) == "summary text"
    assert stub.calls[0]["reasoning_effort"] is None


@pytest.mark.parametrize(
    "result", [_response(content="ok"), LLMError("boom"), asyncio.CancelledError()]
)
async def test_client_is_closed_after_each_call(result: LLMResponse | BaseException) -> None:
    stub, runner = _stub_runner(_model(allowed_efforts=["low"]), result)  # type: ignore[arg-type]
    if isinstance(result, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await runner(_HISTORY)
    else:
        await runner(_HISTORY)
    assert stub.closed == 1


async def test_shared_model_keeps_its_retries() -> None:
    model = _model(allowed_efforts=["low"], max_retries=2)
    stub, runner = _stub_runner(model, _response(content="ok"))
    await runner(_HISTORY)
    await runner(_HISTORY)
    assert model.max_retries == 2
    assert len(stub.calls) == 2


async def test_llm_error_returns_none() -> None:
    _, runner = _stub_runner(_model(allowed_efforts=["low"]), LLMError("boom"))
    assert await runner(_HISTORY) is None


async def test_llm_transient_error_returns_none() -> None:
    _, runner = _stub_runner(_model(allowed_efforts=["low"]), LLMTransientError("temporarily down"))
    assert await runner(_HISTORY) is None


async def test_cancelled_error_propagates() -> None:
    _, runner = _stub_runner(_model(allowed_efforts=["low"]), asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await runner(_HISTORY)


async def test_chat_call_arguments() -> None:
    stub, runner = _stub_runner(
        _model(allowed_efforts=["medium", "low"]),
        _response(content="summary text", finish_reason="stop"),
    )
    await runner(_HISTORY)
    call = stub.calls[0]
    assert call["messages"] == _HISTORY
    assert call["tools"] == []
    assert call["reasoning_effort"] == "low"
    assert call["max_tokens"] == SUMMARY_MAX_TOKENS
    assert call["on_delta"] is None


async def test_max_tokens_capped_by_model_budget() -> None:
    stub, runner = _stub_runner(
        _model(allowed_efforts=["low"], max_tokens=512),
        _response(content="summary text", finish_reason="stop"),
    )
    await runner(_HISTORY)
    assert stub.calls[0]["max_tokens"] == 512


async def test_no_retry_on_server_error() -> None:
    request_count = 0
    seen_models: list[ResolvedModel] = []
    clients: list[LLMClient] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(500, text="boom")

    def factory(model: ResolvedModel) -> LLMClient:
        seen_models.append(model)
        client = LLMClient(model)
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    runner = make_summary_runner(
        _model(allowed_efforts=["low"], max_retries=2), client_factory=factory
    )
    assert await runner(_HISTORY) is None
    assert request_count == 1
    assert seen_models[0].max_retries == 0
    assert clients[0]._http is None


def test_summary_effort_returns_low_when_allowed() -> None:
    assert summary_effort(_model(allowed_efforts=["medium", "low"])) == "low"


def test_summary_effort_returns_none_when_not_supported() -> None:
    assert summary_effort(_model(supported=False, allowed_efforts=[])) is None


def test_summary_effort_returns_first_allowed() -> None:
    assert summary_effort(_model(allowed_efforts=["medium", "high"])) == "medium"
