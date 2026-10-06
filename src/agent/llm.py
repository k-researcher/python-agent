from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

import httpx

from agent.llm_errors import (
    LLMErrorKind,
    ProviderErrorInfo,
    classify_exception,
    classify_http_error,
)
from agent.model_config import SUPPORTED_TRANSPORTS, ResolvedModel
from agent.reasoning import ReasoningDecision, wire_parameters
from agent.redaction import redact

__all__ = ["LLMClient", "LLMError", "LLMResponse", "LLMTransientError", "redact"]

# Receives ("content" | "reasoning", text) for each streamed fragment.
DeltaCallback = Callable[[str, str], Awaitable[None]]


class LLMError(RuntimeError):
    """A failed LLM request. ``info`` holds the typed classification.

    ``started`` is True when the reply stream already sent content, reasoning or a tool call.
    After that, no retry, fallback or context retry is permitted.
    """

    def __init__(
        self, message: str, *, info: ProviderErrorInfo | None = None, started: bool = False
    ) -> None:
        super().__init__(message)
        self.info = info or ProviderErrorInfo(
            LLMErrorKind.unknown, None, None, message[:500], isinstance(self, LLMTransientError)
        )
        self.started = started

    @property
    def kind(self) -> LLMErrorKind:
        return self.info.kind


class LLMTransientError(LLMError):
    """The provider is temporarily unavailable. A fallback model can take the request."""


def error_for(info: ProviderErrorInfo, message: str, *, started: bool = False) -> LLMError:
    """Return the exception class that matches the classification."""
    error = LLMTransientError if info.retryable else LLMError
    return error(message, info=info, started=started)


@dataclass(slots=True)
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    # None: the provider did not report usage. Unknown is not zero.
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning: str | None = None


class LLMClient:
    """OpenAI-compatible /chat/completions client bound to one resolved model."""

    def __init__(self, model: ResolvedModel) -> None:
        self.model = model
        self._http: httpx.AsyncClient | None = None

    @property
    def profile_name(self) -> str:
        return self.model.id

    @property
    def endpoint(self) -> str:
        return f"{self.model.base_url.rstrip('/')}/chat/completions"

    def _client(self) -> httpx.AsyncClient:
        # One pooled client per model keeps connections (and TLS sessions) alive between steps.
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.model.timeout_seconds, connect=15.0),
                follow_redirects=False,
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        reasoning_effort: str | None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model.model,
            "messages": messages,
            "max_tokens": max_tokens or self.model.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        effort = reasoning_effort or self.model.reasoning.default_effort
        if effort in self.model.reasoning.allowed_efforts:
            decision = ReasoningDecision(effort, effort, "message", "caller")
            payload.update(wire_parameters(self.model, decision, payload["max_tokens"]))
        return payload

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> LLMResponse:
        """Send one request. With ``on_delta`` the reply is streamed (SSE) and each text or
        reasoning fragment goes to the callback; the return value is the same."""
        if self.model.kind not in SUPPORTED_TRANSPORTS:
            raise LLMError(f"Provider type {self.model.kind} is not supported yet")
        if not self.model.configured:
            raise LLMError(f"Model {self.model.id} is not configured: API key is missing")

        headers = {"Content-Type": "application/json"}
        if self.model.api_key:
            headers["Authorization"] = f"Bearer {self.model.api_key}"
        payload = self._payload(messages, tools, reasoning_effort, max_tokens)
        if on_delta is not None:
            return await self._stream(headers, payload, on_delta)
        client = self._client()
        retries = self.model.max_retries

        secrets = (self.model.api_key,)
        for attempt in range(retries + 1):
            try:
                response = await client.post(self.endpoint, headers=headers, json=payload)
                retryable = response.status_code == 429 or response.status_code >= 500
                if retryable and attempt < retries:
                    await asyncio.sleep(min(8.0, 2.0**attempt))
                    continue
                response.raise_for_status()
                return self._parse_response(response.json())
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt >= retries:
                    info = _redacted(classify_exception(exc), secrets)
                    raise error_for(info, f"LLM network error: {info.message}") from exc
                await asyncio.sleep(min(8.0, 2.0**attempt))
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                info = _redacted(classify_http_error(status, exc.response.text), secrets)
                raise error_for(info, f"LLM returned HTTP {status}: {info.message}") from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMError("LLM returned an invalid response") from exc

        raise LLMTransientError("LLM request failed")

    async def _stream(
        self, headers: dict[str, str], payload: dict[str, Any], on_delta: DeltaCallback
    ) -> LLMResponse:
        """Stream one request. A retry is possible only before the first received fragment."""
        from agent.sse import ChatStreamAccumulator, SSEDecoder

        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        secrets = (self.model.api_key,)
        retries = self.model.max_retries
        for attempt in range(retries + 1):
            started = False
            try:
                async with self._client().stream(
                    "POST", self.endpoint, headers=headers, json=payload
                ) as response:
                    if response.status_code >= 400:
                        status = response.status_code
                        body = (await response.aread()).decode(errors="replace")
                        info = _redacted(classify_http_error(status, body), secrets)
                        if info.retryable and attempt < retries:
                            await asyncio.sleep(min(8.0, 2.0**attempt))
                            continue
                        raise error_for(info, f"LLM returned HTTP {status}: {info.message}")
                    decoder, accumulator = SSEDecoder(), ChatStreamAccumulator()
                    async for chunk in response.aiter_bytes():
                        for data in decoder.feed(chunk):
                            for event in accumulator.add(data):
                                if event.kind == "tool_call":
                                    started = True
                                elif event.kind in {"content", "reasoning"} and event.text:
                                    started = True
                                    await on_delta(event.kind, event.text)
                    for data in decoder.close():
                        accumulator.add(data)
                    return accumulator.result()
            except LLMError as exc:
                # An error event inside the stream: keep its classification, add the state.
                exc.started = exc.started or started
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                info = _redacted(classify_exception(exc), secrets)
                if started:
                    message = f"LLM stream was interrupted: {info.message}"
                    raise error_for(info, message, started=True) from exc
                if attempt >= retries:
                    raise error_for(info, f"LLM network error: {info.message}") from exc
                await asyncio.sleep(min(8.0, 2.0**attempt))
        raise LLMTransientError("LLM request failed")

    @staticmethod
    def _parse_response(body: dict[str, Any]) -> LLMResponse:
        choice = body["choices"][0]
        message = choice.get("message", {})
        usage = body.get("usage", {})
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        return LLMResponse(
            content=message.get("content"),
            tool_calls=message.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=token_count(usage, "prompt_tokens"),
            completion_tokens=token_count(usage, "completion_tokens"),
            reasoning=reasoning if isinstance(reasoning, str) else None,
        )


def token_count(usage: Any, name: str) -> int | None:
    """Return a reported token count, or None when it is missing or not valid."""
    value = usage.get(name) if isinstance(usage, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def _redacted(info: ProviderErrorInfo, secrets: tuple[str | None, ...]) -> ProviderErrorInfo:
    """Return the classification with the configured key removed from the message."""
    known = tuple(secret for secret in secrets if secret)
    return replace(info, message=redact(info.message, known))


def stream_error_info(error: Any) -> ProviderErrorInfo:
    """Classify an error event that arrives inside a successful SSE stream."""
    body = json.dumps({"error": error}, ensure_ascii=False, default=str)
    status = error.get("status") if isinstance(error, dict) else None
    return classify_http_error(status if isinstance(status, int) else 0, body)
