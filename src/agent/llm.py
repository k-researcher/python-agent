from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from agent.model_config import SUPPORTED_TRANSPORTS, ResolvedModel
from agent.reasoning import ReasoningDecision, wire_parameters

# Receives ("content" | "reasoning", text) for each streamed fragment.
DeltaCallback = Callable[[str, str], Awaitable[None]]


class LLMError(RuntimeError):
    pass


class LLMTransientError(LLMError):
    """The provider is temporarily unavailable. A fallback model can take the request."""


# Generic credential shapes; the configured key is also removed by exact match.
_SECRET_PATTERNS = (
    re.compile(r"\b(sk|pk|rk|key)-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{8,}=*"),
)


def redact(text: str, secrets: tuple[str, ...] = ()) -> str:
    """Remove credentials from provider text before it reaches logs, audit or the UI."""
    for secret in secrets:
        if len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


@dataclass(slots=True)
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
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
                    detail = redact(str(exc), secrets)
                    raise LLMTransientError(f"LLM network error: {detail}") from exc
                await asyncio.sleep(min(8.0, 2.0**attempt))
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                detail = redact(exc.response.text[:1000], secrets)
                message = f"LLM returned HTTP {status}: {detail}"
                if status == 429 or status >= 500:
                    raise LLMTransientError(message) from exc
                raise LLMError(message) from exc
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
                        detail = redact(body[:1000], secrets)
                        retryable = status == 429 or status >= 500
                        if retryable and attempt < retries:
                            await asyncio.sleep(min(8.0, 2.0**attempt))
                            continue
                        error = LLMTransientError if retryable else LLMError
                        raise error(f"LLM returned HTTP {status}: {detail}")
                    decoder, accumulator = SSEDecoder(), ChatStreamAccumulator()
                    async for chunk in response.aiter_bytes():
                        for data in decoder.feed(chunk):
                            for event in accumulator.add(data):
                                if event.kind in {"content", "reasoning"} and event.text:
                                    started = True
                                    await on_delta(event.kind, event.text)
                    for data in decoder.close():
                        accumulator.add(data)
                    return accumulator.result()
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                detail = redact(str(exc), secrets)
                if started:
                    raise LLMTransientError(f"LLM stream was interrupted: {detail}") from exc
                if attempt >= retries:
                    raise LLMTransientError(f"LLM network error: {detail}") from exc
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
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            reasoning=reasoning if isinstance(reasoning, str) else None,
        )
