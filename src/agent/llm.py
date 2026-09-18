from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from agent.config import Settings


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    def __init__(self, settings: Settings, profile_name: str | None = None) -> None:
        self.profile_name = profile_name or settings.default_llm_profile
        self.profile = settings.resolve_llm_profile(self.profile_name)

    @property
    def endpoint(self) -> str:
        return f"{self.profile.base_url.rstrip('/')}/chat/completions"

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        if not self.profile.api_key:
            raise LLMError(f"LLM API key is not configured for profile {self.profile_name}")

        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": messages,
            "max_tokens": self.profile.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.profile.api_key}",
            "Content-Type": "application/json",
        }
        url = self.endpoint
        timeout = httpx.Timeout(self.profile.timeout_seconds, connect=15.0)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            for attempt in range(self.profile.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                    retryable = response.status_code == 429 or response.status_code >= 500
                    if retryable and attempt < self.profile.max_retries:
                        await asyncio.sleep(min(8.0, 2.0**attempt))
                        continue
                    response.raise_for_status()
                    body = response.json()
                    return self._parse_response(body)
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt >= self.profile.max_retries:
                        raise LLMError(f"LLM network error: {exc}") from exc
                    await asyncio.sleep(min(8.0, 2.0**attempt))
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text[:1000]
                    message = f"LLM returned HTTP {exc.response.status_code}: {detail}"
                    raise LLMError(message) from exc
                except (KeyError, TypeError, ValueError) as exc:
                    raise LLMError("LLM returned an invalid response") from exc

        raise LLMError("LLM request failed")

    @staticmethod
    def _parse_response(body: dict[str, Any]) -> LLMResponse:
        choice = body["choices"][0]
        message = choice.get("message", {})
        usage = body.get("usage", {})
        return LLMResponse(
            content=message.get("content"),
            tool_calls=message.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
