"""Async context summary request sent to the light model (stage 3 compression)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from agent.context_summary import SUMMARY_MAX_TOKENS
from agent.llm import LLMClient, LLMError
from agent.model_config import ResolvedModel

__all__ = ["SummaryRunner", "make_summary_runner", "summary_effort"]

# Summarizes a message list into text, or returns None when the paid step is declined.
SummaryRunner = Callable[[list[dict[str, Any]]], Awaitable[str | None]]


def summary_effort(model: ResolvedModel) -> str | None:
    """Return "low" when allowed, else the first allowed effort, else None."""
    allowed = model.reasoning.allowed_efforts
    if not model.reasoning.supported or not allowed:
        return None
    if "low" in allowed:
        return "low"
    return allowed[0]


def make_summary_runner(
    model: ResolvedModel,
    client_factory: Callable[[ResolvedModel], LLMClient] = LLMClient,
) -> SummaryRunner:
    """Return a runner that sends the summary request to a dedicated client.

    The runner owns its own ``LLMClient`` built from ``model`` with
    ``max_retries`` forced to 0. A single failed summary call must decline the
    paid summarization step (deterministic fallback) instead of silently
    retrying, so the session's shared client is never reused or modified.
    """
    dedicated = replace(model, max_retries=0)

    async def run(messages: list[dict[str, Any]]) -> str | None:
        # One client per call: it is closed after the call, and parallel calls do not share it.
        client = client_factory(dedicated)
        try:
            response = await client.chat(
                messages,
                [],
                # None is valid: a model without reasoning gets no reasoning parameter.
                reasoning_effort=summary_effort(dedicated),
                max_tokens=min(SUMMARY_MAX_TOKENS, dedicated.max_tokens),
            )
        except LLMError:
            return None
        finally:
            await client.aclose()
        if response.tool_calls or response.finish_reason == "length" or not response.content:
            return None
        return response.content

    return run
