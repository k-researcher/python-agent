"""Track LLM call token usage and derive its monetary cost."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from agent.model_config import PricingSpec

_ROUND_PLACES = Decimal("0.000001")


@dataclass(frozen=True)
class UsageRecord:
    """Token counts reported for a single LLM call."""

    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None


def usage_from_response(usage: dict[str, Any] | None) -> UsageRecord:
    """Convert an OpenAI-style usage object; missing fields stay None."""
    if usage is None:
        return UsageRecord(prompt_tokens=None, completion_tokens=None)
    completion_details = usage.get("completion_tokens_details")
    prompt_details = usage.get("prompt_tokens_details")
    reasoning = (
        completion_details.get("reasoning_tokens")
        if isinstance(completion_details, dict)
        else None
    )
    cached = prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
    return UsageRecord(
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=reasoning,
        cached_tokens=cached,
    )


def cost(usage: UsageRecord, pricing: PricingSpec | None) -> Decimal | None:
    """Return money spent on a call, or None when it cannot be computed.

    Reasoning tokens are already part of completion_tokens, so they are
    never added a second time.
    """
    if pricing is None or usage.prompt_tokens is None or usage.completion_tokens is None:
        return None
    prompt = Decimal(usage.prompt_tokens) * Decimal(str(pricing.input_per_mtok))
    completion = Decimal(usage.completion_tokens) * Decimal(str(pricing.output_per_mtok))
    return ((prompt + completion) / Decimal(1_000_000)).quantize(_ROUND_PLACES)


def total(costs: Iterable[Decimal | None]) -> tuple[Decimal, bool]:
    """Sum known costs and report whether any input value was unknown."""
    known = Decimal(0)
    has_unknown = False
    for item in costs:
        if item is None:
            has_unknown = True
        else:
            known += item
    return known, has_unknown
