"""Read durable session usage without repricing historical HTTP attempts."""

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.model_config import PricingSpec
from agent.models import OutboundAudit
from agent.usage import UsageRecord, exact_cost, valid_token_count


@dataclass(frozen=True, slots=True)
class SessionUsageSummary:
    attempts: int
    known_tokens: int
    known_cost: Decimal
    has_unknown_tokens: bool
    has_unknown_cost: bool


def attempt_detail(max_output_tokens: int) -> str:
    """Return the audit detail of an LLM attempt: the data of its upper bound."""
    return json.dumps({"max_output_tokens": max_output_tokens})


def _attempt_bound(
    payload_bytes: Any, detail: Any, pricing_snapshot: Any
) -> tuple[int | None, Decimal | None]:
    """Return the upper bound of an attempt without usage: (tokens, cost).

    The input is at most one token per request byte; the output is at most the requested
    maximum. A part that cannot be bounded is None.
    """
    request_bytes = valid_token_count(payload_bytes)
    try:
        max_output = valid_token_count(json.loads(detail).get("max_output_tokens"))
    except (TypeError, ValueError, AttributeError):
        max_output = None
    if request_bytes is None or max_output is None:
        return None, None
    try:
        pricing = (
            PricingSpec.model_validate(pricing_snapshot) if pricing_snapshot is not None else None
        )
    except ValidationError:
        pricing = None
    return request_bytes + max_output, exact_cost(UsageRecord(request_bytes, max_output), pricing)


async def summarize_session_usage(db: AsyncSession, session_id: str) -> SessionUsageSummary:
    """Include unfinished and failed attempts; missing values are never free usage.

    An attempt that did not complete and has no usage (a timeout, a transport error, a
    cancel or a crash) counts at its upper bound when the bound is known. Otherwise its
    usage is unknown.
    """
    rows = await db.execute(
        select(
            OutboundAudit.prompt_tokens,
            OutboundAudit.completion_tokens,
            OutboundAudit.cost_amount,
            OutboundAudit.status,
            OutboundAudit.payload_bytes,
            OutboundAudit.detail,
            OutboundAudit.pricing_snapshot,
        ).where(OutboundAudit.session_id == session_id, OutboundAudit.category == "llm")
    )
    attempts = 0
    known_tokens = 0
    known_cost = Decimal(0)
    unknown_tokens = False
    unknown_cost = False
    with localcontext() as context:
        context.prec = 1000
        for prompt, completion, amount, status, payload_bytes, detail, pricing in rows:
            attempts += 1
            counts = [valid_token_count(prompt), valid_token_count(completion)]
            try:
                parsed = Decimal(amount) if amount is not None else None
            except InvalidOperation:
                parsed = None
            if parsed is not None and (not parsed.is_finite() or parsed < 0):
                parsed = None
            if status != "completed" and (None in counts or parsed is None):
                bound_tokens, bound_cost = _attempt_bound(payload_bytes, detail, pricing)
                if None in counts and bound_tokens is not None:
                    counts = [bound_tokens, 0]
                if parsed is None:
                    parsed = bound_cost
            for count in counts:
                if count is None:
                    unknown_tokens = True
                else:
                    known_tokens += count
            if parsed is None:
                unknown_cost = True
            else:
                known_cost += parsed
    return SessionUsageSummary(attempts, known_tokens, known_cost, unknown_tokens, unknown_cost)
