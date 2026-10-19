"""Read durable session usage without repricing historical HTTP attempts."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.models import OutboundAudit
from agent.usage import valid_token_count


@dataclass(frozen=True, slots=True)
class SessionUsageSummary:
    attempts: int
    known_tokens: int
    known_cost: Decimal
    has_unknown_tokens: bool
    has_unknown_cost: bool


async def summarize_session_usage(db: AsyncSession, session_id: str) -> SessionUsageSummary:
    """Include unfinished and failed attempts; missing values are never free usage."""
    rows = await db.execute(
        select(
            OutboundAudit.prompt_tokens,
            OutboundAudit.completion_tokens,
            OutboundAudit.cost_amount,
        ).where(OutboundAudit.session_id == session_id, OutboundAudit.category == "llm")
    )
    attempts = 0
    known_tokens = 0
    known_cost = Decimal(0)
    unknown_tokens = False
    unknown_cost = False
    with localcontext() as context:
        context.prec = 1000
        for prompt, completion, amount in rows:
            attempts += 1
            for count in (prompt, completion):
                valid = valid_token_count(count)
                if valid is None:
                    unknown_tokens = True
                else:
                    known_tokens += valid
            try:
                parsed = Decimal(amount) if amount is not None else None
            except InvalidOperation:
                parsed = None
            if parsed is None or not parsed.is_finite() or parsed < 0:
                unknown_cost = True
            else:
                known_cost += parsed
    return SessionUsageSummary(attempts, known_tokens, known_cost, unknown_tokens, unknown_cost)
