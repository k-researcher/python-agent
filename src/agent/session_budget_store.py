"""Durable session budget state: limits snapshot and consumed usage.

This module reads and freezes the per-session budget state stored next to the
session rows. It performs no decision making (see ``agent.session_budget``) and
never commits; the caller owns the transaction and the enforcement loop.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agent.config import Settings
from agent.model_config import PricingSpec
from agent.models import Session
from agent.schemas import SessionBudgetRequest
from agent.session_budget import BudgetLimits, BudgetUsage, check_budget
from agent.usage import UsageRecord, exact_cost
from agent.usage_ledger import summarize_session_usage


class BudgetExceeded(Exception):
    """A durable session budget boundary was rejected in runtime.

    Raised to abort the current inference attempt or tool call. It is not an
    LLM error, a tool error, a retry condition or a fallback: it propagates
    unchanged to the run loop, which stops the session.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def budget_limits(settings: Settings, session: Session) -> BudgetLimits:
    """Return the session budget limits from the frozen snapshot, or the server defaults.

    The snapshot lives in ``budget_state["limits"]`` because enforced limits must not
    follow the copied ``configuration`` of a fork: a fork (and an old session) has no
    snapshot and therefore inherits the server ceilings only.
    """
    snapshot = (session.budget_state or {}).get("limits") or {}
    max_steps = snapshot.get("max_steps") or settings.max_inference_steps
    max_tokens = snapshot.get("max_tokens")
    if max_tokens is None:
        max_tokens = settings.max_session_tokens
    max_cost = _decimal_or_none(snapshot.get("max_cost"))
    if max_cost is None:
        max_cost = settings.max_session_cost
    deadline_seconds = _float_or_none(snapshot.get("deadline_seconds"))
    if deadline_seconds is None:
        deadline_seconds = settings.max_session_seconds
    return BudgetLimits(
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_cost=max_cost,
        deadline_seconds=deadline_seconds,
    )


def limits_snapshot(settings: Settings, limits: BudgetLimits) -> dict[str, Any]:
    """Serialize enforced limits to the durable snapshot dictionary."""
    return {
        "max_steps": limits.max_steps,
        "max_tokens": limits.max_tokens,
        "max_cost": str(limits.max_cost) if limits.max_cost is not None else None,
        "deadline_seconds": limits.deadline_seconds,
    }


def server_budget_limits(settings: Settings) -> BudgetLimits:
    """Return the server-default budget: ceilings as limits."""
    return BudgetLimits(
        max_steps=settings.max_inference_steps,
        max_tokens=settings.max_session_tokens,
        max_cost=settings.max_session_cost,
        deadline_seconds=settings.max_session_seconds,
    )


def resolve_budget_limits(settings: Settings, budget: SessionBudgetRequest) -> BudgetLimits:
    """Resolve a client budget against the server ceilings.

    Each requested value must not exceed its server ceiling; missing fields fall back
    to the server values. A violation raises ValueError so the API can answer 422.
    """
    if budget.max_steps is not None and budget.max_steps > settings.max_inference_steps:
        raise ValueError(
            f"max_steps must not exceed the server ceiling {settings.max_inference_steps}"
        )
    if (
        budget.max_tokens is not None
        and settings.max_session_tokens is not None
        and budget.max_tokens > settings.max_session_tokens
    ):
        raise ValueError(
            f"max_tokens must not exceed the server ceiling {settings.max_session_tokens}"
        )
    if (
        budget.max_cost is not None
        and settings.max_session_cost is not None
        and Decimal(budget.max_cost) > settings.max_session_cost
    ):
        raise ValueError(f"max_cost must not exceed the server ceiling {settings.max_session_cost}")
    if (
        budget.deadline_seconds is not None
        and settings.max_session_seconds is not None
        and budget.deadline_seconds > settings.max_session_seconds
    ):
        raise ValueError(
            f"deadline_seconds must not exceed the server ceiling {settings.max_session_seconds}"
        )
    return BudgetLimits(
        max_steps=budget.max_steps or settings.max_inference_steps,
        max_tokens=(
            budget.max_tokens if budget.max_tokens is not None else settings.max_session_tokens
        ),
        max_cost=(
            Decimal(budget.max_cost) if budget.max_cost is not None else settings.max_session_cost
        ),
        deadline_seconds=(
            budget.deadline_seconds
            if budget.deadline_seconds is not None
            else settings.max_session_seconds
        ),
    )


async def load_budget_usage(db: AsyncSession, session: Session, now: datetime) -> BudgetUsage:
    """Return the durable usage: steps and start time come from the snapshot, token
    and cost from the ledger. A state left by an older run generation (or no state)
    means a fresh start: zero steps and ``now`` as the start, while tokens and cost
    stay session-wide. A ledger error raises instead of pretending the usage is zero.
    """
    state = session.budget_state or {}
    same_run = "run_generation" in state and (state.get("run_generation") or 0) == (
        session.run_generation or 0
    )
    summary = await summarize_session_usage(db, session.id)
    return BudgetUsage(
        steps=_positive_steps(state.get("steps")) if same_run else 0,
        known_tokens=summary.known_tokens,
        known_cost=summary.known_cost,
        has_unknown_tokens=summary.has_unknown_tokens,
        has_unknown_cost=summary.has_unknown_cost,
        started_at=(_aware_utc(state.get("started_at")) if same_run else now),
    )


async def record_step(db: AsyncSession, session_id: str, generation: int) -> int:
    """Record one inference step and return the new value for that run. No commit.

    The row is locked with ``SELECT ... FOR UPDATE``, so concurrent records of the
    same session serialize and cannot lose steps. The executor holds the fence; the
    row lock is what serializes the write on PostgreSQL. When the snapshot belongs
    to an older generation the counters start again for the new run: steps reset to
    one, the start timestamp is captured, the stale stop reason is dropped and the
    limits snapshot is kept. Otherwise the step counter grows by one.
    """
    moment = datetime.now(UTC).isoformat()
    row = (
        await db.execute(
            select(Session.budget_state, Session.run_generation)
            .where(Session.id == session_id)
            .with_for_update()
        )
    ).one()
    state = dict(row.budget_state or {})
    if (state.get("run_generation") or 0) != generation:
        new_state: dict[str, Any] = {
            "run_generation": generation,
            "steps": 1,
            "started_at": moment,
        }
        if "limits" in state:
            new_state["limits"] = state["limits"]
        state = new_state
    else:
        state["steps"] = (state.get("steps") or 0) + 1
        state.setdefault("started_at", moment)
        state["run_generation"] = generation
    await db.execute(update(Session).where(Session.id == session_id).values(budget_state=state))
    steps = state.get("steps")
    return steps if isinstance(steps, int) and not isinstance(steps, bool) else 0


async def mark_budget_stop(db: AsyncSession, session_id: str, generation: int, reason: str) -> None:
    """Freeze the stop reason of the current run in the budget state. No commit.

    Omitted on an older generation: the session may run again, and the stale reason
    must not surface in the API.
    """
    row = (
        await db.execute(
            select(Session.budget_state, Session.run_generation)
            .where(Session.id == session_id)
            .with_for_update()
        )
    ).one()
    if (row.run_generation or 0) != generation:
        return
    state = dict(row.budget_state or {})
    state["stop_reason"] = reason
    state.setdefault("run_generation", generation)
    await db.execute(update(Session).where(Session.id == session_id).values(budget_state=state))


async def budget_block_reason(
    db: AsyncSession,
    settings: Settings,
    session: Session,
    now: datetime,
    *,
    new_run: bool,
) -> str | None:
    """Return the pure-policy rejection reason, or None when the session may run.

    ``new_run`` is True when the caller starts a new run generation (sending a
    message, Start): the new run receives fresh steps and a fresh start, but not
    fresh money. Decisions on an approval of the current run pass ``new_run=False``:
    they keep the live deadline but ignore the step counter, because the tools of the
    last allowed step may still run; the runtime denies the next inference.
    """
    if new_run:
        summary = await summarize_session_usage(db, session.id)
        usage = BudgetUsage(
            steps=0,
            known_tokens=summary.known_tokens,
            known_cost=summary.known_cost,
            has_unknown_tokens=summary.has_unknown_tokens,
            has_unknown_cost=summary.has_unknown_cost,
            started_at=_aware_utc(now),
        )
        return check_budget(budget_limits(settings, session), usage, now).reason
    usage = await load_budget_usage(db, session, now)
    return check_budget(budget_limits(settings, session), replace(usage, steps=0), now).reason


async def check_attempt_budget(
    db: AsyncSession,
    settings: Settings,
    session: Session,
    *,
    request_bytes: int,
    max_output_tokens: int,
    pricing: PricingSpec | None,
    now: datetime,
) -> None:
    """Raise BudgetExceeded before one HTTP attempt when the attempt cannot fit the budget.

    The decision uses the durable ledger (session-wide tokens and money) because the
    attempt audit row is written only after this check. The worst case of the attempt is
    ``request_bytes`` input tokens (an upper bound over the actual prompt) plus
    ``max_output_tokens`` output tokens; its cost is priced with ``exact_cost``. Both are
    unknown -- missing usage counts, a missing or invalid pricing -- refuse the attempt
    while a money ceiling is configured.
    """
    summary = await summarize_session_usage(db, session.id)
    limits = budget_limits(settings, session)
    usage = await load_budget_usage(db, session, now)
    decision = check_budget(limits, replace(usage, steps=0), now)
    if decision.reason is not None:
        raise BudgetExceeded(decision.reason)

    if limits.max_tokens is not None:
        if summary.has_unknown_tokens or not _valid_bytes(request_bytes):
            raise BudgetExceeded("token_usage_unknown")
        reserved = int(request_bytes + max_output_tokens)
        if summary.known_tokens > limits.max_tokens - reserved:
            raise BudgetExceeded("tokens_exhausted")

    if limits.max_cost is not None:
        amount = exact_cost(UsageRecord(request_bytes, max_output_tokens), pricing)
        if summary.has_unknown_cost or amount is None:
            raise BudgetExceeded("cost_unknown")
        if summary.known_cost > limits.max_cost - amount:
            raise BudgetExceeded("cost_exhausted")


def _valid_bytes(value: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _positive_steps(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _aware_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None
