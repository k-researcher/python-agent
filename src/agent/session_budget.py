"""Pure decision function for durable session budget limits.

This module decides whether a session may proceed to its next inference or
tool boundary. It performs no I/O, makes no wall-clock calls, and touches no
storage. The caller supplies an aware ``now`` timestamp and owns all runtime,
database, and enforcement responsibilities. Enforcing a session budget is
never performed by this module alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from math import isfinite

_STEPS_EXHAUSTED = "steps_exhausted"
_TOKENS_EXHAUSTED = "tokens_exhausted"
_COST_EXHAUSTED = "cost_exhausted"
_DEADLINE_EXCEEDED = "deadline_exceeded"
_TOKEN_USAGE_UNKNOWN = "token_usage_unknown"
_COST_UNKNOWN = "cost_unknown"
_START_TIME_UNKNOWN = "start_time_unknown"


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    """Upper bounds of a session budget."""

    max_steps: int
    max_tokens: int | None = None
    max_cost: Decimal | None = None
    deadline_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    """Accumulated consumption within a session."""

    steps: int
    known_tokens: int
    known_cost: Decimal
    has_unknown_tokens: bool = False
    has_unknown_cost: bool = False
    started_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Permission for the next boundary and the stable rejection reason."""

    allowed: bool
    reason: str | None


def check_budget(limits: BudgetLimits, usage: BudgetUsage, now: datetime) -> BudgetDecision:
    """Decide whether the next inference or tool boundary is permitted."""
    _validate_limits(limits)
    _validate_usage(usage, now)

    if usage.steps >= limits.max_steps:
        return BudgetDecision(allowed=False, reason=_STEPS_EXHAUSTED)

    if limits.max_tokens is not None and usage.known_tokens >= limits.max_tokens:
        return BudgetDecision(allowed=False, reason=_TOKENS_EXHAUSTED)

    if limits.max_tokens is not None and usage.has_unknown_tokens:
        return BudgetDecision(allowed=False, reason=_TOKEN_USAGE_UNKNOWN)

    if limits.max_cost is not None and usage.known_cost >= limits.max_cost:
        return BudgetDecision(allowed=False, reason=_COST_EXHAUSTED)

    if limits.max_cost is not None and usage.has_unknown_cost:
        return BudgetDecision(allowed=False, reason=_COST_UNKNOWN)

    if limits.deadline_seconds is not None and usage.started_at is not None:
        elapsed_seconds = (now.astimezone(UTC) - usage.started_at.astimezone(UTC)).total_seconds()
        if elapsed_seconds >= limits.deadline_seconds:
            return BudgetDecision(allowed=False, reason=_DEADLINE_EXCEEDED)

    if limits.deadline_seconds is not None and usage.started_at is None:
        return BudgetDecision(allowed=False, reason=_START_TIME_UNKNOWN)

    return BudgetDecision(allowed=True, reason=None)


def _validate_limits(limits: BudgetLimits) -> None:
    _require_positive_int(limits.max_steps, "max_steps")
    if limits.max_tokens is not None:
        _require_positive_int(limits.max_tokens, "max_tokens")
    if limits.max_cost is not None:
        _require_positive_decimal(limits.max_cost, "max_cost")
    if limits.deadline_seconds is not None:
        _require_positive_finite_float(limits.deadline_seconds, "deadline_seconds")


def _validate_usage(usage: BudgetUsage, now: datetime) -> None:
    _require_nonnegative_int(usage.steps, "steps")
    _require_nonnegative_int(usage.known_tokens, "known_tokens")
    _require_nonnegative_decimal(usage.known_cost, "known_cost")
    _require_bool(usage.has_unknown_tokens, "has_unknown_tokens")
    _require_bool(usage.has_unknown_cost, "has_unknown_cost")
    _require_aware_datetime(now, "now")
    if usage.started_at is not None:
        _require_aware_datetime(usage.started_at, "started_at")
        if usage.started_at.astimezone(UTC) > now.astimezone(UTC):
            raise ValueError("started_at must not be in the future")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")


def _require_positive_decimal(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be a positive finite Decimal")


def _require_nonnegative_decimal(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite Decimal")


def _require_positive_finite_float(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (isinstance(value, float) and not isfinite(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite float or int")


def _require_aware_datetime(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} must be an aware datetime")


def _require_bool(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
