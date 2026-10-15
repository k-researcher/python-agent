"""Tests for the pure session_budget policy module."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from decimal import Decimal
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from agent.session_budget import (
    BudgetDecision,
    BudgetLimits,
    BudgetUsage,
    check_budget,
)

_EAST = timezone(timedelta(hours=-5))
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _default_limits() -> BudgetLimits:
    return BudgetLimits(max_steps=100)


def _default_usage() -> BudgetUsage:
    return BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"))


def _assert_allowed(decision: BudgetDecision) -> None:
    assert decision.allowed is True
    assert decision.reason is None


def _assert_rejected(decision: BudgetDecision, reason: str) -> None:
    assert decision.allowed is False
    assert decision.reason == reason


def test_steps_below_boundary_allows() -> None:
    limits = BudgetLimits(max_steps=10)
    usage = BudgetUsage(steps=9, known_tokens=0, known_cost=Decimal("0"))
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_steps_at_boundary_rejects() -> None:
    limits = BudgetLimits(max_steps=10)
    usage = BudgetUsage(steps=10, known_tokens=0, known_cost=Decimal("0"))
    _assert_rejected(check_budget(limits, usage, _NOW), "steps_exhausted")


def test_steps_above_boundary_rejects() -> None:
    limits = BudgetLimits(max_steps=10)
    usage = BudgetUsage(steps=11, known_tokens=0, known_cost=Decimal("0"))
    _assert_rejected(check_budget(limits, usage, _NOW), "steps_exhausted")


def test_tokens_below_boundary_allows() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=1000)
    usage = BudgetUsage(steps=0, known_tokens=999, known_cost=Decimal("0"))
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_tokens_at_boundary_rejects() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=1000)
    usage = BudgetUsage(steps=0, known_tokens=1000, known_cost=Decimal("0"))
    _assert_rejected(check_budget(limits, usage, _NOW), "tokens_exhausted")


def test_tokens_above_boundary_rejects() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=1000)
    usage = BudgetUsage(steps=0, known_tokens=1001, known_cost=Decimal("0"))
    _assert_rejected(check_budget(limits, usage, _NOW), "tokens_exhausted")


def test_cost_below_boundary_allows() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=Decimal("10.00"))
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("9.99"))
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_cost_at_boundary_rejects() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=Decimal("10.00"))
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("10.00"))
    _assert_rejected(check_budget(limits, usage, _NOW), "cost_exhausted")


def test_cost_above_boundary_rejects() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=Decimal("10.00"))
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("10.01"))
    _assert_rejected(check_budget(limits, usage, _NOW), "cost_exhausted")


def test_deadline_below_boundary_allows() -> None:
    started_at = datetime(2024, 6, 1, 11, 59, 1, tzinfo=UTC)
    limits = BudgetLimits(max_steps=100, deadline_seconds=60.0)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=started_at)
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_deadline_at_boundary_rejects() -> None:
    started_at = datetime(2024, 6, 1, 11, 59, 0, tzinfo=UTC)
    limits = BudgetLimits(max_steps=100, deadline_seconds=60.0)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=started_at)
    _assert_rejected(check_budget(limits, usage, _NOW), "deadline_exceeded")


def test_deadline_above_boundary_rejects() -> None:
    started_at = datetime(2024, 6, 1, 11, 58, 30, tzinfo=UTC)
    limits = BudgetLimits(max_steps=100, deadline_seconds=60.0)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=started_at)
    _assert_rejected(check_budget(limits, usage, _NOW), "deadline_exceeded")


def test_max_tokens_disabled_does_not_reject_high_tokens() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=None)
    usage = BudgetUsage(steps=0, known_tokens=999999, known_cost=Decimal("0"))
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_max_cost_disabled_does_not_reject_high_cost() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=None)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("9999.99"))
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_deadline_disabled_does_not_reject_missing_start() -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=None)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=None)
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_unknown_tokens_fail_closed_when_max_tokens_enabled() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=1000)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), has_unknown_tokens=True)
    _assert_rejected(check_budget(limits, usage, _NOW), "token_usage_unknown")


def test_unknown_tokens_ignored_when_max_tokens_disabled() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=None)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), has_unknown_tokens=True)
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_unknown_cost_fail_closed_when_max_cost_enabled() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=Decimal("10.00"))
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), has_unknown_cost=True)
    _assert_rejected(check_budget(limits, usage, _NOW), "cost_unknown")


def test_unknown_cost_ignored_when_max_cost_disabled() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=None)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), has_unknown_cost=True)
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_unknown_start_time_fail_closed_when_deadline_enabled() -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=60.0)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=None)
    _assert_rejected(check_budget(limits, usage, _NOW), "start_time_unknown")


def test_unknown_start_time_ignored_when_deadline_disabled() -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=None)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=None)
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_missing_usage_is_not_zero_for_tokens() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=1000)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), has_unknown_tokens=True)
    _assert_rejected(check_budget(limits, usage, _NOW), "token_usage_unknown")


def test_missing_usage_is_not_zero_for_cost() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=Decimal("10.00"))
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), has_unknown_cost=True)
    _assert_rejected(check_budget(limits, usage, _NOW), "cost_unknown")


def test_timezone_equivalent_instants_give_same_decision() -> None:
    started_utc = datetime(2024, 6, 1, 11, 0, 0, tzinfo=UTC)
    limits = BudgetLimits(max_steps=100, deadline_seconds=7200.0)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=started_utc)

    now_utc = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    now_east = datetime(2024, 6, 1, 7, 0, 0, tzinfo=_EAST)
    assert now_utc == now_east

    decision_utc = check_budget(limits, usage, now_utc)
    decision_east = check_budget(limits, usage, now_east)
    assert decision_utc == decision_east
    _assert_allowed(decision_utc)


def test_timezone_equivalent_instants_reject_identically() -> None:
    started_utc = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
    limits = BudgetLimits(max_steps=100, deadline_seconds=3600.0)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=started_utc)

    now_utc = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    now_east = datetime(2024, 6, 1, 7, 0, 0, tzinfo=_EAST)
    assert now_utc == now_east

    decision_utc = check_budget(limits, usage, now_utc)
    decision_east = check_budget(limits, usage, now_east)
    assert decision_utc == decision_east
    _assert_rejected(decision_utc, "deadline_exceeded")


def test_exact_deadline_boundary_is_rejected() -> None:
    started_at = datetime(2024, 6, 1, 11, 59, 0, 0, tzinfo=UTC)
    limits = BudgetLimits(max_steps=100, deadline_seconds=60.0)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=started_at)
    _assert_rejected(check_budget(limits, usage, _NOW), "deadline_exceeded")


def test_steps_precede_tokens_in_precedence() -> None:
    limits = BudgetLimits(max_steps=10, max_tokens=100)
    usage = BudgetUsage(steps=10, known_tokens=100, known_cost=Decimal("0"))
    _assert_rejected(check_budget(limits, usage, _NOW), "steps_exhausted")


def test_tokens_exhausted_precedes_unknown_tokens() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=100)
    usage = BudgetUsage(steps=0, known_tokens=100, known_cost=Decimal("0"), has_unknown_tokens=True)
    _assert_rejected(check_budget(limits, usage, _NOW), "tokens_exhausted")


def test_cost_exhausted_precedes_unknown_cost() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=Decimal("10.00"))
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("10.00"), has_unknown_cost=True)
    _assert_rejected(check_budget(limits, usage, _NOW), "cost_exhausted")


def test_naive_now_rejected() -> None:
    naive_now = datetime(2024, 6, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="now must be an aware datetime"):
        check_budget(_default_limits(), _default_usage(), naive_now)


def test_naive_started_at_rejected() -> None:
    usage = BudgetUsage(
        steps=0,
        known_tokens=0,
        known_cost=Decimal("0"),
        started_at=datetime(2024, 6, 1, 11, 0, 0),
    )
    with pytest.raises(ValueError, match="started_at must be an aware datetime"):
        check_budget(_default_limits(), usage, _NOW)


def test_negative_steps_rejected() -> None:
    usage = BudgetUsage(steps=-1, known_tokens=0, known_cost=Decimal("0"))
    with pytest.raises(ValueError, match="steps must be a nonnegative int"):
        check_budget(_default_limits(), usage, _NOW)


def test_negative_known_tokens_rejected() -> None:
    usage = BudgetUsage(steps=0, known_tokens=-1, known_cost=Decimal("0"))
    with pytest.raises(ValueError, match="known_tokens must be a nonnegative int"):
        check_budget(_default_limits(), usage, _NOW)


def test_negative_known_cost_rejected() -> None:
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("-0.01"))
    with pytest.raises(ValueError, match="known_cost must be a nonnegative finite Decimal"):
        check_budget(_default_limits(), usage, _NOW)


def test_zero_max_steps_rejected() -> None:
    limits = BudgetLimits(max_steps=0)
    with pytest.raises(ValueError, match="max_steps must be a positive int"):
        check_budget(limits, _default_usage(), _NOW)


def test_negative_max_steps_rejected() -> None:
    limits = BudgetLimits(max_steps=-1)
    with pytest.raises(ValueError, match="max_steps must be a positive int"):
        check_budget(limits, _default_usage(), _NOW)


def test_zero_max_tokens_rejected() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens must be a positive int"):
        check_budget(limits, _default_usage(), _NOW)


def test_zero_max_cost_rejected() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=Decimal("0"))
    with pytest.raises(ValueError, match="max_cost must be a positive finite Decimal"):
        check_budget(limits, _default_usage(), _NOW)


def test_zero_deadline_seconds_rejected() -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=0.0)
    with pytest.raises(ValueError, match="deadline_seconds must be a positive finite float"):
        check_budget(limits, _default_usage(), _NOW)


def test_negative_deadline_seconds_rejected() -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=-5.0)
    with pytest.raises(ValueError, match="deadline_seconds must be a positive finite float"):
        check_budget(limits, _default_usage(), _NOW)


@pytest.mark.parametrize("deadline", [3600, 3600.0, 10**400])
def test_numeric_deadline_is_accepted(deadline: float) -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=deadline)
    usage = replace(_default_usage(), started_at=_NOW - timedelta(seconds=60))
    _assert_allowed(check_budget(limits, usage, _NOW))


def test_integer_deadline_boundary_is_rejected() -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=3600)
    usage = replace(_default_usage(), started_at=_NOW - timedelta(seconds=3600))
    _assert_rejected(check_budget(limits, usage, _NOW), "deadline_exceeded")


@pytest.mark.parametrize("deadline", [True, False, 0, -1, "3600", Decimal("3600")])
def test_invalid_numeric_deadline_is_rejected(deadline: object) -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=cast(float, deadline))
    with pytest.raises(ValueError, match="deadline_seconds must be a positive finite"):
        check_budget(limits, _default_usage(), _NOW)


def test_bool_as_steps_rejected() -> None:
    usage = BudgetUsage(steps=True, known_tokens=0, known_cost=Decimal("0"))
    with pytest.raises(ValueError, match="steps must be a nonnegative int"):
        check_budget(_default_limits(), usage, _NOW)


def test_bool_as_known_tokens_rejected() -> None:
    usage = BudgetUsage(steps=0, known_tokens=True, known_cost=Decimal("0"))
    with pytest.raises(ValueError, match="known_tokens must be a nonnegative int"):
        check_budget(_default_limits(), usage, _NOW)


def test_bool_as_max_steps_rejected() -> None:
    limits = BudgetLimits(max_steps=True)
    with pytest.raises(ValueError, match="max_steps must be a positive int"):
        check_budget(limits, _default_usage(), _NOW)


def test_bool_as_max_tokens_rejected() -> None:
    limits = BudgetLimits(max_steps=100, max_tokens=True)
    with pytest.raises(ValueError, match="max_tokens must be a positive int"):
        check_budget(limits, _default_usage(), _NOW)


def test_nonfinite_max_cost_rejected() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=Decimal("Infinity"))
    with pytest.raises(ValueError, match="max_cost must be a positive finite Decimal"):
        check_budget(limits, _default_usage(), _NOW)


def test_nan_max_cost_rejected() -> None:
    limits = BudgetLimits(max_steps=100, max_cost=Decimal("NaN"))
    with pytest.raises(ValueError, match="max_cost must be a positive finite Decimal"):
        check_budget(limits, _default_usage(), _NOW)


def test_nonfinite_known_cost_rejected() -> None:
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("NaN"))
    with pytest.raises(ValueError, match="known_cost must be a nonnegative finite Decimal"):
        check_budget(_default_limits(), usage, _NOW)


def test_nonfinite_deadline_seconds_rejected() -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=float("inf"))
    with pytest.raises(ValueError, match="deadline_seconds must be a positive finite float"):
        check_budget(limits, _default_usage(), _NOW)


def test_nan_deadline_seconds_rejected() -> None:
    limits = BudgetLimits(max_steps=100, deadline_seconds=float("nan"))
    with pytest.raises(ValueError, match="deadline_seconds must be a positive finite float"):
        check_budget(limits, _default_usage(), _NOW)


def test_future_started_at_rejected() -> None:
    started_at = datetime(2024, 6, 1, 13, 0, 0, tzinfo=UTC)
    usage = BudgetUsage(steps=0, known_tokens=0, known_cost=Decimal("0"), started_at=started_at)
    with pytest.raises(ValueError, match="started_at must not be in the future"):
        check_budget(_default_limits(), usage, _NOW)


class _NoOffsetTimezone(tzinfo):
    def utcoffset(self, value: datetime | None) -> None:
        return None


def test_timezone_without_offset_is_naive() -> None:
    no_offset = _NOW.replace(tzinfo=_NoOffsetTimezone())
    with pytest.raises(ValueError, match="now must be an aware datetime"):
        check_budget(_default_limits(), _default_usage(), no_offset)
    usage = replace(_default_usage(), started_at=no_offset)
    with pytest.raises(ValueError, match="started_at must be an aware datetime"):
        check_budget(_default_limits(), usage, _NOW)


@pytest.mark.parametrize("invalid_value", [0, 1, "", "false", None])
@pytest.mark.parametrize("field_name", ["has_unknown_tokens", "has_unknown_cost"])
def test_unknown_usage_flags_require_bool(field_name: str, invalid_value: object) -> None:
    usage = replace(_default_usage(), **{field_name: cast(bool, invalid_value)})
    with pytest.raises(ValueError, match=f"{field_name} must be a bool"):
        check_budget(_default_limits(), usage, _NOW)


def test_deadline_counts_elapsed_time_during_repeated_hour() -> None:
    eastern = ZoneInfo("America/New_York")
    started_at = datetime(2024, 11, 3, 1, 30, tzinfo=eastern, fold=0)
    now = datetime(2024, 11, 3, 1, 30, tzinfo=eastern, fold=1)
    limits = BudgetLimits(max_steps=100, deadline_seconds=3600.0)
    usage = replace(_default_usage(), started_at=started_at)
    _assert_rejected(check_budget(limits, usage, now), "deadline_exceeded")


def test_deadline_does_not_count_skipped_hour() -> None:
    eastern = ZoneInfo("America/New_York")
    started_at = datetime(2024, 3, 10, 1, 30, tzinfo=eastern)
    now = datetime(2024, 3, 10, 3, 30, tzinfo=eastern)
    limits = BudgetLimits(max_steps=100, deadline_seconds=7200.0)
    usage = replace(_default_usage(), started_at=started_at)
    _assert_allowed(check_budget(limits, usage, now))


def test_future_start_is_rejected_during_repeated_hour() -> None:
    eastern = ZoneInfo("America/New_York")
    started_at = datetime(2024, 11, 3, 1, 15, tzinfo=eastern, fold=1)
    now = datetime(2024, 11, 3, 1, 45, tzinfo=eastern, fold=0)
    usage = replace(_default_usage(), started_at=started_at)
    with pytest.raises(ValueError, match="started_at must not be in the future"):
        check_budget(_default_limits(), usage, now)
